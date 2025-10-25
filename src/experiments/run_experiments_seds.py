from __future__ import annotations
import argparse, json, time, sys, pickle
from pathlib import Path
import numpy as np
# NOTE: JAX/Equinox imports removed as they are not needed for SEDS

# --- Add project root to path ---
# This allows finding `SEDS.seds_core` and `src.data.lasa`
_SCRIPT_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _SCRIPT_DIR.parent.parent
if str(_ROOT_DIR) not in sys.path:
    print(f"Could not find SEDS module. Adding project root to path: {_ROOT_DIR}")
    sys.path.append(str(_ROOT_DIR))
# --- End Path Hack ---

try:
    from SEDS.seds_core import SEDS
except ImportError as e:
    print(f"Error: {e}. Could not import SEDS.")
    print("Please ensure SEDS/seds_core.py exists and the project root is in PYTHONPATH.")
    sys.exit(1)

from src.data.lasa import load_shape, resample
from src.experiments.targets import TargetDTW, TargetLeastEffort
from src.experiments.robust_ctrl import L1Adaptive
from src.experiments.disturbances import big_mid_pulse, two_mid_pulses # direct disturbance (no_llc only)
# IMPORTANT: Assumes simulate function exists and works as expected from the provided file
from src.experiments.simulator import SimConfig, simulate
from src.experiments.metrics_plots import dtw_distance, plot_all_together_with_dist, bar_with_ci


# ------------------------ tiny logger ------------------------
def _now(): return time.strftime("%H:%M:%S")
def log(msg: str, *, enabled: bool = True):
    if enabled: print(f"[{_now()}] {msg}", flush=True)


# ------------------------ helpers ------------------------
def rollout_seds_reference(model: SEDS, z0_model_coords: np.ndarray, t_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Rollout SEDS reference using the model's internal simulator, starting
    from an initial position defined in the model's coordinate frame (target at origin).
    Returns both position and the *predicted* velocity in model coordinates.
    Ensures output arrays match the length of t_grid.
    """
    t = np.asarray(t_grid).reshape(-1)
    output_len = len(t)
    if output_len < 2:
        # Handle edge case of single point t_grid
        pos_ref = z0_model_coords.reshape(1, -1)
        vel_ref = model.predict(z0_model_coords).reshape(1, -1)
        # Ensure correct shape even for single point
        if pos_ref.shape[0] != output_len: pos_ref = np.tile(pos_ref, (output_len, 1))
        if vel_ref.shape[0] != output_len: vel_ref = np.tile(vel_ref, (output_len, 1))
        return pos_ref, vel_ref

    # Calculate dt based on the full t_grid duration and number of steps
    # Note: Using output_len - 1 intervals for output_len points
    dt = float((t[-1] - t[0]) / (output_len - 1))
    n_steps_required = output_len - 1 # Number of steps needed to generate output_len points

    log(f"  Rollout: dt={dt:.5f}, n_steps={n_steps_required}", enabled=True) # Verbose logging

    # Use SEDS internal simulator which now returns (positions, velocities)
    # Input z0 must be in model coordinates. Output will be in model coordinates.
    # The simulate function returns arrays of length n_steps + 1
    pos_sim, vel_sim = model.simulate(z0_model_coords.copy(), dt=dt, n_steps=n_steps_required)
    # pos_sim, vel_sim = model.simulate(z0_model_coords.copy(), dt=0.01, n_steps=1000)


    # Ensure the output lengths match t_grid precisely
    pos_ref = np.zeros((output_len, model.d))
    vel_ref = np.zeros((output_len, model.d))

    # Copy the simulated data, truncating or padding as needed
    actual_sim_len = len(pos_sim) # Should be n_steps_required + 1 = output_len
    copy_len = min(actual_sim_len, output_len)

    pos_ref[:copy_len] = pos_sim[:copy_len]
    vel_ref[:copy_len] = vel_sim[:copy_len]

    # If simulation stopped early (reached target), fill the rest
    if actual_sim_len < output_len:
        log(f"  Warning: SEDS reference simulation stopped early at step {actual_sim_len-1}/{n_steps_required}. Padding output.", enabled=True)
        pos_ref[actual_sim_len:] = model.target # Fill with target position (0,0)
        vel_ref[actual_sim_len:] = 0.0         # Fill with zero velocity

    # --- DEBUG: Check reference validity ---
    if np.any(np.isnan(pos_ref)) or np.any(np.isinf(pos_ref)):
        print("ERROR: NaNs or Infs found in generated pos_ref!")
    if np.any(np.isnan(vel_ref)) or np.any(np.isinf(vel_ref)):
        print("ERROR: NaNs or Infs found in generated vel_ref!")
    if pos_ref.shape != (output_len, model.d):
        print(f"ERROR: Incorrect pos_ref shape: {pos_ref.shape}, expected {(output_len, model.d)}")
    if vel_ref.shape != (output_len, model.d):
        print(f"ERROR: Incorrect vel_ref shape: {vel_ref.shape}, expected {(output_len, model.d)}")
    # --- END DEBUG ---

    return pos_ref, vel_ref

class SEDSWrapper:
    """Wraps SEDS.predict() to match the 'model.func(t, y, args)' interface expected by simulator."""
    def __init__(self, seds_model: SEDS):
        self.seds_model = seds_model
        # Ensure model is centered at origin (as trained)
        assert hasattr(self.seds_model, 'target'), "SEDS model missing 'target' attribute."
        assert np.allclose(self.seds_model.target, 0.0), "SEDS model must be trained with target at origin"
        self.call_count = 0 # Debug counter

    def func(self, t: float, y: np.ndarray, args) -> np.ndarray:
        """
        t: time (ignored, SEDS is autonomous)
        y: position state, shape (2,) - Expected in MODEL coordinates
        args: ignored
        Returns velocity in MODEL coordinates
        """
        self.call_count += 1
        # --- DEBUG: Print input to wrapper ---
        # if self.call_count % 100 == 0: # Print every 100 calls
        #      log(f"SEDSWrapper.func called at t={t:.3f} with y={y}", enabled=True)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
             print(f"ERROR: SEDSWrapper received NaN/Inf input y={y} at t={t:.3f}")
             return np.zeros_like(y) # Return zero velocity on error
             # raise ValueError("NaN/Inf input")
        # --- END DEBUG ---

        # SEDS.predict expects (N,d) or (d,) and returns numpy
        # simulator.py always passes numpy arrays
        v_hat = self.seds_model.predict(y)

        # --- DEBUG: Print output from wrapper ---
        if np.any(np.isnan(v_hat)) or np.any(np.isinf(v_hat)):
             print(f"ERROR: SEDSWrapper produced NaN/Inf output v_hat={v_hat} for y={y}")
             v_hat = np.zeros_like(y) # Return zero velocity on error
             # raise ValueError("NaN/Inf output")
        # --- END DEBUG ---

        return v_hat

# ------------------------ label builders ------------------------
# (No changes needed in _condition_tags)
def _condition_tags(args) -> dict:
    """Return tags/strings for filenames and titles based on LLC + disturbance flags."""
    with_llc = bool(args.with_llc)
    mode_tag = "with_llc" if with_llc else "no_llc"
    mode_title = "LLC" if with_llc else "No LLC"

    if with_llc:
        if args.matched and args.unmatched:
            dist_mode = "matched+unmatched"
            dist_tag = f"matched-{args.matched_type}_unmatched-{args.unmatched_type}"
            dist_title = f"Matched ({args.matched_type}) + Unmatched ({args.unmatched_type})"
        elif args.matched:
            dist_mode = "matched"
            dist_tag = f"matched-{args.matched_type}"
            dist_title = f"Matched ({args.matched_type})"
        elif args.unmatched:
            dist_mode = "unmatched"
            dist_tag = f"unmatched-{args.unmatched_type}"
            dist_title = f"Unmatched ({args.unmatched_type})"
        else:
            dist_mode = "none"
            dist_tag = "none"
            dist_title = "No disturbance"
    else:
        # direct disturbance path (we currently use mid-pulse)
        dist_mode = "direct"
        dist_tag = "direct-midpulse"
        dist_title = "Direct disturbance (mid-pulse)"

    base_tag = f"{mode_tag}_{dist_tag}"
    title = f"{mode_title} — Disturbance: {dist_title}"
    return dict(
        mode_tag=mode_tag, mode_title=mode_title,
        dist_mode=dist_mode, dist_tag=dist_tag, dist_title=dist_title,
        base_tag=base_tag,
        fig_title=title
    )


# ------------------------ main ------------------------
def main():
    ap = argparse.ArgumentParser(description="Run SEDS model experiments with CLF/L1.")
    ap.add_argument("--model_path", type=str, required=True, help="Explicit .pkl path to SEDS model")
    ap.add_argument("--shape", type=str, required=True, help="LASA shape name (e.g. 'Worm')")
    ap.add_argument("--nsamples", type=int, default=1000, help="Number of samples for resampling and simulation time grid.")
    ap.add_argument("--ntrain", type=int, default=4, help="Number of training demos used to calculate average demo.")
    ap.add_argument("--out", type=str, default="outputs/experiments_seds/default_run")
    ap.add_argument("--verbose", action="store_true", help="Enable detailed logging.")
    ap.add_argument(
        "--selector",
        type=str,
        default="dtw",
        choices=["dtw", "least_effort"],
        help="Target selection policy. Default=dtw."
    )

    # plant/LLC flags
    ap.add_argument("--with_llc", action="store_true",
                    help="Use plant + PD low-level controller (LLC) instead of direct/no_llc mode.")
    ap.add_argument("--matched", action="store_true",
                    help="Enable matched lower-level disturbance (acts on acceleration). Only used with --with_llc.")
    ap.add_argument("--unmatched", action="store_true",
                    help="Enable unmatched lower-level disturbance (acts on position rate). Only used with --with_llc.")
    ap.add_argument("--matched_type", type=str, default="sine", choices=["sine","pulse","const"], help="Type of matched disturbance.")
    ap.add_argument("--unmatched_type", type=str, default="sine", choices=["sine","pulse","const"], help="Type of unmatched disturbance.")

    args = ap.parse_args()

    # --- 1. Load SEDS model ---
    model_path = Path(args.model_path)
    log(f"Using model: {model_path}", enabled=args.verbose)
    try:
        seds_model = SEDS.load_model(str(model_path))
        if seds_model is None: raise ValueError("load_model returned None")
        log(f"Model loaded successfully (K={seds_model.K}, d={seds_model.d}).", enabled=args.verbose)
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file not found at {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR: Error loading model from {model_path}: {e}")
        raise ValueError(f"Failed to load SEDS model from {model_path}")

    # --- 2. Create the wrapper for the simulator and selector ---
    # This wrapper makes SEDS usable by components expecting a 'func(t, y, args)' signature.
    # It ensures operations happen in the MODEL coordinate frame.
    try:
        wrapped_model = SEDSWrapper(seds_model)
    except AssertionError as e:
        print(f"FATAL ERROR: Model compatibility check failed: {e}")
        print("Please ensure the loaded SEDS model was trained with the target at the origin.")
        sys.exit(1)


    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {out_dir}", enabled=args.verbose)

    # --- 3. Load and prepare LASA data ---
    log(f"Loading LASA shape: {args.shape}", enabled=args.verbose)
    try:
        lasa = load_shape(args.shape)
    except AttributeError:
        print(f"FATAL ERROR: Shape '{args.shape}' not found in pyLasaDataset.")
        sys.exit(1)

    # Resample to get consistent number of points and time grid
    pos_rs, vel_rs, t_rs = resample(lasa, nsamples=args.nsamples) # t_rs is normalized [0,1]
    log(f"Resampled data to {args.nsamples} points.", enabled=args.verbose)

    # --- 4. Define Coordinate Systems and References ---
    # SEDS model operates in MODEL coordinates (target at origin 0,0)
    # LASA data is in GLOBAL coordinates. We need to shift.
    target_pos_global = lasa.pos[0, -1, :] # Global target position (endpoint of first demo)
    log(f"Global target position: {target_pos_global}", enabled=args.verbose)

    # Calculate the average demo from the first 'ntrain' resampled demos
    if args.ntrain < 1 or args.ntrain > pos_rs.shape[0]:
        print(f"Warning: ntrain ({args.ntrain}) is out of valid range [1, {pos_rs.shape[0]}]. Using ntrain=1.")
        args.ntrain = 1
    train_idxs = list(range(args.ntrain)) # Use first ntrain demos
    avg_demo_pos_global = np.mean(pos_rs[train_idxs], axis=0)   # (nsamples, d) in GLOBAL coords
    avg_demo_vel_global = np.mean(vel_rs[train_idxs], axis=0)   # (nsamples, d) in GLOBAL coords

    # Shift Average Demo to MODEL coordinates for comparisons and reference generation
    # All subsequent operations use these model coordinates.
    avg_demo_pos_model = avg_demo_pos_global - target_pos_global
    avg_demo_vel_model = avg_demo_vel_global # Velocity is invariant to translation
    log(f"Average demo start (model coords): {avg_demo_pos_model[0]}", enabled=args.verbose)


    # Define the simulation time grid based on resampled data
    # This grid determines the duration and steps for both reference and simulation
    demo_t = t_rs # Use the time grid from resample (normalized [0, 1])

    # --- 5. Initial condition for simulation (in MODEL coordinates) ---
    init_pos_model = avg_demo_pos_model[0]
    init_vel_model = avg_demo_vel_model[0]
    # Simulator expects [px, py, vx, vy], but uses only pos for order=1
    init_state_simulator = np.hstack([init_pos_model, init_vel_model])

    log(f"Initial simulator state (model coords): {init_state_simulator}", enabled=args.verbose)

    # --- 6. Generate SEDS reference trajectory (in MODEL coordinates) ---
    # This is the trajectory the model *would* follow from the average start point
    log(f"Generating SEDS reference trajectory: {len(demo_t)} points...", enabled=args.verbose)
    seds_ref_pos_model, seds_ref_vel_model = rollout_seds_reference(seds_model, init_pos_model, demo_t)

    # --- 7. SAVE references (all in MODEL coordinates) ---
    # Note: These files contain data centered at the origin.
    avg_demo_filename = f"{args.shape}_avg_demo_model_coords.npz"
    avg_demo_path = out_dir / avg_demo_filename
    np.savez_compressed(avg_demo_path, pos=avg_demo_pos_model, vel=avg_demo_vel_model, t=demo_t,
                        target_global=target_pos_global,
                        comment="Average demo shifted to model coordinates (target at origin)")

    seds_ref_filename = f"{args.shape}_seds_ref_model_coords.npz"
    seds_ref_path = out_dir / seds_ref_filename
    np.savez_compressed(seds_ref_path, pos=seds_ref_pos_model, vel=seds_ref_vel_model, t=demo_t,
                        target_global=target_pos_global,
                        comment="SEDS rollout reference from avg demo start, in model coordinates")

    log(f"Saved references (in model coords): {avg_demo_filename}, {seds_ref_filename}", enabled=args.verbose)

    # --- 8. Setup Target Selectors ---
    # Selectors operate in MODEL coordinates using references generated above.
    selector_kind = args.selector.lower()
    class _SelectorFactoryHelper:
        """ Helper class to create selector instances with L1 logic, matching NODE script pattern. """
        def __init__(self, dt: float, enable_l1: bool):
            self.dt = float(dt)
            self.enable_l1 = enable_l1
            # log(f"  Initializing selector factory '{selector_kind}' with dt={dt:.5f}, L1={enable_l1}", enabled=args.verbose)

            # Create the core selector instance (DTW or LeastEffort)
            if selector_kind == "least_effort":
                self._selector_core = TargetLeastEffort(
                    model=wrapped_model, # Needs model.func() interface
                    dt=self.dt,
                    t_span=float(demo_t[-1] - demo_t[0]), # Should be 1.0
                    lookahead_N=35,
                    y0_seed=seds_ref_pos_model[0], # Initial pos in model coords
                    wrap=False
                )
            else: # Default: DTW
                try:
                    self._selector_core = TargetDTW(
                        seds_ref_pos_model, seds_ref_vel_model, # References in model coords
                        W=50, H=40
                    )
                    self._selector_core.init_from(seds_ref_pos_model[0]) # Initial pos in model coords
                except Exception as e_dtw:
                    print(f"FATAL ERROR: Failed to initialize TargetDTW: {e_dtw}")
                    raise

            # Create L1 instance if needed, also operates in MODEL coordinates
            if self.enable_l1:
                self._l1_instance = L1Adaptive(Ts=self.dt, a=10.0, omega=20.0, x0=seds_ref_pos_model[0])
            else:
                self._l1_instance = None

        def get_instance(self):
            # Return the core selector instance, attaching L1 if enabled
            # This matches the pattern expected by the NODE simulator's use of selector factory
            if self._l1_instance:
                setattr(self._selector_core, 'l1', self._l1_instance)
            elif hasattr(self._selector_core, 'l1'):
                 # Ensure l1 attribute doesn't exist if not enabled
                 delattr(self._selector_core, 'l1')
            # log(f"    Selector instance created (L1 attached: {hasattr(self._selector_core, 'l1')})", enabled=args.verbose)
            return self._selector_core

    # --- 9. Setup Vector Field for Plotting ---
    # field_fn expects input in MODEL coordinates
    def field_fn(p_xy_model: np.ndarray):
        # Handle potential batch vs single point for plotting
        if p_xy_model.ndim == 1:
            p_xy_model = p_xy_model[np.newaxis, :]
        v_model = wrapped_model.func(0.0, p_xy_model, None)
        # Ensure output matches input shape expectation if single point was passed
        return v_model.squeeze() if p_xy_model.shape[0] == 1 else v_model


    # Plot bounds are based on the average demo (in MODEL coordinates)
    x_range = float(np.ptp(avg_demo_pos_model[:, 0])); y_range = float(np.ptp(avg_demo_pos_model[:, 1]))
    pad_factor = 0.20 # Increased padding slightly
    pad = pad_factor * max(x_range, y_range) if max(x_range, y_range) > 0 else 0.15
    plot_bounds_model = (
        (float(np.min(avg_demo_pos_model[:, 0])) - pad, float(np.max(avg_demo_pos_model[:, 0])) + pad),
        (float(np.min(avg_demo_pos_model[:, 1])) - pad, float(np.max(avg_demo_pos_model[:, 1])) + pad),
    )

    # --- 10. Setup Simulation Conditions ---
    tags = _condition_tags(args)
    sel_tag = "sel-dtw" if selector_kind == "dtw" else "sel-le"
    log(f"Condition Tag: {tags['base_tag']}", enabled=args.verbose)
    log(f"Condition Title: {tags['fig_title']}", enabled=args.verbose)

    # Direct disturbance function (applied in simulator, magnitude matters)
    d_fn = None
    if tags["mode_tag"] == "no_llc":
        log("Setting up direct disturbance (no_llc mode)", enabled=args.verbose)
        d_fn = two_mid_pulses(
            center1=0.30, width1=0.20, mag1=0.0, ax_gain1=(1.0, -1.0), # Non-zero magnitude
            center2=0.80, width2=0.50, mag2=00.0, ax_gain2=(1.0,  1.0), # Non-zero magnitude
        )
    else: # with_llc mode
        log("LLC mode active. Direct disturbance disabled.", enabled=args.verbose)
        if args.matched or args.unmatched:
             log(f"  LLC disturbances: Matched={args.matched} ({args.matched_type}), Unmatched={args.unmatched} ({args.unmatched_type})", enabled=args.verbose)
        else:
             log("  LLC disturbances: None", enabled=args.verbose)


    # --- 11. Run Simulations ---
    controllers = [
        # name, flags (clf, l1), enable_l1 for factory
        ("SEDS",         dict(use_clf=False, use_l1=False), False),
        ("SEDS+CLF",     dict(use_clf=True,  use_l1=False), False),
        ("SEDS+CLF+L1",  dict(use_clf=True,  use_l1=True ), True),
    ]

    rollout_results_model_coords = [] # Store results (pos_model, name) for plotting
    dtw_names, dtw_values = [], []
    last_sim_logs = None # Store logs from the last successful simulation

    # Calculate the simulation dt based on the reference time grid
    if len(demo_t) < 2:
        sim_dt = 0.01 # Default fallback dt
        log(f"Warning: demo_t has < 2 points. Using default sim_dt={sim_dt}", enabled=True)
    else:
        # dt between points in the time grid
        sim_dt = (demo_t[-1] - demo_t[0]) / (len(demo_t) - 1)
        print("HERE SIM DT", sim_dt)
    log(f"Target simulation dt: {sim_dt:.6f}", enabled=args.verbose)

    for i, (name, sim_flags, l1_enabled) in enumerate(controllers, start=1):
        log(f"--- [{i}/{len(controllers)}] Running Controller: {name} ---", enabled=args.verbose)

        # Create the selector factory function for this specific controller run
        # This lambda captures the current dt and l1_enabled flag
        # It creates the _SelectorFactoryHelper which then creates the actual selector instance when called
        selector_factory_func = lambda dt=sim_dt, enable_l1=l1_enabled: _SelectorFactoryHelper(dt, enable_l1).get_instance()


        target_mode = selector_kind # Pass selector type to simulator config

        # Configure the simulation based on flags for this controller
        sim_config = SimConfig(
            dt=None, T=None, # Let simulator use t_grid
            target_mode=target_mode,
            no_llc=(tags["mode_tag"] == "no_llc"),
            # Flags from the controller definition
            use_clf=sim_flags["use_clf"], use_l1=sim_flags["use_l1"],
            matched=args.matched, unmatched=args.unmatched,
            matched_type=args.matched_type, unmatched_type=args.unmatched_type,
            # llc_substeps=10 # Example: Use 10 inner steps per outer step if needed
        )
        log(f"  SimConfig: {sim_config}", enabled=args.verbose)

        # Define output path for simulation logs
        base_filename = f"{args.shape}_{tags['base_tag']}_{sel_tag}"
        npz_filename = f"{base_filename}_{name.replace('+','_')}.npz"
        npz_path = out_dir / npz_filename

        t_start_sim = time.time()
        # Run the simulation - initial state and model operate in MODEL coordinates
        sim_logs = None # Reset logs for this run
        try:
            # Call simulate matching the NODE script's positional/keyword args
            sim_logs = simulate(
                wrapped_model,             # model (positional 1)
                None,                      # f_oracle (positional 2) - Not used for SEDS
                selector_factory_func,     # get_selector (positional 3) - lambda creates instance
                sim_config,                # cfg (positional 4)
                init_state_simulator,      # init_state (positional 5) - [pos_model, vel_model]
                order=1,                   # order (keyword) - SEDS is order 1
                direct_dist_fn=d_fn,       # direct_dist_fn (keyword)
                save_npz_path=str(npz_path), # save_npz_path (keyword)
                t_grid=demo_t              # t_grid (keyword)
            )
            sim_elapsed = time.time() - t_start_sim
            log(f"  Simulation successful ({sim_elapsed:.2f}s).", enabled=args.verbose)

        except Exception as sim_error:
             sim_elapsed = time.time() - t_start_sim
             log(f"!!! ERROR during simulation for controller '{name}' after {sim_elapsed:.2f}s !!!", enabled=True)
             log(f"Error message: {sim_error}", enabled=True)
             import traceback
             traceback.print_exc() # Print full traceback
             log(f"Skipping controller '{name}' due to error.", enabled=True)
             continue # Go to the next controller

        # Store results (positions are in logs["z"], should be in MODEL coords)
        rollout_pos_model = sim_logs["z"]
        rollout_results_model_coords.append((rollout_pos_model, name))

        # Calculate DTW against the average demo (both in MODEL coords)
        try:
            # Ensure avg_demo_pos_model has same length as rollout if needed for DTW
            ref_len = len(avg_demo_pos_model)
            rollout_len = len(rollout_pos_model)
            if ref_len != rollout_len:
                 log(f"  Warning: DTW length mismatch. Ref={ref_len}, Rollout={rollout_len}. Using common length {min(ref_len, rollout_len)}.", enabled=True)
            comp_len = min(ref_len, rollout_len)
            current_dtw = dtw_distance(rollout_pos_model[:comp_len], avg_demo_pos_model[:comp_len])

            dtw_names.append(name); dtw_values.append(current_dtw)
            log(f"  DTW vs avg demo (model coords): {current_dtw:.3f}", enabled=args.verbose)
        except Exception as dtw_err:
            log(f"  Warning: Could not calculate DTW for '{name}': {dtw_err}", enabled=True)
            dtw_names.append(name); dtw_values.append(np.nan)


        last_sim_logs = sim_logs # Keep logs for plotting disturbances

        log(f"Saved: {npz_path.name} | steps={len(sim_logs['t'])}", enabled=args.verbose)

    # --- 12. Plotting ---
    if not rollout_results_model_coords:
         log("No successful simulations to plot. Exiting.", enabled=True)
         # Save metadata even if plotting fails
         # (Call metadata saving logic here, simplified for brevity)
         sys.exit(0) # Exit cleanly

    # All plotting happens in MODEL coordinates
    log("Generating plots...", enabled=args.verbose)
    fig_title = f"{args.shape} — {tags['fig_title']} ({'DTW' if selector_kind=='dtw' else 'Least-Effort'} selector)"
    fig_filename = f"fig_{args.shape}_{tags['base_tag']}_{sel_tag}.png"
    fig_path = out_dir / fig_filename

    # Define reference curves for the plot (all in MODEL coordinates)
    ref_curves_model_coords = [
        ("SEDS ref (model coords)", seds_ref_pos_model),
        ("Avg demo (model coords)", avg_demo_pos_model),
    ]

    try:
        plot_all_together_with_dist(
            rollouts=rollout_results_model_coords, # Simulation results (model coords)
            demo=None, # Not plotting individual demo here
            field_fn=field_fn,                     # Vector field function (expects model coords)
            field_bounds=plot_bounds_model,        # Plot bounds (model coords)
            subtitle=fig_title,
            outpath=fig_path,
            # Pass disturbance info if available in logs (check last_sim_logs exists)
            t_norm=last_sim_logs.get("t_norm", None) if last_sim_logs else None,
            d_direct=last_sim_logs.get("d_direct", None) if last_sim_logs else None,
            ref_curves=ref_curves_model_coords,     # Reference curves (model coords)
            d_matched=last_sim_logs.get("sigma", None) if last_sim_logs else None,
            d_unmatched=last_sim_logs.get("d_p", None) if last_sim_logs else None,
        )
        log(f"Figure saved: {fig_path.name}", enabled=args.verbose)
    except Exception as plot_err:
        log(f"!!! WARNING: Plotting failed: {plot_err} !!!", enabled=True)
        import traceback
        traceback.print_exc()


    # --- 13. DTW Chart ---
    if dtw_values: # Only plot if we have values
        log("Generating DTW chart...", enabled=args.verbose)
        chart_filename = f"chart_{args.shape}_{tags['base_tag']}_{sel_tag}_dtw_avg.png"
        chart_path = out_dir / chart_filename
        try:
            # Filter out potential NaNs before plotting
            valid_dtw_indices = [i for i, v in enumerate(dtw_values) if not np.isnan(v)]
            valid_names = [dtw_names[i] for i in valid_dtw_indices]
            valid_values = np.array([dtw_values[i] for i in valid_dtw_indices])

            if valid_values.size > 0:
                bar_with_ci(
                    valid_names, valid_values, np.zeros_like(valid_values), # Assuming no std dev for now
                    ylabel="DTW(trajectory, avg demo) [Model Coords]",
                    title=f"{args.shape} — {tags['fig_title']} ({'DTW' if selector_kind=='dtw' else 'Least-Effort'} selector)",
                    outpath=chart_path
                )
                log(f"Chart saved:  {chart_path.name}", enabled=args.verbose)
            else:
                 log("No valid DTW values to plot chart.", enabled=True)
        except Exception as chart_err:
            log(f"!!! WARNING: DTW chart generation failed: {chart_err} !!!", enabled=True)
            import traceback
            traceback.print_exc()

    else:
        log("No DTW values to plot.", enabled=True)


    # --- 14. Metadata ---
    log("Saving metadata...", enabled=args.verbose)
    meta = dict(
        model_info=dict(
            path=str(model_path),
            seds_K=seds_model.K,
            seds_dim=seds_model.d
        ),
        data_info=dict(
            shape=args.shape,
            nsamples=args.nsamples,
            ntrain_avg=args.ntrain,
            global_target_pos=target_pos_global.tolist()
        ),
        reference_info=dict(
            comment="References are stored in MODEL coordinates (target at origin)",
            avg_demo_npz=str(avg_demo_path.name),
            seds_ref_npz=str(seds_ref_path.name),
            dtw_reference="average demo (model coords)",
            selector_reference="SEDS rollout (model coords)"
        ),
        simulation_info=dict(
            selector_type=selector_kind,
            condition_tag=tags['base_tag'],
            condition_details=dict(
                with_llc=(tags["mode_tag"]=="with_llc"),
                disturbance_mode=tags["dist_mode"],
                matched=args.matched, unmatched=args.unmatched,
                matched_type=args.matched_type, unmatched_type=args.unmatched_type,
            ),
            sim_dt_used=sim_dt,
            t_grid_duration=float(demo_t[-1] - demo_t[0]),
            num_steps = len(demo_t) - 1
        ),
        outputs=dict(
            # Use names from successful runs only
            figure=str(fig_path.name) if rollout_results_model_coords else None,
            dtw_chart=str(chart_path.name) if dtw_values and not np.all(np.isnan(dtw_values)) else None,
            simulation_logs=[f"{base_filename}_{n.replace('+','_')}.npz" for pos, n in rollout_results_model_coords]
        ),
        # Use names from successful runs only
        dtw_results = {name: val for name, val in zip(dtw_names, dtw_values) if not np.isnan(val)}
    )

    meta_path = out_dir / "meta_experiment_summary.json"
    try:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        log(f"Metadata saved: {meta_path.name}", enabled=args.verbose)
    except Exception as meta_err:
        log(f"!!! WARNING: Failed to save metadata: {meta_err} !!!", enabled=True)


    log("Experiment run complete.", enabled=args.verbose)


if __name__ == "__main__":
    # Add a top-level try-except to catch unexpected errors during setup
    try:
        main()
    except Exception as main_err:
        log(f"--- FATAL ERROR IN MAIN ---", enabled=True)
        log(f"{main_err}", enabled=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)






# from __future__ import annotations
# import argparse, json, time, sys, pickle
# from pathlib import Path
# import numpy as np
# import jax, jax.numpy as jnp
# import equinox as eqx

# # --- Add project root to path ---
# # This is a bit of a hack to make imports work when running as a script
# # or as a module. It allows us to find `SEDS.seds_core` and `src.data.lasa`
# _SCRIPT_DIR = Path(__file__).parent.resolve()
# _ROOT_DIR = _SCRIPT_DIR.parent.parent
# if str(_ROOT_DIR) not in sys.path:
#     print(f"Could not find SEDS module. Adding project root to path: {_ROOT_DIR}")
#     sys.path.append(str(_ROOT_DIR))
# # --- End Path Hack ---

# try:
#     from SEDS.seds_core import SEDS
# except ImportError as e:
#     print(f"Error: {e}. Could not import SEDS.")
#     print("Please ensure SEDS/seds_core.py exists and the project root is in PYTHONPATH.")
#     sys.exit(1)

# from src.data.lasa import load_shape, resample
# from src.experiments.targets import TargetDTW, TargetLeastEffort
# from src.experiments.robust_ctrl import L1Adaptive
# from src.experiments.disturbances import big_mid_pulse, two_mid_pulses # direct disturbance (no_llc only)
# from src.experiments.simulator import SimConfig, simulate
# from src.experiments.metrics_plots import dtw_distance, plot_all_together_with_dist, bar_with_ci


# # ------------------------ tiny logger ------------------------
# def _now(): return time.strftime("%H:%M:%S")
# def log(msg: str, *, enabled: bool = True):
#     if enabled: print(f"[{_now()}] {msg}", flush=True)


# # ------------------------ helpers ------------------------
# def rollout_seds_reference(model: SEDS, z0: np.ndarray, t_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """Rollout SEDS reference and estimate velocity."""
#     t = np.asarray(t_grid).reshape(-1)
#     dt = float((t[-1]-t[0]) / max(1, len(t)-1))
#     n_steps = len(t) - 1
    
#     # Use SEDS internal simulator
#     pos_ref = model.simulate(z0.copy(), dt=dt, n_steps=n_steps)
    
#     # Estimate velocity from position reference
#     # We add an extra point at the end to match length, velocities[last] = 0
#     vel_ref = np.zeros_like(pos_ref)
#     vel_ref[:-1] = np.gradient(pos_ref[:-1], dt, axis=0) # Use [:-1] to avoid edge effects at end
    
#     # Ensure velocity at the target (origin) is zero
#     vel_ref[-1] = 0.0
    
#     return pos_ref, vel_ref

# class SEDSWrapper:
#     """Wraps SEDS.predict() to match the 'model.func()' interface expected by simulator."""
#     def __init__(self, seds_model: SEDS):
#         self.seds_model = seds_model
#         # Ensure model is centered at origin (as trained)
#         assert np.allclose(self.seds_model.target, 0.0), "SEDS model must be trained with target at origin"

#     def func(self, t: float, y: jnp.ndarray | np.ndarray, args) -> jnp.ndarray | np.ndarray:
#         """
#         t: time (ignored, SEDS is autonomous)
#         y: position state, shape (2,)
#         args: ignored
#         """
#         # SEDS.predict expects (N,d) or (d,) and returns numpy
#         is_jax = isinstance(y, jnp.ndarray)
#         if is_jax:
#             y_np = np.array(y)
#         else:
#             y_np = y
            
#         v_hat = self.seds_model.predict(y_np)
        
#         if is_jax:
#             return jnp.array(v_hat)
#         else:
#             return v_hat

# # ------------------------ label builders ------------------------
# def _condition_tags(args) -> dict:
#     """Return tags/strings for filenames and titles based on LLC + disturbance flags."""
#     with_llc = bool(args.with_llc)
#     mode_tag = "with_llc" if with_llc else "no_llc"
#     mode_title = "LLC" if with_llc else "No LLC"

#     if with_llc:
#         if args.matched and args.unmatched:
#             dist_mode = "matched+unmatched"
#             dist_tag = f"matched-{args.matched_type}_unmatched-{args.unmatched_type}"
#             dist_title = f"Matched ({args.matched_type}) + Unmatched ({args.unmatched_type})"
#         elif args.matched:
#             dist_mode = "matched"
#             dist_tag = f"matched-{args.matched_type}"
#             dist_title = f"Matched ({args.matched_type})"
#         elif args.unmatched:
#             dist_mode = "unmatched"
#             dist_tag = f"unmatched-{args.unmatched_type}"
#             dist_title = f"Unmatched ({args.unmatched_type})"
#         else:
#             dist_mode = "none"
#             dist_tag = "none"
#             dist_title = "No disturbance"
#     else:
#         # direct disturbance path (we currently use mid-pulse)
#         dist_mode = "direct"
#         dist_tag = "direct-midpulse"
#         dist_title = "Direct disturbance (mid-pulse)"

#     base_tag = f"{mode_tag}_{dist_tag}"
#     title = f"{mode_title} — Disturbance: {dist_title}"
#     return dict(
#         mode_tag=mode_tag, mode_title=mode_title,
#         dist_mode=dist_mode, dist_tag=dist_tag, dist_title=dist_title,
#         base_tag=base_tag,
#         fig_title=title
#     )


# # ------------------------ main ------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model_path", type=str, required=True, help="Explicit .pkl path to SEDS model")
#     ap.add_argument("--shape", type=str, required=True, help="LASA shape name (e.g. 'Worm')")
#     ap.add_argument("--nsamples", type=int, default=1000, help="Number of samples for resampling.")
#     ap.add_argument("--ntrain", type=int, default=4, help="Number of training demos (for avg).")
#     ap.add_argument("--out", type=str, default="outputs/experiments_seds/Worm_suite")
#     ap.add_argument("--verbose", action="store_true")
#     ap.add_argument(
#         "--selector",
#         type=str,
#         default="dtw",
#         choices=["dtw", "least_effort"],
#         help="Target selection policy. Default=dtw."
#     )

#     # plant/LLC flags
#     ap.add_argument("--with_llc", action="store_true",
#                     help="Use plant + PD low-level controller (LLC) instead of direct/no_llc mode.")
#     ap.add_argument("--matched", action="store_true",
#                     help="Enable matched lower-level disturbance (acts on acceleration).")
#     ap.add_argument("--unmatched", action="store_true",
#                     help="Enable unmatched lower-level disturbance (acts on position rate).")
#     ap.add_argument("--matched_type", type=str, default="sine", choices=["sine","pulse","const"])
#     ap.add_argument("--unmatched_type", type=str, default="sine", choices=["sine","pulse","const"])

#     args = ap.parse_args()

#     # --- load model
#     model_path = Path(args.model_path)
#     log(f"Using model: {model_path}", enabled=args.verbose)
#     try:
#         seds_model = SEDS.load_model(str(model_path))
#         log(f"Model loaded successfully (K={seds_model.K}).", enabled=args.verbose)
#     except Exception as e:
#         print(f"Error loading model from {model_path}: {e}")
#         try:
#             # Try original pickle load as fallback
#             with open(model_path, 'rb') as f:
#                 seds_model = pickle.load(f)
#             log(f"Model loaded via fallback pickle (K={seds_model.K}).", enabled=args.verbose)
#         except Exception as e2:
#             print(f"Fallback pickle load failed: {e2}")
#             raise ValueError(f"Failed to load SEDS model from {model_path}")
    
#     # --- Create the wrapper for the simulator and selector ---
#     wrapped_model = SEDSWrapper(seds_model)
    
#     out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

#     # --- LASA data + resample
#     lasa = load_shape(args.shape)
#     pos_rs, vel_rs, _ = resample(lasa, nsamples=args.nsamples)

#     # --- training subset & average demo
#     # We must offset the demo data to match the model's coordinate frame (target at origin)
#     target_pos = lasa.pos[0, -1, :] # Global target position
    
#     train_idxs = list(range(1, args.ntrain)) if args.ntrain > 1 else [0]
    
#     # Get average demo in GLOBAL coordinates
#     demo_avg_pos_global = np.mean([pos_rs[i] for i in train_idxs], axis=0)   # (N,2)
#     demo_avg_vel_global = np.mean([vel_rs[i] for i in train_idxs], axis=0)   # (N,2)
    
#     # Shift to MODEL coordinates (target at 0,0)
#     demo_avg_pos = demo_avg_pos_global - target_pos
#     demo_avg_vel = demo_avg_vel_global # Velocity unchanged
#     demo_t = np.linspace(0.0, 1.0, demo_avg_pos.shape[0], dtype=float)

#     # --- initial condition from the average demo (in MODEL coordinates)
#     init_pos = demo_avg_pos[0]
#     init_vel = demo_avg_vel[0]
#     init_state = np.hstack([init_pos, init_vel]) # Shape (4,) [px, py, vx, vy]
    
#     # Note: simulator.py handles 1st vs 2nd order state based on `order` flag.
#     # We pass order=1, so it will only use the first 2 elements (position)
#     # as the initial state `z_true`.
#     log(f"Simulating from init state (pos): {init_pos}", enabled=args.verbose)


#     # --- SEDS rollout reference for CLF/L1 (in MODEL coordinates)
#     log(f"Generating SEDS reference: {len(demo_t)} steps, dt={demo_t[1]-demo_t[0]:.4f}", enabled=args.verbose)
#     seds_ref_pos, seds_ref_vel = rollout_seds_reference(seds_model, init_pos, 10*demo_t)

#     # --- SAVE references (in MODEL coordinates)
#     avg_demo_path = out_dir / f"{args.shape}_avg_demo.npz"
#     np.savez_compressed(avg_demo_path, pos=demo_avg_pos, vel=demo_avg_vel, t=demo_t)

#     node_ref_path = out_dir / f"{args.shape}_seds_ref_from_avg_init.npz"
#     np.savez_compressed(node_ref_path, pos=seds_ref_pos, vel=seds_ref_vel, t=demo_t)

#     log(f"Saved references: {avg_demo_path.name}, {node_ref_path.name}", enabled=args.verbose)

#     # ---- selector factories (DTW default; Least-Effort optional)
#     selector_kind = args.selector.lower()  # "dtw" or "least_effort"

#     class _SelWrap:
#         def __init__(self, dt: float):
#             self.dt = float(dt)
#             if selector_kind == "least_effort":
#                 # Build least-effort selector from the learned model directly.
#                 # Use same init as SEDS ref for consistency.
#                 self.sel = TargetLeastEffort(
#                     model=wrapped_model, # Use the wrapped model
#                     dt=self.dt,
#                     t_span=float(demo_t[-1] - demo_t[0]),   # normalized to 1.0
#                     lookahead_N=35,
#                     y0_seed=seds_ref_pos[0],
#                     wrap=False
#                 )
#             else:
#                 # Default: DTW on the SEDS reference
#                 self.sel = TargetDTW(seds_ref_pos, seds_ref_vel, W=50, H=40)
#                 self.sel.init_from(seds_ref_pos[0])

#             # L1 at the same (outer) dt
#             self.l1 = L1Adaptive(Ts=self.dt, a=10.0, omega=20.0, x0=seds_ref_pos[0])

#         def get(self, inp):
#             # simulate() passes history (DTW) or current pos (Least-Effort)
#             if selector_kind == "least_effort":
#                 if isinstance(inp, (list, tuple)):
#                     x = np.asarray(inp[-1], dtype=float)
#                 else:
#                     x = np.asarray(inp, dtype=float)
#                 return self.sel.get(x)
#             else:
#                 return self.sel.get(inp)

#     def selector_with_l1(dt: float): return _SelWrap(dt)
#     def selector_no_l1(dt: float):
#         s = _SelWrap(dt)
#         if hasattr(s, "l1"):
#             delattr(s, "l1")
#         return s

#     # ---- vector field function
#     def field_fn(p_xy: np.ndarray):
#         # SEDSWrapper expects (d,) or (N,d)
#         return wrapped_model.func(0.0, p_xy, None)

#     # ---- plot bounds around average demo
#     x_range = float(np.ptp(demo_avg_pos[:, 0])); y_range = float(np.ptp(demo_avg_pos[:, 1]))
#     pad = 0.15 * max(x_range, y_range) if max(x_range, y_range) > 0 else 0.1
#     bounds = (
#         (float(np.min(demo_avg_pos[:, 0])) - pad, float(np.max(demo_avg_pos[:, 0])) + pad),
#         (float(np.min(demo_avg_pos[:, 1])) - pad, float(np.max(demo_avg_pos[:, 1])) + pad),
#     )

#     # ---- condition tags & titles
#     tags = _condition_tags(args)
#     sel_tag = "sel-dtw" if selector_kind == "dtw" else "sel-le"
#     log(f"Condition: {tags['fig_title']}", enabled=args.verbose)

#     # ---- direct disturbance for no_llc (ignored when with_llc)
#     d_fn = None
#     if tags["mode_tag"] == "no_llc":
#         d_fn = two_mid_pulses(
#             center1=0.30, width1=0.20, mag1=0.0, ax_gain1=(1.0, -1.0),
#             center2=0.80, width2=0.50, mag2=0.0, ax_gain2=(1.0,  1.0),
#         )

#     # ---- simulate all controllers
#     controllers = [
#         ("SEDS",         dict(use_clf=False, use_l1=False), selector_no_l1),
#         ("SEDS+CLF",     dict(use_clf=True,  use_l1=False), selector_no_l1),
#         ("SEDS+CLF+L1",  dict(use_clf=True,  use_l1=True ), selector_with_l1),
#     ]

#     cols = []
#     names, dtw_vals = [], []
#     last_logs = None

#     for i, (name, flags, sel_factory) in enumerate(controllers, start=1):
#         log(f"[{i}/{len(controllers)}] Running controller: {name}", enabled=args.verbose)

#         target_mode = "dtw" if selector_kind == "dtw" else "least_effort"

#         cfg = SimConfig(
#             dt=None, T=None, target_mode=target_mode,
#             no_llc=(tags["mode_tag"] == "no_llc"),
#             use_clf=flags["use_clf"], use_l1=flags["use_l1"],
#             matched=args.matched, unmatched=args.unmatched,
#             matched_type=args.matched_type, unmatched_type=args.unmatched_type,
#         )

#         base = f"{args.shape}_{tags['base_tag']}_{sel_tag}"
#         npz_name = f"{base}_{name.replace('+','_')}.npz"
#         npz_path = out_dir / npz_name

#         t0 = time.time()
        
#         # Calculate dt based on t_grid
#         sim_dt = (demo_t[-1] - demo_t[0]) / (len(demo_t) - 1)
#         print("sim_dt", sim_dt) # Debug print
        
#         logs = simulate(
#             wrapped_model, None, lambda: sel_factory(dt=sim_dt),
#             cfg, init_state, order=1, # Force order=1 for SEDS (pos->vel)
#             direct_dist_fn=d_fn,
#             save_npz_path=str(npz_path),
#             t_grid=demo_t
#         )
#         elapsed = time.time() - t0

#         cols.append((logs["z"], name))
#         d = dtw_distance(logs["z"], demo_avg_pos)  # DTW vs average training demo
#         names.append(name); dtw_vals.append(d)
#         last_logs = logs

#         log(f"Saved: {npz_path.name} | steps={len(logs['t'])} | DTW(avg demo)={d:.3f} | elapsed={elapsed:.2f}s",
#             enabled=args.verbose)

#     # ---- figure (overlay + disturbance subplot)
#     fig_title = f"{args.shape} — {tags['fig_title']} ({'DTW' if selector_kind=='dtw' else 'Least-Effort'} selector)"
#     fig_path = out_dir / f"fig_{args.shape}_{tags['base_tag']}_{sel_tag}.png"

#     ref_curves = [
#         ("SEDS ref (from avg init)", seds_ref_pos),
#         ("Avg training demo",        demo_avg_pos),
#     ]

#     plot_all_together_with_dist(
#         rollouts=cols,
#         demo=None,
#         field_fn=field_fn,
#         field_bounds=bounds,
#         subtitle=fig_title,
#         outpath=fig_path,
#         t_norm=last_logs.get("t_norm", None),
#         d_direct=last_logs.get("d_direct", None),
#         ref_curves=ref_curves,
#         d_matched=last_logs.get("sigma", None),
#         d_unmatched=last_logs.get("d_p", None),
#     )
#     log(f"Figure saved: {fig_path}", enabled=args.verbose)

#     # ---- DTW chart (numeric labels)
#     chart_path = out_dir / f"chart_{args.shape}_{tags['base_tag']}_{sel_tag}_dtw_avg.png"
#     bar_with_ci(
#         names, np.array(dtw_vals), np.zeros_like(dtw_vals),
#         ylabel="DTW(trajectory, avg demo)",
#         title=f"{args.shape} — {tags['fig_title']} ({'DTW' if selector_kind=='dtw' else 'Least-Effort'} selector)",
#         outpath=chart_path
#     )
#     log(f"Chart saved:  {chart_path}", enabled=args.verbose)

#     # ---- meta
#     meta = dict(
#         model=str(model_path), 
#         seds_K=seds_model.K,
#         shape=args.shape, nsamples=args.nsamples, ntrain=args.ntrain,
#         demo_ref="SEDS rollout from average-demo init",
#         dtw_ref="average of training demos",
#         selector=selector_kind,
#         condition=dict(
#             with_llc=(tags["mode_tag"]=="with_llc"),
#             disturbance_mode=tags["dist_mode"],
#             matched=args.matched, unmatched=args.unmatched,
#             matched_type=args.matched_type, unmatched_type=args.unmatched_type,
#         ),
#         outputs=dict(
#             fig=str(fig_path), chart=str(chart_path),
#             logs=[f"{args.shape}_{tags['base_tag']}_{sel_tag}_{n.replace('+','_')}.npz" for n,_,_ in controllers]
#         )
#     )

#     meta["references"] = dict(
#         avg_demo=str(avg_demo_path),
#         seds_ref=str(node_ref_path),
#     )

#     with open(out_dir / "meta.json", "w") as f:
#         json.dump(meta, f, indent=2)
#     log("Done.", enabled=args.verbose)


# if __name__ == "__main__":
#     main()

