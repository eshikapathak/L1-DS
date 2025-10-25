
import argparse
import pickle
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Path Fix ---
# Add the project root to the path so we can use absolute imports
# (e.g., from SEDS.seds_core import SEDS)
_SCRIPT_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _SCRIPT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.append(str(_ROOT_DIR))
# --- End Fix ---

# --- Import Fix ---
# Use absolute imports so the pickle file stores the correct module path
# Import resample as well
from SEDS.lasa_data import load_shape, get_shape_names, resample
from SEDS.seds_core import SEDS
# --- End Fix ---


def main(args):
    """
    Main training and plotting loop.
    """
    # --- 1. Load Data ---
    print(f"Loading LASA dataset for shape: {args.shape}")
    lasa_shape = load_shape(args.shape)

    # Original Data dimensions
    n_demos_orig, n_samples_orig, d_orig = lasa_shape.pos.shape
    print(f"Original data: {n_demos_orig} demos, {n_samples_orig} samples, {d_orig} dimensions")

    # --- 1b. Optional Resampling ---
    if args.nsamples_train is not None and args.nsamples_train > 0:
        print(f"Resampling demonstrations to {args.nsamples_train} points...")
        try:
            # resample returns (pos_rs, vel_rs, t_rs)
            # pos_rs: (D, nsamples, 2), vel_rs: (D, nsamples, 2), t_rs: (nsamples,)
            pos_data, vel_data, t_data_resampled = resample(lasa_shape, nsamples=args.nsamples_train)
            n_samples = args.nsamples_train # Update n_samples count
            print(f"Resampled data: {pos_data.shape[0]} demos, {n_samples} samples, {pos_data.shape[2]} dimensions")
            print(f"time", len(t_data_resampled))
        except Exception as e:
            print(f"Error during resampling: {e}")
            print("Proceeding with original data.")
            pos_data = lasa_shape.pos
            vel_data = lasa_shape.vel
            n_samples = n_samples_orig # Use original sample count
    else:
        # Use original data if no resampling requested
        pos_data = lasa_shape.pos
        vel_data = lasa_shape.vel
        n_samples = n_samples_orig # Use original sample count
        print("Using original data samples.")


    # --- 1c. Center Data at Target ---
    # Find the target (common endpoint from original, non-resampled data)
    # Using the original ensures consistency even if resampling changes the exact endpoint slightly
    target = lasa_shape.pos[0, -1, :]
    print(f"Centering data around target: {target}")

    # Center all data (original or resampled) at the target
    demos_x = pos_data - target
    demos_dx = vel_data # Velocities are unaffected by translation

    n_demos, n_samples_final, d = demos_x.shape # d is n_dim
    # print(f"Data ready for training: {n_demos} demos, {n_samples_final} samples, {d} dimensions") # Redundant?

    # --- 2. Train Model ---
    seds = SEDS(n_gaussians=args.k, dim=d)
    print(f"Initializing SEDS with {args.k} Gaussians.")

    print("Training SEDS model... (This may take a minute)")
    success = seds.train(demos_x, demos_dx)

    if not success:
        print("Model training failed. Exiting.")
        sys.exit(1)

    print("Training complete.")

    # --- 3. Simulate and Plot ---
    print("Simulating trajectories from demonstration starting points...")
    sim_trajs = []
    for i in range(n_demos):
        x_start = demos_x[i, 0, :] # Starting point of demo i (in model coords)
        # Simulate returns (positions, velocities)
        traj, _ = seds.simulate(x_start, dt=args.dt, n_steps=len(t_data_resampled))#args.n_steps)
        sim_trajs.append(traj)

    print("Plotting results...")

    # --- Plotting Section ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(f'SEDS Training for "{args.shape}" with {args.k} Gaussians ({"Resampled to " + str(args.nsamples_train) + " pts" if args.nsamples_train else "Original Samples"})', fontsize=16)

    # Plot 1: Original Demonstrations (Centered)
    # Always plot the *training* data (potentially resampled)
    ax1 = axes[0]
    ax1.set_title("Training Demonstrations (Centered)")
    for i, demo in enumerate(demos_x): # Use demos_x which is centered and maybe resampled
        ax1.plot(demo[:, 0], demo[:, 1], 'b-', alpha=0.6, label="Demonstration" if i == 0 else None)
        ax1.scatter(demo[0, 0], demo[0, 1], c='b', marker='o') # Start
    ax1.scatter(0, 0, c='r', marker='*', s=200, label='Target (Origin)') # Target
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, linestyle='--', alpha=0.6)


    # Plot 2: SEDS Reproductions
    ax2 = axes[1]
    ax2.set_title("SEDS Reproductions")
    for i, traj in enumerate(sim_trajs):
        ax2.plot(traj[:, 0], traj[:, 1], 'g-', alpha=0.8, label="SEDS Simulation" if i == 0 else None)
        ax2.scatter(traj[0, 0], traj[0, 1], c='g', marker='o') # Start (should match demo starts)
    ax2.scatter(0, 0, c='r', marker='*', s=200, label='Target (Origin)') # Target
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True, linestyle='--', alpha=0.6)


    # Plot 3: Learned Vector Field
    ax3 = axes[2]
    ax3.set_title("Learned Dynamical System")

    # Create a grid to visualize the vector field based on Plot 1 limits
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    # Adjust padding based on data range to avoid excessive empty space
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    padding = 0.15 * max(x_range, y_range) if max(x_range, y_range) > 0 else 0.5
    padding = max(padding, 0.5) # Ensure minimum padding

    x_grid_min, x_grid_max = xlim[0] - padding, xlim[1] + padding
    y_grid_min, y_grid_max = ylim[0] - padding, ylim[1] + padding

    grid_density = 25 # Increased density slightly
    x_grid = np.linspace(x_grid_min, x_grid_max, grid_density)
    y_grid = np.linspace(y_grid_min, y_grid_max, grid_density)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T # Shape (grid_density*grid_density, d)

    # Predict velocities on the grid
    try:
        DX = seds.predict(grid_points)
        DX_grid = DX.reshape(X.shape[0], X.shape[1], d) # Use d (n_dim)

        # Plot streamlines
        speed = np.sqrt(DX_grid[:, :, 0]**2 + DX_grid[:, :, 1]**2)
        lw = 1.5 * speed / speed.max() if speed.max() > 0 else np.ones_like(speed)*0.5 # Line width based on speed
        ax3.streamplot(X, Y, DX_grid[:, :, 0], DX_grid[:, :, 1], color='k', linewidth=lw, density=1.8, arrowstyle='->', arrowsize=1.0)

    except Exception as e_plot:
        print(f"Could not plot vector field: {e_plot}")

    # Overlay original demos (training data) for context
    for demo in demos_x:
        ax3.plot(demo[:, 0], demo[:, 1], 'b-', alpha=0.2) # Make more transparent
    ax3.scatter(0, 0, c='r', marker='*', s=200, label='Target (Origin)') # Target
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")
    ax3.legend()
    ax3.axis('equal')
    ax3.set_xlim(x_grid_min, x_grid_max)
    ax3.set_ylim(y_grid_min, y_grid_max)
    ax3.grid(True, linestyle='--', alpha=0.6)


    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
    # --- End Plotting Section ---

    # --- 4. Save Artifacts ---
    if args.outdir:
        try:
            save_dir = Path(args.outdir) # Use pathlib
            save_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            print(f"Saving artifacts to: {save_dir}")

            # Auto-generate file names based on hyperparams
            resample_tag = f"_resamp{args.nsamples_train}" if args.nsamples_train else ""
            base_name = f"{args.shape}_k{args.k}{resample_tag}"
            model_path = save_dir / f"{base_name}_model.pkl"
            plot_path = save_dir / f"{base_name}_plots.png"

            # Save the model object
            seds.save_model(model_path) # Use the class method
            # print(f"Model saved to {model_path}") # save_model already prints

            # Save the figure
            fig.savefig(plot_path, dpi=150) # Set DPI for better resolution
            print(f"Plots saved to {plot_path}")

        except Exception as e:
            print(f"Error saving artifacts: {e}")

    # 5. Show plots unless disabled
    if not args.no_show_plots:
        print("Displaying plots...")
        plt.show()
    else:
        plt.close(fig) # Close the figure if not showing


# --- Main execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SEDS on LASA handwriting dataset.")

    # Get available shapes for help text
    try:
        shape_names = get_shape_names()
        shape_help = f"Name of the LASA shape. Choices: {', '.join(shape_names)}"
    except Exception:
        shape_names = []
        shape_help = "Name of the LASA shape (e.g., 'Worm')."

    parser.add_argument("--shape", type=str, default="Worm", help=shape_help) # Changed default
    parser.add_argument("-k", type=int, default=6, help="Number of Gaussian components (K). Default: 6")
    parser.add_argument("--nsamples_train", type=int, default=1000, help="Optional: Resample demonstrations to this many points before training.")

    # Simulation params
    parser.add_argument("--dt", type=float, default=0.001, help="Simulation time step for generating reproductions. Default: 0.01")
    parser.add_argument("--n_steps", type=int, default=1000, help="Number of simulation steps for generating reproductions. Default: 500") # Adjusted default

    # Saving
    parser.add_argument("-o", "--outdir", type=str, default=None, help="Directory to save the trained model (.pkl) and plots (.png).")
    parser.add_argument("--no_show_plots", action='store_true', help="Do not show the plot window at the end.")

    parsed_args = parser.parse_args()

    # Re-check shape name now that imports are loaded
    if parsed_args.shape not in shape_names and len(shape_names) > 0:
        print(f"\n--- ERROR: Shape '{parsed_args.shape}' not found ---", file=sys.stderr)
        print("Available shapes are:", file=sys.stderr)
        print(", ".join(shape_names), file=sys.stderr)
        print("------------------------------------------", file=sys.stderr)
        sys.exit(1)

    main(parsed_args)



# # SEDS/seds_train.py
# import argparse
# import pickle
# import sys
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # --- Path Fix ---
# # Add the project root to the path so we can use absolute imports
# # (e.g., from SEDS.seds_core import SEDS)
# _SCRIPT_DIR = Path(__file__).parent.resolve()
# _ROOT_DIR = _SCRIPT_DIR.parent
# if str(_ROOT_DIR) not in sys.path:
#     sys.path.append(str(_ROOT_DIR))
# # --- End Fix ---

# # --- Import Fix ---
# # Use absolute imports so the pickle file stores the correct module path
# from SEDS.lasa_data import load_shape, get_shape_names
# from SEDS.seds_core import SEDS
# # --- End Fix ---


# def main(args):
#     """
#     Main training and plotting loop.
#     """
#     # --- 1. Load Data ---
#     # print(f"Loading LASA dataset for shape: {args.shape}")
#     # all_shapes = get_shape_names()
#     # if not all_shapes:
#     #      print("\n--- ERROR ---", file=sys.stderr)
#     #      print("Could not find any shapes in 'pyLasaDataset'.", file=sys.stderr)
#     #      print("Please ensure the package is installed correctly.", file=sys.stderr)
#     #      print("-------------", file=sys.stderr)
#     #      sys.exit(1)
         
#     # if args.shape not in all_shapes:
#     #     print(f"\n--- ERROR: Shape '{args.shape}' not found ---", file=sys.stderr)
#     #     print("Available shapes are:", file=sys.stderr)
#     #     print(", ".join(all_shapes), file=sys.stderr)
#     #     print("------------------------------------------", file=sys.stderr)
#     #     sys.exit(1)

#     lasa_shape = load_shape(args.shape)
    
#     # Data is (n_demos, n_samples, dim). SEDS paper assumes target is at origin.
#     pos_data = lasa_shape.pos
#     vel_data = lasa_shape.vel

#     # Find the target (common endpoint)
#     # Use the last point of the first demo as the target
#     target = pos_data[0, -1, :]
    
#     # Center all data at the target
#     demos_x = pos_data - target
#     demos_dx = vel_data # Velocities are unaffected by translation

#     n_demos, n_samples, d = demos_x.shape # d is n_dim
#     print(f"Data loaded: {n_demos} demos, {n_samples} samples, {d} dimensions")

#     # --- 2. Train Model ---
#     seds = SEDS(n_gaussians=args.k, dim=d)
#     print(f"Initializing SEDS with {args.k} Gaussians.")
    
#     print("Training SEDS model... (This may take a minute)")
#     success = seds.train(demos_x, demos_dx)
    
#     if not success:
#         print("Model training failed. Exiting.")
#         sys.exit(1)
    
#     print("Training complete.")

#     # --- 3. Simulate and Plot ---
#     print("Simulating trajectories from demonstration starting points...")
#     sim_trajs = []
#     for i in range(n_demos):
#         x_start = demos_x[i, 0, :] # Starting point of demo i
#         traj, _ = seds.simulate(x_start, dt=args.dt, n_steps=args.n_steps)
#         sim_trajs.append(traj)

#     print("Plotting results...")
    
#     # --- New Plotting Section ---
#     fig, axes = plt.subplots(1, 3, figsize=(24, 7))
#     fig.suptitle(f'SEDS Training for "{args.shape}" with {args.k} Gaussians', fontsize=16) # Use args.k

#     # Plot 1: Original Demonstrations
#     ax1 = axes[0]
#     ax1.set_title("Original Demonstrations (Centered)")
#     for i, demo in enumerate(demos_x):
#         ax1.plot(demo[:, 0], demo[:, 1], 'b-', alpha=0.6, label="Demonstration" if i == 0 else None)
#         ax1.scatter(demo[0, 0], demo[0, 1], c='b', marker='o') # Start
#     ax1.scatter(0, 0, c='r', marker='*', s=200, label='Target') # Target
#     ax1.set_xlabel("x1")
#     ax1.set_ylabel("x2")
#     ax1.legend()
#     ax1.axis('equal')

#     # Plot 2: SEDS Reproductions
#     ax2 = axes[1]
#     ax2.set_title("SEDS Reproductions")
#     for i, traj in enumerate(sim_trajs):
#         ax2.plot(traj[:, 0], traj[:, 1], 'g-', alpha=0.8, label="SEDS Simulation" if i == 0 else None)
#         ax2.scatter(traj[0, 0], traj[0, 1], c='g', marker='o') # Start
#     ax2.scatter(0, 0, c='r', marker='*', s=200, label='Target') # Target
#     ax2.set_xlabel("x1")
#     ax2.set_ylabel("x2")
#     ax2.legend()
#     ax2.axis('equal')

#     # Plot 3: Learned Vector Field
#     ax3 = axes[2]
#     ax3.set_title("Learned Dynamical System")
    
#     # Create a grid to visualize the vector field
#     xlim = ax1.get_xlim()
#     ylim = ax1.get_ylim()
#     padding = 1.0
#     x_grid = np.linspace(xlim[0] - padding, xlim[1] + padding, 20)
#     y_grid = np.linspace(ylim[0] - padding, ylim[1] + padding, 20)
#     X, Y = np.meshgrid(x_grid, y_grid)
#     grid_points = np.vstack([X.ravel(), Y.ravel()]).T
    
#     # Predict velocities on the grid
#     DX = seds.predict(grid_points)
#     DX_grid = DX.reshape(X.shape[0], X.shape[1], d) # Use d (n_dim)
    
#     # Plot streamlines
#     ax3.streamplot(X, Y, DX_grid[:, :, 0], DX_grid[:, :, 1], color='k', linewidth=0.5, density=1.5)
    
#     # Overlay original demos for context
#     for demo in demos_x:
#         ax3.plot(demo[:, 0], demo[:, 1], 'b-', alpha=0.3)
#     ax3.scatter(0, 0, c='r', marker='*', s=200, label='Target') # Target
#     ax3.set_xlabel("x1")
#     ax3.set_ylabel("x2")
#     ax3.legend()
#     ax3.axis('equal')
#     ax3.set_xlim(xlim[0] - padding, xlim[1] + padding)
#     ax3.set_ylim(ylim[0] - padding, ylim[1] + padding)
    
#     plt.tight_layout()
#     # --- End New Plotting Section ---

#     # --- 4. Save Artifacts ---
#     if args.outdir:
#         try:
#             # Create the full output directory path
#             save_dir = os.path.join(args.outdir)
            
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#                 print(f"Created directory: {save_dir}")

#             # Auto-generate file names based on hyperparams
#             base_name = f"{args.shape}_k{args.k}" # Use args.k
#             model_path = os.path.join(save_dir, f"{base_name}_model.pkl")
#             plot_path = os.path.join(save_dir, f"{base_name}_plots.png")
                
#             # Save the model object
#             seds.save_model(model_path) # Use the class method
#             print(f"Model saved to {model_path}")
            
#             # Save the figure
#             fig.savefig(plot_path)
#             print(f"Plots saved to {plot_path}")
            
#         except Exception as e:
#             print(f"Error saving artifacts: {e}")

#     # 5. Show plots unless disabled
#     if not args.no_show_plots:
#         plt.show()

# # --- Main execution ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train SEDS on LASA handwriting dataset.")
    
#     # Get available shapes for help text
#     try:
#         shape_names = get_shape_names()
#         shape_help = "Name of the LASA shape to train on."
#     except Exception:
#         shape_names = []
#         shape_help = "Name of the LASA shape to train on (e.g., 'Worm')."

#     parser.add_argument("--shape", type=str, default="Sshape", help=shape_help)
#     parser.add_argument("-k", "--k", type=int, default=6, dest="k", help="Number of Gaussian components (K). Default: 6")
    
#     # Simulation params
#     parser.add_argument("--dt", type=float, default=0.01, help="Simulation time step. Default: 0.01")
#     parser.add_argument("--n_steps", type=int, default=1000, help="Number of simulation steps. Default: 300")

#     # Saving
#     parser.add_argument("-o", "--outdir", type=str, default=None, help="Directory to save the trained model (.pkl) and plots (.png).")
#     parser.add_argument("--no_show_plots", action='store_true', help="Do not show the plot window at the end.")

#     parsed_args = parser.parse_args()
    
#     # Re-check shape name now that imports are loaded
#     if parsed_args.shape not in shape_names and len(shape_names) > 0:
#         print(f"\n--- ERROR: Shape '{parsed_args.shape}' not found ---", file=sys.stderr)
#         print("Available shapes are:", file=sys.stderr)
#         print(", ".join(shape_names), file=sys.stderr)
#         print("------------------------------------------", file=sys.stderr)
#         sys.exit(1)
        
#     main(parsed_args)

