import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
import pickle
import warnings # Import warnings
from pathlib import Path

class SEDS:
    """
    Implementation of the Stable Estimator of Dynamical Systems (SEDS)
    using Gaussian Mixture Models (GMM) and constrained optimization,
    as described in the paper:
    "Learning Stable Nonlinear Dynamical Systems With Gaussian Mixture Models"
    (Khansari-Zadeh and Billard, 2011)

    This implementation uses the SEDS-Likelihood approach (Section V-A).
    """
    def __init__(self, n_gaussians, dim=2, random_state=42):
        self.K = n_gaussians  # Number of Gaussian components
        self.d = dim          # Dimensionality of the data (e.g., 2 for 2D points)
        self.target = np.zeros(self.d) # Target/attractor, assumed to be at the origin
        self.random_state = random_state

        # Model parameters to be learned (Eq. 8)
        self.priors = None  # pi_k (K,)
        self.mu = None      # mu_k (K, 2*d)
        self.sigma = None   # Sigma_k (K, 2*d, 2*d)

        # Optimized, stable parameters (Eq. 13)
        self.A_k = np.zeros((self.K, self.d, self.d))  # (K, d, d)
        self.b_k = np.zeros((self.K, self.d))         # (K, d)

        # Pre-calculated values for simulation
        self.sigma_xi_inv = np.zeros((self.K, self.d, self.d)) # (K, d, d)

        # Internal state for optimization progress
        self._optimizer_iteration = 0


    def gmm_init(self, demos_x, demos_dx):
        """
        Initializes the GMM parameters (priors, mu, sigma) using
        scikit-learn's GaussianMixture model on the concatenated data.

        Args:
            demos_x (np.ndarray): Position data (N_total_samples, d)
            demos_dx (np.ndarray): Velocity data (N_total_samples, d)
        """
        X_gmm = np.hstack([demos_x, demos_dx]) # Shape (N_total_samples, 2*d)

        try:
            # Fit a GMM to the data
            gmm = GaussianMixture(
                n_components=self.K,
                covariance_type='full',
                random_state=self.random_state,
                n_init=3 # Add n_init for robustness
            )
            gmm.fit(X_gmm)

            # Store GMM parameters
            self.priors = gmm.weights_
            self.mu = gmm.means_
            self.sigma = gmm.covariances_

            # Pre-calculate Sigma_xi_inv for h_k (Eq. 8)
            for k in range(self.K):
                sigma_xi = self.sigma[k, :self.d, :self.d]
                # Ensure symmetry before inversion
                sigma_xi_sym = 0.5 * (sigma_xi + sigma_xi.T)
                # Add regularization to avoid singular matrix errors
                sigma_xi_reg = sigma_xi_sym + np.eye(self.d) * 1e-6
                try:
                    # Check positive definiteness before inverting
                    np.linalg.cholesky(sigma_xi_reg)
                    self.sigma_xi_inv[k] = np.linalg.inv(sigma_xi_reg)
                except np.linalg.LinAlgError:
                    warnings.warn(f"GMM init Sigma_xi[{k}] not positive definite after regularization. Using Identity.")
                    self.sigma_xi_inv[k] = np.eye(self.d)


        except Exception as e:
            print(f"--- ERROR in GMM Initialization ---")
            print(f"Failed to fit GaussianMixture model. Error: {e}")
            print(f"Data shape: {X_gmm.shape}")
            if np.any(np.isnan(X_gmm)) or np.any(np.isinf(X_gmm)):
                print("Error: Data contains NaN or Inf values.")
            return False
        return True


    def _calculate_A_b_unconstrained(self):
        """
        Calculates the unconstrained A_k and b_k parameters (Eq. 8)
        from the initialized GMM parameters. This is the starting
        point for the constrained optimization.
        """
        # Mu_k = [mu_xi_k, mu_dx_k]
        mu_xi = self.mu[:, :self.d]    # (K, d)
        mu_dx = self.mu[:, self.d:]    # (K, d)

        # Sigma_k = [[Sigma_xi,  Sigma_xidx],
        #            [Sigma_dxxi, Sigma_dx]]
        # Sigma_xi = self.sigma[:, :self.d, :self.d]        # (K, d, d) # Not needed directly
        # Sigma_xidx = self.sigma[:, :self.d, self.d:]      # (K, d, d) # Not needed directly
        Sigma_dxxi = self.sigma[:, self.d:, :self.d]      # (K, d, d)
        # Sigma_dx = self.sigma[:, self.d:, self.d:]        # (K, d, d) # Not needed here

        # A_k = Sigma_dx_xi * Sigma_xi^-1
        # b_k = mu_dx_k - A_k * mu_xi_k
        A_k_init = np.zeros((self.K, self.d, self.d))
        b_k_init = np.zeros((self.K, self.d))

        for k in range(self.K):
            # We pre-calculated sigma_xi_inv in gmm_init
            inv_sigma_xi_k = self.sigma_xi_inv[k] # Already regularized and inverted
            sigma_dxxi_k = Sigma_dxxi[k]

            A_k_init[k] = sigma_dxxi_k @ inv_sigma_xi_k
            b_k_init[k] = mu_dx[k] - A_k_init[k] @ mu_xi[k]

        return A_k_init, b_k_init


    def _pack_params(self, A_k, b_k):
        """Helper to flatten A_k and b_k into a 1D vector for the optimizer."""
        return np.concatenate([A_k.ravel(), b_k.ravel()])

    def _unpack_params(self, params_vec):
        """Helper to unpack the 1D vector back into A_k and b_k."""
        try:
            A_k_flat = params_vec[:self.K * self.d * self.d]
            b_k_flat = params_vec[self.K * self.d * self.d:]

            A_k = A_k_flat.reshape((self.K, self.d, self.d))
            b_k = b_k_flat.reshape((self.K, self.d))
            return A_k, b_k
        except ValueError as e:
             raise ValueError(f"Error unpacking params_vec with length {len(params_vec)}. Expected {self.K*self.d*self.d + self.K*self.d}.") from e


    def _get_h_k(self, x):
        """
        Calculate the activation probabilities h_k(x) (Eq. 8).
        x shape can be (d,) or (N, d) for batch processing.

        Returns:
            h_k (np.ndarray): Shape (K,) or (N, K)
        """
        is_batch = x.ndim == 2
        x_proc = x if is_batch else x[np.newaxis, :]
        N = x_proc.shape[0]

        mu_xi = self.mu[:, :self.d] # Shape (K, d)

        # Calculate Mahalanobis distances efficiently using einsum
        # delta = x_proc[:, np.newaxis, :] - mu_xi[np.newaxis, :, :] # (N, K, d)
        # mahalanobis_sq = np.einsum('nkd,kdf,nkf->nk', delta, self.sigma_xi_inv, delta) # (N, K)

        # Alternative calculation (sometimes more stable or faster)
        mahalanobis_sq = np.zeros((N, self.K))
        for k in range(self.K):
             delta_k = x_proc - mu_xi[k] # (N, d)
             temp_k = delta_k @ self.sigma_xi_inv[k] # (N, d) @ (d, d) -> (N, d)
             mahalanobis_sq[:, k] = np.sum(temp_k * delta_k, axis=1) # (N,)

        exponent = -0.5 * mahalanobis_sq # Shape (N, K)

        # Calculate Gaussian PDF values (without constant factor initially)
        # N(x | mu, Sigma) propto exp(-0.5 * mahalanobis^2)
        # We need P(k) * P(x|k) = priors[k] * N(x|mu_xi[k], sigma_xi[k])
        # The constant factor cancels in normalization, but helps stability if calculated.
        # sigma_xi_det = np.linalg.det(self.sigma[:, :self.d, :self.d] + np.eye(self.d) * 1e-6) # Regularize determinant calc
        # const_factor = (2 * np.pi)**(-self.d / 2) * sigma_xi_det**(-0.5) # Shape (K,)
        # p_x_given_k = const_factor[np.newaxis, :] * np.exp(exponent) # Shape (N, K)

        # Let's try without the constant factor first as it cancels
        p_x_given_k_unnorm = np.exp(exponent) # Shape (N, K)

        # Numerator: P(k) * P(x|k)
        numerator = self.priors[np.newaxis, :] * p_x_given_k_unnorm # Shape (N, K)

        # Denominator: Sum over k [ P(k) * P(x|k) ]
        denominator = np.sum(numerator, axis=1, keepdims=True) # Shape (N, 1)

        # Add small epsilon for numerical stability
        h_k = numerator / (denominator + 1e-100) # Shape (N, K)

        # Check for NaNs/Infs which can indicate numerical issues
        if np.any(np.isnan(h_k)) or np.any(np.isinf(h_k)):
            warnings.warn("NaN or Inf detected in h_k calculation. Check GMM init or data scaling.")
            # Attempt to clean up? Replace NaNs with uniform? Or let it propagate?
            h_k = np.nan_to_num(h_k, nan=1.0/self.K) # Replace NaNs with uniform
            # Re-normalize after cleanup
            h_k /= (np.sum(h_k, axis=1, keepdims=True) + 1e-100)


        return h_k.squeeze(0) if not is_batch else h_k


    def _objective_function(self, params_vec, demos_x, demos_dx):
        """
        The SEDS-Likelihood objective function (Eq. 19), simplified to
        MSE (Eq. 21) for robustness and speed.

        This function is vectorized to be fast.

        Args:
            params_vec (np.ndarray): 1D vector of A_k and b_k
            demos_x (np.ndarray): (N, d)
            demos_dx (np.ndarray): (N, d)
        """
        try:
            A_k, b_k = self._unpack_params(params_vec)
            N = demos_x.shape[0]

            # --- Vectorized Implementation ---
            h_k_batch = self._get_h_k(demos_x) # (N, K)

            # f_all_k = A_k @ x + b_k : (N, K, d)
            # Use einsum: 'nd,kdj->nkj' = (N,d) @ (K,d,j=d) -> (N,K,j=d)
            f_all_k = np.einsum('nd,kdj->nkj', demos_x, A_k) + b_k[np.newaxis, :, :]

            # dx_pred = sum_k h_k * f_k : (N, d)
            # Use einsum: 'nk,nkd->nd'
            dx_pred_batch = np.einsum('nk,nkd->nd', h_k_batch, f_all_k)

            # Calculate total Mean Squared Error
            diff = demos_dx - dx_pred_batch
            total_mse = np.sum(diff**2) # Sum of squares

            # Return mean squared error
            mse = 0.5 * total_mse / N

            # Check for NaN/Inf in result
            if np.isnan(mse) or np.isinf(mse):
                 warnings.warn(f"Objective function resulted in NaN/Inf. Params: {params_vec[:5]}...")
                 return 1e12 # Return a large finite number instead of Inf

            return mse

        except Exception as e:
            warnings.warn(f"Exception in objective function: {e}. Returning large error.")
            # import traceback
            # traceback.print_exc() # For detailed debugging
            return 1e12 # Return a large finite value on error


    def _stability_constraints(self):
        """
        Defines the stability constraints (Eq. 13) for the optimizer.
        These are functions that take the parameter vector `p`.

        Returns:
            A list of dictionaries, one for each constraint.
        """
        constraints = []
        stability_margin = 1e-5 # Small positive value for strict inequality A+A' < 0

        for k in range(self.K):
            # 1. Constraint: b_k = 0 (since target is origin)
            # type 'eq' means con(p) == 0
            # We constrain the squared norm to be zero
            constraints.append({
                'type': 'eq',
                'fun': lambda p, k_idx=k: np.sum(self._unpack_params(p)[1][k_idx]**2)
            })

            # 2. Constraint: A_k + A_k.T <= -margin * I (Negative Definite)
            # type 'ineq' means con(p) >= 0
            # We require max_eigenvalue(A_k + A_k.T) <= -margin
            # => -max_eigenvalue - margin >= 0
            def neg_def_constraint(p, k_idx=k):
                A_k_local, _ = self._unpack_params(p)
                A_k_sym = 0.5 * (A_k_local[k_idx] + A_k_local[k_idx].T)
                try:
                    # Use eigvalsh for symmetric matrices
                    eigenvalues = np.linalg.eigvalsh(A_k_sym)
                    max_eig = np.max(eigenvalues)
                    # Constraint requires max_eig <= -stability_margin
                    # For solver (>= 0), return -max_eig - stability_margin
                    return -max_eig - stability_margin
                except np.linalg.LinAlgError:
                    warnings.warn(f"LinAlgError checking eigenvalues for A_k[{k_idx}]. Penalizing.")
                    return -1e6 # Return large negative value if eig calculation fails

            constraints.append({
                'type': 'ineq',
                'fun': neg_def_constraint
            })

        return constraints

    def train(self, demos_x, demos_dx):
        """
        Trains the SEDS model.
        1. Initializes GMM parameters.
        2. Sets up constrained optimization problem.
        3. Solves for stable A_k and b_k.

        Args:
            demos_x (np.ndarray): Position data (N_demos, N_samples, d)
            demos_dx (np.ndarray): Velocity data (N_demos, N_samples, d)

        Returns:
            bool: True if optimization finished (even if limit reached), False on critical error.
        """
        try:
            demos_x = np.asarray(demos_x, dtype=np.float64)
            demos_dx = np.asarray(demos_dx, dtype=np.float64)
        except Exception as e:
            print(f"--- ERROR in train(): Failed to convert input data to NumPy arrays. Error: {e}")
            return False

        try:
            n_demos, n_samples, dim = demos_x.shape
            if dim != self.d:
                print(f"--- ERROR in train(): Data dimension ({dim}) does not match model dimension ({self.d}).")
                return False
            demos_x_flat = demos_x.reshape((-1, self.d))
            demos_dx_flat = demos_dx.reshape((-1, self.d))
        except Exception as e:
            print(f"--- ERROR in train(): Failed to reshape data. Expected 3D array (N_demos, N_samples, d).")
            print(f"Got demos_x shape: {demos_x.shape}, Error: {e}")
            return False

        # 1. Initialize GMM (priors, mu, sigma)
        if not self.gmm_init(demos_x_flat, demos_dx_flat):
            print("--- ERROR in train(): GMM initialization failed.")
            return False # GMM init failed

        # 2. Get unconstrained A_k, b_k as starting point
        try:
            A_k_init, b_k_init = self._calculate_A_b_unconstrained()
        except Exception as e:
            print(f"--- ERROR in train(): Failed to calculate initial A_k, b_k: {e}")
            return False

        # 3. Define constraints and initial parameters for optimization
        # Force b_k=0 for stability constraint
        b_k_init.fill(0.0)
        initial_params = self._pack_params(A_k_init, b_k_init)
        constraints = self._stability_constraints() # Constraints are now functions of params_vec

        # --- Callback for progress tracking ---
        self._optimizer_iteration = 0
        def optimizer_callback(params_vec):
            self._optimizer_iteration += 1
            if self._optimizer_iteration % 10 == 0:
                # Use try-except as objective can fail during optimization iterations
                try:
                    mse = self._objective_function(params_vec, demos_x_flat, demos_dx_flat)
                    # Only print if mse is valid
                    if not (np.isnan(mse) or np.isinf(mse)):
                         print(f"[SEDS Train] Iter: {self._optimizer_iteration:4d}, Current MSE: {mse:.6f}")
                except Exception:
                    print(f"[SEDS Train] Iter: {self._optimizer_iteration:4d}, Error calculating MSE for callback.")
                    pass # Continue optimization even if callback MSE fails

        # --- Run the optimization ---
        print("Starting constrained optimization (SLSQP)...")
        optimization_succeeded = False # Flag for success status
        try:
            result = minimize(
                self._objective_function,
                initial_params,
                args=(demos_x_flat, demos_dx_flat),
                method='SLSQP',
                constraints=constraints,
                callback=optimizer_callback,
                # Increased maxiter, adjust ftol if needed
                options={'disp': False, 'maxiter': 1000, 'ftol': 1e-7}
            )

            # --- Check result status ---
            if result.success:
                final_mse = self._objective_function(result.x, demos_x_flat, demos_dx_flat)
                print(f"[SEDS Train] Optimization finished successfully. Final MSE: {final_mse:.6f}")
                optimization_succeeded = True
            # *** MODIFICATION START ***
            # Treat iteration limit not as failure, but save the result
            elif result.status == 9 or 'Iteration limit reached' in result.message:
                 final_mse = self._objective_function(result.x, demos_x_flat, demos_dx_flat)
                 print(f"--- WARNING: Optimization finished: Iteration limit reached ---")
                 print(f"    Message: {result.message}")
                 print(f"    MSE at final iteration: {final_mse:.6f}")
                 optimization_succeeded = True # Consider it a success for proceeding
            # *** MODIFICATION END ***
            else:
                 # Other statuses are treated as failures
                 print(f"--- ERROR: Optimization failed ---")
                 print(f"    Status: {result.status}")
                 print(f"    Message: {result.message}")
                 # You might want to inspect result.x here even on failure
                 optimization_succeeded = False


            # Store the final parameters regardless of exact success status, if we decided to proceed
            if optimization_succeeded:
                self.A_k, self.b_k = self._unpack_params(result.x)
                # Verify constraints on the final result (optional, for debugging)
                # final_constraints = self._stability_constraints()
                # for i, con in enumerate(final_constraints):
                #      val = con['fun'](result.x)
                #      print(f"Constraint {i} ({con['type']}): Value={val}")

            return optimization_succeeded # Return True if successful OR iteration limit reached

        except ValueError as ve:
             print(f"--- ERROR in Optimization (ValueError): {ve}")
             print("This might be due to issues unpacking parameters or constraint evaluation.")
             import traceback
             traceback.print_exc()
             return False
        except Exception as e:
            print(f"--- ERROR in Optimization ---")
            print(f"Optimization failed with unexpected error: {e}")
            print(f"Initial params (first 5): {initial_params[:5]}")
            import traceback
            traceback.print_exc()
            return False


    def predict(self, x):
        """
        Predicts the velocity (dx) at a given position (x).
        This is f(x) from Eq. 9.
        Handles both single points (d,) and batches (N, d).

        Args:
            x (np.ndarray): Position vector(s), shape (d,) or (N, d)

        Returns:
            np.ndarray: Predicted velocity vector(s), shape (d,) or (N, d)
        """
        if self.priors is None or self.A_k is None or self.b_k is None:
            raise RuntimeError("Model parameters are not initialized or trained. Call train() first.")

        try:
            is_batch = x.ndim == 2
            x_proc = x if is_batch else x[np.newaxis, :]
            N = x_proc.shape[0]

            h_k = self._get_h_k(x_proc) # Shape (N, K)

            # Vectorized f_k(x) = A_k*x + b_k
            # Use einsum: 'nd,kdj->nkj'
            f_all_k = np.einsum('nd,kdj->nkj', x_proc, self.A_k) + self.b_k[np.newaxis, :, :] # (N, K, d)

            # Vectorized f(x) = sum_k( h_k(x) * f_k(x) )
            # Use einsum: 'nk,nkd->nd'
            dx_pred = np.einsum('nk,nkd->nd', h_k, f_all_k) # (N, d)

            return dx_pred.squeeze(0) if not is_batch else dx_pred

        except Exception as e:
            print(f"Error during prediction for input shape {x.shape}: {e}")
            # Return zero velocity or raise error?
            return np.zeros_like(x)


    def simulate(self, x_start, dt, n_steps):
        """
        Simulates a trajectory by integrating the learned dynamics using Euler method.

        Args:
            x_start (np.ndarray): Starting position (d,)
            dt (float): Time step for integration
            n_steps (int): Number of steps to simulate (results in n_steps + 1 points)

        Returns:
            (np.ndarray, np.ndarray):
                - trajectory (np.ndarray): The simulated positions (n_steps + 1, d)
                - velocities (np.ndarray): The predicted velocities at each position (n_steps + 1, d)
        """
        if self.A_k is None:
             raise RuntimeError("Model is not trained. Cannot simulate.")

        # Allocate arrays for n_steps + 1 points (time 0 to time n_steps*dt)
        trajectory = np.zeros((n_steps + 1, self.d))
        velocities = np.zeros((n_steps + 1, self.d))

        trajectory[0] = x_start
        current_x = x_start.copy()

        # Convergence thresholds (stricter)
        pos_threshold_sq = (1e-5)**2
        vel_threshold_sq = (1e-5)**2

        # Predict velocity at the start point
        current_dx = self.predict(current_x)
        velocities[0] = current_dx

        for t in range(n_steps): # Loop n_steps times (indices 0 to n_steps-1)
            # Check for invalid state before proceeding
            if np.any(np.isnan(current_x)) or np.any(np.isinf(current_x)):
                 warnings.warn(f"Simulation state became invalid (NaN/Inf) at step {t}. Stopping early.")
                 # Fill remaining steps with last valid state or target? Let's use target.
                 trajectory[t+1:] = self.target
                 velocities[t+1:] = 0.0
                 break

            # Euler integration step
            next_x = current_x + current_dx * dt

            # Predict velocity at the *next* position for the next iteration
            next_dx = self.predict(next_x)

            # Store results for the *next* time step (index t+1)
            trajectory[t+1] = next_x
            velocities[t+1] = next_dx # Store velocity predicted *at* next_x

            # Check for convergence after updating state
            pos_diff_sq = np.sum((next_x - self.target)**2)
            vel_sq = np.sum(next_dx**2)

            if pos_diff_sq < pos_threshold_sq and vel_sq < vel_threshold_sq:
                # If converged, fill remaining steps and stop
                # print(f"Converged at step {t+1}") # Debugging convergence
                trajectory[t+1:] = self.target # Fill remaining positions
                velocities[t+1:] = 0.0         # Fill remaining velocities
                break

            # Update state for the next iteration
            current_x = next_x
            current_dx = next_dx

        return trajectory, velocities


    # def save_model(self, filepath):
    #     """Saves the trained model to a file using pickle."""
    #     # Ensure filepath is a Path object or convert string
    #     filepath = Path(filepath)
    #     try:
    #         with open(filepath, 'wb') as f:
    #             pickle.dump(self, f)
    #         print(f"Model saved successfully to {filepath}")
    #     except Exception as e:
    #         print(f"Error saving model to {filepath}: {e}")

    # @staticmethod
    # def load_model(filepath):
    #     """Loads a trained model from a file."""
    #     # Ensure filepath is a Path object or convert string
    #     filepath = Path(filepath)
    #     try:
    #         with open(filepath, 'rb') as f:
    #             model = pickle.load(f)
    #         # print(f"Model loaded successfully from {filepath}") # Can be noisy
    #         return model
    #     except FileNotFoundError:
    #          print(f"Error loading model: File not found at {filepath}")
    #          return None
    #     except Exception as e:
    #         print(f"Error loading model from {filepath}: {e}")
    #         return None
    def save_model(self, filepath):
        """Saves the trained model to a file using pickle."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"Model saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving model to {filepath}: {e}")

    @staticmethod
    def load_model(filepath):
        """Loads a trained model from a file."""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            # print(f"Model loaded successfully from {filepath}") # Too noisy for exp script
            return model
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            return None

# import numpy as np
# from sklearn.mixture import GaussianMixture
# from scipy.optimize import minimize
# import pickle

# class SEDS:
#     """
#     Implementation of the Stable Estimator of Dynamical Systems (SEDS)
#     using Gaussian Mixture Models (GMM) and constrained optimization,
#     as described in the paper:
#     "Learning Stable Nonlinear Dynamical Systems With Gaussian Mixture Models"
#     (Khansari-Zadeh and Billard, 2011)

#     This implementation uses the SEDS-Likelihood approach (Section V-A).
#     """
#     def __init__(self, n_gaussians, dim=2, random_state=42):
#         self.K = n_gaussians  # Number of Gaussian components
#         self.d = dim          # Dimensionality of the data (e.g., 2 for 2D points)
#         self.target = np.zeros(self.d) # Target/attractor, assumed to be at the origin
#         self.random_state = random_state

#         # Model parameters to be learned (Eq. 8)
#         self.priors = None  # pi_k (K,)
#         self.mu = None      # mu_k (K, 2*d)
#         self.sigma = None   # Sigma_k (K, 2*d, 2*d)

#         # Optimized, stable parameters (Eq. 13)
#         self.A_k = np.zeros((self.K, self.d, self.d))  # (K, d, d)
#         self.b_k = np.zeros((self.K, self.d))         # (K, d)

#         # Pre-calculated values for simulation
#         self.sigma_xi_inv = np.zeros((self.K, self.d, self.d)) # (K, d, d)
        
#         # Internal state for optimization progress
#         self._optimizer_iteration = 0


#     def gmm_init(self, demos_x, demos_dx):
#         """
#         Initializes the GMM parameters (priors, mu, sigma) using
#         scikit-learn's GaussianMixture model on the concatenated data.

#         Args:
#             demos_x (np.ndarray): Position data (N_total_samples, d)
#             demos_dx (np.ndarray): Velocity data (N_total_samples, d)
#         """
#         X_gmm = np.hstack([demos_x, demos_dx]) # Shape (N_total_samples, 2*d)

#         try:
#             # Fit a GMM to the data
#             gmm = GaussianMixture(
#                 n_components=self.K,
#                 covariance_type='full',
#                 random_state=self.random_state
#             )
#             gmm.fit(X_gmm)

#             # Store GMM parameters
#             # FIX: The scikit-learn GMM attribute for priors is .weights_, not .priors_
#             self.priors = gmm.weights_
#             self.mu = gmm.means_
#             self.sigma = gmm.covariances_

#             # Pre-calculate Sigma_xi_inv for h_k (Eq. 8)
#             for k in range(self.K):
#                 sigma_xi = self.sigma[k, :self.d, :self.d]
#                 # Add regularization to avoid singular matrix errors
#                 sigma_xi += np.eye(self.d) * 1e-6
#                 self.sigma_xi_inv[k] = np.linalg.inv(sigma_xi)

#         except Exception as e:
#             print(f"--- ERROR in GMM Initialization ---")
#             print(f"Failed to fit GaussianMixture model. Error: {e}")
#             print(f"Data shape: {X_gmm.shape}")
#             if np.any(np.isnan(X_gmm)) or np.any(np.isinf(X_gmm)):
#                 print("Error: Data contains NaN or Inf values.")
#             return False
#         return True


#     def _calculate_A_b_unconstrained(self):
#         """
#         Calculates the unconstrained A_k and b_k parameters (Eq. 8)
#         from the initialized GMM parameters. This is the starting
#         point for the constrained optimization.
#         """
#         # Mu_k = [mu_xi_k, mu_dx_k]
#         mu_xi = self.mu[:, :self.d]    # (K, d)
#         mu_dx = self.mu[:, self.d:]    # (K, d)

#         # Sigma_k = [[Sigma_xi,  Sigma_xidx],
#         #            [Sigma_dxxi, Sigma_dx]]
#         Sigma_xi = self.sigma[:, :self.d, :self.d]        # (K, d, d)
#         Sigma_xidx = self.sigma[:, :self.d, self.d:]      # (K, d, d)
#         Sigma_dxxi = self.sigma[:, self.d:, :self.d]      # (K, d, d)
#         # Sigma_dx = self.sigma[:, self.d:, self.d:]        # (K, d, d) # Not needed here

#         # A_k = Sigma_dx_xi * Sigma_xi^-1
#         # b_k = mu_dx_k - A_k * mu_xi_k
#         A_k_init = np.zeros((self.K, self.d, self.d))
#         b_k_init = np.zeros((self.K, self.d))

#         for k in range(self.K):
#             # We pre-calculated sigma_xi_inv in gmm_init
#             inv_sigma_xi_k = self.sigma_xi_inv[k]
#             sigma_dxxi_k = Sigma_dxxi[k]

#             A_k_init[k] = sigma_dxxi_k @ inv_sigma_xi_k
#             b_k_init[k] = mu_dx[k] - A_k_init[k] @ mu_xi[k]

#         return A_k_init, b_k_init


#     def _pack_params(self, A_k, b_k):
#         """Helper to flatten A_k and b_k into a 1D vector for the optimizer."""
#         return np.concatenate([A_k.ravel(), b_k.ravel()])

#     def _unpack_params(self, params_vec):
#         """Helper to unpack the 1D vector back into A_k and b_k."""
#         A_k_flat = params_vec[:self.K * self.d * self.d]
#         b_k_flat = params_vec[self.K * self.d * self.d:]
        
#         A_k = A_k_flat.reshape((self.K, self.d, self.d))
#         b_k = b_k_flat.reshape((self.K, self.d))
#         return A_k, b_k

#     def _get_h_k(self, x):
#         """
#         Calculate the activation probabilities h_k(x) (Eq. 8).
#         x shape can be (d,) or (N, d) for batch processing.
        
#         Returns:
#             h_k (np.ndarray): Shape (K,) or (N, K)
#         """
#         # --- Vectorization Fix ---
#         # Handle both single-point (d,) and batch (N, d) inputs
#         is_batch = x.ndim == 2
#         if not is_batch:
#             x = x[np.newaxis, :]  # Promote to (1, d) batch
            
#         N = x.shape[0]
#         # --- End Fix ---
        
#         N_k = np.zeros((N, self.K)) # Shape (N, K)
#         mu_xi = self.mu[:, :self.d] # (K, d)

#         for k in range(self.K):
#             delta = x - mu_xi[k] # (N, d) - (d,) -> (N, d)
            
#             # Vectorized Mahalanobis distance calculation
#             # exponent = -0.5 * delta.T @ self.sigma_xi_inv[k] @ delta # Old non-batch way
#             temp = delta @ self.sigma_xi_inv[k] # (N, d) @ (d, d) -> (N, d)
#             exponent = -0.5 * np.sum(temp * delta, axis=1) # (N,)
            
#             N_k[:, k] = self.priors[k] * np.exp(exponent)

#         # Normalize probabilities for each point in the batch
#         sum_N_k = np.sum(N_k, axis=1) # (N,)
#         safe_sums = np.maximum(sum_N_k, 1e-100) # Avoid division by zero
#         h_k = N_k / safe_sums[:, np.newaxis] # (N, K) / (N, 1) -> (N, K)
        
#         if not is_batch:
#             return h_k.squeeze(0) # Return (K,)
#         return h_k # Return (N, K)

#     def _objective_function(self, params_vec, demos_x, demos_dx):
#         """
#         The SEDS-Likelihood objective function (Eq. 19), simplified to
#         MSE (Eq. 21) for robustness and speed.
        
#         This function is vectorized to be fast.
        
#         Args:
#             params_vec (np.ndarray): 1D vector of A_k and b_k
#             demos_x (np.ndarray): (N, d)
#             demos_dx (np.ndarray): (N, d)
#         """
#         A_k, b_k = self._unpack_params(params_vec)
#         N = demos_x.shape[0]

#         # --- Vectorized Implementation ---
        
#         # 1. Calculate all h_k(x) for all N points at once
#         # h_k_batch shape: (N, K)
#         h_k_batch = self._get_h_k(demos_x)
        
#         # 2. Calculate all f_k(x_n) for all N points and K components
#         # f_all_k shape: (N, K, d)
#         # 'nd,kdj->nkj' = (N,d) @ (K,d,j=d) -> (N,K,j=d)
#         f_all_k = np.einsum('nd,kdj->nkj', demos_x, A_k) + b_k[np.newaxis, :, :]
        
#         # 3. Calculate all predicted velocities f(x_n)
#         # dx_pred_batch shape: (N, d)
#         # (N, K, 1) * (N, K, d) -> sum over K -> (N, d)
#         dx_pred_batch = np.sum(h_k_batch[:, :, np.newaxis] * f_all_k, axis=1)

#         # 4. Calculate total Mean Squared Error
#         total_mse = np.sum((demos_dx - dx_pred_batch)**2)
        
#         return 0.5 * total_mse / N # Return mean squared error
#         # --- End Vectorized Implementation ---


#     def _stability_constraints(self, params_vec):
#         """
#         Defines the stability constraints (Eq. 13) for the optimizer.
        
#         Returns:
#             A list of dictionaries, one for each constraint.
#         """
#         A_k, b_k = self._unpack_params(params_vec)
#         constraints = []

#         for k in range(self.K):
#             # 1. Constraint: b_k = -A_k * target
#             # Since target is origin (0), this simplifies to b_k = 0
#             # We'll enforce this as an equality constraint.
#             # type 'eq' means con(params) == 0
#             def con_b_k(params, k=k):
#                 _, b_k_local = self._unpack_params(params)
#                 return np.sum(b_k_local[k]**2) # Enforce b_k[k] == 0
            
#             constraints.append({'type': 'eq', 'fun': con_b_k})

#             # 2. Constraint: A_k + A_k.T < 0 (Negative Definite)
#             # This is enforced by requiring all eigenvalues of the
#             # symmetric part (A_k + A_k.T) to be negative.
#             # We set a small margin (e.g., -1e-6) to ensure strict negativity.
#             # type 'ineq' means con(params) >= 0
#             def con_A_k_eig(params, k=k):
#                 A_k_local, _ = self._unpack_params(params)
#                 A_k_sym = 0.5 * (A_k_local[k] + A_k_local[k].T)
#                 eigenvalues = np.linalg.eigvalsh(A_k_sym)
#                 # We want eigenvalues <= -margin
#                 # So, -eigenvalues - margin >= 0
#                 margin = 1e-5
#                 return -eigenvalues - margin
            
#             # This creates 'd' inequality constraints for each component k
#             for i in range(self.d):
#                 constraints.append({
#                     'type': 'ineq',
#                     'fun': lambda params, k=k, i=i: con_A_k_eig(params, k)[i]
#                 })

#         return constraints

#     def train(self, demos_x, demos_dx):
#         """
#         Trains the SEDS model.
#         1. Initializes GMM parameters.
#         2. Sets up constrained optimization problem.
#         3. Solves for stable A_k and b_k.
        
#         Args:
#             demos_x (np.ndarray): Position data (N_demos, N_samples, d)
#             demos_dx (np.ndarray): Velocity data (N_demos, N_samples, d)
#         """
        
#         # --- FIX: Convert inputs to numpy arrays ---
#         # This makes the function robust if passed lists instead of arrays
#         try:
#             demos_x = np.asarray(demos_x, dtype=np.float64)
#             demos_dx = np.asarray(demos_dx, dtype=np.float64)
#         except Exception as e:
#             print(f"--- ERROR in train() ---")
#             print(f"Failed to convert input data to NumPy arrays. Error: {e}")
#             return False
#         # --- End Fix ---

#         # Reshape data from (N_demos, N_samples, d) to (N_total_samples, d)
#         # This is the format required by sklearn GMM and the objective function
#         try:
#             n_demos, n_samples, dim = demos_x.shape
#             if dim != self.d:
#                 print(f"--- ERROR in train() ---")
#                 print(f"Data dimension ({dim}) does not match model dimension ({self.d}).")
#                 return False
                
#             demos_x_flat = demos_x.reshape((n_demos * n_samples, self.d))
#             demos_dx_flat = demos_dx.reshape((n_demos * n_samples, self.d))
#         except Exception as e:
#             print(f"--- ERROR in train() ---")
#             print(f"Failed to reshape data. Expected 3D array (N_demos, N_samples, d).")
#             print(f"Got demos_x shape: {demos_x.shape}")
#             print(f"Error: {e}")
#             return False

#         # 1. Initialize GMM (priors, mu, sigma)
#         if not self.gmm_init(demos_x_flat, demos_dx_flat):
#             return False # GMM init failed

#         # 2. Get unconstrained A_k, b_k as starting point
#         A_k_init, b_k_init = self._calculate_A_b_unconstrained()
        
#         # 3. Define constraints
#         # We must satisfy b_k = 0 for stability at origin
#         # We can just set b_k_init to zero for the optimization
#         b_k_init.fill(0.0)
        
#         initial_params = self._pack_params(A_k_init, b_k_init)
        
#         constraints = self._stability_constraints(initial_params)
        
#         # --- Add callback for progress tracking ---
#         self._optimizer_iteration = 0
#         def optimizer_callback(params_vec):
#             self._optimizer_iteration += 1
#             if self._optimizer_iteration % 10 == 0:
#                 mse = self._objective_function(params_vec, demos_x_flat, demos_dx_flat)
#                 print(f"[SEDS Train] Iter: {self._optimizer_iteration}, Current MSE: {mse:.6f}")
#         # --- End callback ---

#         # 4. Run the optimization
#         # We use 'SLSQP' as it handles both equality and inequality constraints
#         try:
#             result = minimize(
#                 self._objective_function,
#                 initial_params,
#                 args=(demos_x_flat, demos_dx_flat),
#                 method='SLSQP',
#                 constraints=constraints,
#                 callback=optimizer_callback, # Add the callback here
#                 options={'disp': False, 'maxiter': 500} # Set disp=False
#             )
            
#             if not result.success:
#                 print(f"--- WARNING: Optimization failed ---")
#                 print(f"Message: {result.message}")
#                 # We can still proceed with the result, but it may not be optimal
#             else:
#                 final_mse = self._objective_function(result.x, demos_x_flat, demos_dx_flat)
#                 print(f"[SEDS Train] Optimization finished. Final MSE: {final_mse:.6f}")

#             # Store the optimized parameters
#             self.A_k, self.b_k = self._unpack_params(result.x)
#             return result.success

#         except Exception as e:
#             print(f"--- ERROR in Optimization ---")
#             print(f"Optimization failed with error: {e}")
#             print(f"Initial A_k[0]:\n{A_k_init[0]}")
#             return False

#     def predict(self, x):
#         """
#         Predicts the velocity (dx) at a given position (x).
#         This is f(x) from Eq. 9.
#         Handles both single points (d,) and batches (N, d).

#         Args:
#             x (np.ndarray): Position vector(s), shape (d,) or (N, d)
        
#         Returns:
#             np.ndarray: Predicted velocity vector(s), shape (d,) or (N, d)
#         """
#         if self.priors is None:
#             raise RuntimeError("Model is not trained. Call train() first.")
        
#         # --- Vectorization Fix ---
#         is_batch = x.ndim == 2
#         if not is_batch:
#             x = x[np.newaxis, :] # Promote to (1, d) batch
        
#         N = x.shape[0]
            
#         h_k = self._get_h_k(x) # Shape (N, K)
        
#         # Vectorized f_k(x) = A_k*x + b_k
#         # f_all_k[n, k, d] = A_k[k] @ x[n] + b_k[k]
#         # Use einsum for batch matrix multiplication: (N, d) @ (K, d, d) -> (N, K, d)
#         # 'nd,kdj->nkj' = (N,d) @ (K,d,j=d) -> (N,K,j=d)
#         f_all_k = np.einsum('nd,kdj->nkj', x, self.A_k) + self.b_k[np.newaxis, :, :]
        
#         # Vectorized f(x) = sum_k( h_k(x) * f_k(x) )
#         # (N, K, 1) * (N, K, d) -> sum over K -> (N, d)
#         dx_pred = np.sum(h_k[:, :, np.newaxis] * f_all_k, axis=1)

#         if not is_batch:
#             return dx_pred.squeeze(0) # Return (d,)
#         return dx_pred # Return (N, d)
#         # --- End Fix ---

#     def simulate(self, x_start, dt, n_steps):
#         """
#         Simulates a trajectory by integrating the learned dynamics.

#         Args:
#             x_start (np.ndarray): Starting position (d,)
#             dt (float): Time step for integration
#             n_steps (int): Number of steps to simulate (e.g., n_steps=1000 means 1001 points)
        
#         Returns:
#             (np.ndarray, np.ndarray): 
#                 - The simulated trajectory (n_steps + 1, d)
#                 - The predicted velocities (n_steps + 1, d)
#         """
#         # Create array for n_steps + 1 points (0 to n_steps inclusive)
#         trajectory = np.zeros((n_steps + 1, self.d))
#         velocities = np.zeros((n_steps + 1, self.d))
        
#         trajectory[0] = x_start
#         current_x = x_start.copy()
        
#         # Get velocity at the start point
#         current_dx = self.predict(current_x)
#         velocities[0] = current_dx
        
#         for t in range(n_steps): # Loop n_steps times (from 0 to n_steps-1)
            
#             # Euler integration
#             current_x = current_x + current_dx * dt
            
#             # Predict velocity at the *new* position
#             current_dx = self.predict(current_x)
            
#             # Stop if we've reached the target
#             # Check velocity as well to prevent premature stopping
#             if np.linalg.norm(current_x - self.target) < 1e-4 and np.linalg.norm(current_dx) < 1e-4:
#                 trajectory[t+1:] = self.target # Fill remaining steps with target
#                 velocities[t+1:] = 0.0         # Fill remaining velocities with zero
#                 break
            
#             trajectory[t+1] = current_x
#             velocities[t+1] = current_dx
                
#         return trajectory, velocities

#     def save_model(self, filepath):
#         """Saves the trained model to a file using pickle."""
#         try:
#             with open(filepath, 'wb') as f:
#                 pickle.dump(self, f)
#             print(f"Model saved successfully to {filepath}")
#         except Exception as e:
#             print(f"Error saving model to {filepath}: {e}")

#     @staticmethod
#     def load_model(filepath):
#         """Loads a trained model from a file."""
#         try:
#             with open(filepath, 'rb') as f:
#                 model = pickle.load(f)
#             # print(f"Model loaded successfully from {filepath}") # Too noisy for exp script
#             return model
#         except Exception as e:
#             print(f"Error loading model from {filepath}: {e}")
#             return None




# # 
# # import numpy as np
# # from sklearn.mixture import GaussianMixture
# # from scipy.optimize import minimize
# # import pickle

# # class SEDS:
# #     """
# #     Implementation of the Stable Estimator of Dynamical Systems (SEDS)
# #     using Gaussian Mixture Models (GMM) and constrained optimization,
# #     as described in the paper:
# #     "Learning Stable Nonlinear Dynamical Systems With Gaussian Mixture Models"
# #     (Khansari-Zadeh and Billard, 2011)

# #     This implementation uses the SEDS-Likelihood approach (Section V-A).
# #     """
# #     def __init__(self, n_gaussians, dim=2, random_state=42):
# #         self.K = n_gaussians  # Number of Gaussian components
# #         self.d = dim          # Dimensionality of the data (e.g., 2 for 2D points)
# #         self.target = np.zeros(self.d) # Target/attractor, assumed to be at the origin
# #         self.random_state = random_state

# #         # Model parameters to be learned (Eq. 8)
# #         self.priors = None  # pi_k (K,)
# #         self.mu = None      # mu_k (K, 2*d)
# #         self.sigma = None   # Sigma_k (K, 2*d, 2*d)

# #         # Optimized, stable parameters (Eq. 13)
# #         self.A_k = np.zeros((self.K, self.d, self.d))  # (K, d, d)
# #         self.b_k = np.zeros((self.K, self.d))         # (K, d)

# #         # Pre-calculated values for simulation
# #         self.sigma_xi_inv = np.zeros((self.K, self.d, self.d)) # (K, d, d)
        
# #         # Internal state for optimization progress
# #         self._optimizer_iteration = 0


# #     def gmm_init(self, demos_x, demos_dx):
# #         """
# #         Initializes the GMM parameters (priors, mu, sigma) using
# #         scikit-learn's GaussianMixture model on the concatenated data.

# #         Args:
# #             demos_x (np.ndarray): Position data (N_total_samples, d)
# #             demos_dx (np.ndarray): Velocity data (N_total_samples, d)
# #         """
# #         X_gmm = np.hstack([demos_x, demos_dx]) # Shape (N_total_samples, 2*d)

# #         try:
# #             # Fit a GMM to the data
# #             gmm = GaussianMixture(
# #                 n_components=self.K,
# #                 covariance_type='full',
# #                 random_state=self.random_state
# #             )
# #             gmm.fit(X_gmm)

# #             # Store GMM parameters
# #             # FIX: The scikit-learn GMM attribute for priors is .weights_, not .priors_
# #             self.priors = gmm.weights_
# #             self.mu = gmm.means_
# #             self.sigma = gmm.covariances_

# #             # Pre-calculate Sigma_xi_inv for h_k (Eq. 8)
# #             for k in range(self.K):
# #                 sigma_xi = self.sigma[k, :self.d, :self.d]
# #                 # Add regularization to avoid singular matrix errors
# #                 sigma_xi += np.eye(self.d) * 1e-6
# #                 self.sigma_xi_inv[k] = np.linalg.inv(sigma_xi)

# #         except Exception as e:
# #             print(f"--- ERROR in GMM Initialization ---")
# #             print(f"Failed to fit GaussianMixture model. Error: {e}")
# #             print(f"Data shape: {X_gmm.shape}")
# #             if np.any(np.isnan(X_gmm)) or np.any(np.isinf(X_gmm)):
# #                 print("Error: Data contains NaN or Inf values.")
# #             return False
# #         return True


# #     def _calculate_A_b_unconstrained(self):
# #         """
# #         Calculates the unconstrained A_k and b_k parameters (Eq. 8)
# #         from the initialized GMM parameters. This is the starting
# #         point for the constrained optimization.
# #         """
# #         # Mu_k = [mu_xi_k, mu_dx_k]
# #         mu_xi = self.mu[:, :self.d]    # (K, d)
# #         mu_dx = self.mu[:, self.d:]    # (K, d)

# #         # Sigma_k = [[Sigma_xi,  Sigma_xidx],
# #         #            [Sigma_dxxi, Sigma_dx]]
# #         Sigma_xi = self.sigma[:, :self.d, :self.d]        # (K, d, d)
# #         Sigma_xidx = self.sigma[:, :self.d, self.d:]      # (K, d, d)
# #         Sigma_dxxi = self.sigma[:, self.d:, :self.d]      # (K, d, d)
# #         # Sigma_dx = self.sigma[:, self.d:, self.d:]        # (K, d, d) # Not needed here

# #         # A_k = Sigma_dx_xi * Sigma_xi^-1
# #         # b_k = mu_dx_k - A_k * mu_xi_k
# #         A_k_init = np.zeros((self.K, self.d, self.d))
# #         b_k_init = np.zeros((self.K, self.d))

# #         for k in range(self.K):
# #             # We pre-calculated sigma_xi_inv in gmm_init
# #             inv_sigma_xi_k = self.sigma_xi_inv[k]
# #             sigma_dxxi_k = Sigma_dxxi[k]

# #             A_k_init[k] = sigma_dxxi_k @ inv_sigma_xi_k
# #             b_k_init[k] = mu_dx[k] - A_k_init[k] @ mu_xi[k]

# #         return A_k_init, b_k_init


# #     def _pack_params(self, A_k, b_k):
# #         """Helper to flatten A_k and b_k into a 1D vector for the optimizer."""
# #         return np.concatenate([A_k.ravel(), b_k.ravel()])

# #     def _unpack_params(self, params_vec):
# #         """Helper to unpack the 1D vector back into A_k and b_k."""
# #         A_k_flat = params_vec[:self.K * self.d * self.d]
# #         b_k_flat = params_vec[self.K * self.d * self.d:]
        
# #         A_k = A_k_flat.reshape((self.K, self.d, self.d))
# #         b_k = b_k_flat.reshape((self.K, self.d))
# #         return A_k, b_k

# #     def _get_h_k(self, x):
# #         """
# #         Calculate the activation probabilities h_k(x) (Eq. 8).
# #         x shape can be (d,) or (N, d) for batch processing.
        
# #         Returns:
# #             h_k (np.ndarray): Shape (K,) or (N, K)
# #         """
# #         # --- Vectorization Fix ---
# #         # Handle both single-point (d,) and batch (N, d) inputs
# #         is_batch = x.ndim == 2
# #         if not is_batch:
# #             x = x[np.newaxis, :]  # Promote to (1, d) batch
            
# #         N = x.shape[0]
# #         # --- End Fix ---
        
# #         N_k = np.zeros((N, self.K)) # Shape (N, K)
# #         mu_xi = self.mu[:, :self.d] # (K, d)

# #         for k in range(self.K):
# #             delta = x - mu_xi[k] # (N, d) - (d,) -> (N, d)
            
# #             # Vectorized Mahalanobis distance calculation
# #             # exponent = -0.5 * delta.T @ self.sigma_xi_inv[k] @ delta # Old non-batch way
# #             temp = delta @ self.sigma_xi_inv[k] # (N, d) @ (d, d) -> (N, d)
# #             exponent = -0.5 * np.sum(temp * delta, axis=1) # (N,)
            
# #             N_k[:, k] = self.priors[k] * np.exp(exponent)

# #         # Normalize probabilities for each point in the batch
# #         sum_N_k = np.sum(N_k, axis=1) # (N,)
# #         safe_sums = np.maximum(sum_N_k, 1e-100) # Avoid division by zero
# #         h_k = N_k / safe_sums[:, np.newaxis] # (N, K) / (N, 1) -> (N, K)
        
# #         if not is_batch:
# #             return h_k.squeeze(0) # Return (K,)
# #         return h_k # Return (N, K)

# #     def _objective_function(self, params_vec, demos_x, demos_dx):
# #         """
# #         The SEDS-Likelihood objective function (Eq. 19), simplified to
# #         MSE (Eq. 21) for robustness and speed.
        
# #         This function is vectorized to be fast.
        
# #         Args:
# #             params_vec (np.ndarray): 1D vector of A_k and b_k
# #             demos_x (np.ndarray): (N, d)
# #             demos_dx (np.ndarray): (N, d)
# #         """
# #         A_k, b_k = self._unpack_params(params_vec)
# #         N = demos_x.shape[0]

# #         # --- Vectorized Implementation ---
        
# #         # 1. Calculate all h_k(x) for all N points at once
# #         # h_k_batch shape: (N, K)
# #         h_k_batch = self._get_h_k(demos_x)
        
# #         # 2. Calculate all f_k(x_n) for all N points and K components
# #         # f_all_k shape: (N, K, d)
# #         # 'nd,kdj->nkj' = (N,d) @ (K,d,j=d) -> (N,K,j=d)
# #         f_all_k = np.einsum('nd,kdj->nkj', demos_x, A_k) + b_k[np.newaxis, :, :]
        
# #         # 3. Calculate all predicted velocities f(x_n)
# #         # dx_pred_batch shape: (N, d)
# #         # (N, K, 1) * (N, K, d) -> sum over K -> (N, d)
# #         dx_pred_batch = np.sum(h_k_batch[:, :, np.newaxis] * f_all_k, axis=1)

# #         # 4. Calculate total Mean Squared Error
# #         total_mse = np.sum((demos_dx - dx_pred_batch)**2)
        
# #         return 0.5 * total_mse / N # Return mean squared error
# #         # --- End Vectorized Implementation ---


# #     def _stability_constraints(self, params_vec):
# #         """
# #         Defines the stability constraints (Eq. 13) for the optimizer.
        
# #         Returns:
# #             A list of dictionaries, one for each constraint.
# #         """
# #         A_k, b_k = self._unpack_params(params_vec)
# #         constraints = []

# #         for k in range(self.K):
# #             # 1. Constraint: b_k = -A_k * target
# #             # Since target is origin (0), this simplifies to b_k = 0
# #             # We'll enforce this as an equality constraint.
# #             # type 'eq' means con(params) == 0
# #             def con_b_k(params, k=k):
# #                 _, b_k_local = self._unpack_params(params)
# #                 return np.sum(b_k_local[k]**2) # Enforce b_k[k] == 0
            
# #             constraints.append({'type': 'eq', 'fun': con_b_k})

# #             # 2. Constraint: A_k + A_k.T < 0 (Negative Definite)
# #             # This is enforced by requiring all eigenvalues of the
# #             # symmetric part (A_k + A_k.T) to be negative.
# #             # We set a small margin (e.g., -1e-6) to ensure strict negativity.
# #             # type 'ineq' means con(params) >= 0
# #             def con_A_k_eig(params, k=k):
# #                 A_k_local, _ = self._unpack_params(params)
# #                 A_k_sym = 0.5 * (A_k_local[k] + A_k_local[k].T)
# #                 eigenvalues = np.linalg.eigvalsh(A_k_sym)
# #                 # We want eigenvalues <= -margin
# #                 # So, -eigenvalues - margin >= 0
# #                 margin = 1e-5
# #                 return -eigenvalues - margin
            
# #             # This creates 'd' inequality constraints for each component k
# #             for i in range(self.d):
# #                 constraints.append({
# #                     'type': 'ineq',
# #                     'fun': lambda params, k=k, i=i: con_A_k_eig(params, k)[i]
# #                 })

# #         return constraints

# #     def train(self, demos_x, demos_dx):
# #         """
# #         Trains the SEDS model.
# #         1. Initializes GMM parameters.
# #         2. Sets up constrained optimization problem.
# #         3. Solves for stable A_k and b_k.
        
# #         Args:
# #             demos_x (np.ndarray): Position data (N_demos, N_samples, d)
# #             demos_dx (np.ndarray): Velocity data (N_demos, N_samples, d)
# #         """
        
# #         # --- FIX: Convert inputs to numpy arrays ---
# #         # This makes the function robust if passed lists instead of arrays
# #         try:
# #             demos_x = np.asarray(demos_x, dtype=np.float64)
# #             demos_dx = np.asarray(demos_dx, dtype=np.float64)
# #         except Exception as e:
# #             print(f"--- ERROR in train() ---")
# #             print(f"Failed to convert input data to NumPy arrays. Error: {e}")
# #             return False
# #         # --- End Fix ---

# #         # Reshape data from (N_demos, N_samples, d) to (N_total_samples, d)
# #         # This is the format required by sklearn GMM and the objective function
# #         try:
# #             n_demos, n_samples, dim = demos_x.shape
# #             if dim != self.d:
# #                 print(f"--- ERROR in train() ---")
# #                 print(f"Data dimension ({dim}) does not match model dimension ({self.d}).")
# #                 return False
                
# #             demos_x_flat = demos_x.reshape((n_demos * n_samples, self.d))
# #             demos_dx_flat = demos_dx.reshape((n_demos * n_samples, self.d))
# #         except Exception as e:
# #             print(f"--- ERROR in train() ---")
# #             print(f"Failed to reshape data. Expected 3D array (N_demos, N_samples, d).")
# #             print(f"Got demos_x shape: {demos_x.shape}")
# #             print(f"Error: {e}")
# #             return False

# #         # 1. Initialize GMM (priors, mu, sigma)
# #         if not self.gmm_init(demos_x_flat, demos_dx_flat):
# #             return False # GMM init failed

# #         # 2. Get unconstrained A_k, b_k as starting point
# #         A_k_init, b_k_init = self._calculate_A_b_unconstrained()
        
# #         # 3. Define constraints
# #         # We must satisfy b_k = 0 for stability at origin
# #         # We can just set b_k_init to zero for the optimization
# #         b_k_init.fill(0.0)
        
# #         initial_params = self._pack_params(A_k_init, b_k_init)
        
# #         constraints = self._stability_constraints(initial_params)
        
# #         # --- Add callback for progress tracking ---
# #         self._optimizer_iteration = 0
# #         def optimizer_callback(params_vec):
# #             self._optimizer_iteration += 1
# #             if self._optimizer_iteration % 10 == 0:
# #                 mse = self._objective_function(params_vec, demos_x_flat, demos_dx_flat)
# #                 print(f"[SEDS Train] Iter: {self._optimizer_iteration}, Current MSE: {mse:.6f}")
# #         # --- End callback ---

# #         # 4. Run the optimization
# #         # We use 'SLSQP' as it handles both equality and inequality constraints
# #         try:
# #             result = minimize(
# #                 self._objective_function,
# #                 initial_params,
# #                 args=(demos_x_flat, demos_dx_flat),
# #                 method='SLSQP',
# #                 constraints=constraints,
# #                 callback=optimizer_callback, # Add the callback here
# #                 options={'disp': False, 'maxiter': 500} # Set disp=False
# #             )
            
# #             if not result.success:
# #                 print(f"--- WARNING: Optimization failed ---")
# #                 print(f"Message: {result.message}")
# #                 # We can still proceed with the result, but it may not be optimal
# #             else:
# #                 final_mse = self._objective_function(result.x, demos_x_flat, demos_dx_flat)
# #                 print(f"[SEDS Train] Optimization finished. Final MSE: {final_mse:.6f}")

# #             # Store the optimized parameters
# #             self.A_k, self.b_k = self._unpack_params(result.x)
# #             return result.success

# #         except Exception as e:
# #             print(f"--- ERROR in Optimization ---")
# #             print(f"Optimization failed with error: {e}")
# #             print(f"Initial A_k[0]:\n{A_k_init[0]}")
# #             return False

# #     def predict(self, x):
# #         """
# #         Predicts the velocity (dx) at a given position (x).
# #         This is f(x) from Eq. 9.
# #         Handles both single points (d,) and batches (N, d).

# #         Args:
# #             x (np.ndarray): Position vector(s), shape (d,) or (N, d)
        
# #         Returns:
# #             np.ndarray: Predicted velocity vector(s), shape (d,) or (N, d)
# #         """
# #         if self.priors is None:
# #             raise RuntimeError("Model is not trained. Call train() first.")
        
# #         # --- Vectorization Fix ---
# #         is_batch = x.ndim == 2
# #         if not is_batch:
# #             x = x[np.newaxis, :] # Promote to (1, d) batch
        
# #         N = x.shape[0]
            
# #         h_k = self._get_h_k(x) # Shape (N, K)
        
# #         # Vectorized f_k(x) = A_k*x + b_k
# #         # f_all_k[n, k, d] = A_k[k] @ x[n] + b_k[k]
# #         # Use einsum for batch matrix multiplication: (N, d) @ (K, d, d) -> (N, K, d)
# #         # 'nd,kdj->nkj' = (N,d) @ (K,d,j=d) -> (N,K,j=d)
# #         f_all_k = np.einsum('nd,kdj->nkj', x, self.A_k) + self.b_k[np.newaxis, :, :]
        
# #         # Vectorized f(x) = sum_k( h_k(x) * f_k(x) )
# #         # (N, K, 1) * (N, K, d) -> sum over K -> (N, d)
# #         dx_pred = np.sum(h_k[:, :, np.newaxis] * f_all_k, axis=1)

# #         if not is_batch:
# #             return dx_pred.squeeze(0) # Return (d,)
# #         return dx_pred # Return (N, d)
# #         # --- End Fix ---

# #     def simulate(self, x_start, dt, n_steps):
# #         """
# #         Simulates a trajectory by integrating the learned dynamics.

# #         Args:
# #             x_start (np.ndarray): Starting position (d,)
# #             dt (float): Time step for integration
# #             n_steps (int): Number of steps to simulate
        
# #         Returns:
# #             np.ndarray: The simulated trajectory (n_steps, d)
# #         """
# #         trajectory = np.zeros((n_steps, self.d))
# #         trajectory[0] = x_start
        
# #         current_x = x_start
        
# #         for t in range(1, n_steps):
# #             # Predict velocity
# #             dx = self.predict(current_x)
            
# #             # Euler integration
# #             current_x = current_x + dx * dt
# #             trajectory[t] = current_x
            
# #             # Stop if we've reached the target
# #             if np.linalg.norm(current_x - self.target) < 1e-3:
# #                 trajectory[t:] = self.target # Fill remaining steps
# #                 break
                
# #         return trajectory

# #     def save_model(self, filepath):
# #         """Saves the trained model to a file using pickle."""
# #         try:
# #             with open(filepath, 'wb') as f:
# #                 pickle.dump(self, f)
# #             print(f"Model saved successfully to {filepath}")
# #         except Exception as e:
# #             print(f"Error saving model to {filepath}: {e}")

# #     @staticmethod
# #     def load_model(filepath):
# #         """Loads a trained model from a file."""
# #         try:
# #             with open(filepath, 'rb') as f:
# #                 model = pickle.load(f)
# #             print(f"Model loaded successfully from {filepath}")
# #             return model
# #         except Exception as e:
# #             print(f"Error loading model from {filepath}: {e}")
# #             return None



