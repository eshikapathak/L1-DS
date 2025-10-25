import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
import pickle

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
                random_state=self.random_state
            )
            gmm.fit(X_gmm)

            # Store GMM parameters
            # FIX: The scikit-learn GMM attribute for priors is .weights_, not .priors_
            self.priors = gmm.weights_
            self.mu = gmm.means_
            self.sigma = gmm.covariances_

            # Pre-calculate Sigma_xi_inv for h_k (Eq. 8)
            for k in range(self.K):
                sigma_xi = self.sigma[k, :self.d, :self.d]
                # Add regularization to avoid singular matrix errors
                sigma_xi += np.eye(self.d) * 1e-6
                self.sigma_xi_inv[k] = np.linalg.inv(sigma_xi)

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
        Sigma_xi = self.sigma[:, :self.d, :self.d]        # (K, d, d)
        Sigma_xidx = self.sigma[:, :self.d, self.d:]      # (K, d, d)
        Sigma_dxxi = self.sigma[:, self.d:, :self.d]      # (K, d, d)
        # Sigma_dx = self.sigma[:, self.d:, self.d:]        # (K, d, d) # Not needed here

        # A_k = Sigma_dx_xi * Sigma_xi^-1
        # b_k = mu_dx_k - A_k * mu_xi_k
        A_k_init = np.zeros((self.K, self.d, self.d))
        b_k_init = np.zeros((self.K, self.d))

        for k in range(self.K):
            # We pre-calculated sigma_xi_inv in gmm_init
            inv_sigma_xi_k = self.sigma_xi_inv[k]
            sigma_dxxi_k = Sigma_dxxi[k]

            A_k_init[k] = sigma_dxxi_k @ inv_sigma_xi_k
            b_k_init[k] = mu_dx[k] - A_k_init[k] @ mu_xi[k]

        return A_k_init, b_k_init


    def _pack_params(self, A_k, b_k):
        """Helper to flatten A_k and b_k into a 1D vector for the optimizer."""
        return np.concatenate([A_k.ravel(), b_k.ravel()])

    def _unpack_params(self, params_vec):
        """Helper to unpack the 1D vector back into A_k and b_k."""
        A_k_flat = params_vec[:self.K * self.d * self.d]
        b_k_flat = params_vec[self.K * self.d * self.d:]
        
        A_k = A_k_flat.reshape((self.K, self.d, self.d))
        b_k = b_k_flat.reshape((self.K, self.d))
        return A_k, b_k

    def _get_h_k(self, x):
        """
        Calculate the activation probabilities h_k(x) (Eq. 8).
        x shape can be (d,) or (N, d) for batch processing.
        
        Returns:
            h_k (np.ndarray): Shape (K,) or (N, K)
        """
        # --- Vectorization Fix ---
        # Handle both single-point (d,) and batch (N, d) inputs
        is_batch = x.ndim == 2
        if not is_batch:
            x = x[np.newaxis, :]  # Promote to (1, d) batch
            
        N = x.shape[0]
        # --- End Fix ---
        
        N_k = np.zeros((N, self.K)) # Shape (N, K)
        mu_xi = self.mu[:, :self.d] # (K, d)

        for k in range(self.K):
            delta = x - mu_xi[k] # (N, d) - (d,) -> (N, d)
            
            # Vectorized Mahalanobis distance calculation
            # exponent = -0.5 * delta.T @ self.sigma_xi_inv[k] @ delta # Old non-batch way
            temp = delta @ self.sigma_xi_inv[k] # (N, d) @ (d, d) -> (N, d)
            exponent = -0.5 * np.sum(temp * delta, axis=1) # (N,)
            
            N_k[:, k] = self.priors[k] * np.exp(exponent)

        # Normalize probabilities for each point in the batch
        sum_N_k = np.sum(N_k, axis=1) # (N,)
        safe_sums = np.maximum(sum_N_k, 1e-100) # Avoid division by zero
        h_k = N_k / safe_sums[:, np.newaxis] # (N, K) / (N, 1) -> (N, K)
        
        if not is_batch:
            return h_k.squeeze(0) # Return (K,)
        return h_k # Return (N, K)

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
        A_k, b_k = self._unpack_params(params_vec)
        N = demos_x.shape[0]

        # --- Vectorized Implementation ---
        
        # 1. Calculate all h_k(x) for all N points at once
        # h_k_batch shape: (N, K)
        h_k_batch = self._get_h_k(demos_x)
        
        # 2. Calculate all f_k(x_n) for all N points and K components
        # f_all_k shape: (N, K, d)
        # 'nd,kdj->nkj' = (N,d) @ (K,d,j=d) -> (N,K,j=d)
        f_all_k = np.einsum('nd,kdj->nkj', demos_x, A_k) + b_k[np.newaxis, :, :]
        
        # 3. Calculate all predicted velocities f(x_n)
        # dx_pred_batch shape: (N, d)
        # (N, K, 1) * (N, K, d) -> sum over K -> (N, d)
        dx_pred_batch = np.sum(h_k_batch[:, :, np.newaxis] * f_all_k, axis=1)

        # 4. Calculate total Mean Squared Error
        total_mse = np.sum((demos_dx - dx_pred_batch)**2)
        
        return 0.5 * total_mse / N # Return mean squared error
        # --- End Vectorized Implementation ---


    def _stability_constraints(self, params_vec):
        """
        Defines the stability constraints (Eq. 13) for the optimizer.
        
        Returns:
            A list of dictionaries, one for each constraint.
        """
        A_k, b_k = self._unpack_params(params_vec)
        constraints = []

        for k in range(self.K):
            # 1. Constraint: b_k = -A_k * target
            # Since target is origin (0), this simplifies to b_k = 0
            # We'll enforce this as an equality constraint.
            # type 'eq' means con(params) == 0
            def con_b_k(params, k=k):
                _, b_k_local = self._unpack_params(params)
                return np.sum(b_k_local[k]**2) # Enforce b_k[k] == 0
            
            constraints.append({'type': 'eq', 'fun': con_b_k})

            # 2. Constraint: A_k + A_k.T < 0 (Negative Definite)
            # This is enforced by requiring all eigenvalues of the
            # symmetric part (A_k + A_k.T) to be negative.
            # We set a small margin (e.g., -1e-6) to ensure strict negativity.
            # type 'ineq' means con(params) >= 0
            def con_A_k_eig(params, k=k):
                A_k_local, _ = self._unpack_params(params)
                A_k_sym = 0.5 * (A_k_local[k] + A_k_local[k].T)
                eigenvalues = np.linalg.eigvalsh(A_k_sym)
                # We want eigenvalues <= -margin
                # So, -eigenvalues - margin >= 0
                margin = 1e-5
                return -eigenvalues - margin
            
            # This creates 'd' inequality constraints for each component k
            for i in range(self.d):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda params, k=k, i=i: con_A_k_eig(params, k)[i]
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
        """
        
        # --- FIX: Convert inputs to numpy arrays ---
        # This makes the function robust if passed lists instead of arrays
        try:
            demos_x = np.asarray(demos_x, dtype=np.float64)
            demos_dx = np.asarray(demos_dx, dtype=np.float64)
        except Exception as e:
            print(f"--- ERROR in train() ---")
            print(f"Failed to convert input data to NumPy arrays. Error: {e}")
            return False
        # --- End Fix ---

        # Reshape data from (N_demos, N_samples, d) to (N_total_samples, d)
        # This is the format required by sklearn GMM and the objective function
        try:
            n_demos, n_samples, dim = demos_x.shape
            if dim != self.d:
                print(f"--- ERROR in train() ---")
                print(f"Data dimension ({dim}) does not match model dimension ({self.d}).")
                return False
                
            demos_x_flat = demos_x.reshape((n_demos * n_samples, self.d))
            demos_dx_flat = demos_dx.reshape((n_demos * n_samples, self.d))
        except Exception as e:
            print(f"--- ERROR in train() ---")
            print(f"Failed to reshape data. Expected 3D array (N_demos, N_samples, d).")
            print(f"Got demos_x shape: {demos_x.shape}")
            print(f"Error: {e}")
            return False

        # 1. Initialize GMM (priors, mu, sigma)
        if not self.gmm_init(demos_x_flat, demos_dx_flat):
            return False # GMM init failed

        # 2. Get unconstrained A_k, b_k as starting point
        A_k_init, b_k_init = self._calculate_A_b_unconstrained()
        
        # 3. Define constraints
        # We must satisfy b_k = 0 for stability at origin
        # We can just set b_k_init to zero for the optimization
        b_k_init.fill(0.0)
        
        initial_params = self._pack_params(A_k_init, b_k_init)
        
        constraints = self._stability_constraints(initial_params)
        
        # --- Add callback for progress tracking ---
        self._optimizer_iteration = 0
        def optimizer_callback(params_vec):
            self._optimizer_iteration += 1
            if self._optimizer_iteration % 10 == 0:
                mse = self._objective_function(params_vec, demos_x_flat, demos_dx_flat)
                print(f"[SEDS Train] Iter: {self._optimizer_iteration}, Current MSE: {mse:.6f}")
        # --- End callback ---

        # 4. Run the optimization
        # We use 'SLSQP' as it handles both equality and inequality constraints
        try:
            result = minimize(
                self._objective_function,
                initial_params,
                args=(demos_x_flat, demos_dx_flat),
                method='SLSQP',
                constraints=constraints,
                callback=optimizer_callback, # Add the callback here
                options={'disp': False, 'maxiter': 500} # Set disp=False
            )
            
            if not result.success:
                print(f"--- WARNING: Optimization failed ---")
                print(f"Message: {result.message}")
                # We can still proceed with the result, but it may not be optimal
            else:
                final_mse = self._objective_function(result.x, demos_x_flat, demos_dx_flat)
                print(f"[SEDS Train] Optimization finished. Final MSE: {final_mse:.6f}")

            # Store the optimized parameters
            self.A_k, self.b_k = self._unpack_params(result.x)
            return result.success

        except Exception as e:
            print(f"--- ERROR in Optimization ---")
            print(f"Optimization failed with error: {e}")
            print(f"Initial A_k[0]:\n{A_k_init[0]}")
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
        if self.priors is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        
        # --- Vectorization Fix ---
        is_batch = x.ndim == 2
        if not is_batch:
            x = x[np.newaxis, :] # Promote to (1, d) batch
        
        N = x.shape[0]
            
        h_k = self._get_h_k(x) # Shape (N, K)
        
        # Vectorized f_k(x) = A_k*x + b_k
        # f_all_k[n, k, d] = A_k[k] @ x[n] + b_k[k]
        # Use einsum for batch matrix multiplication: (N, d) @ (K, d, d) -> (N, K, d)
        # 'nd,kdj->nkj' = (N,d) @ (K,d,j=d) -> (N,K,j=d)
        f_all_k = np.einsum('nd,kdj->nkj', x, self.A_k) + self.b_k[np.newaxis, :, :]
        
        # Vectorized f(x) = sum_k( h_k(x) * f_k(x) )
        # (N, K, 1) * (N, K, d) -> sum over K -> (N, d)
        dx_pred = np.sum(h_k[:, :, np.newaxis] * f_all_k, axis=1)

        if not is_batch:
            return dx_pred.squeeze(0) # Return (d,)
        return dx_pred # Return (N, d)
        # --- End Fix ---

    def simulate(self, x_start, dt, n_steps):
        """
        Simulates a trajectory by integrating the learned dynamics.

        Args:
            x_start (np.ndarray): Starting position (d,)
            dt (float): Time step for integration
            n_steps (int): Number of steps to simulate (e.g., n_steps=1000 means 1001 points)
        
        Returns:
            (np.ndarray, np.ndarray): 
                - The simulated trajectory (n_steps + 1, d)
                - The predicted velocities (n_steps + 1, d)
        """
        # Create array for n_steps + 1 points (0 to n_steps inclusive)
        trajectory = np.zeros((n_steps + 1, self.d))
        velocities = np.zeros((n_steps + 1, self.d))
        
        trajectory[0] = x_start
        current_x = x_start.copy()
        
        # Get velocity at the start point
        current_dx = self.predict(current_x)
        velocities[0] = current_dx
        
        for t in range(n_steps): # Loop n_steps times (from 0 to n_steps-1)
            
            # Euler integration
            current_x = current_x + current_dx * dt
            
            # Predict velocity at the *new* position
            current_dx = self.predict(current_x)
            
            # Stop if we've reached the target
            # Check velocity as well to prevent premature stopping
            if np.linalg.norm(current_x - self.target) < 1e-4 and np.linalg.norm(current_dx) < 1e-4:
                trajectory[t+1:] = self.target # Fill remaining steps with target
                velocities[t+1:] = 0.0         # Fill remaining velocities with zero
                break
            
            trajectory[t+1] = current_x
            velocities[t+1] = current_dx
                
        return trajectory, velocities

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




# 
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
#             n_steps (int): Number of steps to simulate
        
#         Returns:
#             np.ndarray: The simulated trajectory (n_steps, d)
#         """
#         trajectory = np.zeros((n_steps, self.d))
#         trajectory[0] = x_start
        
#         current_x = x_start
        
#         for t in range(1, n_steps):
#             # Predict velocity
#             dx = self.predict(current_x)
            
#             # Euler integration
#             current_x = current_x + dx * dt
#             trajectory[t] = current_x
            
#             # Stop if we've reached the target
#             if np.linalg.norm(current_x - self.target) < 1e-3:
#                 trajectory[t:] = self.target # Fill remaining steps
#                 break
                
#         return trajectory

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
#             print(f"Model loaded successfully from {filepath}")
#             return model
#         except Exception as e:
#             print(f"Error loading model from {filepath}: {e}")
#             return None



