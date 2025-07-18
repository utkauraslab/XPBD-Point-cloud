import numpy as np
import pickle
import torch
import torch.autograd.functional as F
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve
import networkx as nx
from scipy.linalg import svd, inv, norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import warnings
from typing import List, Tuple, Optional
import time
from typing import Optional 


try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found. For progress bars, please install it: pip install tqdm")
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    from torch.func import vmap, jacrev
except ImportError:
    print("torch.func not available. Please update your PyTorch version (>=1.13).")
    # Provide a fallback so the program doesn't crash, although it will be very slow.
    def vmap(func, in_dims=0, out_dims=0):
        def _vmap_wrapper(batch_tensor):
            return torch.stack([func(tensor) for tensor in batch_tensor])
        return _vmap_wrapper

class CompleteXPBDGaussianModeling:
    def __init__(self, tau=8, k_neighbors=8, tau_g=0.01, sigma_weight=0.03, epsilon=1e-4, 
                 heat_sigma=None, diffusion_time=None, use_autograd=True, device='auto',
                 debug_max_step =None):
        """
        XPBD-based Per-Point Gaussian Modeling
        
        Args:
            tau: Half-window size for temporal averaging
            k_neighbors: Number of neighbors for k-NN graph
            tau_g: Geodesic distance threshold (in normalized units)
            sigma_weight: Gaussian weight parameter
            epsilon: Numerical stability constant
            heat_sigma: Sigma parameter for Heat Method (auto-computed if None)
            diffusion_time: Diffusion time for Heat Method (auto-computed if None)
            use_autograd: Whether to use PyTorch automatic differentiation
            device: 'cpu', 'cuda', or 'auto'
        """
        self.tau = tau
        self.tau_trunc = 1e-3
        global eps_pinv
        eps_pinv = 1e-8
        self.k_neighbors = k_neighbors
        self.tau_g = tau_g
        self.sigma_weight = sigma_weight
        self.epsilon = epsilon
        self.heat_sigma = heat_sigma
        self.diffusion_time = diffusion_time
        self.use_autograd = use_autograd
        self.debug_max_step = debug_max_step

        # PyTorch Setup
        if use_autograd:
            try:
                if device == 'auto':
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                else:
                    self.device = torch.device(device)
                
                self.dtype = torch.float32
                torch.set_default_dtype(self.dtype)
                
                print(f"Using PyTorch autograd; device: {self.device}")
                
                if self.device.type == 'cuda':
                    print(f"✅✅✅ CUDA enabled: {torch.cuda.get_device_name(self.device)}")
                else:
                    print("⚠️⚠️⚠️ Currently running on CPU (no GPU detected)")
            except Exception as e:
                self.use_autograd = False
        


        self._laplacian_cache = None
        self._mass_cache = None
        self._sigma_cache = None
        self._last_points_hash = None
        
        self._last_valid_F   = {}    # Cache for the last valid F per point
        self._global_F_cache = None  # Global fallback F cache
    
    def load_tracking_data(self, pkl_path):
        print(f"Loading data from: {pkl_path}")
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print("Loading tissue (background) points...")
        # Ensure data is loaded as float32 to match torch dtype
        self.tracked_pts = data['positions'].astype(np.float32)
        # self.rgbs = data['rgbs_bg'][0]
        # self.frids = data['frids_bg'][0]
        
        self.num_pts, self.num_frames, _ = self.tracked_pts.shape
        
        print(f"Successfully loaded data:")
        print(f"  - Frames: {self.num_frames}")
        print(f"  - Points per frame: {self.num_pts}")
        print(f"  - Tracked points shape: {self.tracked_pts.shape}")
        
        self._check_data_quality()
        
        return self.tracked_pts
    
    def _check_data_quality(self):
        """Perform a basic quality check on the loaded data."""
        print(f"\n=== Data Quality Check ===")
        
        nan_count = np.sum(np.isnan(self.tracked_pts))
        inf_count = np.sum(np.isinf(self.tracked_pts))
        print(f"NaN values: {nan_count}")
        print(f"Inf values: {inf_count}")
        
        all_points = self.tracked_pts.reshape(-1, 3)
        print(f"Point cloud bounds:")
        print(f"  X: [{np.min(all_points[:, 0]):.4f}, {np.max(all_points[:, 0]):.4f}]")
        print(f"  Y: [{np.min(all_points[:, 1]):.4f}, {np.max(all_points[:, 1]):.4f}]")
        print(f"  Z: [{np.min(all_points[:, 2]):.4f}, {np.max(all_points[:, 2]):.4f}]")
        
        
        if self.num_frames > 1:
            frame_motions = []
            for t in range(1, self.num_frames):
                motion = np.linalg.norm(self.tracked_pts[:,t,:] - self.tracked_pts[:,t-1,:], axis=1)
                frame_motions.append(np.mean(motion))
            print(f"Average frame-to-frame motion: {np.mean(frame_motions):.6f}")
    
    def step0_estimate_reference_position(self, t):
        """
        Step 0: Estimate reference position by averaging a temporal window.
        """
        t_start = max(0, t - self.tau)
        t_end = min(self.num_frames, t + self.tau)
        
        window_pts = self.tracked_pts[:,t_start:t_end,:]
        
        reference_positions = np.mean(window_pts, axis=1)  
        
        return reference_positions
    
    def _construct_robust_laplacian(self, points, k_neighbors=None, sigma=None):
        """
        Construct a robust Laplace matrix for the point cloud.
        """
        if k_neighbors is None:
            k_neighbors = min(self.k_neighbors, len(points) - 1)
        
        n_points = len(points)
        
        # Construct k-NN graph
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, n_points))
        nbrs.fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # Adaptive sigma estimation
        if sigma is None:
            local_scales = []
            for i in range(n_points):
                neighbor_dists = distances[i, 1:min(8, len(distances[i]))]
                if len(neighbor_dists) > 0:
                    harmonic_mean = len(neighbor_dists) / np.sum(1.0 / (neighbor_dists + 1e-10))
                    local_scales.append(harmonic_mean)
            
            if len(local_scales) > 0:
                sigma = np.percentile(local_scales, 75)
            else:
                sigma = 1e-3
        
        # Construct symmetric weight matrix
        row_indices = []
        col_indices = []
        weights = []
        
        seen_edges = set()
        
        for i in range(n_points):
            for j in range(1, len(indices[i])):
                neighbor_idx = indices[i][j]
                
                edge_key = tuple(sorted([i, neighbor_idx]))
                
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    
                    dist_ij = distances[i][j]
                    
                    # Look up reverse distance for symmetry
                    reverse_neighbors = indices[neighbor_idx]
                    reverse_distances = distances[neighbor_idx]
                    reverse_idx = np.where(reverse_neighbors == i)[0]
                    
                    if len(reverse_idx) > 0:
                        dist_ji = reverse_distances[reverse_idx[0]]
                        avg_dist = (dist_ij + dist_ji) / 2.0
                    else:
                        avg_dist = dist_ij
                    
                    weight = np.exp(-avg_dist**2 / (sigma**2))
                    
                    row_indices.extend([i, neighbor_idx])
                    col_indices.extend([neighbor_idx, i])
                    weights.extend([weight, weight])
        
        # Construct weight matrix W
        W = csr_matrix((weights, (row_indices, col_indices)), 
                       shape=(n_points, n_points))
        
        # Degree Matrix D
        degree = np.array(W.sum(axis=1)).flatten()
        D = csr_matrix((degree, (range(n_points), range(n_points))), 
                       shape=(n_points, n_points))
        
        # Laplacian Matrix L = D - W
        L = D - W
        
        # Mass Matrix M
        mass_weights = np.sqrt(degree + 1e-8)
        
        for i in range(n_points):
            neighbor_dists = distances[i, 1:min(4, len(distances[i]))]
            if len(neighbor_dists) > 0:
                local_density = 1.0 / (np.mean(neighbor_dists) + 1e-8)
                mass_weights[i] *= np.sqrt(local_density)
        
        total_mass = np.sum(mass_weights)
        if total_mass > 0:
            mass = mass_weights / total_mass
        else:
            mass = np.ones(n_points) / n_points
        
        M = csr_matrix((mass, (range(n_points), range(n_points))), 
                       shape=(n_points, n_points))
        
        return L, M, sigma
    
    def _construct_euclidean_neighborhood_robust(self, reference_positions):

        print(f"  Constructing enhanced Euclidean neighborhood...")
        n_points = len(reference_positions)
        neighborhoods = []
        
        # Distance analysis
        all_distances = pdist(reference_positions)
        percentiles = np.percentile(all_distances, [25, 50, 75, 90])
        median_distance = percentiles[1]
        
        # Adaptive threshold
        iqr = percentiles[2] - percentiles[0]
        if iqr > 0:
            distance_threshold = median_distance + 0.5 * iqr
        else:
            distance_threshold = median_distance * 1.5
        
        print(f"    Distance summary: median={median_distance:.6f}, IQR={iqr:.6f}")
        print(f"    Adaptive threshold: {distance_threshold:.6f}")
        
        for i in range(n_points):
            distances = np.linalg.norm(reference_positions - reference_positions[i], axis=1)
            distances[i] = np.inf
            
            within_threshold = np.where(distances <= distance_threshold)[0]
            
            min_neighbors = max(8, int(self.k_neighbors * 0.8))
            max_neighbors = self.k_neighbors * 3
            
            if len(within_threshold) < min_neighbors:
                neighborhood = np.argsort(distances)[:max(self.k_neighbors, min_neighbors)].tolist()
            else:
                neighbor_distances = distances[within_threshold]
                sorted_indices = np.argsort(neighbor_distances)
                actual_max = min(len(within_threshold), max_neighbors)
                neighborhood = [within_threshold[j] for j in sorted_indices[:actual_max]]
            
            neighborhoods.append(neighborhood)
        
        neighborhood_sizes = [len(n) for n in neighborhoods]
        print(f"      Enhanced Euclidean neighborhood stats:")
        print(f"      Mean size: {np.mean(neighborhood_sizes):.2f}")
        print(f"      Size range: [{np.min(neighborhood_sizes)}, {np.max(neighborhood_sizes)}]")
        
        return neighborhoods
    
    def step1_construct_geodesic_neighborhood_heat_method(self, reference_positions):
        """
        Step 1: Construct geodesic neighborhoods using the Heat Method.
        """
        print(f"Step 1: Constructing geodesic neighborhoods via Heat Method...")
        
        n_points = len(reference_positions)
    
        # Point cloud analysis
        distances = pdist(reference_positions)
        if distances.size == 0:
            return [[] for _ in range(n_points)] # Handle case with < 2 points
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # Enhanced fallback condition
        use_euclidean = (
            n_points < 20 or 
            min_dist < 1e-6 or 
            (min_dist > 0 and max_dist/min_dist > 2000) or
            (mean_dist > 0 and std_dist/mean_dist > 2.0)
        )
        
        if use_euclidean:
            print(f"  Using robust Euclidean fallback due to point cloud properties.")
            return self._construct_euclidean_neighborhood_robust(reference_positions)
        
        # Try Heat Method
        try:
            L, M, sigma = self._construct_robust_laplacian(reference_positions, sigma=self.heat_sigma)
            self.heat_sigma = sigma
            
            # Adaptive geodesic radius
            point_scale = np.std(reference_positions, axis=0)
            mean_point_scale = np.mean(point_scale)
            adaptive_tau_g = max(self.tau_g, mean_point_scale * 0.15, mean_dist * 0.3)
            
            neighborhoods = []
            failed_points = 0
            
            for i in tqdm(range(n_points), desc="  Building Geodesic Neighborhoods"):
                try:
                    u = self._solve_heat_diffusion(L, M, i, self.diffusion_time)
                    X_hat = self._compute_gradient_field(L, u)
                    geodesic_distances = self._recover_distances_poisson(L, X_hat)
                    
                    if np.any(np.isnan(geodesic_distances)) or np.any(np.isinf(geodesic_distances)):
                        raise ValueError("Invalid geodesic distances")
                    
                    neighborhood_indices = np.where(geodesic_distances <= adaptive_tau_g)[0]
                    neighborhood = neighborhood_indices.tolist()
                    
                    if i in neighborhood:
                        neighborhood.remove(i)
                    
                    # Neighborhood size control
                    min_neighbors = max(3, self.k_neighbors // 3)
                    max_neighbors = self.k_neighbors * 2
                    
                    if len(neighborhood) < min_neighbors:
                        neighbor_distances = geodesic_distances.copy()
                        neighbor_distances[i] = np.inf
                        closest_indices = np.argsort(neighbor_distances)[:max(self.k_neighbors, 10)]
                        neighborhood = closest_indices.tolist()
                    elif len(neighborhood) > max_neighbors:
                        neighbor_distances = geodesic_distances[neighborhood]
                        sorted_indices = np.argsort(neighbor_distances)
                        neighborhood = [neighborhood[j] for j in sorted_indices[:self.k_neighbors]]
                    
                    neighborhoods.append(neighborhood)
                    
                except Exception as e:
                    failed_points += 1
                    # Fallback to Euclidean neighborhood
                    distances_to_i = np.linalg.norm(reference_positions - reference_positions[i], axis=1)
                    distances_to_i[i] = np.inf
                    closest_indices = np.argsort(distances_to_i)[:self.k_neighbors]
                    neighborhoods.append(closest_indices.tolist())
            
            if failed_points > 0:
                print(f"    Warning: {failed_points} points fell back to Euclidean neighborhoods")
                
        except Exception as e:
            print(f"  Enhanced Heat Method failed: {e}")
            print(f"  Falling back to robust Euclidean neighborhoods...")
            return self._construct_euclidean_neighborhood_robust(reference_positions)
        
        # Statistics
        neighborhood_sizes = [len(n) for n in neighborhoods]
        print(f"  Geodesic neighborhood stats: Avg size={np.mean(neighborhood_sizes):.2f}, Range=[{np.min(neighborhood_sizes)}, {np.max(neighborhood_sizes)}]")
        
        return neighborhoods
    
    def _solve_heat_diffusion(self, L, M, source_idx, diffusion_time=None):
        """Solve the heat diffusion equation."""
        n_points = L.shape[0]
        
        if diffusion_time is None:
            if self.heat_sigma is not None:
                diffusion_time = self.heat_sigma**2
            else:
                mean_edge_length = np.mean([w for w in L.data if w < 0]) if L.nnz > 0 else 1e-4
                diffusion_time = abs(mean_edge_length) * 0.1
            diffusion_time = max(diffusion_time, 1e-6)
        
        delta_s = np.zeros(n_points)
        delta_s[source_idx] = 1.0
        rhs = M @ delta_s
        
        system_matrix = M + diffusion_time * L
        reg_strength = max(self.epsilon, 1e-8 * np.mean(np.abs(system_matrix.data)))
        system_matrix += reg_strength * identity(n_points)
        
        try:
            u = spsolve(system_matrix, rhs)
            
            if np.any(np.isnan(u)) or np.any(np.isinf(u)):
                raise ValueError("Heat diffusion solution contains NaN/Inf")
                
        except Exception as e:
            u = np.exp(-np.arange(n_points, dtype=float) / max(1, n_points))
            u[source_idx] = 1.0
        
        return u
    
    def _compute_gradient_field(self, L, u, epsilon=1e-8):
        """Compute the normalized gradient field of a scalar function u."""
        X = -L @ u
        X_norm = np.linalg.norm(X) + epsilon
        X_hat = X / X_norm
        
        if np.any(np.isnan(X_hat)) or np.any(np.isinf(X_hat)):
            X_hat = np.ones_like(X) / np.sqrt(len(X))
        
        return X_hat
    
    def _recover_distances_poisson(self, L, X_hat):
        """Recover distances by solving the Poisson equation."""
        n_points = L.shape[0]
        
        L_pinned = L.copy()
        L_pinned[0, 0] += 1.0
        
        try:
            phi = spsolve(L_pinned, X_hat)
            
            if np.any(np.isnan(phi)) or np.any(np.isinf(phi)):
                raise ValueError("Poisson solution contains NaN/Inf")
                
        except Exception as e:
            phi = np.abs(X_hat)
            phi = phi / (np.max(phi) + 1e-8)
        
        phi = phi - np.min(phi)
        phi = np.maximum(phi, 0)
        
        return phi
    
    def compute_deformation_gradient_torch(
            self, positions, reference_positions, neighborhoods, point_idx, sigma_weight):
        """
        Compute the local deformation gradient F_i for a single point.
        Automatically caches the last valid F to prevent fallback to identity.
        """
        # --- Prepare Tensors ---
        if self.use_autograd:
            if isinstance(positions, np.ndarray):
                positions = torch.tensor(positions, dtype=self.dtype,
                                        device=self.device, requires_grad=True)
            if isinstance(reference_positions, np.ndarray):
                reference_positions = torch.tensor(reference_positions,
                                                dtype=self.dtype,
                                                device=self.device, requires_grad=False)

        neighbors = neighborhoods[point_idx]
        if len(neighbors) < 3:
            # Degenerate case: use cache
            if point_idx in self._last_valid_F:
                return self._last_valid_F[point_idx]
            if self._global_F_cache is not None:
                return self._global_F_cache
            return torch.eye(3, dtype=self.dtype, device=self.device)   # Final fallback

        # --- Weighted Least Squares ---
        A_list, B_list, W_list = [], [], []
        pi_r = reference_positions[point_idx]
        pi_c = positions[point_idx]

        for j in neighbors:
            if j == point_idx:
                continue
            dr = reference_positions[j] - pi_r
            dc = positions[j] - pi_c
            norm_r = torch.linalg.norm(dr) if self.use_autograd else np.linalg.norm(dr)
            if norm_r < 1e-8:
                continue
            
            w = torch.exp(-norm_r**2 / (2 * sigma_weight**2)) if self.use_autograd \
                else np.exp(-norm_r**2 / (2 * sigma_weight**2))

            if w < 1e-6:
                continue
            A_list.append(dr)
            B_list.append(dc)
            W_list.append(w)

        if len(A_list) < 3:
            if point_idx in self._last_valid_F:
                return self._last_valid_F[point_idx]
            if self._global_F_cache is not None:
                return self._global_F_cache
            return torch.eye(3, dtype=self.dtype, device=self.device)

        if self.use_autograd:
            A = torch.stack(A_list)
            B = torch.stack(B_list)
            W = torch.diag(torch.stack(W_list))
            WA = W @ A
            WB = W @ B
            
            try:
                U, S, Vt = torch.linalg.svd(WA, full_matrices=False)
                S_inv = torch.where(S > 1e-6 * S[0], 1.0 / S, torch.zeros_like(S))
                F_i = (Vt.T @ torch.diag(S_inv) @ U.T @ WB).T
                
                det_F = torch.det(F_i)
                # Fallback if deformation is extreme or invalid
                if det_F <= 1e-6 or det_F > 100 or torch.any(torch.isnan(F_i)) or torch.any(torch.isinf(F_i)):
                     if point_idx in self._last_valid_F:
                         return self._last_valid_F[point_idx]
                     if self._global_F_cache is not None:
                         return self._global_F_cache
                     return torch.eye(3, dtype=self.dtype, device=self.device)

                # Cache valid F
                if not torch.allclose(F_i, torch.eye(3, device=self.device, dtype=self.dtype), atol=1e-6):
                    self._last_valid_F[point_idx] = F_i.detach()
                    if self._global_F_cache is None:
                        self._global_F_cache = F_i.detach()
            
            except torch.linalg.LinAlgError:
                if point_idx in self._last_valid_F:
                    return self._last_valid_F[point_idx]
                if self._global_F_cache is not None:
                    return self._global_F_cache
                return torch.eye(3, dtype=self.dtype, device=self.device)

            return F_i
        
        else:   # ===== NumPy Branch =====
            A = np.array(A_list)              # (k,3)
            B = np.array(B_list)
            W = np.diag(np.array(W_list))     # (k,k)

            WA = W @ A
            WB = W @ B

            try:
                U, S, Vt = np.linalg.svd(WA, full_matrices=False)   # WA = U S Vᵀ
                S_inv = np.where(S > 1e-6 * S[0], 1.0 / S, 0.0)     # Truncate small singular values
                F_i = (Vt.T @ np.diag(S_inv) @ U.T @ WB).T          # Least squares solution
                det_F  = np.linalg.det(F_i)
                fro_F  = np.linalg.norm(F_i, 'fro')

                # Cache valid F
                if not np.allclose(F_i, np.eye(3), atol=1e-6):
                    self._last_valid_F[point_idx] = F_i.copy()
                    if self._global_F_cache is None:
                        self._global_F_cache = F_i.copy()

                # Validate and fallback
                if (det_F <= 1e-6 or det_F > 100 or fro_F > 10 or
                    np.any(np.isnan(F_i)) or np.any(np.isinf(F_i))):
                    F_i = np.eye(3)

            except Exception as e:
                if point_idx < 5:
                    print(f"SVD failed (point {point_idx}): {e}")
                F_i = np.eye(3)

            return F_i
    
    
    def step2_estimate_deformation_gradient(self, t, reference_positions, neighborhoods, sigma_weight):
        """
        Step 2: Estimate per-point deformation gradients.
        """
        current_positions = self.tracked_pts[:, t, :]
        deformation_gradients = []
        
        # Global statistics
        ref_center = np.mean(reference_positions, axis=0)
        cur_center = np.mean(current_positions, axis=0)
        ref_cov = np.cov(reference_positions.T)
        cur_cov = np.cov(current_positions.T)

        ref_scale = np.sqrt(np.trace(ref_cov))
        cur_scale = np.sqrt(np.trace(cur_cov))
        scale_ratio = cur_scale / (ref_scale + 1e-8)

        # Fallback counters
        total_fallback_count = 0
        extreme_fallback_count = 0
        local_fallback_count = 0
        local_degenerate_count = 0

        # Iterate over all points
        for i in range(self.num_pts):
            # Detect extreme deformation
            is_extreme_deformation = (
                scale_ratio < 0.05 or scale_ratio > 20
            )

            if is_extreme_deformation:
                deformation_gradients.append(np.eye(3))
                total_fallback_count += 1
                extreme_fallback_count += 1
                continue

            try:
                F_i = self.compute_deformation_gradient_torch(
                    current_positions, reference_positions, neighborhoods, i, sigma_weight
                )

                if isinstance(F_i, torch.Tensor):
                    F_i_np = F_i.detach().cpu().numpy()
                else:
                    F_i_np = F_i

                u, s, vt = np.linalg.svd(F_i_np)
                if np.min(s) < 1e-6:
                    local_degenerate_count += 1

                deformation_gradients.append(F_i_np)

            except Exception as e:
                deformation_gradients.append(np.eye(3))
                total_fallback_count += 1
                local_fallback_count += 1

        return np.array(deformation_gradients)

 
    
    def compute_constraint_residuals(self, F):
        """
        Compute corrected constraint residuals (hydrostatic and deviatoric).
        """
        if self.use_autograd and isinstance(F, torch.Tensor):
            F = F.clone()
            
            # C_hydro
            det_F = torch.det(F)
            C_hydro = det_F - 1.0
            
            # C_devia
            FtF = F.T @ F
            trace_FtF = torch.trace(FtF)
            deviatoric_part = FtF - (trace_FtF / 3.0) * torch.eye(3, device=self.device, dtype=self.dtype)
            C_devia = torch.norm(deviatoric_part, p='fro')
            
        else:
            if isinstance(F, torch.Tensor):
                F = F.detach().cpu().numpy()
            
            # C_hydro
            det_F = np.linalg.det(F)
            C_hydro = det_F - 1.0
            
            # C_devia
            FtF = F.T @ F
            trace_FtF = np.trace(FtF)
            deviatoric_part = FtF - (trace_FtF / 3.0) * np.eye(3)
            C_devia = np.linalg.norm(deviatoric_part, 'fro')
        
        return C_hydro, C_devia
    
    def step3_compute_constraint_residuals(self, deformation_gradients):
        """
        Step 3: Compute constraint residuals for all points.
        """
        hydrostatic_constraints = []
        deviatoric_constraints = []
        
        for i in range(len(deformation_gradients)):
            F_i = deformation_gradients[i]
            C_hydro, C_devia = self.compute_constraint_residuals(F_i)
            
            if isinstance(C_hydro, torch.Tensor):
                C_hydro = C_hydro.detach().cpu().numpy()
            if isinstance(C_devia, torch.Tensor):
                C_devia = C_devia.detach().cpu().numpy()
            
            hydrostatic_constraints.append(float(C_hydro))
            deviatoric_constraints.append(float(C_devia))
        
        hydrostatic_constraints = np.array(hydrostatic_constraints)
        deviatoric_constraints = np.array(deviatoric_constraints)
        
        print(f"  C_hydro stats: range=[{np.min(hydrostatic_constraints):.6f}, {np.max(hydrostatic_constraints):.6f}], mean={np.mean(hydrostatic_constraints):.6f}")
        print(f"  C_devia stats: range=[{np.min(deviatoric_constraints):.6f}, {np.max(deviatoric_constraints):.6f}], mean={np.mean(deviatoric_constraints):.6f}")
        
        return hydrostatic_constraints, deviatoric_constraints
    

    def constraint_function_torch(self, positions_flat, reference_positions, neighborhoods, sigma_weight):
        """
        Computes all constraint residuals for a flattened position tensor.
        [FIXED] Removed in-place requires_grad_() for jacrev compatibility.
        """
        N_pts = self.num_pts
        
        positions = positions_flat.reshape(N_pts, 3)
        
        constr = []
        for gidx in range(N_pts):
            F_i = self.compute_deformation_gradient_torch(
                positions, reference_positions, neighborhoods, gidx, sigma_weight
            )
            C_h, C_d = self.compute_constraint_residuals(F_i)
            constr.extend([C_h, C_d])
        
        return torch.stack(constr)

    def compute_jacobian_autograd(self, positions, reference_positions, neighborhoods, sigma_weight):
        """
        --- PERFORMANCE OPTIMIZATION ---
        Compute the Jacobian using the highly optimized torch.func.jacrev, which calculates
        the entire matrix in a single, parallelized operation on the GPU. This is much
        faster than the previous row-by-row VJP loop for this problem size.
        """
        if isinstance(positions, torch.Tensor):
            positions_torch = positions
        else:                   
            positions_torch = torch.as_tensor(
                positions, dtype=self.dtype, device=self.device)
        if isinstance(reference_positions, torch.Tensor):
            reference_torch = reference_positions
        else:
            reference_torch = torch.as_tensor(
                reference_positions, dtype=self.dtype, device=self.device)
        
        N_pts = self.num_pts
        if N_pts == 0:
            return np.zeros((0, 0)), np.zeros(0)

        positions_flat = positions_torch.reshape(-1)
        positions_flat.requires_grad_(True)
        
        try:
            # Define the function whose Jacobian we want. It maps the flat 3N vector
            # of positions to the 2N vector of constraint residuals.
            constraint_func = lambda x: self.constraint_function_torch(x, reference_torch, neighborhoods, sigma_weight)

            # Compute the entire Jacobian in one shot. This is the core optimization.
            print("  Computing Jacobian with torch.func.jacrev (this may take a moment)...")
            start_time = time.time()
            J_torch = jacrev(constraint_func)(positions_flat)
            print(f"  Jacobian computation finished in {time.time() - start_time:.2f} seconds.")
            
            # Validate Jacobian
            if torch.any(torch.isnan(J_torch)) or torch.any(torch.isinf(J_torch)):
                raise ValueError("Jacobian contains NaN or Inf")
            
            J = J_torch.detach().cpu().numpy()
            
            # Compute constraint residuals (re-using the function is fine)
            with torch.no_grad():
                constraints_torch = constraint_func(positions_flat)
                constraints = constraints_torch.detach().cpu().numpy()
            
            return J, constraints
            
        except Exception as e:
            print(f"Autodiff with jacrev failed: {e}")
            print("Falling back to numerical differentiation")
            return self._compute_jacobian_numerical(positions, reference_positions, neighborhoods, sigma_weight)
        
    def _svd_pinv_solve(self, J, residual, tau=1e-3, lm=1e-3):
        """
        Solve the linear system using Truncated SVD and Levenberg-Marquardt damping.
        """
        if J.size == 0:
            return np.zeros(0)

        # Gradient norm clipping
        grad_norm = np.linalg.norm(residual)
        if grad_norm > 1e3:
            residual = residual * (1e3 / grad_norm)

        U, s, Vt = np.linalg.svd(J, full_matrices=False)

        # Truncate small singular values
        keep = s > tau * s[0]
        if not np.any(keep):
            return np.zeros(J.shape[1])

        s_kept   = s[keep]
        U_kept   = U[:, keep]
        Vt_kept  = Vt[keep, :]

        # Adaptive Levenberg-Marquardt damping
        res_norm     = np.linalg.norm(residual)
        adaptive_lm  = max(lm, 1e-4 * res_norm)   # More damping for larger residuals
        s_damped     = s_kept / (s_kept**2 + adaptive_lm)

        delta = -(Vt_kept.T * s_damped) @ (U_kept.T @ residual)

        return delta

    
    def _compute_jacobian_numerical(self, positions, reference_positions, neighborhoods, sigma_weight):
        """
        Fallback to numerical differentiation if autograd fails.
        [FIXED] Now handles torch.Tensor inputs correctly.
        """
        # --- FIX: Ensure input is a NumPy array ---
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().numpy()

        N_pts = self.num_pts
        
        # Compute base constraints
        base_constraints = []
        for point_idx in range(N_pts):
            F_i = self.compute_deformation_gradient_torch(
                positions, reference_positions, neighborhoods, point_idx, sigma_weight)
            C_hydro, C_devia = self.compute_constraint_residuals(F_i)
            
            if isinstance(C_hydro, torch.Tensor):
                C_hydro = C_hydro.detach().cpu().numpy()
            if isinstance(C_devia, torch.Tensor):
                C_devia = C_devia.detach().cpu().numpy()
                
            base_constraints.extend([float(C_hydro), float(C_devia)])
        
        base_constraints = np.array(base_constraints)
        
        # Adaptive step size
        pos_scale = np.std(positions)
        h = max(1e-8, pos_scale * 1e-6)
        
        # Numerical differentiation
        J = np.zeros((2 * N_pts, 3 * N_pts))
        
        for i in tqdm(range(N_pts), desc="  Computing Jacobian (Numerical)"):
            for coord in range(3):
                # Perturb position
                positions_perturbed = positions.copy()
                positions_perturbed[i, coord] += h
                
                # Recompute all constraints
                perturbed_constraints = []
                for k in range(N_pts):
                    F_j = self.compute_deformation_gradient_torch(
                        positions_perturbed, reference_positions, neighborhoods, k, sigma_weight)
                    C_hydro, C_devia = self.compute_constraint_residuals(F_j)
                    
                    if isinstance(C_hydro, torch.Tensor):
                        C_hydro = C_hydro.detach().cpu().numpy()
                    if isinstance(C_devia, torch.Tensor):
                        C_devia = C_devia.detach().cpu().numpy()
                        
                    perturbed_constraints.extend([float(C_hydro), float(C_devia)])
                
                perturbed_constraints = np.array(perturbed_constraints)
                
                # Numerical gradient
                grad = (perturbed_constraints - base_constraints) / h
                J[:, 3*i + coord] = grad
        
        return J, base_constraints
    
    def step4_xpbd_constraint_projection(
        self,
        positions: torch.Tensor,
        reference_positions: torch.Tensor,
        neighborhoods: list,
        sigma_weight: float):
        """
        Step 4: Project positions to satisfy constraints using XPBD.
        """

        corrected_pos = positions.clone()
        corrected_pos.requires_grad_(True)

        max_iter, tol = 10, 1e-3
        res_hist      = []
        consecutive_up = 0

        # ========== XPBD main loop ================
        for it in range(max_iter):
            print(f"  [XPBD] Starting iteration {it+1}/{max_iter}...")
            # (a) Compute Jacobian and residuals
            J, r = self.compute_jacobian_autograd(
                corrected_pos, reference_positions,
                neighborhoods, sigma_weight)

            if J.size == 0:
                break

            res_norm = np.linalg.norm(r)
            print(f"  [XPBD] iter {it+1} | Residual Norm = {res_norm:.4e}")

            # Early stopping: if residual is huge after 6 iterations
            if it >= 6 and res_norm > 1e3:
                print("  Stopping early: residual > 1e3 after 6 iterations.")
                break

            # Early stopping: if residual increases for 3 consecutive iterations
            if len(res_hist) and res_norm > res_hist[-1] * 1.20:
                consecutive_up += 1
                if consecutive_up >= 3:
                    print("  Stopping early: residual increased for 3 consecutive iterations.")
                    break
            else:
                consecutive_up = 0
            res_hist.append(res_norm)

            # Convergence check
            if res_norm < tol:
                print("  ✓ Converged.")
                break

            # (b) Compute position update Δp
            δp_flat = self._svd_pinv_solve(
                J, r, tau=self.tau_trunc, lm=1e-3)
            if δp_flat.size == 0:
                print("  Stopping: SVD resulted in zero singular values.")
                break

            δp    = torch.tensor(δp_flat.reshape(-1, 3),
                                 dtype=self.dtype, device=self.device)
            δnorm = torch.linalg.norm(δp).item()

            # (c) Annealed step size strategy
            if it == 0:
                step = min(0.5, 1.0 / (δnorm + 1e-8))          # Conservative first step
            else:
                improving = len(res_hist) < 2 or res_hist[-1] < res_hist[-2]
                step = 1.0 / (δnorm + 1e-8)
                step = step if improving else step * 0.3        # Reduce step if residual worsened
                if res_norm > 1e3:
                    step *= 0.5                                 # Further reduce for large residuals
            step = max(1e-8, step)

            # (d) Update positions
            with torch.no_grad():
                corrected_pos += step * δp
            corrected_pos.requires_grad_(True)

            # Early stopping for tiny updates
            if δnorm * step < tol * 0.1:
                print("  Stopping early: position update is too small.")
                break

        return corrected_pos.detach().cpu().numpy()


    def step5_covariance_estimation(self,
                                    t: int,
                                    reference_positions: np.ndarray,
                                    neighborhoods: list,
                                    corrected_positions: np.ndarray,
                                    sigma_weight: float):

        print("Step 5: Covariance Estimation (Iterative)...")
        
        # Convert inputs to tensors on the correct device
        ref_pos_torch = torch.tensor(reference_positions, device=self.device)
        corr_pos_torch = torch.tensor(corrected_positions, device=self.device)
        
        covariance_matrices_np = []
        
        for i in tqdm(range(self.num_pts), desc="  Computing Covariances"):
            try:
                # Define a function to compute constraints for just point i, 
                # taking only its own position as input.
                def _local_func(pos_i):
                    # Create a temporary full position tensor where only the i-th point's position is changed.
                    temp_positions = corr_pos_torch.clone()
                    temp_positions[i] = pos_i
                    
                    F_i = self.compute_deformation_gradient_torch(
                        temp_positions, ref_pos_torch, neighborhoods, i, sigma_weight
                    )
                    C_h, C_d = self.compute_constraint_residuals(F_i)
                    return torch.stack([C_h, C_d])
                
                # Compute the 2x3 local Jacobian for point i.
                pos_i_torch = corr_pos_torch[i].clone().requires_grad_(True)
                J_local = jacrev(_local_func)(pos_i_torch)

                if J_local.size() == 0 or torch.allclose(J_local, torch.zeros_like(J_local), atol=1e-12):
                    raise ValueError("Local Jacobian is zero or empty")

                # Perform SVD and compute covariance on the GPU
                U, S, Vt = torch.linalg.svd(J_local, full_matrices=False)
                
                s_max = S[0]
                keep = S > self.tau_trunc * s_max
                r = torch.sum(keep).item()
                if r == 0: r = 1 # Keep at least one singular value

                Vr = Vt[:r, :].T
                sr2 = S[:r]**2
                
                Sigma_ti = Vr @ torch.diag(1.0 / sr2) @ Vr.T
                
                # Force Positive Definite on GPU
                eigval, eigvec = torch.linalg.eigh(Sigma_ti)
                eigval = torch.clamp(eigval.real, min=1e-8)
                max_cond = 100.0
                cond_now = eigval[-1] / eigval[0]
                if cond_now > max_cond:
                    max_e = eigval[-1]
                    min_e = max_e / max_cond
                    eigval = torch.clamp(eigval, min=min_e, max=max_e)
                
                Sigma_ti_pd = eigvec @ torch.diag(eigval) @ eigvec.T
                covariance_matrices_np.append(Sigma_ti_pd.cpu().numpy())

            except Exception as e:
                # Fallback to neighborhood covariance
                nb = neighborhoods[i]
                if len(nb) >= 3:
                    nb_cov = np.cov(corrected_positions[nb].T)
                    Sigma_ti = nb_cov * 0.5 + np.eye(3, dtype=np.float32) * 1e-4
                else:
                    Sigma_ti = np.eye(3, dtype=np.float32) * 1e-3
                covariance_matrices_np.append(Sigma_ti)

        return np.array(covariance_matrices_np)

    
    def process_frame(self, t):
        """
        Process a single frame through the complete pipeline.
        """
        print(f"\nProcessing frame {t} (Debug up to step: {self.debug_max_step}) ...")
        
        # Step 0: Estimate reference positions 
        print("Step 0: Estimate reference positions...")
        reference_positions = self.step0_estimate_reference_position(t)
        if self.debug_max_step == 0:
            return {'reference_positions': reference_positions}
    
        positions_torch = torch.tensor(
            self.tracked_pts[:,t,:], dtype=self.dtype,
            device=self.device, requires_grad=True)

        reference_torch = torch.tensor(
            reference_positions, dtype=self.dtype,
            device=self.device) 
        
        # Step 1: Build geodesic neighborhoods
        print("Step 1: Build geodesic neighborhoods...")
        neighborhoods = self.step1_construct_geodesic_neighborhood_heat_method(reference_positions)
        if self.debug_max_step == 1:
            return {
                'reference_positions': reference_positions,
                'neighborhoods': neighborhoods
            }
        
        # !!!!! Calculate adaptive sigma here to be passed to subsequent steps
        all_neighbor_distances = []
        for i in range(self.num_pts):
            neighbors_i = neighborhoods[i]
            if not neighbors_i:
                continue
            pi_r = reference_positions[i]
            for j in neighbors_i:
                dist = np.linalg.norm(reference_positions[j] - pi_r)
                all_neighbor_distances.append(dist)

        if all_neighbor_distances:
            adaptive_sigma = np.median(all_neighbor_distances).astype(np.float32)
            adaptive_sigma = max(adaptive_sigma, 1e-5) 
            print(f"\nAdaptive sigma_weight for frame {t} computed: {adaptive_sigma:.6f}")
        else:
            adaptive_sigma = self.sigma_weight # Fallback
            print(f"\nCould not compute adaptive sigma for frame {t}, using default: {adaptive_sigma:.6f}")

        # Step 2: Estimate deformation gradients
        print("Step 2: Estimate deformation gradients...")
        deformation_gradients = self.step2_estimate_deformation_gradient(
            t, reference_positions, neighborhoods, adaptive_sigma) # Pass sigma
        
        if self.debug_max_step == 2:
            return {
                'reference_positions': reference_positions,
                'neighborhoods': neighborhoods,
                'deformation_gradients': deformation_gradients
            }
            
        # Step 3: Compute constraint residuals
        print("Step 3: Compute constraint residuals...")
        hydrostatic_constraints, deviatoric_constraints = \
            self.step3_compute_constraint_residuals(deformation_gradients)
        
        if self.debug_max_step == 3:
            return {
                'reference_positions': reference_positions,
                'neighborhoods': neighborhoods,
                'deformation_gradients': deformation_gradients,
                'hydrostatic_constraints': hydrostatic_constraints,
                'deviatoric_constraints': deviatoric_constraints
            }
        
        # Step 4: XPBD constraint projection
        print("Step 4: XPBD constraint projection...")
        corrected_means = self.step4_xpbd_constraint_projection(
            positions_torch, reference_torch, neighborhoods, adaptive_sigma) # Pass sigma
        
        if self.debug_max_step == 4:
            return {
                'reference_positions': reference_positions,
                'neighborhoods': neighborhoods,
                'deformation_gradients': deformation_gradients,
                'hydrostatic_constraints': hydrostatic_constraints,
                'deviatoric_constraints': deviatoric_constraints,
                'corrected_means': corrected_means
            }
    
        # Step 5: Covariance estimation
        covariance_matrices = self.step5_covariance_estimation(
            t, reference_positions, neighborhoods, corrected_means, adaptive_sigma) # Pass sigma
        
        # Recompute final statistics
        print("Finalizing statistics...")
        
        final_deformation_gradients = []
        final_hydrostatic_constraints = []
        final_deviatoric_constraints = []
        
        for i in range(self.num_pts):
            F_i = self.compute_deformation_gradient_torch(
                corrected_means, reference_positions, neighborhoods, i, adaptive_sigma) # Pass sigma
            
            if isinstance(F_i, torch.Tensor):
                F_i = F_i.detach().cpu().numpy()
            
            final_deformation_gradients.append(F_i)
            
            C_hydro, C_devia = self.compute_constraint_residuals(F_i)
            if isinstance(C_hydro, torch.Tensor):
                C_hydro = C_hydro.detach().cpu().numpy()
            if isinstance(C_devia, torch.Tensor):
                C_devia = C_devia.detach().cpu().numpy()
            
            final_hydrostatic_constraints.append(float(C_hydro))
            final_deviatoric_constraints.append(float(C_devia))
        
        return {
            'means': corrected_means,
            'covariances': covariance_matrices,
            'reference_positions': reference_positions,
            'deformation_gradients': np.array(final_deformation_gradients),
            'hydrostatic_constraints': np.array(final_hydrostatic_constraints),
            'deviatoric_constraints': np.array(final_deviatoric_constraints),
            'neighborhoods': neighborhoods
        }
    


    def visualize_gaussians(
            self,
            frame_result,
            frame_idx: int,
            sample_points: int = 200,
            raw_pts: Optional[np.ndarray] = None):
        """
        Visualize Gaussians in a two-subplot view:
        - ax1: raw point cloud
        - ax2: XPBD-projected means + principal axes
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

        means      = frame_result["means"]
        covs       = frame_result["covariances"]
        hydro_c    = frame_result["hydrostatic_constraints"]
        devia_c    = frame_result["deviatoric_constraints"]
        rgbs       = frame_result.get("rgbs") # Use .get for safety


        N_pts = len(means)
        sample_count = min(sample_points, N_pts)
        np.random.seed(42)
        sampled_idx = np.random.choice(N_pts, sample_count, replace=False)
        
        fig = plt.figure(figsize=(12, 6))
        ax_raw  = fig.add_subplot(121, projection="3d")
        ax_gaus = fig.add_subplot(122, projection="3d")

        # Raw point cloud ---
        if raw_pts is not None and len(raw_pts) == len(means):
            ax_raw.scatter(*raw_pts.T,  c="royalblue", s=30, alpha=0.9)
            ax_raw.set_title(f"Raw Point Cloud – Frame {frame_idx}")
        else:
            ax_raw.set_title("Raw Cloud (Not Provided)")

        for ax in (ax_raw, ax_gaus):
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

        # XPBD Projected Gaussians ---
        for idx in sampled_idx:
            μ   = means[idx]
            Σ   = covs[idx]
            # Use a default color if rgbs is not available
            rgb = np.clip(rgbs[idx], 0, 1) if rgbs is not None else [0.5, 0.5, 0.8]
            α   = 0.3 + 0.7 * np.clip((abs(hydro_c[idx]) + abs(devia_c[idx])) / 2, 0, 1)

            ax_gaus.scatter(*μ, c=[rgb], s=80, alpha=α, edgecolors="k", linewidths=0.6)

            # Principal axes
            eigval, eigvec = np.linalg.eigh(Σ)
            eigval = np.real(eigval)
            scale  = 0.05 / np.sqrt(eigval.max() + 1e-12)
            for i in np.argsort(eigval)[::-1]:
                length    = 2 * np.sqrt(max(0, eigval[i]) * scale) # Ensure non-negative
                direction = eigvec[:, i] * length
                ax_gaus.plot([μ[0]-direction[0], μ[0]+direction[0]],
                            [μ[1]-direction[1], μ[1]+direction[1]],
                            [μ[2]-direction[2], μ[2]+direction[2]],
                            color=rgb, alpha=α*0.8, linewidth=2 - i*0.4)



        ax_gaus.set_title(
            f"XPBD-Projected Gaussians – Frame {frame_idx}\n"
            f"({sample_count}/{N_pts} points visualized)"
        )
        plt.tight_layout()
        return fig



    
    def create_gaussian_animation_enhanced(self, frame_list, output_filename="complete_gaussian_animation.mp4", 
                                         fps=8, sample_points=30):
        """
        Create an animation of the Gaussian distributions over time.
        """
        import matplotlib.animation as animation
        
        print(f"Creating full Gaussian distribution animation...")
        print(f"  Frames to process: {len(frame_list)}")
        print(f"  Output file: {output_filename}")
        print(f"  FPS: {fps}")
        
        frame_results = {}
        valid_frames = []
        
        start_time = time.time()
        
        for i, frame_idx in enumerate(frame_list):
            
            if i % max(1, len(frame_list) // 10) == 0:
                elapsed = time.time() - start_time
                if i > 0:
                    eta = elapsed * (len(frame_list) - i) / i
                    print(f"  Processing frame {frame_idx} ({i+1}/{len(frame_list)})  - ETA: {eta/60:.1f}m")
                else:
                    print(f"  Processing frame {frame_idx} ({i+1}/{len(frame_list)})")
            

                
            try:
                frame_result = self.process_frame(frame_idx)
                
                essential_result = {
                    'means': frame_result['means'],
                    'covariances': frame_result['covariances'],
                    'rgbs': frame_result['rgbs'],
                    'hydrostatic_constraints': frame_result['hydrostatic_constraints'],
                    'deviatoric_constraints': frame_result['deviatoric_constraints']
                }
                
                frame_results[frame_idx] = essential_result
                valid_frames.append(frame_idx)
                
            except Exception as e:
                print(f"    ✗ Frame {frame_idx} failed to process: {e}")
                continue
        
        processing_time = time.time() - start_time
        print(f"Successfully processed {len(valid_frames)} frames in {processing_time/60:.1f} minutes.")
        
        if len(valid_frames) < 2:
            print("Error: Not enough valid frames to create an animation.")
            return None
        
        # Calculate global bounds for consistent view
        print("Calculating global bounds...")
        all_means = []
        sample_frames = valid_frames[::max(1, len(valid_frames)//10)]
        
        for frame_idx in sample_frames:
            if frame_idx in frame_results:
                means = frame_results[frame_idx]['means']
                all_means.extend(means)
        
            
        all_means = np.array(all_means)
        bounds_min = np.min(all_means, axis=0)
        bounds_max = np.max(all_means, axis=0)
        bounds_center = (bounds_min + bounds_max) / 2
        bounds_range = np.max(bounds_max - bounds_min) * 0.6
        
        print(f"Scene bounds: center={bounds_center}, range={bounds_range:.4f}")
        
        # Setup plot
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Animation function
        def animate(frame_num):
            ax.clear()
            
            frame_idx = valid_frames[frame_num]
            frame_result = frame_results.get(frame_idx)
            
            if frame_result is None:
                ax.text2D(0.5, 0.5, f'Frame {frame_idx}\nProcessing Failed', 
                         transform=ax.transAxes, ha='center', va='center')
                return
            
            means = frame_result['means']
            covariances = frame_result['covariances']
            rgbs = frame_result['rgbs']
            hydro_constraints = frame_result['hydrostatic_constraints']
            devia_constraints = frame_result['deviatoric_constraints']
            
            # Sample points for visualization
            N = len(means)
            n_sample = min(sample_points, N)
            sampled_idx = np.random.choice(N, n_sample, replace=False)
            
            for idx in sampled_idx:
                mean = means[idx]
                cov = covariances[idx]
                rgb_color = np.clip(rgbs[idx], 0, 1)
                
                constraint_magnitude = (abs(hydro_constraints[idx]) + 
                                      abs(devia_constraints[idx]))
                alpha_val = 0.5 + 0.5 * np.clip(constraint_magnitude / 2.0, 0, 1)
                
                ax.scatter(mean[0], mean[1], mean[2], c=[rgb_color], s=150, 
                          alpha=alpha_val, edgecolors='black', linewidths=1)
                
                # Draw ellipsoid axes
                try:
                    eigenvals, eigenvecs = np.linalg.eigh(cov)
                    eigenvals = np.real(eigenvals)
                    eigenvecs = np.real(eigenvecs)
                    
                    max_eigenval = np.max(eigenvals)
                    if max_eigenval < 1e-6:
                        vis_eigenvals = np.array([0.01, 0.008, 0.006]) * bounds_range / 10
                    else:
                        target_size = bounds_range * 0.05
                        scale_factor = target_size / np.sqrt(max_eigenval)
                        vis_eigenvals = eigenvals * scale_factor
                    
                    sorted_indices = np.argsort(vis_eigenvals)[::-1]
                    
                    for i in range(3):
                        axis_idx = sorted_indices[i]
                        if vis_eigenvals[axis_idx] > 0:
                            axis_length = 2 * np.sqrt(vis_eigenvals[axis_idx])
                            axis_dir = eigenvecs[:, axis_idx] * axis_length
                            
                            line_alpha = alpha_val * (0.9 - i * 0.2)
                            linewidth = 6 - i * 1.5
                            
                            ax.plot([mean[0] - axis_dir[0], mean[0] + axis_dir[0]],
                                   [mean[1] - axis_dir[1], mean[1] + axis_dir[1]],
                                   [mean[2] - axis_dir[2], mean[2] + axis_dir[2]], 
                                   color=rgb_color, alpha=line_alpha, linewidth=linewidth)
                
                except Exception:
                    pass
    
            
            # Set consistent view
            ax.set_xlim(bounds_center[0] - bounds_range/2, bounds_center[0] + bounds_range/2)
            ax.set_ylim(bounds_center[1] - bounds_range/2, bounds_center[1] + bounds_range/2)
            ax.set_zlim(bounds_center[2] - bounds_range/2, bounds_center[2] + bounds_range/2)
            
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            ax.set_zlabel('Z', fontsize=12)
            
            # Add frame info
            total_points = len(means)
            
            avg_hydro = np.mean(np.abs(hydro_constraints))
            avg_devia = np.mean(np.abs(devia_constraints))
            
            title_text = f'Complete Gaussian Distributions - Frame {frame_idx}\n'

            title_text += f'Hydro: {avg_hydro:.3f} | Devia: {avg_devia:.3f}'
            
            ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
            
            # Progress indicator
            progress = (frame_num + 1) / len(valid_frames)
            progress_text = f'Progress: {frame_num + 1}/{len(valid_frames)} ({progress*100:.1f}%)'
            
            # Info box
            info_text = f'Frame: {frame_idx}\n'

            info_text += f'Total: {total_points}\nMethod: Complete XPBD'
            
            ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.text2D(0.98, 0.02, progress_text, transform=ax.transAxes,
                     fontsize=10, horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Create and save animation
        print(f"Creating animation with {len(valid_frames)} frames...")
        animation_start = time.time()
        
        anim = animation.FuncAnimation(fig, animate, frames=len(valid_frames), 
                                     interval=1000//fps, blit=False, repeat=True)
        
        print(f"Saving animation to {output_filename}...")
        try:
            if len(valid_frames) > 500:
                dpi = 80
                print("  Using lower DPI for large animation.")
            else:
                dpi = 100
                
            anim.save(output_filename, writer='ffmpeg', fps=fps, 
                     extra_args=['-vcodec', 'libx264'], dpi=dpi)
            
            animation_time = time.time() - animation_start
            total_time = time.time() - start_time
            
            print(f"  Animation successfully saved: {output_filename}")
            print(f"  Animation creation time: {animation_time/60:.1f} minutes")
            print(f"  Total time: {total_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"Failed to save MP4: {e}")
            gif_filename = output_filename.replace('.mp4', '.gif')
            try:
                print(f"Attempting to save as GIF: {gif_filename}")
                anim.save(gif_filename, writer='pillow', fps=min(fps, 10), dpi=80)
                print(f"Successfully saved as GIF: {gif_filename}")
            except Exception as e2:
                print(f"Failed to save GIF: {e2}")
                return None
        
        plt.close(fig)
        return anim
    
    def visualize_debug_info_enhanced(self, frame_result, frame_idx):
        """
        Enhanced debug visualization - supports all points mode.
        """
        fig = plt.figure(figsize=(24, 8))
        
        means = frame_result['means']
        reference_positions = frame_result['reference_positions']
        
        print(f"Frame {frame_idx}:", f"  Total points: {len(means)}")
        
        # 1. Current Point Distribution
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        
        ax1.scatter(means[:, 0], means[:, 1], means[:, 2], 
                    c='limegreen', s=50, alpha=0.7, label='Current (Corrected)')
        
        ax1.scatter(reference_positions[:, 0], reference_positions[:, 1], reference_positions[:, 2], 
                    c='royalblue', s=20, alpha=0.3, label='Reference')
        
        ax1.set_title('Current & Reference')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # 3. Deformation Vectors (Sample)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        total_points = len(means)
        sample_size = min(50, total_points)
        sample_idx = np.random.choice(total_points, sample_size, replace=False)
            
        for idx in sample_idx:
            ax2.plot([reference_positions[idx, 0], means[idx, 0]], 
                    [reference_positions[idx, 1], means[idx, 1]], 
                    [reference_positions[idx, 2], means[idx, 2]], 
                    'g-', alpha=0.6, linewidth=1)
        
        ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
        ax2.set_title('Sample Deformation Vectors')
        ax2.legend()
        

        plt.tight_layout()
        return fig
    
    
    
    def visualize_geodesic_distances(self, reference_positions, source_idx=0):
        """Visualize the geodesic distance field from a source point."""
        print(f"\nVisualizing geodesic distance field (source: {source_idx})...")
        
        fig = plt.figure(figsize=(15, 5))
        
        # Compute geodesic distances
        L, M, sigma = self._construct_robust_laplacian(reference_positions)
        u = self._solve_heat_diffusion(L, M, source_idx)
        X_hat = self._compute_gradient_field(L, u)
        geodesic_distances = self._recover_distances_poisson(L, X_hat)
        
        # 1. 3D point cloud with distance colormap
        ax1 = fig.add_subplot(131, projection='3d')
        scatter = ax1.scatter(reference_positions[:, 0], 
                             reference_positions[:, 1], 
                             reference_positions[:, 2],
                             c=geodesic_distances, 
                             cmap='viridis', s=50)
        ax1.scatter(*reference_positions[source_idx], c='red', s=200, marker='*')
        plt.colorbar(scatter, ax=ax1, label='Geodesic Distance')
        ax1.set_title(f'Geodesic Distances from Point {source_idx}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # 2. Distance histogram
        ax2 = fig.add_subplot(132)
        ax2.hist(geodesic_distances, bins=50, alpha=0.7, color='blue')
        ax2.axvline(x=self.tau_g, color='red', linestyle='--', 
                    label=f'τ_g threshold = {self.tau_g:.3f}')
        ax2.set_xlabel('Geodesic Distance')
        ax2.set_ylabel('Count')
        ax2.set_title('Distance Distribution')
        ax2.legend()
        
        # 3. Geodesic neighborhood visualization
        ax3 = fig.add_subplot(133, projection='3d')
        neighborhood = np.where(geodesic_distances <= self.tau_g)[0]
        
        # Points within the neighborhood
        ax3.scatter(reference_positions[neighborhood, 0],
                    reference_positions[neighborhood, 1],
                    reference_positions[neighborhood, 2],
                    c='green', s=100, alpha=0.8, label='Geodesic Neighborhood')
        
        # Points outside the neighborhood
        outside = np.setdiff1d(range(len(reference_positions)), neighborhood)
        ax3.scatter(reference_positions[outside, 0],
                    reference_positions[outside, 1],
                    reference_positions[outside, 2],
                    c='gray', s=20, alpha=0.3)
        
        # Source point
        ax3.scatter(*reference_positions[source_idx], 
                    c='red', s=200, marker='*', label='Source')
        
        ax3.set_title(f'Geodesic Neighborhood (|N|={len(neighborhood)})')
        ax3.legend()
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        plt.tight_layout()
        return fig

    def compare_geodesic_euclidean(self, reference_positions, sample_pairs=20):
        """Compare geodesic and Euclidean distances."""
        print(f"\nComparing Geodesic vs. Euclidean distances ({sample_pairs} pairs)...")
        
        n_points = len(reference_positions)
        
        # Randomly select point pairs
        np.random.seed(42)
        pairs = [(np.random.randint(n_points), np.random.randint(n_points)) 
                 for _ in range(sample_pairs)]
        
        geodesic_dists = []
        euclidean_dists = []
        
        L, M, sigma = self._construct_robust_laplacian(reference_positions)
        
        for i, (src, tgt) in enumerate(pairs):
            if src == tgt:
                continue
            
            if i % 5 == 0:
                print(f"  Processing pair {i+1}/{len(pairs)}...")
                
            # Geodesic distance
            u = self._solve_heat_diffusion(L, M, src)
            X_hat = self._compute_gradient_field(L, u)
            geo_dists = self._recover_distances_poisson(L, X_hat)
            geodesic_dists.append(geo_dists[tgt])
            
            # Euclidean distance
            euclidean_dists.append(np.linalg.norm(
                reference_positions[src] - reference_positions[tgt]))
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot comparison
        ax1.scatter(euclidean_dists, geodesic_dists, alpha=0.6)
        ax1.plot([0, max(euclidean_dists)], [0, max(euclidean_dists)], 
                 'r--', label='y=x (identical)')
        ax1.set_xlabel('Euclidean Distance')
        ax1.set_ylabel('Geodesic Distance')
        ax1.set_title('Geodesic vs Euclidean Distances')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Ratio distribution
        ratios = np.array(geodesic_dists) / (np.array(euclidean_dists) + 1e-8)
        ax2.hist(ratios, bins=30, alpha=0.7, color='green')
        ax2.axvline(x=1.0, color='red', linestyle='--', label='Ratio = 1')
        ax2.set_xlabel('Geodesic/Euclidean Ratio')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Distance Ratio Distribution\nMean ratio: {np.mean(ratios):.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def evaluate_neighborhood_quality(self, reference_positions, num_samples=10):
        """Evaluate the quality of the generated neighborhoods."""
        neighborhoods = self.step1_construct_geodesic_neighborhood_heat_method(
            reference_positions)
        
        fig = plt.figure(figsize=(12, 10))
        
        # 1. Neighborhood Size Distribution
        ax1 = plt.subplot(2, 2, 1)
        sizes = [len(n) for n in neighborhoods]
        ax1.hist(sizes, bins=30, alpha=0.7, color='blue')
        ax1.axvline(x=self.k_neighbors, color='red', linestyle='--', 
                    label=f'Target k={self.k_neighbors}')
        ax1.set_xlabel('Neighborhood Size')
        ax1.set_ylabel('Count')
        ax1.set_title(f'Neighborhood Size Distribution\nMean: {np.mean(sizes):.1f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Randomly pick some samples to visualize
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        sample_indices = np.random.choice(len(neighborhoods), 
                                         min(num_samples, len(neighborhoods)), 
                                         replace=False)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(sample_indices)))
        
        for i, idx in enumerate(sample_indices):
            center = reference_positions[idx]
            neighbors = neighborhoods[idx]
            
            # Center points
            ax2.scatter(*center, c=[colors[i]], s=200, marker='*', 
                       edgecolors='black', linewidths=2)
            
            # Neighbor points
            if len(neighbors) > 0:
                neighbor_pts = reference_positions[neighbors]
                ax2.scatter(neighbor_pts[:, 0], neighbor_pts[:, 1], 
                           neighbor_pts[:, 2], c=[colors[i]], s=50, alpha=0.6)
                
                # Connection lines
                for n_idx in neighbors[:5]:
                    ax2.plot([center[0], reference_positions[n_idx, 0]],
                            [center[1], reference_positions[n_idx, 1]],
                            [center[2], reference_positions[n_idx, 2]],
                            c=colors[i], alpha=0.3, linewidth=1)
        
        ax2.set_title('Sample Geodesic Neighborhoods')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # 3. Neighborhood connectivity stats
        ax3 = plt.subplot(2, 2, 3)
        ax3.text(0.1, 0.5, 
                 f"Geodesic Neighborhood Statistics:\n\n"
                 f"• Mean size: {np.mean(sizes):.2f}\n"
                 f"• Std size: {np.std(sizes):.2f}\n"
                 f"• Min size: {np.min(sizes)}\n"
                 f"• Max size: {np.max(sizes)}\n"
                 f"• Empty neighborhoods: {sum(1 for s in sizes if s == 0)}\n"
                 f"• Target geodesic radius: {self.tau_g:.4f}",
                 fontsize=12, transform=ax3.transAxes)
        ax3.axis('off')
        
        # 4. Heat diffusion parameters
        ax4 = plt.subplot(2, 2, 4)
        ax4.text(0.1, 0.5,
                 f"Heat Method Parameters:\n"
                 f"• Sigma (adaptive): {self.heat_sigma:.6f}\n"
                 f"• Diffusion time: {self.diffusion_time}\n"
                 f"• k-neighbors for graph: {self.k_neighbors}\n"
                 f"• Robust Laplacian: Yes\n"
                 f"• Point cloud size: {len(reference_positions)}",
                 fontsize=12, transform=ax4.transAxes)
        ax4.axis('off')
        
        plt.tight_layout()
        return fig



def main():
    print("=== COMPLETE XPBD Gaussian Modeling - All Features ===")

    data_file = "downsampled_pc_2000_filtered.pkl"
    if not Path(data_file).exists():
        print(f"Data file not found: {data_file}")
        return
    
    # Debugging until ..........
    try:
        user_step = input(
            "➤ Select max debug step (0=step0, ..., 5=step5, Enter=all): "
        ).strip()
        debug_max_step = int(user_step) if user_step else None
    except:
        debug_max_step = None

    try:
        generate_animation_input = input("➤ Generate animation? (y/n, default=n): ").strip().lower()
        generate_animation = generate_animation_input == 'y'
    except:
        generate_animation = False
    
    frame_step = 5  # Default
    if generate_animation:
        try:
            default_step = 5
            user_in = input(f"➤ Process every N frames? (Enter for default={default_step}): ").strip()
            frame_step = int(user_in) if user_in else default_step
            assert frame_step > 0, "Frame step must be a positive integer."
        except (ValueError, AssertionError) as e:
            print(f"Invalid input, using default {default_step}.")
            frame_step = default_step

    # Check PyTorch
    use_autograd = True
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        print("PyTorch not available, using NumPy.")
        use_autograd = False
        device = 'cpu'

    # Initialize XPBD model
    xpbd_model = CompleteXPBDGaussianModeling(
        tau=8, k_neighbors=20, tau_g=0.15, sigma_weight=0.01, epsilon=1e-4,
        use_autograd=use_autograd, device=device,
        debug_max_step = debug_max_step 
    )
         
    try:
        # 1. Load data
        xpbd_model.load_tracking_data(data_file)
        
        target_frame = xpbd_model.num_frames // 3
        print(f"\nProcessing example frame {target_frame} ...")
        frame_result = xpbd_model.process_frame(target_frame)
        
        if frame_result is None:
             print("Frame processing failed.")
             return

        reference_positions = frame_result.get('reference_positions')
        output_dir = Path("geodesic_results")
        output_dir.mkdir(exist_ok=True)
        
        
        fig1 = xpbd_model.visualize_geodesic_distances(reference_positions, source_idx=0)
        fig1.savefig(output_dir / 'geodesic_distance_field.png', dpi=150, bbox_inches='tight')
        plt.close(fig1)
        
        fig2 = xpbd_model.compare_geodesic_euclidean(reference_positions, sample_pairs=30)
        fig2.savefig(output_dir / 'geodesic_vs_euclidean.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)
        
        fig3 = xpbd_model.evaluate_neighborhood_quality(reference_positions, num_samples=15)
        fig3.savefig(output_dir / 'neighborhood_quality.png', dpi=150, bbox_inches='tight')
        plt.close(fig3)
        
        if 'means' in frame_result:
            fig6 = xpbd_model.visualize_gaussians(
                frame_result, 
                frame_idx=target_frame, 
                sample_points=50,
                raw_pts=xpbd_model.tracked_pts[:,target_frame,:]
            )
            fig6.savefig(output_dir / f'frame_{target_frame}_gaussians.png', dpi=150, bbox_inches='tight')
            print(f"   ✓ Saved visualization: frame_{target_frame}_gaussians.png")
            plt.close(fig6)
                
        if generate_animation:
            frame_list = list(range(0, xpbd_model.num_frames, frame_step))
            anim_file = f"xpbd_gaussians_every_{frame_step}f.mp4"
            xpbd_model.create_gaussian_animation_enhanced(
                frame_list      = frame_list,
                output_filename = anim_file,
                fps             = 8,
                sample_points   = 50
            )

        print("\n" + "="*60)
        print("All results generated successfully!")
        print(f"Output directory: {output_dir.absolute()}")
        print("Generated files:")
        for file in sorted(output_dir.glob('*.png')):
            print(f"   - {file.name}")
        if 'generate_animation' in locals() and generate_animation and 'anim_file' in locals() and Path(anim_file).exists():
            print(f"   - {anim_file} (Animation)")
        print("="*60)

        return xpbd_model, frame_result

    except Exception as e:
        print(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    xpbd_model, frame_result = main()
