"""
turbo_m.py

Authors: natelgrw, dkochar
Last Edited: 02/13/2026

TuRBO-M sampling algorithm implementation using BoTorch and GPyTorch.
"""

from dataclasses import dataclass

import math
MAX_UINT32_SEED = (2 ** 32) - 1
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from torch.quasirandom import SobolEngine

from simulator import globalsy

try:
    from algorithms.sobol.generator import (
        TECH_CONSTANTS,
        _read_model_l_bounds,
        discrete_linear_grid,
        pick_from_grid_by_u,
        nearest_grid_u,
        e_series_grid,
    )
except Exception:
    TECH_CONSTANTS = {}

    def _read_model_l_bounds(fet_num):
        """
        Fallback model-card bounds when Sobol generator helpers are unavailable.
        
        Parameters:
        -----------
        fet_num (int): Technology node to infer nominal L bounds.

        Returns:
        --------
        tuple: (lmin, lmax) in meters.
        """
        return (10e-9, 30e-9)

    def discrete_linear_grid(lb, ub, step):
        """
        Fallback linear grid utility matching Sobol helper behavior.
        
        Parameters:
        -----------
        lb (float): Lower bound.
        ub (float): Upper bound.
        step (float): Step size.

        Returns:
        --------
        np.array: Array of discrete values.
        """
        if step <= 0:
            return np.array([lb])
        count = int(math.floor((ub - lb) / step))
        grid = lb + np.arange(0, count + 1) * step
        if grid.size == 0:
            return np.array([lb])
        if grid[-1] < ub - 1e-12:
            grid = np.append(grid, ub)
        return grid

    def pick_from_grid_by_u(grid, u):
        """
        Fallback grid picker that maps normalized coordinate to nearest index.

        Parameters:
        -----------
        grid (np.array): Grid of values.
        u (float): Normalized value in [0,1].

        Returns:
        --------
        tuple: (value, index)
        """
        if grid is None or len(grid) == 0:
            return None, None
        idx = int(u * len(grid))
        if idx >= len(grid):
            idx = len(grid) - 1
        return float(grid[idx]), idx

    def nearest_grid_u(grid, val):
        """
        Fallback inverse grid mapping helper.

        Parameters:
        -----------
        grid (np.array): Grid of values.
        val (float): Value to find nearest grid point for.

        Returns:
        --------
        float: Normalized u in [0,1] corresponding to nearest grid point.
        """
        if grid is None or len(grid) == 0:
            return 0.5
        idx = int(np.argmin(np.abs(grid - float(val))))
        if len(grid) <= 1:
            return 0.0
        return float(idx) / max(1, (len(grid) - 1))


# ===== TurboState Dataclass ===== #


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = float("inf")
    restart_triggered: bool = False
    
    # track the center of this trust region relative to global X
    center_idx: int = -1

    def __post_init__(self):
        """
        Fill derived defaults that depend on configured dimensionality.
        """
        if math.isnan(self.failure_tolerance):
            self.failure_tolerance = 2 * self.dim


# ===== ASPECTOR TuRBO-M Optimizer Class ===== #


class ASPECTOR_TurboM:
    def __init__(
        self,
        dim,
        specs_weights=None,
        num_trust_regions=5,
        batch_size=64,
        failure_tolerance=None,
        seed=None,
        device=None,
        dtype=torch.double,
        **kwargs,
    ):
        """
        Initialize ASPECTOR TuRBO-M optimizer state.

        Parameters:
        -----------
        dim (int): Optimization dimensionality in normalized unit space.
        specs_weights (dict | None): Optional scalarization weights per spec.
        num_trust_regions (int): Number of parallel trust regions.
        batch_size (int): Total candidate points generated per ask call.
        failure_tolerance (int | None): Optional fixed TR shrink tolerance.
        seed (int | None): Base seed for deterministic Sobol candidate generation.
        device (torch.device | None): Torch device for model/data tensors.
        dtype (torch.dtype): Tensor floating-point type.
        """
        self.dim = dim
        self.num_trust_regions = num_trust_regions
        self.batch_size = batch_size
        self.seed = seed
        self._sobol_seed_counter = 0
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # initialize trust regions
        self.state = [
            TurboState(dim=dim, batch_size=batch_size)
            for _ in range(num_trust_regions)
        ]
        if failure_tolerance is not None:
            for s in self.state:
                s.failure_tolerance = failure_tolerance

        # data storage
        self.X = torch.empty((0, dim), dtype=dtype, device=self.device)
        self.Y = torch.empty((0, 1), dtype=dtype, device=self.device)
        
        # robust scalarization statistics
        self.spec_stats = {}
        self.weights = specs_weights if specs_weights else self._get_default_weights()

    def _next_sobol_engine(self):
        """
        Build a SobolEngine using a reproducible advancing seed stream when configured.

        Returns:
        --------
        SobolEngine: Torch Sobol engine for candidate generation.
        """
        sobol_seed = None
        if self.seed is not None:
            sobol_seed = (int(self.seed) + int(self._sobol_seed_counter)) % (MAX_UINT32_SEED + 1)
            self._sobol_seed_counter += 1
        return SobolEngine(self.dim, scramble=True, seed=sobol_seed)

    def params_bounds_from_globals(self, param_names, fet_num=None, vdd_nominal=None):
        """
        Build lower and upper bounds arrays from globalsy parameter metadata.

        Parameters:
        -----------
        param_names (list[str]): Ordered parameter names to bound.
        fet_num (int | None): Optional technology node used to infer nominal VDD.
        vdd_nominal (float | None): Optional fallback nominal VDD.

        Returns:
        --------
        tuple: (lbs, ubs) as NumPy arrays in the same order as param_names.
        """
        lbs = []
        ubs = []

        # determine a nominal VDD if possible.
        current_vdd = None
        if fet_num is not None:
            current_vdd = TECH_CONSTANTS.get(fet_num, {}).get('vdd_nom')
        if current_vdd is None and vdd_nominal is not None:
            current_vdd = vdd_nominal
        if current_vdd is None:
            current_vdd = 1.0

        for p in param_names:
            if p.startswith('nA'):
                bp = globalsy.basic_params.get('L', None)
                if bp:
                    try:
                        lb = float(bp.get('lb'))
                        ub = float(bp.get('ub'))
                    except Exception:
                        lb, ub = (10e-9, 30e-9)
                else:
                    lb, ub = (10e-9, 30e-9)

            # fin count
            elif p.startswith('nB'):
                bp = globalsy.basic_params.get('Nfin', None)
                if bp:
                    lb = int(bp.get('lb'))
                    ub = int(bp.get('ub'))
                else:
                    lb, ub = (1, 256)

            # biases
            elif 'biasn' in p or p.startswith('vbiasn'):
                key = p if p in globalsy.basic_params else 'vbiasn0'
                info = globalsy.basic_params.get(key, None)
                if info:
                    try:
                        lb = float(eval(info.get('lb'), {"__builtins__": None}, {"vdd_nominal": current_vdd}))
                        ub = float(eval(info.get('ub'), {"__builtins__": None}, {"vdd_nominal": current_vdd}))
                    except Exception:
                        lb, ub = (0.35 * current_vdd, 0.65 * current_vdd)
                else:
                    lb, ub = (0.35 * current_vdd, 0.65 * current_vdd)
                try:
                    step = 0.01
                    lb = round(lb / step) * step
                    ub = round(ub / step) * step
                    if ub < lb:
                        ub = lb
                except Exception:
                    pass

            elif 'biasp' in p or p.startswith('vbiasp'):
                key = p if p in globalsy.basic_params else 'vbiasp0'
                info = globalsy.basic_params.get(key, None)
                if info:
                    try:
                        lb = float(eval(info.get('lb'), {"__builtins__": None}, {"vdd_nominal": current_vdd}))
                        ub = float(eval(info.get('ub'), {"__builtins__": None}, {"vdd_nominal": current_vdd}))
                    except Exception:
                        lb, ub = (0.30 * current_vdd, 0.75 * current_vdd)
                else:
                    lb, ub = (0.30 * current_vdd, 0.75 * current_vdd)
                try:
                    step = 0.01
                    lb = round(lb / step) * step
                    ub = round(ub / step) * step
                    if ub < lb:
                        ub = lb
                except Exception:
                    pass

            # internal capacitors
            elif p.startswith('nC'):
                bp = globalsy.basic_params.get('C_internal', None)
                if bp:
                    try:
                        lb = float(bp.get('lb'))
                        ub = float(bp.get('ub'))
                    except Exception:
                        lb, ub = (100e-15, 5e-12)
                else:
                    lb, ub = (100e-15, 5e-12)

            # internal resistors
            elif p.startswith('nR'):
                bp = globalsy.basic_params.get('R_internal', None)
                if bp:
                    try:
                        lb = float(bp.get('lb'))
                        ub = float(bp.get('ub'))
                    except Exception:
                        lb, ub = (500.0, 100e3)
                else:
                    lb, ub = (500.0, 100e3)

            else:
                # fallback wide bounds
                lb, ub = (-1e9, 1e9)

            lbs.append(lb)
            ubs.append(ub)

        return np.array(lbs, dtype=float), np.array(ubs, dtype=float)

    def physical_to_unit(self, phys_dicts, param_names, fet_num=None, vdd_nominal=None):
        """
        Convert physical parameter dictionaries to normalized unit coordinates.

        Parameters:
        -----------
        phys_dicts (list[dict]): Physical parameter dictionaries.
        param_names (list[str]): Ordered parameter names.
        fet_num (int | None): Optional technology node for node-specific mappings.
        vdd_nominal (float | None): Optional nominal VDD override.

        Returns:
        --------
        torch.Tensor: Tensor shaped [N, len(param_names)] in [0, 1].
        """
        import math

        lbs, ubs = self.params_bounds_from_globals(param_names, fet_num=fet_num, vdd_nominal=vdd_nominal)

        rows = []
        for d in phys_dicts:
            urow = []
            for i, p in enumerate(param_names):
                val = d.get(p)
                lb = float(lbs[i])
                ub = float(ubs[i])
                if val is None:
                    u = 0.5
                else:
                    # discrete L mapping
                    if p.startswith('nA'):
                        try:
                            fet = fet_num if fet_num is not None else globalsy.testbench_params['Fet_num'][0]
                            model_lmin, model_lmax = _read_model_l_bounds(fet)
                            step = globalsy.basic_params.get('L', {}).get('step', 2e-9)
                            grid = np.arange(model_lmin, model_lmax + 1e-15, step)
                            if grid.size == 0:
                                u = 0.5
                            else:
                                idx = int(np.argmin(np.abs(grid - float(val))))
                                u = float(idx) / max(1, (grid.size - 1))
                        except Exception:
                            u = (float(val) - lb) / (ub - lb) if ub != lb else 0.5

                    elif p.startswith('nC') or p.startswith('nR'):
                        # log mapping
                        try:
                            u = (math.log10(float(val)) - math.log10(lb)) / (math.log10(ub) - math.log10(lb))
                        except Exception:
                            u = 0.5
                    else:
                        # linear mapping
                        try:
                            u = (float(val) - lb) / (ub - lb) if ub != lb else 0.5
                        except Exception:
                            u = 0.5

                # clamp
                if u is None or np.isnan(u):
                    u = 0.5
                u = max(0.0, min(1.0, float(u)))
                urow.append(u)
            rows.append(urow)

        return torch.tensor(rows, dtype=self.dtype, device=self.device)

    def unit_to_physical(self, X_tensor, param_names, fet_num=None, vdd_nominal=None):
        """
        Convert normalized unit coordinates to physical parameter dictionaries.

        Parameters:
        -----------
        X_tensor (torch.Tensor): Normalized samples with shape [N, D].
        param_names (list[str]): Ordered parameter names.
        fet_num (int | None): Optional technology node for node-specific mappings.
        vdd_nominal (float | None): Optional nominal VDD override.

        Returns:
        --------
        list[dict]: Physical parameter dictionaries.
        """
        X = X_tensor.detach().cpu().numpy()
        lbs, ubs = self.params_bounds_from_globals(param_names, fet_num=fet_num, vdd_nominal=vdd_nominal)
        out = []
        for row in X:
            d = {}
            for i, p in enumerate(param_names):
                u = float(row[i])
                lb = float(lbs[i])
                ub = float(ubs[i])
                if p.startswith('nA'):
                    try:
                        fet = fet_num if fet_num is not None else globalsy.testbench_params['Fet_num'][0]
                        model_lmin, model_lmax = _read_model_l_bounds(fet)
                        step = globalsy.basic_params.get('L', {}).get('step', 2e-9)
                        grid = np.arange(model_lmin, model_lmax + 1e-15, step)
                        idx = int(round(u * (grid.size - 1)))
                        idx = max(0, min(idx, grid.size - 1))
                        val = float(grid[idx])
                    except Exception:
                        val = lb + u * (ub - lb)
                elif p.startswith('nC') or p.startswith('nR'):
                    try:
                        val = 10 ** (np.log10(lb) + u * (np.log10(ub) - np.log10(lb)))
                    except Exception:
                        val = lb + u * (ub - lb)
                elif p.startswith('vbias') or 'bias' in p:
                    # discrete bias mapping
                    try:
                        step = 0.01
                        if step <= 0:
                            val = lb + u * (ub - lb)
                        else:
                            grid = discrete_linear_grid(lb, ub, step)
                            idx = int(round(u * (max(1, len(grid) - 1))))
                            idx = max(0, min(idx, len(grid) - 1))
                            val = float(grid[idx])
                    except Exception:
                        val = lb + u * (ub - lb)
                elif p == 'vdd' or p == 'vcm':
                    # discrete VDD/VCM mapping using absolute 0.01 V grid
                    try:
                        step = 0.01
                        if step <= 0:
                            val = lb + u * (ub - lb)
                        else:
                            grid = discrete_linear_grid(lb, ub, step)
                            idx = int(round(u * (max(1, len(grid) - 1))))
                            idx = max(0, min(idx, len(grid) - 1))
                            val = float(grid[idx])
                    except Exception:
                        val = lb + u * (ub - lb)
                else:
                    val = lb + u * (ub - lb)
                d[p] = val
            out.append(d)
        return out

    def load_state(self, X_init, Y_init=None):
        """
        Warm-start or resume optimizer state from previous data.

        Parameters:
        -----------
        X_init (torch.Tensor | np.ndarray | dict): Warm-start points or full state dict.
        Y_init (torch.Tensor | np.ndarray | None): Optional warm-start objective values.
        """
        if isinstance(X_init, dict):
            # full resume from serialized state dict
            state = X_init
            self.state = state['state']
            self.X = state['X']
            self.Y = state['Y']
            self.spec_stats = state.get('spec_stats', {})
            self.weights = state.get('weights', self.weights)
            self.seed = state.get('seed', self.seed)
            self._sobol_seed_counter = state.get('sobol_seed_counter', self._sobol_seed_counter)
            return
        if not isinstance(X_init, torch.Tensor):
            X_init = torch.tensor(X_init, dtype=self.dtype, device=self.device)
        if not isinstance(Y_init, torch.Tensor):
            Y_init = torch.tensor(Y_init, dtype=self.dtype, device=self.device)
        self.X = torch.cat([self.X, X_init], dim=0)
        self.Y = torch.cat([self.Y, Y_init], dim=0)
        if len(self.Y) > 0:
            k = min(len(self.Y), len(self.state))
            _, best_indices = torch.topk(self.Y.flatten(), k, largest=False)
            for i, state in enumerate(self.state):
                idx = i % len(best_indices)
                state.center_idx = best_indices[idx].item()
                state.best_value = self.Y[state.center_idx].item()

    def _get_default_weights(self):
        """
        Return default multi-objective scalarization weights.

        Returns:
        --------
        dict: Mapping from spec names to scalarization weights.
        """
        return {
            # High importance on stability constraints.
            'gain_ol': 1.0,
            'ugbw': 1.0,
            'pm': 100.0,
            'power': 2.0, 'estimated_area': 1.0, 'cmrr': 1.0, 'psrr': 1.0,
            'vos': 5.0, 'output_voltage_swing': 1.0, 'integrated_noise': 1.0,
            'slew_rate': 1.0, 'settle_time': 1.0, 'thd': 1.0
        }

    def _update_spec_stats(self, specs_list):
        """
        Update running statistics used for objective normalization.

        Parameters:
        -----------
        specs_list (list[dict | None]): Batch of extracted spec dictionaries.
        """
        for key in self.weights.keys():
            if key not in self.spec_stats:
                self.spec_stats[key] = {'vals': []}
            
            vals = []
            for s in specs_list:
                if s and s.get('valid', False) and s.get(key) is not None:
                    v = s.get(key)
                    # handle tuple metrics
                    if isinstance(v, (list, tuple)):
                        v = abs(v[1] - v[0])
                    vals.append(float(v))
            
            if vals:
                # append to history 
                self.spec_stats[key]['vals'].extend(vals)
                
                # compute robust stats
                data = torch.tensor(self.spec_stats[key]['vals'])
                self.spec_stats[key]['mean'] = data.mean().item()
                self.spec_stats[key]['std'] = data.std().item() + 1e-9

    def scalarize_specs(self, specs_list, update_stats=True):
        """
        Scalarize batched spec dictionaries into a single objective value.

        Parameters:
        -----------
        specs_list (list[dict | None]): Batch of per-design spec dictionaries.
        update_stats (bool): Whether to refresh running normalization statistics.

        Returns:
        --------
        torch.Tensor: Column vector [N, 1] of scalarized objective values.
        """
        if update_stats:
            self._update_spec_stats(specs_list)
            
        y_vals = []
        for specs in specs_list:
            if specs is None or not specs.get('valid', False):
                y_vals.append([1e6])  # Penalty.
                continue

            cost = 0.0

            # smooth constraint penalties (quadratic hinge)
            # defaults preserve prior intent while giving smoother gradients
            lam_pm = self.weights.get('_lam_pm', 2.0e3)
            lam_gain = self.weights.get('_lam_gain', 2.0e3)
            lam_ugbw = self.weights.get('_lam_ugbw', 2.0e3)

            def qhinge_geq(val, target, lam):
                try:
                    violation = max(0.0, (float(target) - float(val)) / max(abs(float(target)), 1e-12))
                except Exception:
                    violation = 1.0
                return float(lam) * (violation ** 2)

            def qhinge_nonneg(val, lam):
                try:
                    violation = max(0.0, -float(val))
                except Exception:
                    violation = 1.0
                return float(lam) * (violation ** 2)
            
            pm = specs.get('pm', 0.0)
            cost += qhinge_geq(pm, 45.0, lam_pm)

            gain = specs.get('gain_ol', 0.0)
            cost += qhinge_geq(gain, 35.0, lam_gain)

            ugbw = specs.get('ugbw', 0.0)
            cost += qhinge_geq(ugbw, 1e6, lam_ugbw)

            # normalized objectives
            pm_target = self.weights.get('_pm_target', None)
            pm_range = self.weights.get('_pm_range', 0.0)

            for key, weight in self.weights.items():
                if key.startswith('_'):
                    continue

                if key == 'pm':
                    continue

                if key == 'psrr' and specs.get('is_diff', False):
                    continue

                raw_val = specs.get(key, 0.0)
                if isinstance(raw_val, (list, tuple)):
                    raw_val = abs(raw_val[1] - raw_val[0])
                if raw_val is None:
                    raw_val = 0.0

                mu = self.spec_stats[key].get('mean', 0.0)
                sigma = self.spec_stats[key].get('std', 1.0)

                z_val = (raw_val - mu) / sigma

                if key in ['gain_ol', 'ugbw', 'cmrr', 'psrr', 'slew_rate', 'output_voltage_swing']:
                    term = -1.0 * z_val
                else:
                    term = 1.0 * z_val

                cost += weight * term

            # optional targeted metrics
            if pm_target is not None and pm_range is not None and pm_target > 0:
                diff = abs(pm - pm_target)
                if diff > pm_range:
                    cost += self.weights.get('pm', 0.0) * (diff - pm_range)

            y_vals.append([cost])
            
        return torch.tensor(y_vals, dtype=self.dtype, device=self.device)

    def tell(self, X_new, specs_new):
        """
        Ingest a new evaluated batch and update trust-region state.

        Parameters:
        -----------
        X_new (torch.Tensor | np.ndarray): Candidate locations in unit space.
        specs_new (list[dict | None]): Extracted spec dictionaries for X_new.
        """
        if not isinstance(X_new, torch.Tensor):
            X_new = torch.tensor(X_new, dtype=self.dtype, device=self.device)
        
        # scalarize.
        Y_new = self.scalarize_specs(specs_new, update_stats=True)
        
        # append to database.
        self.X = torch.cat([self.X, X_new], dim=0)
        self.Y = torch.cat([self.Y, Y_new], dim=0)
        
        for state in self.state:
            if state.restart_triggered:
                continue
            
            # identify the center of trust region
            if state.center_idx >= 0 and state.center_idx < len(self.X):
                center = self.X[state.center_idx].unsqueeze(0)
            else:
                continue
            
            dists = torch.cdist(X_new, center).squeeze()
            
            mask = dists <= (state.length * math.sqrt(self.dim)) 
            
            relevant_indices = torch.where(mask)[0]
            
            if len(relevant_indices) > 0:
                # check whether this batch improved region
                best_in_batch = Y_new[relevant_indices].min().item()
                
                if best_in_batch < state.best_value - 1e-4:
                    state.success_counter += 1
                    state.failure_counter = 0
                    state.best_value = best_in_batch
                else:
                    state.success_counter = 0
                    state.failure_counter += 1
            else:
                pass

            # update trust-region hyperparameters
            if state.success_counter >= state.success_tolerance:
                state.length = min(2.0 * state.length, state.length_max)
                state.success_counter = 0
            elif state.failure_counter >= state.failure_tolerance:
                state.length /= 2.0
                state.failure_counter = 0
                
            if state.length < state.length_min:
                state.restart_triggered = True

    def ask(self, n_samples=None):
        """
        Generate a new candidate batch (optimizer ask API).

        Parameters:
        -----------
        n_samples (int | None): Optional override for total batch size.

        Returns:
        --------
        torch.Tensor: Candidate points in unit space.
        """
        if n_samples is not None:
            # override the total batch size split across regions
            self.batch_size = n_samples

        return self.generate_batch()

    def generate_batch(self, n_candidates=2000):
        """
        Generate candidate points using dynamic multi-TR allocation and TS.

        Parameters:
        -----------
        n_candidates (int): Number of local TS candidates per active trust region.

        Returns:
        --------
        torch.Tensor: New candidate points in unit space.
        """
        X_next = torch.empty((0, self.dim), dtype=self.dtype, device=self.device)
        
        # dynamic allocation
        scores = np.array([1.0 / (1.0 + s.failure_counter) for s in self.state])
        probs = scores / scores.sum()
        
        # discrete allocation
        counts = np.random.multinomial(self.batch_size, probs)
        
        for i, state in enumerate(self.state):
            n_samples = counts[i]
            if n_samples == 0:
                continue
            
            # handle restart.
            if state.restart_triggered:
                # reset region state.
                state.length = 0.8
                state.failure_counter = 0
                state.success_counter = 0
                state.best_value = float('inf')
                state.restart_triggered = False
                
                # Sobol restart
                sobol = self._next_sobol_engine()
                cand = sobol.draw(n_samples).to(dtype=self.dtype, device=self.device)
                X_next = torch.cat([X_next, cand], dim=0)
                
                continue
            
            # determine center
            if len(self.X) > 0:
                if state.center_idx == -1:
                    # first time initialization
                    state.center_idx = self.Y.argmin().item()

                old_center = self.X[state.center_idx].unsqueeze(0)
                dists = torch.cdist(self.X, old_center).squeeze()
                mask = dists <= state.length * 2.0
                
                if mask.any():
                    # find minimum Y 
                    subset_y = self.Y[mask]
                    subset_indices = torch.where(mask)[0]
                    local_best_local_idx = subset_y.argmin()
                    state.center_idx = subset_indices[local_best_local_idx].item()
                    
                x_center = self.X[state.center_idx].unsqueeze(0)
                
                # model fitting
                dists = torch.cdist(self.X, x_center).squeeze()
                
                n_train = min(len(self.X), 256)
                _, indices = torch.topk(dists, n_train, largest=False)
                
                train_X = self.X[indices]
                train_Y = self.Y[indices]
                
                # standardize Y locally for GP stability
                y_std = train_Y.std()
                if torch.isnan(y_std) or y_std == 0.0:
                    y_std = 1.0
                train_Y_std = (train_Y - train_Y.mean()) / (y_std + 1e-6)
                
                # fit the surrogate model
                covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim, lengthscale_prior=GammaPrior(3.0, 6.0)))
                gp = SingleTaskGP(train_X, train_Y_std, covar_module=covar)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                try:
                    fit_gpytorch_model(mll)
                except Exception:
                    pass
                
                # Thompson sampling
                with torch.no_grad():
                    sobol = self._next_sobol_engine()
                    pert = sobol.draw(n_candidates).to(dtype=self.dtype, device=self.device)
                    
                    tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
                    tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)
                    
                    cand_X = tr_lb + (tr_ub - tr_lb) * pert
                    
                    posterior = gp.posterior(cand_X)
                    samples = posterior.rsample(sample_shape=torch.Size([1])).squeeze()
                    
                    _, best_idxs = torch.topk(samples, n_samples, largest=False)
                    X_tr = cand_X[best_idxs]
                    
                    X_next = torch.cat([X_next, X_tr], dim=0)

            else:
                # cold start
                sobol = self._next_sobol_engine()
                X_next = torch.cat([X_next, sobol.draw(n_samples).to(dtype=self.dtype, device=self.device)], dim=0)

        return X_next
