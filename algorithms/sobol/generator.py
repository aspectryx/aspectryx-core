"""
generator.py

Author: natelgrw
Last Edited: 02/05/2026

Sobol sequence generator for circuit sizing parameters.
Generates valid design points respecting globalsy constraints and technology rules.
"""

import math
import os
import re
import sys

import numpy as np
from scipy.stats import qmc

# add project root to path so globalsy can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from simulator import globalsy


# ===== Constants ===== #


TECH_CONSTANTS = {
    7:  {'lmin': 1e-8, 'lmax': 3e-8, 'vdd_nom': 0.70},
    10: {'lmin': 1e-8, 'lmax': 3e-8, 'vdd_nom': 0.75},
    14: {'lmin': 1e-8, 'lmax': 3e-8, 'vdd_nom': 0.80},
    16: {'lmin': 1e-8, 'lmax': 3e-8, 'vdd_nom': 0.80},
    20: {'lmin': 1e-8, 'lmax': 2.4e-8, 'vdd_nom': 0.90},
}

# standard e-series base values for component selection
E12 = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
E24 = [
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
    3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1,
]


# ===== Sampling Functions ===== #


def e_series_grid(lb, ub, series='E24'):
    """
    Generate a sorted list of standard E-series values between lb and ub.

    Parameters:
    -----------
    lb (float): Lower bound of the range.
    ub (float): Upper bound of the range.
    series (str): 'E12' or 'E24' for the desired E-series.

    Returns:
    --------
    np.array: Sorted array of E-series values within the specified range.
    """
    bases = E24 if series == 'E24' else E12
    if lb <= 0:
        lb = 1e-15
    min_exp = int(math.floor(math.log10(lb)))
    max_exp = int(math.ceil(math.log10(ub)))
    vals = []
    for e in range(min_exp - 1, max_exp + 1):
        for b in bases:
            v = b * (10 ** e)
            if v >= lb - 1e-30 and v <= ub + 1e-30:
                vals.append(v)
    vals = sorted(set(vals))
    return np.array(vals)


def discrete_linear_grid(lb, ub, step):
    """
    Generates a discrete linear grid from lb to ub with given step.

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
    # ensure inclusion of upper bound
    count = int(math.floor((ub - lb) / step))
    grid = lb + np.arange(0, count + 1) * step
    if grid.size == 0:
        return np.array([lb])
    # ensure ub included
    if grid[-1] < ub - 1e-12:
        grid = np.append(grid, ub)
    return grid


def pick_from_grid_by_u(grid, u):
    """
    Picks a value from grid based on normalized u in [0,1].

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
    # map u in [0,1] to an index in [0, len(grid)-1]
    try:
        u_f = float(u)
    except Exception:
        u_f = 0.0
    if u_f < 0.0:
        u_f = 0.0
    if u_f > 1.0:
        u_f = 1.0
    idx = int(u_f * max(1, (len(grid) - 1)))
    if idx >= len(grid):
        idx = len(grid) - 1
    return float(grid[idx]), idx


def nearest_grid_u(grid, val):
    """
    Given a grid and a value, returns the normalized u in [0,1] corresponding
    to the nearest grid point.

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


def _read_model_l_bounds(fet_num):
    """
    Reads lmin/lmax from local model card files under the `lstp/` folder.
    If parsing fails, falls back to TECH_CONSTANTS or hardcoded defaults.

    Parameters:
    -----------
    fet_num (int): Technology node (e.g., 7, 10, 14, 16, 20).

    Returns:
    --------
    tuple: (lmin, lmax) in meters.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    lstp_dir = os.path.join(project_root, 'lstp')
    candidates = [f"{fet_num}nfet.pm", f"{fet_num}pfet.pm", f"{fet_num}fpm.pm"]
    lmin = None
    lmax = None
    for fn in candidates:
        path = os.path.join(lstp_dir, fn)
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r') as f:
                text = f.read()
            m_min = re.search(r"lmin\s*=\s*([0-9\.eE+\-]+)", text)
            m_max = re.search(r"lmax\s*=\s*([0-9\.eE+\-]+)", text)
            if m_min:
                lmin = float(m_min.group(1))
            if m_max:
                lmax = float(m_max.group(1))
            if lmin is not None and lmax is not None:
                return lmin, lmax
        except Exception:
            continue

    # fall back to hard-coded constants if parsing fails
    t = TECH_CONSTANTS.get(fet_num, None)
    if t is not None:
        lmin = t.get('lmin', None)
        lmax = t.get('lmax', None)
    if lmin is None:
        lmin = 10e-9
    if lmax is None:
        # conservative fallback: 3x lmin
        lmax = max(3 * lmin, lmin + 2e-9)
    return lmin, lmax


# ===== Sobol Generator Class ===== #


class SobolSizingGenerator:

    def __init__(self, sizing_params_list, seed=None, topology=None):
        """
        Initialize Sobol generator state for sizing + environment dimensions.

        Parameters:
        -----------
        sizing_params_list (list): Parameter names to sample for circuit sizing.
        seed (int | None): Optional Sobol scramble seed.
        topology (str | None): Optional topology name used for VCM fallback bounds.
        """
        self.sizing_params = sizing_params_list
        self.seed = seed
        self.tech_nodes = globalsy.testbench_params['Fet_num']
        self.topology = topology

        # fixed testbench parameters
        self.fixed_params = [
            'fet_num', 'vdd', 'vcm', 'tempc', 'cload_val'
        ]

        # total dimensions = fixed params + sizing params
        self.dim = len(self.fixed_params) + len(self.sizing_params)
        self.dim_sizing = len(self.sizing_params)

        # initialize Sobol engine
        self.engine = qmc.Sobol(d=self.dim, scramble=True, seed=seed)

    def _log_sample(self, u, lb, ub):
        """
        Sample logarithmically between lower and upper bounds.

        Parameters:
        -----------
        u (float): Normalized coordinate in [0, 1].
        lb (float): Lower bound (> 0).
        ub (float): Upper bound (> 0).

        Returns:
        --------
        float: Log-space sampled value.
        """
        log_lb = np.log10(lb)
        log_ub = np.log10(ub)
        val_log = log_lb + float(u) * (log_ub - log_lb)
        return 10 ** val_log

    def _pick_tb_val(self, param_key, series_key, u_val):
        """
        Pick a testbench value from E-series grid or log fallback.

        Parameters:
        -----------
        param_key (str): Key under globalsy.testbench_params.
        series_key (str | None): Key under globalsy.sampling_metadata (or explicit series).
        u_val (float): Normalized coordinate in [0, 1].

        Returns:
        --------
        float | None: Sampled value, or None if bounds are unavailable.
        """
        try:
            pinfo = globalsy.testbench_params[param_key]
            lb = pinfo['lb']
            ub = pinfo['ub']
        except Exception:
            return None
        series_name = globalsy.sampling_metadata.get(series_key, None) if isinstance(series_key, str) else series_key
        if series_name is None:
            return self._log_sample(u_val, lb, ub)
        grid = e_series_grid(lb, ub, series=series_name)
        val, _ = pick_from_grid_by_u(grid, u_val)
        if val is None:
            val = self._log_sample(u_val, lb, ub)
        return val

    def _pick_temp_from_u(self, u_val):
        """
        Map normalized value to integer temperature bounds.

        Parameters:
        -----------
        u_val (float): Normalized coordinate in [0, 1].

        Returns:
        --------
        int: Temperature in degrees C, clamped to configured bounds.
        """
        temp_lb = int(round(globalsy.testbench_params['Tempc']['lb']))
        temp_ub = int(round(globalsy.testbench_params['Tempc']['ub']))
        t = int(round(temp_lb + float(u_val) * (temp_ub - temp_lb)))
        return max(temp_lb, min(temp_ub, t))

    def generate(self, n_samples=None, u_samples=None, robust_env=False, start_idx=0):
        """
        Generates n_samples of valid design configurations with environment as context inputs.

        Parameters:
        -----------
        n_samples (int): Number of samples to generate.
        u_samples (np.array): Pre-generated [0,1] samples of shape (n, dim) or (n, dim_sizing).
        robust_env (bool): If True, environment params are randomized per evaluation. If False, use nominal env.
        start_idx (int): Index to resume Sobol sequence from.

        Returns:
        --------
        list[dict]: List of dictionaries with both sizing and environment parameters.
        """
        # expand sizing-only inputs with environment dimensions when needed
        if u_samples is not None:
            n_samples = len(u_samples)
            input_dim = len(u_samples[0]) if isinstance(u_samples, list) else u_samples.shape[1]

            if input_dim == self.dim:
                u_samples_expanded = u_samples
            elif input_dim == self.dim_sizing:
                if robust_env:
                    rng = np.random.default_rng()
                    u_env = rng.uniform(0, 1, size=(n_samples, len(self.fixed_params)))
                    u_samples_expanded = np.hstack([u_samples, u_env])
                else:
                    u_env = np.full((n_samples, len(self.fixed_params)), 0.5)
                    u_samples_expanded = np.hstack([u_samples, u_env])
            else:
                raise ValueError(
                    f"Provided samples dim {input_dim} matches neither full ({self.dim}) nor sizing ({self.dim_sizing})"
                )
        else:
            if n_samples is None:
                raise ValueError("Must provide n_samples if u_samples is None")
            if start_idx > 0:
                self.engine.fast_forward(start_idx)
                u_samples_expanded = self.engine.random(n=n_samples)
            else:
                # prefer 2^m Sobol blocks for balance, then truncate
                m = math.ceil(math.log2(n_samples)) if n_samples > 0 else 0
                if m <= 0:
                    u_samples_expanded = self.engine.random(n=n_samples)
                else:
                    u_full = self.engine.random_base2(m)
                    u_samples_expanded = u_full[:n_samples]
        
        configs = []
        n_env_dims = len(self.fixed_params)
        
        for i in range(n_samples):
            row = u_samples_expanded[i]
            config = {}
            
            # environment parameters
            env_col_idx = 0
            u_fet = row[env_col_idx]
            env_col_idx += 1
            fet_idx = int(u_fet * len(self.tech_nodes))
            fet_idx = min(fet_idx, len(self.tech_nodes) - 1)
            fet_num = self.tech_nodes[fet_idx]
            config['fet_num'] = fet_num
            
            # node-dependent constants
            t_const = TECH_CONSTANTS[fet_num]
            vdd_nom = t_const['vdd_nom']
            
            # VDD
            u_vdd = row[env_col_idx]
            env_col_idx += 1
            vdd_lb = 0.9 * vdd_nom
            vdd_ub = 1.1 * vdd_nom
            abs_step = 0.01
            try:
                step_vdd = abs_step
                vdd_lb = round(vdd_lb / step_vdd) * step_vdd
                vdd_ub = round(vdd_ub / step_vdd) * step_vdd
                if vdd_ub < vdd_lb:
                    vdd_ub = vdd_lb
            except Exception:
                pass
            vdd_grid = discrete_linear_grid(vdd_lb, vdd_ub, step_vdd)
            vdd, _ = pick_from_grid_by_u(vdd_grid, u_vdd)
            config['vdd'] = vdd
            
            # VCM
            u_vcm = row[env_col_idx]
            env_col_idx += 1

            def eval_bound_expr(expr, vdd_nominal):
                """
                Evaluate a bound that may be numeric or an expression string.

                Parameters:
                -----------
                expr (float | str): Bound value or expression.
                vdd_nominal (float): Nominal VDD used by expressions.

                Returns:
                --------
                float | None: Parsed numeric value, or None on failure.
                """
                try:
                    if isinstance(expr, str):
                        return float(eval(expr, {"__builtins__": None}, {"vdd_nominal": vdd_nominal}))
                    return float(expr)
                except Exception:
                    return None

            vcm_lb = None
            vcm_ub = None
            try:
                tb_vcm = globalsy.testbench_params.get('VCM', None)
                if tb_vcm and isinstance(tb_vcm, dict):
                    vcm_lb = eval_bound_expr(tb_vcm.get('lb'), vdd_nom)
                    vcm_ub = eval_bound_expr(tb_vcm.get('ub'), vdd_nom)
            except Exception:
                vcm_lb = None
                vcm_ub = None

            # fallback to topology-aware defaults if explicit bounds are unavailable
            if vcm_lb is None or vcm_ub is None:
                try:
                    vcm_lb_auto, vcm_ub_auto = globalsy.vcm_bounds_for_topology(self.topology, vdd_nom)
                    vcm_lb = vcm_lb if vcm_lb is not None else vcm_lb_auto
                    vcm_ub = vcm_ub if vcm_ub is not None else vcm_ub_auto
                except Exception:
                    if vcm_lb is None:
                        vcm_lb = 0.45 * vdd_nom
                    if vcm_ub is None:
                        vcm_ub = 0.55 * vdd_nom
            if vcm_ub < vcm_lb:
                vcm_ub = vcm_lb
            abs_step = 0.01
            try:
                step_vcm = abs_step
                vcm_lb = round(vcm_lb / step_vcm) * step_vcm
                vcm_ub = round(vcm_ub / step_vcm) * step_vcm
                if vcm_ub < vcm_lb:
                    vcm_ub = vcm_lb
            except Exception:
                pass
            vcm_grid = discrete_linear_grid(vcm_lb, vcm_ub, step_vcm)
            vcm, _ = pick_from_grid_by_u(vcm_grid, u_vcm)
            config['vcm'] = vcm

            # temperature
            u_temp = row[env_col_idx]
            env_col_idx += 1
            tempc = self._pick_temp_from_u(u_temp)
            config['tempc'] = tempc

            # load capacitance
            u_cl = row[env_col_idx]
            env_col_idx += 1
            config['cload_val'] = self._pick_tb_val('Cload_val', 'C_series', u_cl)
            
            # sizing parameters
            col_idx = n_env_dims
            for param in self.sizing_params:
                u_p = row[col_idx]
                col_idx += 1
                                
                if param.startswith('nA'):
                    model_lmin, model_lmax = _read_model_l_bounds(fet_num)
                    if model_lmax <= model_lmin:
                        model_lmax = model_lmin + 2e-9

                    step = 2e-9
                    grid = np.arange(model_lmin, model_lmax + 1e-15, step)
                    if grid.size == 0:
                        grid = np.array([model_lmin])

                    # map u_p in [0,1] to an index in the discrete grid
                    idx = int(u_p * grid.size)
                    if idx >= grid.size:
                        idx = grid.size - 1
                    val = float(grid[idx])
                    config[param] = val
                    
                elif param.startswith('nB'):
                    # prefer globalsy.basic_params if available
                    bp_nfin = globalsy.basic_params.get('Nfin', None)
                    if bp_nfin is not None:
                        try:
                            p_lb = int(bp_nfin.get('lb'))
                            p_ub = int(bp_nfin.get('ub'))
                        except Exception:
                            p_lb, p_ub = 4, 128
                    else:
                        p_lb, p_ub = 4, 128
                    # map u_p in [0,1] to integer range [p_lb, p_ub]
                    span = max(0, p_ub - p_lb)
                    val = int(round(p_lb + u_p * span))
                    if val < p_lb:
                        val = p_lb
                    if val > p_ub:
                        val = p_ub
                    config[param] = int(val)
                    
                elif "bias" in param:
                    vdd = config['vdd']

                    def eval_bound(expr):
                        """
                        Evaluate a bound expression safely using the nominal VDD.

                        Parameters:
                        -----------
                        expr (float | str): Bound value or expression.

                        Returns:
                        --------
                        float | None: Parsed numeric value, or None on failure.
                        """
                        try:
                            if isinstance(expr, str):
                                return float(eval(expr, {"__builtins__": None}, {"vdd_nominal": vdd_nom}))
                            else:
                                return float(expr)
                        except Exception:
                            return None

                    # NMOS vbiasn
                    if param.startswith('vbiasn'):
                        suffix = param[len('vbiasn'):]
                        key = 'vbiasn' + (suffix if suffix else '0')
                        binfo = globalsy.basic_params.get(key, None)
                        if binfo:
                            p_lb = eval_bound(binfo.get('lb'))
                            p_ub = eval_bound(binfo.get('ub'))
                        else:
                            tail_lb_frac, tail_ub_frac = (0.35, 0.50)
                            cas_lb_frac, cas_ub_frac = (0.55, 0.75)
                            if suffix in ('0', '1', ''):
                                p_lb = tail_lb_frac * vdd
                                p_ub = tail_ub_frac * vdd
                            elif suffix == '2':
                                p_lb = cas_lb_frac * vdd
                                p_ub = cas_ub_frac * vdd
                            else:
                                p_lb = 0.1 * vdd
                                p_ub = 0.9 * vdd
                        if p_lb is None or p_ub is None:
                            p_lb = 0.1 * vdd
                            p_ub = 0.9 * vdd
                        step_bias = 0.01
                        try:
                            p_lb = round(float(p_lb) / step_bias) * step_bias
                            p_ub = round(float(p_ub) / step_bias) * step_bias
                            if p_ub < p_lb:
                                p_ub = p_lb
                        except Exception:
                            pass
                        bias_grid = discrete_linear_grid(p_lb, p_ub, step_bias)
                        val, _ = pick_from_grid_by_u(bias_grid, u_p)
                        if val is None:
                            val = p_lb + u_p * (p_ub - p_lb)
                        # snap to canonical step to ensure exact multiples
                        try:
                            if step_bias > 0:
                                val = p_lb + round((val - p_lb) / step_bias) * step_bias
                        except Exception:
                            pass
                        # clamp
                        if val < p_lb:
                            val = p_lb
                        if val > p_ub:
                            val = p_ub
                        config[param] = float(val)

                    # PMOS vbiasp
                    elif param.startswith('vbiasp'):
                        suffix = param[len('vbiasp'):]
                        key = 'vbiasp' + (suffix if suffix else '0')
                        binfo = globalsy.basic_params.get(key, None)
                        if binfo:
                            p_lb = eval_bound(binfo.get('lb'))
                            p_ub = eval_bound(binfo.get('ub'))
                        else:
                            tail_lb_frac, tail_ub_frac = (0.55, 0.75)
                            cas_lb_frac, cas_ub_frac = (0.30, 0.45)
                            if suffix in ('0', '2', ''):
                                p_lb = tail_lb_frac * vdd
                                p_ub = tail_ub_frac * vdd
                            elif suffix == '1':
                                p_lb = cas_lb_frac * vdd
                                p_ub = cas_ub_frac * vdd
                            else:
                                p_lb = 0.1 * vdd
                                p_ub = 0.9 * vdd
                        if p_lb is None or p_ub is None:
                            p_lb = 0.1 * vdd
                            p_ub = 0.9 * vdd
                        step_bias = 0.01
                        try:
                            p_lb = round(float(p_lb) / step_bias) * step_bias
                            p_ub = round(float(p_ub) / step_bias) * step_bias
                            if p_ub < p_lb:
                                p_ub = p_lb
                        except Exception:
                            pass
                        bias_grid = discrete_linear_grid(p_lb, p_ub, step_bias)
                        val, _ = pick_from_grid_by_u(bias_grid, u_p)
                        if val is None:
                            val = p_lb + u_p * (p_ub - p_lb)
                        try:
                            if step_bias > 0:
                                val = p_lb + round((val - p_lb) / step_bias) * step_bias
                        except Exception:
                            pass
                        if val < p_lb:
                            val = p_lb
                        if val > p_ub:
                            val = p_ub
                        config[param] = float(val)

                    else:
                        # generic bias fallback
                        p_lb = 0.1 * config['vdd']
                        p_ub = 0.9 * config['vdd']
                        step_bias = 0.01
                        try:
                            p_lb = round(float(p_lb) / step_bias) * step_bias
                            p_ub = round(float(p_ub) / step_bias) * step_bias
                            if p_ub < p_lb:
                                p_ub = p_lb
                        except Exception:
                            pass
                        bias_grid = discrete_linear_grid(p_lb, p_ub, step_bias)
                        val, _ = pick_from_grid_by_u(bias_grid, u_p)
                        if val is None:
                            val = p_lb + u_p * (p_ub - p_lb)
                        config[param] = val

                elif param.startswith('nC'):
                    # discrete capacitor values (E12 series)
                    p_lb = 100e-15
                    p_ub = 5e-12
                    c_grid = e_series_grid(p_lb, p_ub, series='E12')
                    c_val, _ = pick_from_grid_by_u(c_grid, u_p)
                    if c_val is None:
                        c_val = self._log_sample(u_p, p_lb, p_ub)
                    config[param] = c_val

                elif param.startswith('nR'):
                    # discrete resistor values (E24 series)
                    p_lb = 500
                    p_ub = 500e3
                    r_series = globalsy.sampling_metadata.get('R_series', 'E24')
                    r_grid = e_series_grid(p_lb, p_ub, series=r_series)
                    r_val, _ = pick_from_grid_by_u(r_grid, u_p)
                    if r_val is None:
                        r_val = self._log_sample(u_p, p_lb, p_ub)
                    config[param] = r_val
                    
                else:
                    # fallback for unknown parameters
                    config[param] = u_p

            configs.append(config)
            
        return configs

    def inverse_map(self, df):
        """
        Maps physical parameters (from dataframe) back to Unit Hypercube [0,1]^d.
        Includes both environment and sizing parameters as context inputs.
        Used for initializing TuRBO with existing data ("Sight" mode).
        
        Parameters:
        -----------
        df (pd.DataFrame): Dataframe containing 'in_{param}' columns (or without 'in_' prefix).

        Returns:
        --------
        tuple: (X_tensor, valid_idx)
               X_tensor is shape [N, dim] with normalized values in [0,1].
               valid_idx are row indices from the input dataframe that were kept.
        """
        import torch
        
        fet_num = self.tech_nodes[0]
        t_const = TECH_CONSTANTS[fet_num]
        current_vdd = t_const['vdd_nom']
        
        # combine env and sizing params for full inverse mapping
        full_param_list = self.fixed_params + self.sizing_params
        X_rows = []
        valid_idx = []
        
        for i_row, (_, row) in enumerate(df.iterrows()):
            u_row = []
            
            try:
                for param in full_param_list:
                    col_name = f"in_{param}"
                    if col_name not in row:
                        col_name = param
                    
                    if col_name not in row:
                        if param in self.fixed_params:
                            u = 0.5
                        else:
                            continue
                    else:
                        val = row[col_name]
                        u = 0.5
                        
                        if param == 'fet_num':
                            try:
                                node_idx = self.tech_nodes.index(int(val))
                                u = float(node_idx) / max(1, len(self.tech_nodes) - 1)
                            except Exception:
                                u = 0.5
                        
                        elif param == 'vdd':
                            vdd_nom = current_vdd
                            vdd_lb = 0.9 * vdd_nom
                            vdd_ub = 1.1 * vdd_nom
                            vdd_grid = discrete_linear_grid(
                                round(vdd_lb / 0.01) * 0.01,
                                round(vdd_ub / 0.01) * 0.01,
                                0.01
                            )
                            u = nearest_grid_u(vdd_grid, val)
                        
                        elif param == 'vcm':
                            vcm_lb = 0.45 * current_vdd
                            vcm_ub = 0.55 * current_vdd
                            vcm_grid = discrete_linear_grid(
                                round(vcm_lb / 0.01) * 0.01,
                                round(vcm_ub / 0.01) * 0.01,
                                0.01
                            )
                            u = nearest_grid_u(vcm_grid, val)
                        
                        elif param == 'tempc':
                            temp_lb = int(round(globalsy.testbench_params['Tempc']['lb']))
                            temp_ub = int(round(globalsy.testbench_params['Tempc']['ub']))
                            temp_span = max(1, temp_ub - temp_lb)
                            u = (int(val) - temp_lb) / temp_span
                        
                        elif param == 'cload_val':
                            cl_lb = globalsy.testbench_params['Cload_val']['lb']
                            cl_ub = globalsy.testbench_params['Cload_val']['ub']
                            cl_series = globalsy.sampling_metadata.get('C_series', 'E12')
                            cl_grid = e_series_grid(cl_lb, cl_ub, series=cl_series)
                            if len(cl_grid) > 0:
                                u = nearest_grid_u(cl_grid, val)
                            else:
                                u = (np.log10(val) - np.log10(cl_lb)) / (np.log10(cl_ub) - np.log10(cl_lb))
                        
                        elif param.startswith('nA'):
                            model_lmin, model_lmax = _read_model_l_bounds(fet_num)
                            if model_lmax <= model_lmin:
                                model_lmax = model_lmin + 2e-9
                            grid = np.arange(model_lmin, model_lmax + 1e-15, 2e-9)
                            if grid.size == 0:
                                u = 0.5
                            else:
                                idx_nearest = int(np.argmin(np.abs(grid - float(val))))
                                u = float(idx_nearest) / max(1, (grid.size - 1))
                        
                        elif param.startswith('nB'):
                            p_lb = 1
                            p_ub = 256
                            u = (val - p_lb) / (p_ub - p_lb)
                        
                        elif "bias" in param:
                            if param.startswith('vbiasn'):
                                p_lb = 0.35 * current_vdd
                                p_ub = 0.50 * current_vdd
                            elif param.startswith('vbiasp'):
                                p_lb = 0.30 * current_vdd
                                p_ub = 0.75 * current_vdd
                            else:
                                p_lb = 0.1 * current_vdd
                                p_ub = 0.9 * current_vdd
                            
                            try:
                                p_lb = round(float(p_lb) / 0.01) * 0.01
                                p_ub = round(float(p_ub) / 0.01) * 0.01
                                if p_ub < p_lb:
                                    p_ub = p_lb
                            except Exception:
                                pass
                            bias_grid = discrete_linear_grid(p_lb, p_ub, 0.01)
                            u = nearest_grid_u(bias_grid, val)
                        
                        elif param.startswith('nC'):
                            p_lb = 100e-15
                            p_ub = 5e-12
                            c_grid = e_series_grid(p_lb, p_ub, series='E12')
                            u = nearest_grid_u(c_grid, val)
                        
                        elif param.startswith('nR'):
                            p_lb = 500
                            p_ub = 500e3
                            r_series = globalsy.sampling_metadata.get('R_series', 'E24')
                            r_grid = e_series_grid(p_lb, p_ub, series=r_series)
                            u = nearest_grid_u(r_grid, val)
                        
                        else:
                            u = val
                    
                    u = max(0.0, min(1.0, float(u)))
                    u_row.append(u)
                
                if len(u_row) == len(full_param_list):
                    X_rows.append(u_row)
                    valid_idx.append(i_row)
            
            except Exception:
                continue
        
        return torch.tensor(X_rows, dtype=torch.double), valid_idx


# ===== Main ===== #

if __name__ == "__main__":
    test_params = ['nA1', 'nB1', 'vbiasn0', 'nR1']
    gen = SobolSizingGenerator(test_params)
    samples = gen.generate(5)
    
    import json
    print(json.dumps(samples, indent=2))
