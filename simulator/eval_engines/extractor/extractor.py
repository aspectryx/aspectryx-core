"""
extractor.py

Author: natelgrw
Last Edited: 01/24/2026

Extractor engine for processing simulation results and calculating rewards.
Handles mapping of parameters, specification verification, and JSON logging.
"""

import numpy as np
import os
import json
import uuid
import re
import hashlib
import time
import errno
from collections import OrderedDict
from simulator import globalsy
import importlib

def extract_parameter_names(scs_file):
    """
    Extracts all parameter names from a Spectre netlist file.

    Parameters:
    -----------
    scs_file : str
        Path to the Spectre netlist (.scs) file.
    
    Returns:
    --------
    list
        List of parameter names found in the netlist parameters declaration.
    """
    excluded_params = {
        "dc_offset", "use_tran", "use_sine", "mode_unity", 
        "run_gatekeeper", "run_full_char", "fet_num", "vdd", "vcm", "tempc",
        # Legacy netlist controls still present in some templates; never sample.
        "rfeedback_val", "rsrc_val",
        "sim_tier_1", "sim_tier_2",
        # Control flags present in netlists that should not be sampled
        "swap_polarity"
    }

    with open(scs_file, "r") as file:
        for line in file:
            if line.strip().startswith("parameters"):
                matches = re.findall(r'(\w+)=', line)
                # Filter out excluded parameters (control flags and env vars)
                return [m for m in matches if m not in excluded_params and not m.startswith("sim_")]
    return []

def build_bounds(params_id, vdd=None, vcm=None, tempc=None, fet_num=None):
    """
    Constructs parameter bounds for optimization using `globalsy` parameter dicts.

    Args:
        params_id (iterable): list of parameter names
        vdd, vcm, tempc, fet_num: environment/testbench context values used to
            evaluate any expression-based bounds (e.g., '0.35 * vdd_nominal').

                    existing_main_specs = existing.get('specs', main_specs)
                    # Ensure returned existing record always contains a 'specs' field
                    if 'specs' not in existing:
                        try:
                            existing['specs'] = existing_main_specs
                        except Exception:
                            existing['specs'] = {}
                    return reward1, existing_main_specs, existing
    """
    full_lb, full_ub = [], []
    opt_lb, opt_ub, opt_params = [], [], []
    
    for pname in params_id:
        
        # --- Basic Parameters ---
        if pname.startswith("nA"): # Length (L) - use explicit basic_params if available
            bp = globalsy.basic_params.get('L', None)
            if bp is not None:
                try:
                    low = float(bp.get('lb'))
                    high = float(bp.get('ub'))
                except Exception:
                    low, high = (10e-9, 100e-9)
            else:
                low, high = (10e-9, 100e-9)
        elif pname.startswith("nB"): # Nfin
            bp = globalsy.basic_params.get('Nfin', None)
            if bp is not None:
                try:
                    low = int(bp.get('lb'))
                    high = int(bp.get('ub'))
                except Exception:
                    low, high = (1, 256)
            else:
                low, high = (1, 256)
        elif "biasn" in pname or pname.startswith('vbiasn'):
            # Try explicit per-bias basic_params entries first (vbiasn0, vbiasn1, ...)
            key = pname if pname in globalsy.basic_params else None
            if key is None:
                # fallback to generic vbiasn0/nominal fractions
                # use vbiasn0 if present
                bp0 = globalsy.basic_params.get('vbiasn0', None)
                if bp0 is not None:
                    try:
                        low = float(eval(bp0.get('lb'), {"__builtins__":None}, {"vdd_nominal": vdd}))
                        high = float(eval(bp0.get('ub'), {"__builtins__":None}, {"vdd_nominal": vdd}))
                    except Exception:
                        low, high = (0.35 * vdd, 0.65 * vdd)
                else:
                    low, high = (0.35 * vdd, 0.65 * vdd)
            else:
                info = globalsy.basic_params.get(key)
                try:
                    low = float(eval(info.get('lb'), {"__builtins__":None}, {"vdd_nominal": vdd}))
                    high = float(eval(info.get('ub'), {"__builtins__":None}, {"vdd_nominal": vdd}))
                except Exception:
                    low, high = (0.35 * vdd, 0.65 * vdd)
        elif "biasp" in pname or pname.startswith('vbiasp'):
            key = pname if pname in globalsy.basic_params else None
            if key is None:
                bp0 = globalsy.basic_params.get('vbiasp0', None)
                if bp0 is not None:
                    try:
                        low = float(eval(bp0.get('lb'), {"__builtins__":None}, {"vdd_nominal": vdd}))
                        high = float(eval(bp0.get('ub'), {"__builtins__":None}, {"vdd_nominal": vdd}))
                    except Exception:
                        low, high = (0.15 * vdd, 0.50 * vdd)
                else:
                    low, high = (0.15 * vdd, 0.50 * vdd)
            else:
                info = globalsy.basic_params.get(key)
                try:
                    low = float(eval(info.get('lb'), {"__builtins__":None}, {"vdd_nominal": vdd}))
                    high = float(eval(info.get('ub'), {"__builtins__":None}, {"vdd_nominal": vdd}))
                except Exception:
                    low, high = (0.15 * vdd, 0.50 * vdd)
        elif pname.startswith("nC"): # C_internal
            bp = globalsy.basic_params.get('C_internal', None)
            if bp is not None:
                try:
                    low = float(bp.get('lb'))
                    high = float(bp.get('ub'))
                except Exception:
                    low, high = (100e-15, 5e-12)
            else:
                low, high = (100e-15, 5e-12)
        elif pname.startswith("nR"): # R_internal
            bp = globalsy.basic_params.get('R_internal', None)
            if bp is not None:
                try:
                    low = float(bp.get('lb'))
                    high = float(bp.get('ub'))
                except Exception:
                    low, high = (500, 100e3)
            else:
                low, high = (500, 100e3)
            
        # --- Testbench Parameters ---
        elif pname == "cload_val":
            tb = globalsy.testbench_params.get('Cload_val', {})
            low = float(tb.get('lb', 10e-15))
            high = float(tb.get('ub', 5e-12))
            
        # --- Fixed/Environment Variables (pass-through) ---
        elif pname.startswith("vdd"):
            low, high = (vdd, vdd)
        elif pname.startswith("vcm"):
            low, high = (vcm, vcm)
        elif pname.startswith("tempc"):
            low, high = (tempc, tempc)
        elif pname.startswith("fet_num"):
            low, high = (fet_num, fet_num)
        else:
            print(f"   [!] Parameter {pname} using generic fallback bounds.")
            low, high = (-1e9, 1e9)
        
        # Add to full bounds (for config_env)
        full_lb.append(low)
        full_ub.append(high)
        
        # Only add to optimization bounds if not fixed
        if low < high:
            opt_lb.append(low)
            opt_ub.append(high)
            opt_params.append(pname)

    return np.array(full_lb), np.array(full_ub), np.array(opt_lb), np.array(opt_ub), opt_params

def classify_opamp_type(file_path):
    """
    Classifies op-amp topology based on netlist structure.

    Parameters:
    -----------
    file_path : str
        Path to the Spectre netlist file.
    
    Returns:
    --------
    str
        Either "differential" or "single_ended" based on presence of Voutn.
    """
    with open(file_path, "r") as file:
        for line in file:
            if "Voutn" in line:
                return "differential"
        else:
            return "single_ended"


class Extractor:
    """
    Objective function wrapper for circuit simulation.

    Encapsulates netlist simulation and performance evaluation for use with
    Sobol sampling. Converts raw simulation results to reward (penalty) scores for logging.
    
    Initialization Parameters:
    --------------------------
    dim : int
        Dimension of parameter space.
    params_id : list
        Parameter names.
    specs_id : list
        Specification names (from optimization target list - often empty now).
    specs_ideal : list
        Target specification values.
    vcm : float
        Common mode voltage.
    vdd : float
        Supply voltage.
    tempc : float
        Temperature in Celsius.
    ub : numpy array
        Upper bounds for parameters.
    lb : numpy array
        Lower bounds for parameters.
    config : dict
        Configuration dictionary for the testbench and target specs.
    fet_num : int
        Transistor size in nm.
    """

    def __init__(self, dim, opt_params, params_id, specs_id, specs_ideal, specs_weights, sim_flags, vcm, vdd, tempc, ub, lb, config, fet_num, results_dir, netlist_name, size_map, rfeedback=1e7, rsrc=50, cload=1e-12, mode="test_drive", sim_mode="complete"):
        
        self.mode = mode # "test_drive" or "mass_collection"
        self.sim_mode = sim_mode # "efficient" or "complete"
        self.dim = dim
        self.opt_params = opt_params        # Parameters being varied
        self.params_id = params_id          # All parameters for netlist
        self.specs_id = specs_id
        self.specs_ideal = specs_ideal
        self.specs_weights = specs_weights
        self.sim_flags = sim_flags
        self.vcm = vcm
        self.vdd = vdd
        self.tempc = tempc
        self.ub = ub
        self.lb = lb
        # configuration dict for measurement/testbench (preferred)
        self.config = config
        self.fet_num = fet_num
        self.results_dir = results_dir
        self.netlist_name = netlist_name
        self.size_map = size_map
        
        # Passive component defaults
        self.rfeedback = rfeedback
        self.rsrc = rsrc
        self.cload = cload

    def lookup(self, spec, goal_spec):
        """
        Calculate normalized performance deviation from target specifications.
        Kept for backward compatibility if we want to calculate rewards for analysis,
        even if not used for active optimization loop steering.
        """
        norm_specs = []
        for s, g in zip(spec, goal_spec):
            # Check for range measurement (tuple/list of length 2)
            if isinstance(s, (list, tuple, np.ndarray)) and len(s) == 2:
                 # Check if target is also a range
                 if isinstance(g, (list, tuple, np.ndarray)) and len(g) == 2:
                     # Containment Optimization
                     val_min = (g[0] - s[0]) / (abs(g[0]) + abs(s[0]) + 1e-9)
                     val_max = (s[1] - g[1]) / (abs(g[1]) + abs(s[1]) + 1e-9)
                     norm_specs.append(min(val_min, val_max))
                 else:
                     # Width Optimization
                     width_s = abs(s[1] - s[0])
                     width_g = float(g)
                     val = (width_s - width_g) / (abs(width_g) + abs(width_s) + 1e-9)
                     norm_specs.append(val)
            else:
                 # Scalar Optimization
                 s_val = float(s) if s is not None else 0.0
                 g_val = float(g)
                 val = (s_val - g_val) / (abs(g_val) + abs(s_val) + 1e-9)
                 norm_specs.append(val)

        return np.array(norm_specs)

    def reward(self, spec, goal_spec, specs_id, specs_weights):
        """
        Calculate the penalty-based reward (cost) from specifications.
        """
        if not specs_id:
            return 0.0
            
        rel_specs = self.lookup(spec, goal_spec)
        reward = 0
        
        # Define Optimization Direction (Maximize vs Minimize)
        minimize_specs = ["power", "integrated_noise", "settling_time", "settle_time", "vos", "estimated_area", "area"]
        
        for i, rel_spec in enumerate(rel_specs):
            s_name = specs_id[i]
            s_weight = specs_weights[i]
            
            if s_name in minimize_specs:
                if rel_spec > 0: # Bad
                    reward += s_weight * np.abs(rel_spec)
            else: # Maximize (gain, ugbw, pm, cmrr, etc.)
                if rel_spec < 0: # Bad
                    reward += s_weight * np.abs(rel_spec)

        return reward

    def _get_default_bad_specs(self, params):
        # Calculate estimated area in µm² using the same first-order
        # passive assumptions as the measurement managers.
        area_m2 = 0.0
        node_map = {
            7:  {'cpp': 54e-9,  'pitch': 22e-9, 'cap_density': 25e-15 / 1e-12, 'res_coeff': (50e-9)**2  / 600},
            10: {'cpp': 66e-9,  'pitch': 28e-9, 'cap_density': 18e-15 / 1e-12, 'res_coeff': (80e-9)**2  / 400},
            14: {'cpp': 78e-9,  'pitch': 32e-9, 'cap_density': 12e-15 / 1e-12, 'res_coeff': (100e-9)**2 / 250},
            16: {'cpp': 90e-9,  'pitch': 42e-9, 'cap_density': 10e-15 / 1e-12, 'res_coeff': (130e-9)**2 / 200},
            20: {'cpp':110e-9,  'pitch': 60e-9, 'cap_density':  8e-15 / 1e-12, 'res_coeff': (200e-9)**2 / 150},
        }
        try:
            tech_node = int(params.get('fet_num', 10))
        except Exception:
            tech_node = 10
        tech = node_map.get(tech_node, node_map[10])

        for key, val in params.items():
            if key.startswith('nA'):
                suffix = key[2:]
                width_key = 'nB' + suffix
                if width_key in params:
                    try:
                        L_eff = max(float(val), tech['cpp'])
                        width = float(params[width_key]) * tech['pitch']
                        area_m2 += L_eff * width
                    except Exception:
                        pass
        area_m2 *= 3.0

        cap_density = tech['cap_density']
        res_area_coeff = tech['res_coeff']
        for key, val in params.items():
            if key.startswith('nC') and val > 0:
                area_m2 += (val / cap_density)
            elif key.startswith('nR') and val > 0:
                area_m2 += (val * res_area_coeff)
        
        # Convert m² → µm² for readable/stable training values
        estimated_area = area_m2 * 1e12
                
        # Determine MM instance names by scanning both the generated netlist and
        # the original topology file in `topologies_mtlcad` (take the union).
        mm_names = []
        try:
            mm_idxs = set()
            # 1) Scan generated netlist for any MM#:param occurrences. We search
            # the whole netlist because the `save` line can be wrapped/continued
            # across multiple lines in generated files.
            try:
                with open(self.netlist_name, 'r') as nf:
                    lines = nf.readlines()
                nettext = ''.join(lines)
                mm_idxs.update(re.findall(r'\bMM(\d+):', nettext))
            except Exception:
                pass

            # 2) Also scan the matching topology file (preferred canonical source)
            try:
                # Support multiple topology directories (topologies*, e.g. topologies_mtlcad, topologies_mtlcad2)
                cwd = os.getcwd()
                # If caller specified a topology directory in params, prefer it
                preferred = []
                for key in ('topology_dir', 'topologies_dir', 'topo_path'):
                    try:
                        val = params.get(key)
                    except Exception:
                        val = None
                    if val and os.path.isdir(val):
                        preferred.append(val)

                topo_dirs = [os.path.join(cwd, d) for d in os.listdir(cwd) if d.startswith('topologies') and os.path.isdir(os.path.join(cwd, d))]
                # Prepend preferred dirs, avoid duplicates
                for p in reversed(preferred):
                    if p not in topo_dirs:
                        topo_dirs.insert(0, p)
                if preferred:
                    try:
                        print(f"   [debug] using preferred topology dir: {preferred[0]}")
                    except Exception:
                        pass
                # Attempt to extract topology identifier from the generated netlist
                topo_name = None
                try:
                    with open(self.netlist_name, 'r') as nf2:
                        for ln in nf2:
                            m = re.match(r"^\*---\s+(\S+)", ln.strip())
                            if m:
                                # capture the token after '*---'
                                topo_name = m.group(1)
                                # ignore generic marker lines like 'TOPOLOGY' or 'BETA'
                                if topo_name.upper() in ('TOPOLOGY', 'SAVE', 'BETA'):
                                    topo_name = None
                                    continue
                                break
                except Exception:
                    topo_name = None

                if topo_name:
                    # Look for the topology file in any topo_dirs
                    for td in topo_dirs:
                        topo_path = os.path.join(td, topo_name) if os.path.isdir(td) else os.path.join(td, topo_name + '.scs')
                        if not os.path.isfile(topo_path):
                            # try with .scs suffix and basename match
                            topo_path = os.path.join(td, topo_name + '.scs')
                        if os.path.isfile(topo_path):
                            with open(topo_path, 'r') as tf:
                                tlines = tf.readlines()
                            tstripped = [ln.strip() for ln in tlines if ln.strip()]
                            tline = tstripped[-2] if len(tstripped) >= 2 else (tstripped[0] if len(tstripped) == 1 else '')
                            mm_idxs.update(re.findall(r'\bMM(\d+):', tline))
                            if not mm_idxs:
                                mm_idxs.update(re.findall(r'\bMM(\d+):', ''.join(tlines)))
                            break
                else:
                    # No clear topology name. Attempt to choose topology file by
                    # matching topology basename against the generated netlist filename
                    # (many generated netlists include the topology name as a prefix).
                    try:
                        netbase = os.path.basename(self.netlist_name)
                        chosen = None
                        for td in topo_dirs:
                            for fname in os.listdir(td):
                                if not fname.endswith('.scs'):
                                    continue
                                base = os.path.splitext(fname)[0]
                                if base in netbase:
                                    chosen = os.path.join(td, fname)
                                    break
                            if chosen:
                                break
                        if chosen and os.path.isfile(chosen):
                            with open(chosen, 'r') as tf:
                                tlines = tf.readlines()
                            tstripped = [ln.strip() for ln in tlines if ln.strip()]
                            tline = tstripped[-2] if len(tstripped) >= 2 else (tstripped[0] if len(tstripped) == 1 else '')
                            mm_idxs.update(re.findall(r'\bMM(\d+):', tline))
                            if not mm_idxs:
                                mm_idxs.update(re.findall(r'\bMM(\d+):', ''.join(tlines)))
                    except Exception:
                        pass
            except Exception:
                pass
            # Debug: log detected MM indices for troubleshooting (concise)
            try:
                detected = sorted({int(i) for i in mm_idxs})
                if detected:
                    print(f"   [debug] detected MM indices: {detected} (from netlist{'+' if topo_name else ''}topology)")
                else:
                    print("   [debug] no MM indices detected in netlist/topology scan")
            except Exception:
                pass

            # Build sorted MM names list (if any found). Do NOT fall back to nB count.
            mm_names = [f'MM{int(i)}' for i in sorted({int(i) for i in mm_idxs})]
        except Exception:
            mm_names = []

        bad_specs = {
            'gain_ol': -1000.0,
            'ugbw': 1.0,
            'pm': -180.0,
            'estimated_area': estimated_area,
            'power': 1.0,
            'vos': 10.0,
            'cmrr': -1000.0,
            'psrr': -1000.0,
            'thd': 1000.0,
            'output_voltage_swing': 0.0,
            'integrated_noise': 10.0,
            'slew_rate': 1e-6,
            'settle_time': 1e12,
            'zregion_of_operation_MM': {mm: 0.0 for mm in mm_names},
            'zzids_MM': {mm: 0.0 for mm in mm_names},
            'zzvds_MM': {mm: 0.0 for mm in mm_names},
            'zzvgs_MM': {mm: 0.0 for mm in mm_names}
        }
        return bad_specs

    def _return_bad_specs(self, params):
        bad_specs = self._get_default_bad_specs(params)
        
        globalsy.counterrrr += 1
        if self.mode == "mass_collection":
            # Build per-transistor op_points from the z-prefixed entries in bad_specs
            op_points = {}
            key_map = {'zregion_of_operation_MM': 'region_of_operation',
                       'zzids_MM': 'ids',
                       'zzvds_MM': 'vds', 'zzvgs_MM': 'vgs'}
            for zkey, param_name in key_map.items():
                for comp, val in bad_specs.get(zkey, {}).items():
                    if comp not in op_points:
                        op_points[comp] = {}
                    op_points[comp][param_name] = val
            sim_result = {
                "id": f"bad_{globalsy.counterrrr}",
                "specs": {k: v for k, v in bad_specs.items() if not k.startswith('z')},
                "operating_points": op_points,
            }
            return 0.0, bad_specs, sim_result
        return 0.0, bad_specs

    def _build_sizing_env_bias(self, full_params, final_sizing, iter_num=None):
        """
        Construct sizing, env, and bias dictionaries used for sim_key generation
        and result records. Returns a dict with keys: topology_id, netlist,
        sim_id (if provided), parameters, bias, env.
        """
        env_dict = {
            "vdd": self.vdd,
            "vcm": self.vcm,
            "tempc": self.tempc,
            "fet_num": self.fet_num
        }

        bias_dict = {}
        for k, v in full_params.items():
            if k not in env_dict:
                if k.lower().startswith('vbias') or k.lower().startswith('ibias') or 'bias' in k.lower():
                    bias_dict[k] = v

        sizing_data = {
            "topology_id": self.netlist_name,
            "netlist": self.netlist_name,
            "parameters": final_sizing,
            "sizing": final_sizing,
            "bias": bias_dict,
            "env": env_dict
        }
        if iter_num is not None:
            sizing_data['sim_id'] = iter_num
        return sizing_data

    def sim_key_for_params(self, full_params, final_sizing=None):
        """
        Compute the deterministic sim_key (SHA1 hex) for a given parametrization
        without executing the simulation. `final_sizing` can be provided to
        avoid recomputing structured sizing; otherwise the method will attempt
        to build it similarly to the run path.
        """
        try:
            # Reconstruct final_sizing if not provided (best-effort)
            if final_sizing is None:
                # Attempt to mirror the structured sizing logic
                structured_sizing = {}
                if self.size_map:
                    for comp, params in self.size_map.items():
                        comp_props = {}
                        for prop, val_expr in params.items():
                            clean_var = val_expr.replace('{{', '').replace('}}', '').strip()
                            if clean_var in full_params:
                                comp_props[prop] = full_params[clean_var]
                            else:
                                try:
                                    comp_props[prop] = float(clean_var)
                                except Exception:
                                    comp_props[prop] = clean_var
                        structured_sizing[comp] = comp_props
                final_sizing = structured_sizing if structured_sizing else full_params

            sizing_data = {
                "netlist": self.netlist_name,
                "sizing": final_sizing,
                "bias": {k: v for k, v in full_params.items() if (k.lower().startswith('vbias') or k.lower().startswith('ibias') or 'bias' in k.lower())},
                "env": {"vdd": full_params.get('vdd', self.vdd), "vcm": full_params.get('vcm', self.vcm), "tempc": full_params.get('tempc', self.tempc), "fet_num": full_params.get('fet_num', self.fet_num)}
            }
            key_source = json.dumps(sizing_data, sort_keys=True, separators=(',', ':'))
            return hashlib.sha1(key_source.encode('utf-8')).hexdigest()
        except Exception:
            return str(uuid.uuid4())

    def __call__(self, x, sim_id=None):
        """
        Main execution function for Sobol sampling loop.

        Converts parameter vector to circuit netlist, runs Spectre simulation,
        logs evaluation metadata.

        Parameters:
        -----------
        x : numpy array or dict
            Parameter vector of length self.dim.
        sim_id : int, optional
            Explicit simulation ID. If None, uses globalsy.counterrrr.
        
        Returns:
        --------
        float
            Dummy reward score (0.0).
        """
        # Handle Dictionary Input (Direct from Generator) - PREFERRED PATH
        if isinstance(x, dict):
            full_params = x.copy()
            
            # Ensure discrete params are int
            for pname, val in full_params.items():
                if pname.startswith("nB"):
                    full_params[pname] = int(round(val))
                    
        else:
            # Legacy/Vector Input Path
            assert len(x) == self.dim
            assert x.ndim == 1

            sample = x.copy()
            # ... (rest of legacy mapping logic if ever needed) ...
            # For now, let's just implement the vector mapping assuming legacy usage
            
            for i, param in enumerate(self.opt_params):
                 if param.startswith('nB'): 
                    sample[i] = round(sample[i])

            full_params = {}
            opt_idx = 0
            
            for pname in self.params_id:
                if pname in self.opt_params:
                    full_params[pname] = sample[opt_idx]
                    opt_idx += 1
            
            # Helper for leftovers
            while opt_idx < len(self.opt_params):
                pname = self.opt_params[opt_idx]
                if pname not in full_params:
                     full_params[pname] = sample[opt_idx]
                opt_idx += 1

        # Common Logic: creation of cadence simulation environment
        # Build the measurement environment from the provided config dict.
        sim_env = None
        try:
            if not isinstance(self.config, dict):
                raise RuntimeError("Extractor requires a configuration dict, not a file path.")
            tb_module_name = self.config['measurement']['testbenches']['ac_dc']['tb_module']
            meas_module = importlib.import_module(tb_module_name)
            OpampMeasMan = meas_module.OpampMeasMan
            # Pass the config dict directly to the measurement manager
            sim_env = OpampMeasMan(self.config)
        except Exception as e:
            # Fail fast: configuration or measurement manager instantiation failed.
            raise RuntimeError(f"Failed to build sim_env from config: {e}")

        # Print iteration steps
        if sim_id is not None:
             iter_num = sim_id
        else:
             iter_num = globalsy.counterrrr + 1 
        
        print(f"\n {'='*60}")
        print(f" Iteration {iter_num}")
        print(f" {'='*60}")
        print(f"   [+] Running simulation...")

        # Now Update self state from full_params if present (Sobol mode)
        # OR use current self state if not present (Fixed mode)
        if "fet_num" in full_params: self.fet_num = full_params["fet_num"]
        else: full_params["fet_num"] = self.fet_num
            
        if "vdd" in full_params: self.vdd = full_params["vdd"]
        else: full_params["vdd"] = self.vdd
            
        if "vcm" in full_params: self.vcm = full_params["vcm"]
        else: full_params["vcm"] = self.vcm
            
        if "tempc" in full_params: self.tempc = full_params["tempc"]
        else: full_params["tempc"] = self.tempc

        # Ensure passive defaults if not swept/provided by generator
        if "vcm" in full_params: self.vcm = full_params["vcm"]
        if "tempc" in full_params: self.tempc = full_params["tempc"]

        # Ensure passive defaults if not swept/provided by generator
        if "cload_val" not in full_params: full_params["cload_val"] = self.cload
        
        # Inject Simulation Control Flags dependent on user selection
        # run_gatekeeper typically enables basic startup checks (DC, STB)
        full_params["run_gatekeeper"] = 1 # Always run primary gatekeeper

        # Track simulation outcome for dataset metadata
        # 0 = dcOp/Spectre failure, 1 = Tier1 only (bad ops/AC), 2 = full char (passed gatekeeper)
        sim_status = 0

        # --- ALWAYS run Tier-1 (gatekeeper) first ---
        full_params["run_full_char"] = 0
        param_val = [OrderedDict(full_params)]
        sim_env.ver_specs['results_dir'] = self.results_dir
        eval_result = sim_env.evaluate(param_val)

        if not eval_result or len(eval_result) == 0:
            eval_result = [(None, {}, 1)]

        # If the underlying simulation returned a non-zero info code, treat as DCOP failure
        try:
            sim_info = int(eval_result[0][2]) if len(eval_result[0]) > 2 else 0
        except Exception:
            sim_info = 0

        dcop_failed = False
        if sim_info != 0:
            print(f"   [!] Underlying simulation reported non-zero info={sim_info} — DCOP failure.")
            # Force sim_status = 0 and emit default bad specs
            sim_status = 0
            bad_specs_dict = self._get_default_bad_specs(full_params)
            eval_result[0] = (eval_result[0][0], bad_specs_dict, sim_info)
            dcop_failed = True

        specs = eval_result[0][1]

        if not dcop_failed:
            # Diagnostic: count how many spec values are non-None
            real_vals = sum(1 for v in specs.values() if v is not None)
            print(f"   [GATEKEEPER] Tier 1: {real_vals}/{len(specs)} specs extracted")
            if real_vals == 0:
                print(f"   [!] Tier 1 returned NO data — Spectre likely failed.")
                # dcOp failed outright
                sim_status = 0
                # ensure bad specs are created below and will include all MM entries
            else:
                # we have at least some Tier-1 data; do not assume gatekeeper pass yet
                # sim_status will be set to 1 (gatekeeper fail) or 2 (gatekeeper pass)
                sim_status = 0
        else:
            # DCOP failed earlier; skip gatekeeper processing and keep sim_status=0
            print("   [i] Skipping gatekeeper checks due to DCOP failure.")

        # Gatekeeper and Tier-2 decision logic is skipped if DCOP already failed
        if not dcop_failed:
            # Build list of MM instance names using the same detection logic as
            # `_get_default_bad_specs` so we always enumerate all topology MMs.
            mm_names = []
            try:
                defaults = self._get_default_bad_specs(full_params)
                mm_names = list(defaults.get('zregion_of_operation_MM', {}).keys())
            except Exception:
                mm_names = []

            region_MM = specs.get('zregion_of_operation_MM', {})
            ids_MM = specs.get('zzids_MM', {})

            ops_good = True
            # require that every MM is present and shows non-zero operating values
            if not mm_names or not region_MM:
                ops_good = False
            else:
                for mm in mm_names:
                    region = region_MM.get(mm)
                    ids = ids_MM.get(mm)
                    # Missing entries or invalid region
                    if region is None or region == 0.0 or region == 4.0:
                        ops_good = False
                        break
                    # Require positive gm and ids for transistor to be considered operating
                    try:
                        if (ids is None) or float(ids) <= 0.0:
                            ops_good = False
                            break
                    except Exception:
                        ops_good = False
                        break

            if not ops_good:
                print(f"   [-] Transistor ops bad — skipping Tier 2.")

            # Check AC performance using strict gatekeeper thresholds requested by user
            gain_ol = specs.get('gain_ol')
            pm = specs.get('pm')
            ugbw = specs.get('ugbw')

            gain_ol = gain_ol if gain_ol is not None else -1000.0
            pm = pm if pm is not None else -180.0
            ugbw = ugbw if ugbw is not None else 0.0

            # User requested: require gain, pm, ugbw > 1 to pass gatekeeper
            if ops_good and not (gain_ol > 1.0 and pm > 1.0 and ugbw > 1.0):
                print(f"   [-] Gatekeeper AC thresholds failed (gain={gain_ol}, pm={pm}, ugbw={ugbw}) — skipping Tier 2.")
                ops_good = False

            if ops_good:
                # Gatekeeper passed: run full characterization
                sim_status = 2
                full_params["run_full_char"] = 1
                param_val = [OrderedDict(full_params)]
                sim_env.ver_specs['results_dir'] = self.results_dir # Inject dir for meas man
                eval_result = sim_env.evaluate(param_val)
            else:
                # Gatekeeper failed but Tier-1 produced some data: mark as Tier-1-only (sim_status=1)
                if real_vals > 0:
                    sim_status = 1
                else:
                    sim_status = 0
                # Merge Tier 1 specs with default bad specs for Tier 2 (or for failure)
                bad_specs_dict = self._get_default_bad_specs(full_params)
                for k, v in specs.items():
                    if v is None:
                        continue
                    # If this is a z-prefixed per-MM dict, merge per-transistor entries
                    if k.startswith('z') and isinstance(v, dict) and isinstance(bad_specs_dict.get(k), dict):
                        for comp, val in v.items():
                            bad_specs_dict[k][comp] = val
                    else:
                        bad_specs_dict[k] = v
                eval_result[0] = (eval_result[0][0], bad_specs_dict)
        else:
            # dcop_failed True: keep sim_status=0 and eval_result as set above
            pass
        
        # Error handling: check if evaluation returned valid results
        if not eval_result or len(eval_result) == 0:
            print(f"   [!] Simulation returned empty results.")
            bad_specs_dict = self._get_default_bad_specs(full_params)
            eval_result = [(None, bad_specs_dict)]
        
        cur_specs = OrderedDict(sorted(eval_result[0][1].items(), key=lambda k:k[0]))

        # --- Print formatted spec results ---
        # `valid` flag deprecated: use `sim_status` (2 == full characterization pass)
        is_valid = (sim_status == 2)
        status_tag = "PASS" if is_valid else "FAIL"
        print(f"   [+] Measurement complete — [{status_tag}]")
        # Print sim_status for clarity in terminal output
        sim_status_label = {0: 'DCOP_FAIL', 1: 'TIER1_ONLY', 2: 'FULL_CHAR'}.get(sim_status, 'UNKNOWN')
        print(f"   [i] sim_status: {sim_status} ({sim_status_label})")
        print(f"   {'-'*50}")

        # Separate main specs from op-point dicts
        op_keys = [k for k in cur_specs if k.startswith('z')]
        main_keys = [k for k in cur_specs if not k.startswith('z') and k != 'valid']

        # Print main specs in a compact table
        if main_keys:
            col_w = max(len(k) for k in main_keys) + 2
            for k in main_keys:
                v = cur_specs[k]
                if isinstance(v, float):
                    print(f"   {k:<{col_w}} {v:>14.4g}")
                else:
                    print(f"   {k:<{col_w}} {str(v):>14}")

        # Print operating points compactly
        if op_keys:
            # Collect transistor names from first op dict
            sample_dict = cur_specs.get(op_keys[0], {})
            if isinstance(sample_dict, dict) and sample_dict:
                mm_names = sorted(sample_dict.keys())
                # Header
                # Build human labels (remove leading 'z' and trailing '_MM')
                param_labels = []
                for ok in op_keys:
                    label = ok.lstrip('z')
                    if label.endswith('_MM'):
                        label = label[:-3]
                    param_labels.append(label)

                # Split capacitance params into a separate table for readability
                cap_params = {'cgg', 'cgs', 'cdd', 'cgd', 'css'}
                core_labels = [lbl for lbl in param_labels if lbl not in cap_params]
                cap_labels = [lbl for lbl in param_labels if lbl in cap_params]

                # Mapping from label -> original op_key (z-prefixed)
                label_to_key = {}
                for ok in op_keys:
                    lab = ok.lstrip('z')
                    if lab.endswith('_MM'):
                        lab = lab[:-3]
                    label_to_key[lab] = ok

                # Print core parameters table
                header_core = f"   {'FET':<6}" + "".join(f"{lbl:>12}" for lbl in core_labels)
                print(f"\n   Operating Points:")
                print(header_core)
                print(f"   {'-'*(6 + 12 * len(core_labels))}")
                for mm in mm_names:
                    row = f"   {mm:<6}"
                    for lbl in core_labels:
                        ok = label_to_key.get(lbl)
                        val = cur_specs.get(ok, {}).get(mm, 0.0) if ok else 0.0
                        if isinstance(val, float):
                            row += f"{val:>12.4g}"
                        else:
                            row += f"{str(val):>12}"
                    print(row)

                # Print capacitances table (if any)
                if cap_labels:
                    header_cap = f"   {'FET':<6}" + "".join(f"{lbl:>12}" for lbl in cap_labels)
                    print(f"\n   Capacitances:")
                    print(header_cap)
                    print(f"   {'-'*(6 + 12 * len(cap_labels))}")
                    for mm in mm_names:
                        row = f"   {mm:<6}"
                        for lbl in cap_labels:
                            ok = label_to_key.get(lbl)
                            val = cur_specs.get(ok, {}).get(mm, 0.0) if ok else 0.0
                            if isinstance(val, float):
                                row += f"{val:>12.4g}"
                            else:
                                row += f"{str(val):>12}"
                        print(row)
        print(f"   {'-'*50}")
        
        # Calculate Reward (Optional/Dummy if specs_id empty)
        reward_input = []
        if self.specs_id:
            for s_name in self.specs_id:
                val = cur_specs.get(s_name)
                if val is None:
                    if s_name in ["power", "integrated_noise", "settling_time", "vos"]:
                        val = 1e9 # High value bad
                    else:
                        val = -1e9 # Low value bad
                reward_input.append(val)
            
            reward1 = self.reward(reward_input, self.specs_ideal, self.specs_id, self.specs_weights)
        else:
            reward1 = 0.0

        # Build structured sizing from map
        structured_sizing = {}
        if self.size_map:
            for comp, params in self.size_map.items():
                comp_props = {}
                for prop, val_expr in params.items():
                    # Handle Jinja variables {{var}} or direct var
                    clean_var = val_expr.replace('{{', '').replace('}}', '').strip()
                    if clean_var in full_params:
                        comp_props[prop] = full_params[clean_var]
                    else:
                        # Try literal
                        try:
                            comp_props[prop] = float(clean_var)
                        except ValueError:
                            comp_props[prop] = clean_var
                structured_sizing[comp] = comp_props
        
        final_sizing = structured_sizing if structured_sizing else full_params

        # Build sizing, env, and bias dictionaries
        sizing_data = self._build_sizing_env_bias(full_params, final_sizing, iter_num)
        
        # Cleanup full_params vs bias - remove bias from sizing section if present
        for k in ["vdd", "vcm", "tempc", "fet_num"]:
            if k in sizing_data["sizing"]:
                del sizing_data["sizing"][k]

        # --- Custom Cleanup per User Request ---
        # 1. Remove unwanted components (Rshunt, Runity, Rsw) and individual Rsrc
        comps_to_remove = []
        for comp in sizing_data["sizing"]:
            if comp.startswith("Rshunt") or comp.startswith("Rsw") or comp.startswith("R_unity") or comp.startswith("Rsrc_"):
                comps_to_remove.append(comp)
        
        for comp in comps_to_remove:
            del sizing_data["sizing"][comp]

           # 2. Add/Consolidate desired components (Cload)
           if "cload_val" in full_params:
               sizing_data["sizing"]["Cload"] = {"c": full_params["cload_val"]}
        # ---------------------------------------
        
        # Generate deterministic simulation key from parameters (unique per parametrization)
        try:
            key_source = json.dumps({
                "netlist": self.netlist_name,
                "sizing": sizing_data["sizing"],
                "bias": sizing_data["bias"],
                "env": sizing_data["env"]
            }, sort_keys=True, separators=(',', ':'))
            sim_key = hashlib.sha1(key_source.encode('utf-8')).hexdigest()
        except Exception:
            # Fallback to UUID if hashing fails for any reason
            sim_key = str(uuid.uuid4())

        # Process Specs and Operating Points
        raw_specs = eval_result[0][1] if eval_result and len(eval_result) > 0 and len(eval_result[0]) > 1 else {}
        
        main_specs = {}
        op_points = {}
        
        # Helper to safely merge dictionary data
        def merge_op_data(target_dict, param_name, data_dict):
            # data_dict is expected to be { "MM0": 1.23, "MM1": 4.56 ... }
            if isinstance(data_dict, dict):
                for comp_name, val in data_dict.items():
                    if comp_name not in target_dict:
                        target_dict[comp_name] = {}
                    # Preserve full numeric precision when storing operating points
                    try:
                        if isinstance(val, (int, float)):
                            target_dict[comp_name][param_name] = float(val)
                        else:
                            target_dict[comp_name][param_name] = val
                    except Exception:
                        target_dict[comp_name][param_name] = val
            elif isinstance(data_dict, list):
                 # Fallback if meas_man.py wasn't updated or returns list
                 pass

        for k, v in raw_specs.items():
            if k.startswith("z"):
                # Clean up key name: remove 'zz' prefix and '_MM' suffix
                # keys are like zzgm_MM, zzids_MM
                # v is now a dictionary { "MM0": x, ... }
                
                clean_k = k.lstrip('z')
                if clean_k.endswith('_MM'):
                    clean_k = clean_k[:-3] # remove _MM
                    
                # We want op_points to be { "MM0": { "gm": ..., "ids": ... } }
                merge_op_data(op_points, clean_k, v)
            else:
                # Preserve numeric precision for main specs as well
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    try:
                        main_specs[k] = float(v)
                    except Exception:
                        main_specs[k] = v
                elif isinstance(v, tuple):
                    lst = []
                    for x in v:
                        try:
                            lst.append(float(x) if isinstance(x, (int, float)) else x)
                        except Exception:
                            lst.append(x)
                    main_specs[k] = lst
                else:
                    main_specs[k] = v

        # Consolidate Simulation Result into Single Object
        # Sanitize parameters for per-sim JSONs: remove large/unwanted fields
        parameters = dict(sizing_data.get("sizing", {}))
        for _k in ("R_cmfb_pole", "C_cmfb_pole"):
            parameters.pop(_k, None)

        simulation_result = {
            "id": sim_key,
            "sim_id": sim_key,
            "topology_id": 1,
            "netlist": self.netlist_name,
            "parameters": parameters,
            "bias": sizing_data["bias"],
            "env": sizing_data["env"],
            "specs": main_specs,
            "operating_points": op_points,
            "sim_status": sim_status  # 0=convergence_fail, 1=tier1_only, 2=full_char
        }

        # Use deterministic file naming to ensure a parametrization is written exactly once
        # The actual mass-collection persistence is handled by the DataCollector
        # (it buffers and writes batch JSON files). The extractor will not write
        # per-sim marker files here to avoid extra artifacts.

        # Non mass-collection mode: write a full per-sim JSON into results_dir
        result_fname = os.path.join(self.results_dir, f"{sim_key}.json")
        lock_fname = result_fname + ".lock"

        # If result already exists from a prior run, load and return it (idempotent)
        try:
            if os.path.exists(result_fname):
                try:
                    with open(result_fname, 'r') as rf:
                        existing = json.load(rf)
                    existing_main_specs = existing.get('specs', main_specs)
                    return reward1, existing_main_specs, existing
                except Exception:
                    # If file exists but is unreadable, proceed to attempt creation via lock
                    pass

            # Attempt to create a lock atomically. If we succeed we are the owner and will run the sim.
            lock_fd = None
            try:
                lock_fd = os.open(lock_fname, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(lock_fd, f"pid:{os.getpid()}\nstart:{time.time()}\n".encode('utf-8'))
            except OSError as e:
                lock_fd = None
                if e.errno == errno.EEXIST:
                    # Another process is running this parametrization. Wait for the result to appear.
                    # Allow configuration via self.config['sim_control'] if present
                    sim_ctrl = self.config.get('sim_control', {}) if isinstance(self.config, dict) else {}
                    WAIT_TIMEOUT = float(sim_ctrl.get('wait_timeout', 30.0))
                    POLL = float(sim_ctrl.get('poll_interval', 0.2))
                    STALE_LOCK_AGE = float(sim_ctrl.get('stale_lock_age', 600.0))

                    # If the lock file is stale (older than STALE_LOCK_AGE), remove it and retry lock acquisition
                    try:
                        mtime = os.path.getmtime(lock_fname)
                        age = time.time() - mtime
                        if age > STALE_LOCK_AGE:
                            try:
                                os.remove(lock_fname)
                                # try to acquire lock again immediately
                                lock_fd = os.open(lock_fname, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                                os.write(lock_fd, f"pid:{os.getpid()}\nstart:{time.time()}\n".encode('utf-8'))
                            except Exception:
                                lock_fd = None
                    except Exception:
                        pass

                    waited = 0.0
                    while waited < WAIT_TIMEOUT:
                        if os.path.exists(result_fname):
                            try:
                                with open(result_fname, 'r') as rf:
                                    existing = json.load(rf)
                                existing_main_specs = existing.get('specs', main_specs)
                                if 'specs' not in existing:
                                    try:
                                        existing['specs'] = existing_main_specs
                                    except Exception:
                                        existing['specs'] = {}
                                return reward1, existing_main_specs, existing
                            except Exception:
                                pass
                        time.sleep(POLL)
                        waited += POLL

                    # Timeout waiting for result; do NOT attempt to re-run - treat as failure
                    bad_specs = self._get_default_bad_specs(sizing_data['parameters'] if 'parameters' in sizing_data else sizing_data.get('sizing', {}))
                    fail_result = {
                        "id": sim_key,
                        "sim_id": sim_key,
                        "netlist": self.netlist_name,
                        "parameters": sizing_data.get("parameters", sizing_data.get("sizing", {})),
                        "bias": sizing_data.get("bias", {}),
                        "env": sizing_data.get("env", {}),
                        "specs": bad_specs,
                        "operating_points": {},
                        "sim_status": 0
                    }
                    return 0.0, bad_specs, fail_result

            # We are the lock owner (we created the lock file). Run the simulation and write atomically.
            try:
                if self.mode == "mass_collection":
                    # In mass collection mode we no longer write per-sim markers.
                    # Collector is responsible for producing batch JSON files.
                    return reward1, main_specs, simulation_result
                else:
                    tmp_path = result_fname + ".tmp"
                    with open(tmp_path, 'w') as tf:
                        json.dump(simulation_result, tf, indent=2)
                    os.replace(tmp_path, result_fname)
                    print(f"   [+] Exported result: {sim_key}")
                    return reward1, main_specs
            finally:
                try:
                    if lock_fd:
                        os.close(lock_fd)
                    if os.path.exists(lock_fname):
                        os.remove(lock_fname)
                except Exception:
                    pass
        except Exception as e:
            # Ensure any unexpected error does not raise to the worker — return a single failed result
            bad_specs = self._get_default_bad_specs(sizing_data['sizing'])
            fail_result = {
                "id": sim_key,
                "sim_id": sim_key,
                "netlist": self.netlist_name,
                "parameters": sizing_data["sizing"],
                "bias": sizing_data["bias"],
                "env": sizing_data["env"],
                "specs": bad_specs,
                "operating_points": {},
                "sim_status": 0
            }
            return 0.0, bad_specs, fail_result
