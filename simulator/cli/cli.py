"""
cli.py

Author: natelgrw
Last Edited: 01/24/2026

Command Line Interface for ASPECTOR Core.
Provides interactive setup for running circuit optimization pipelines.
Supports Parallel Execution.
"""

import os
import sys
import argparse
import numpy as np
import time
import multiprocessing as mp
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from simulator import globalsy
from simulator.compute.runner import run_parallel_simulations
from simulator.compute.collector import DataCollector
from algorithms.sobol.generator import SobolSizingGenerator
from algorithms.turbo_m import ASPECTOR_TurboM
from simulator.eval_engines.utils.design_reps import extract_sizing_map
from simulator.eval_engines.spectre.configs.config_env import EnvironmentConfig
from simulator.eval_engines.extractor.extractor import Extractor, extract_parameter_names, classify_opamp_type
import torch
import json
import subprocess
import hashlib
import random


# ===== Introductory Text ===== #


class Style:
    CHECK = "[+]"
    X = "[!]"
    INFO = "..."
    ARROW = ">>"
    LINE = "-" * 70
    DOUBLE_LINE = "=" * 70


def clear_screen():
    """
    Clears the terminal screen.
    """
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """
    Prints the stylized header for the CLI.
    """
    print()
    print(Style.DOUBLE_LINE)
    print("       TITAN FOUNDATION MODEL - CIRCUIT OPTIMIZATION PIPELINE       ".center(70))
    print(Style.DOUBLE_LINE)
    print()


def print_section(title):
    """
    Prints a stylized section header.

    Parameters:
    -----------
    title (str): The title of the section to print.
    """
    print(f"\n{Style.LINE}")
    print(f" {title.upper()} ".center(70))
    print(f"{Style.LINE}\n")


def print_success(message):
    """
    Prints a success message with a checkmark.

    Parameters:
    -----------
    message (str): The success message to print.
    """
    print(f" {Style.CHECK} {message}")


def print_info(message):
    """
    Prints an informational message.

    Parameters:
    -----------
    message (str): The informational message to print.
    """
    print(f" {Style.INFO} {message}")


def print_error(message):
    """ 
    Prints an error message with an X. 

    Parameters:
    -----------
    message (str): The error message to print.
    """
    print(f" {Style.X} Error: {message}")


MAX_UINT32_SEED = (2 ** 32) - 1


def normalize_seed(seed):
    """
    Normalize integer seeds into the inclusive uint32 range required by NumPy/SciPy.

    Parameters:
    -----------
    seed (int | None): Input seed.

    Returns:
    --------
    int | None: Normalized seed or None.
    """
    if seed is None:
        return None
    return int(seed) % (MAX_UINT32_SEED + 1)


def configure_reproducibility(global_seed=None, numpy_seed=None, torch_seed=None):
    """
    Apply deterministic seeds for NumPy and PyTorch when provided.

    Parameters:
    -----------
    global_seed (int | None): Fallback seed used if specific library seed is missing.
    numpy_seed (int | None): Explicit NumPy RNG seed.
    torch_seed (int | None): Explicit PyTorch RNG seed.

    Returns:
    --------
    dict: Applied seed values.
    """
    applied_numpy_seed = normalize_seed(numpy_seed if numpy_seed is not None else global_seed)
    applied_torch_seed = normalize_seed(torch_seed if torch_seed is not None else global_seed)

    if applied_numpy_seed is not None:
        np.random.seed(int(applied_numpy_seed))
    if applied_torch_seed is not None:
        torch.manual_seed(int(applied_torch_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(applied_torch_seed))

    return {
        'numpy_seed': applied_numpy_seed,
        'torch_seed': applied_torch_seed,
    }


def save_run_config_snapshot(output_dir, algorithm, personas_weights, run_config, lambdas=None, netlist_name="", seed_info=None):
    """
    Save a snapshot of the run configuration for reproducibility and ablation studies.

    Parameters:
    -----------
    output_dir (str): Directory to save the config snapshot.
    algorithm (str): 'sobol' or 'turbo_m'.
    personas_weights (dict): Dictionary of persona -> weights for TuRBO, or None for Sobol.
    run_config (dict): Full run configuration dict.
    lambdas (dict, optional): Smooth penalty lambdas dict. Defaults included if None.
    netlist_name (str, optional): Name of the netlist being optimized.
    """
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                            cwd=os.path.dirname(__file__)).decode().strip()[:8]
    except Exception:
        git_commit = "unknown"
    
    config_snapshot = {
        'timestamp': time.time(),
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'code_version': git_commit,
        'algorithm': algorithm,
        'netlist_name': netlist_name,
        'run_config': {k: v for k, v in run_config.items()
                       if k != 'api_key'
                       and not (algorithm == 'sobol' and k in ('turbo_batch_size', 'num_trust_regions', 'selected_weights'))},
    }

    # smooth penalty lambdas are TuRBO-specific — omit from Sobol snapshots
    if algorithm == 'turbo_m':
        if lambdas is None:
            lambdas = {
                '_lam_pm': 2.0e3,
                '_lam_gain': 2.0e3,
                '_lam_ugbw': 2.0e3,
            }
        config_snapshot['smooth_penalty_lambdas'] = lambdas

    if seed_info:
        config_snapshot['seed_info'] = seed_info
    
    if algorithm == 'turbo_m' and personas_weights:
        config_snapshot['personas'] = personas_weights
    
    os.makedirs(output_dir, exist_ok=True)
    run_stamp = time.strftime('%Y%m%d_%H%M%S')
    metadata_path = os.path.join(output_dir, 'metadata.json')
    metadata_snapshot_path = os.path.join(output_dir, f'metadata_{run_stamp}.json')
    # Backward-compatible filenames still written for older tooling.
    config_path = os.path.join(output_dir, 'run_config.json')
    snapshot_path = os.path.join(output_dir, f'run_config_{run_stamp}.json')

    for path in [metadata_path, metadata_snapshot_path, config_path, snapshot_path]:
        with open(path, 'w') as f:
            json.dump(config_snapshot, f, indent=2)

    print_success(
        f"Saved metadata snapshot to {metadata_path} and {metadata_snapshot_path} "
        f"(compat: {config_path}, {snapshot_path})"
    )


# ===== User Input Functions ===== #


def get_valid_int(prompt, min_val, max_val, default):
    """
    Prompt for an integer within a range.

    Parameters:
    -----------
    prompt (str): The prompt to display to the user.
    """
    while True:
        user_input = input(f" {Style.ARROW} {prompt} [{default}]: ").strip()
        
        if not user_input:
            return default
            
        try:
            val = int(user_input)
            if min_val <= val <= max_val:
                return val
            else:
                print_error(f"Value must be between {min_val} and {max_val}.")
        except ValueError:
            print_error("Invalid integer.")


def get_netlist_selection():
    """
    Allows user to select a custom directory or use the default.

    Returns:
    --------
    list of tuples: [(netlist_name_base, full_path), ...] for selected netlists. Can be one or many if batch mode is chosen.
    """
    print_section("Netlist Selection")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    default_path = os.path.join(project_root, "topologies")
    
    print(f" Default Directory: {default_path}")
    print(" (Press ENTER to use default, or paste a custom folder path)")
    
    custom_path = input(f" {Style.ARROW} Directory: ").strip()
    
    base_path = default_path
    if custom_path:
        # handle relative paths
        expanded_path = os.path.expanduser(custom_path)
        expanded_path = os.path.abspath(expanded_path)
        
        if os.path.isdir(expanded_path):
            base_path = expanded_path
            print_success(f"Using directory: {base_path}")
        else:
            print_error(f"Directory not found: {custom_path}. Reverting to default.")

    available_files = []
    if os.path.exists(base_path):
        available_files = sorted([f for f in os.listdir(base_path) if f.endswith('.scs')])
        
        if not available_files:
            print_error(f"No .scs files found in {base_path}")
            return []
            
        print("\n Available Netlists:")
        for i, f in enumerate(available_files):
            print(f"   {i+1}. {f[:-4]}")
        print(f"   {len(available_files)+1}. [BATCH] Run ALL Netlists in folder")
        print()
    else:
        print_error(f"Path does not exist: {base_path}")
        return get_netlist_selection()

    while True:
        prompt = f" {Style.ARROW} Enter selection (Number or Name): "
        user_input = input(prompt).strip()
        
        if not user_input:
            continue
            
        # check for selection
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(available_files):
                # SINGLE FILE
                fname = available_files[idx]
                return [(fname[:-4], os.path.join(base_path, fname))]
            elif idx == len(available_files):
                # ALL
                print_info(f"Selected Batch Mode: {len(available_files)} netlists")
                return [(f[:-4], os.path.join(base_path, f)) for f in available_files]
            else:
                 print_error("Invalid selection number")
                 continue
        
        # fallback to name string match
        if not user_input.endswith('.scs'):
            target_f = user_input + ".scs"
        else:
            target_f = user_input
            
        full_path = os.path.join(base_path, target_f)
            
        if os.path.exists(full_path):
            print_success(f"Netlist loaded: {user_input}")
            return [(user_input.replace('.scs',''), full_path)]
        else:
            print_error(f"File not found: {target_f}")


def get_turbo_mode(use_turbo):
    """
    Prompts for Blind vs Sight (Warm Start from Sobol) mode.
    
    Parameters:
    -----------
    use_turbo (bool): Whether to use TuRBO mode.

    Returns:
    --------
    str: 'blind' or 'sight' based on user selection.
    """
    if not use_turbo:
        return "blind"

    print("\n TuRBO Initialization Mode:")
    print("   1. Blind (Default) - Start from scratch")
    print("   2. Sight (Warm Start) - Load Sobol data from results directory")
    
    while True:
        try:
            sel = input(f" {Style.ARROW} Enter selection [1]: ").strip()
            if not sel:
                sel = "1"
            
            if sel == "1":
                return "blind"
            elif sel == "2":
                print_success("Sight mode selected. Sobol parquets will be loaded from the results directory.")
                return "sight"
            else:
                print_error("Invalid selection.")
        except ValueError:
            pass


# ===== Data Processing Functions ===== #


def _parquet_to_specs(df):
    """
    Reconstruct spec dicts from a Parquet DataFrame (out_ columns).
    
    Parameters:
    -----------
    df (pd.DataFrame): DataFrame loaded from a Parquet file, containing columns of values

    Returns:
    --------
    list of dicts: Each dict corresponds to a row in the DataFrame and contains the extracted specs with cleaned keys
    """
    spec_cols = [c for c in df.columns if c.startswith('out_')]

    # detect swing columns
    swing_min = 'out_output_voltage_swing_min' if 'out_output_voltage_swing_min' in df.columns else None
    swing_max = 'out_output_voltage_swing_max' if 'out_output_voltage_swing_max' in df.columns else None
    
    specs_list = []
    for _, row in df.iterrows():
        s = {}
        s['valid'] = bool(row.get('valid', False))
        for col in spec_cols:
            if 'output_voltage_swing' in col:
                continue
            key = col[4:]
            if key == 'area':
                key = 'estimated_area'
            val = row[col]
            if pd.notna(val):
                s[key] = float(val)
        # reconstruct swing tuple
        if swing_min and swing_max and pd.notna(row[swing_min]) and pd.notna(row[swing_max]):
            s['output_voltage_swing'] = (float(row[swing_min]), float(row[swing_max]))
        specs_list.append(s)
    return specs_list


def _find_sobol_parquets(netlist_name_base, results_dir):
    """
    Scan the results directory for Sobol Parquet files matching this netlist.
    Looks in {results_dir}/sobol/ for any .parquet files.
    
    Parameters:
    -----------
    netlist_name_base (str): The base name of the netlist (without .scs)
    results_dir (str): The directory where results are stored

    Returns:
    --------
    list of str: Paths to found Sobol Parquet files (may be empty).
    """
    sobol_dir = os.path.join(results_dir, "sobol")
    if not os.path.isdir(sobol_dir):
        return []
    
    found = []
    for f in sorted(os.listdir(sobol_dir)):
        if f.endswith('.parquet'):
            found.append(os.path.join(sobol_dir, f))
    return found


def run_optimization_task(netlist_name_base, scs_file_path, run_config):
    """
    Executes the optimization pipeline for a single netlist.

    Parameters:
    -----------
    netlist_name_base (str): Base name of the netlist (without .scs)
    scs_file_path (str): Full path to the .scs netlist file
    run_config (dict): Configuration dictionary containing parameters for the run
    """
    print_section(f"Processing: {netlist_name_base}")
    
    # unpack Config
    n_workers = run_config['n_workers']
    adaptive_workers = run_config.get('adaptive_workers', False)
    use_turbo = run_config['use_turbo']
    use_mass_collection = run_config['use_mass_collection']
    

    # pipeline setup
    results_dir = os.path.join(run_config['output_dir'], netlist_name_base)
    os.makedirs(results_dir, exist_ok=True)

    # subdirectories for sobol, turbo_m
    sobol_dir = os.path.join(results_dir, "sobol")
    turbo_dir = os.path.join(results_dir, "turbo_m")

    os.makedirs(sobol_dir, exist_ok=True)
    os.makedirs(turbo_dir, exist_ok=True)

    collector = None
    sobol_parquet = os.path.join(sobol_dir, f"{netlist_name_base}_sobol.parquet")
    sobol_state = os.path.join(sobol_dir, "sobol_state.txt")
    turbo_parquet = os.path.join(turbo_dir, f"{netlist_name_base}_turbo_m.parquet")
    turbo_state = os.path.join(turbo_dir, "turbo_state.pt")

    # choose collector output dir based on algorithm
    if use_mass_collection:
        if use_turbo:
            collector = DataCollector(output_dir=turbo_dir, buffer_size=1000, parquet_name=f"{netlist_name_base}_turbo_m.parquet")
        else:
            collector = DataCollector(output_dir=sobol_dir, buffer_size=1000, parquet_name=f"{netlist_name_base}_sobol.parquet")

    size_map = extract_sizing_map(scs_file_path)

    # extract parameters
    params_id = extract_parameter_names(scs_file_path)
    ignored_params = ['fet_num', 'vdd', 'vcm', 'tempc', 'cload_val']
    sizing_params_for_gen = [p for p in params_id if p not in ignored_params]
    
    # initialize Generator (Mapping Layer)
    base_global_seed = run_config.get('global_seed', 20260319)
    seed_scope = run_config.get('seed_scope', 'per_netlist')
    seed_mode = run_config.get('seed_mode', 'specified')
    netlist_seed_offset = 0
    if seed_scope == 'per_netlist':
        # stable per-topology offset so each netlist has unique but reproducible seeds
        netlist_seed_offset = int(hashlib.sha256(netlist_name_base.encode('utf-8')).hexdigest()[:8], 16)
    global_seed = normalize_seed(int(base_global_seed) + int(netlist_seed_offset))
    sobol_seed = normalize_seed(run_config.get('sobol_seed', global_seed + 1000))
    turbo_seed_base = normalize_seed(run_config.get('turbo_seed_base', global_seed + 2000))
    applied_seeds = configure_reproducibility(
        global_seed=global_seed,
        numpy_seed=run_config.get('numpy_seed'),
        torch_seed=run_config.get('torch_seed'),
    )
    seed_info = {
        'seed_mode': seed_mode,
        'seed_scope': seed_scope,
        'base_global_seed': base_global_seed,
        'netlist_seed_offset': netlist_seed_offset,
        'global_seed': global_seed,
        'sobol_seed': sobol_seed,
        'numpy_seed': applied_seeds['numpy_seed'],
        'torch_seed': applied_seeds['torch_seed'],
    }
    if use_turbo:
        seed_info['turbo_seed_base'] = turbo_seed_base

    generator = SobolSizingGenerator(sizing_params_for_gen, seed=sobol_seed)
    
    opt_dim = generator.dim_sizing if use_turbo else generator.dim

    # configure extractor
    sim_flags = {'ac': True, 'dc': True, 'noise': True, 'tran': True}
    opamp_type = classify_opamp_type(scs_file_path)
    
    full_lb = [-1e9] * len(params_id)
    full_ub = [ 1e9] * len(params_id)
    
    config_env = EnvironmentConfig(scs_file_path, opamp_type, {}, params_id, full_lb, full_ub, results_dir=turbo_dir if use_turbo else sobol_dir)
    config_dict = config_env.get_config_dict()
    
    # if running multiple personas re-initialize the extractor and agent inside the loop
    if not use_turbo:
        specs_id = ["gain_ol", "ugbw", "pm", "power", "vos"]
        specs_ideal = [0.0] * len(specs_id) 
        specs_weights = [1.0, 1.0, 10.0, 10.0, 10.0]
        
        extractor = Extractor(
            dim=len(params_id),
            opt_params=params_id,
            params_id=params_id,
            specs_id=specs_id,            
            specs_ideal=specs_ideal,
            specs_weights=specs_weights,
            sim_flags=sim_flags,
            vcm=0,                 
            vdd=0,                 
            tempc=27,              
            ub=full_ub,            
            lb=full_lb,            
            config=config_dict,
            fet_num=0,             
            results_dir=sobol_dir, 
            netlist_name=netlist_name_base,
            size_map=size_map,
            mode="mass_collection" if use_mass_collection else "test_drive",
            sim_mode=run_config.get('sim_mode', 'complete')
        )

    # shared execution loop variables
    n_max_evals = run_config['n_max_evals']
    interrupted = False

    try:
        if not use_turbo:
            print(f" {Style.INFO} Mode: Sobol Exploration (One-shot)")
            
            # Save run config snapshot for reproducibility
            save_run_config_snapshot(
                sobol_dir, 
                algorithm='sobol',
                personas_weights=None,
                run_config=run_config,
                netlist_name=netlist_name_base,
                seed_info=seed_info
            )
            
            start_idx = 0
            if os.path.exists(sobol_state):
                try:
                    with open(sobol_state, 'r') as f:
                        start_idx = int(f.read().strip())
                    print_info(f"Resuming Sobol sequence from index {start_idx}")
                except Exception as e:
                    print_error(f"Failed to read Sobol state: {e}")
            samples = generator.generate(n_max_evals, start_idx=start_idx)
            # inject flags
            for s in samples:
                s['run_gatekeeper'] = 1
                s['run_full_char'] = 1

            try:
                os.environ.pop('ASPECTOR_RESULTS_DIR', None)
            except Exception:
                pass

            print_info("Running pre-flight simulation check (1 sample, no multiprocessing)...")
            try:
                preflight_result = extractor(samples[0], sim_id=0)
                if preflight_result is None:
                    print_error("Pre-flight returned None. Spectre may not be working.")
                else:
                    n_vals = len(preflight_result) if isinstance(preflight_result, tuple) else 0
                    print_success(f"Pre-flight passed! Extractor returned {n_vals}-tuple.")
                    if n_vals >= 2 and isinstance(preflight_result[1], dict):
                        n_specs = len([k for k, v in preflight_result[1].items() if v is not None])
                        print_info(f"  Extracted {n_specs} non-null specs from pre-flight sample.")
            except Exception as e:
                import traceback
                print_error(f"Pre-flight FAILED: {e}")
                traceback.print_exc()
                print_error("Fix the above error before running the batch. Aborting.")
                return
            
            completed = 0
            for (completed, total, elapsed, data) in run_parallel_simulations(samples, extractor, n_workers, adaptive=adaptive_workers):
                rate = completed / elapsed if elapsed > 0 else 0
                percent = (completed / total) * 100
                bar = '#' * int(30 * completed // total) + '-' * (30 - int(30 * completed // total))
                print(f" [{bar}] {percent:5.1f}% | {completed}/{total} | Rate: {rate:4.1f} sim/s", end='\r')
                if data and use_mass_collection and collector:
                    idx, result_val = data
                    if len(result_val) == 3:
                        full_res = result_val[2] or {}
                        flat_config = samples[idx]
                        specs_to_log = None
                        if isinstance(full_res, dict):
                            specs_to_log = full_res.get('specs')
                        if specs_to_log is None and len(result_val) >= 2:
                            specs_to_log = result_val[1] or {}
                        if specs_to_log is None:
                            specs_to_log = {}

                        collector.log(
                            flat_config,
                            specs_to_log,
                            meta={
                                'sim_id': full_res.get('id', None),
                                'sim_status': full_res.get('sim_status', -1) if isinstance(full_res, dict) else -1,
                                'algorithm': 'sobol',
                                'netlist_name': netlist_name_base,
                                'sobol_index': start_idx + idx,
                                'sobol_seed': sobol_seed
                            },
                            operating_points=(full_res.get('operating_points') if isinstance(full_res, dict) else None)
                        )
                    else:
                        idx, result_val = data
                        flat_config = samples[idx]
                        specs_to_log = result_val[1] if isinstance(result_val, (list, tuple)) and len(result_val) > 1 else {}
                        operating_points = None
                        if isinstance(specs_to_log, dict):
                            op_temp = {}
                            for kk, vv in specs_to_log.items():
                                if isinstance(kk, str) and kk.startswith('z') and isinstance(vv, dict):
                                    clean_k = kk.lstrip('z')
                                    if clean_k.endswith('_MM'):
                                        clean_k = clean_k[:-3]
                                    for comp, val in vv.items():
                                        if comp not in op_temp:
                                            op_temp[comp] = {}
                                        try:
                                            if isinstance(val, (int, float)):
                                                op_temp[comp][clean_k] = float(val)
                                            else:
                                                op_temp[comp][clean_k] = val
                                        except Exception:
                                            op_temp[comp][clean_k] = val
                            if op_temp:
                                operating_points = op_temp

                        collector.log(
                            flat_config,
                            specs_to_log or {},
                            meta={
                                'sim_id': None,
                                'sim_status': -1,
                                'algorithm': 'sobol',
                                'netlist_name': netlist_name_base,
                                'sobol_index': start_idx + idx,
                                'sobol_seed': sobol_seed
                            },
                            operating_points=operating_points
                        )
                
                # periodically save Sobol state to survive hard crashes
                if completed % 1000 == 0:
                    try:
                        with open(sobol_state, 'w') as f:
                            f.write(str(start_idx + completed))
                    except Exception:
                        pass

            # save final state
            try:
                with open(sobol_state, 'w') as f:
                    f.write(str(start_idx + n_max_evals))
                print_info(f"Saved Sobol state at index {start_idx + n_max_evals}")
            except Exception as e:
                print_error(f"Failed to save Sobol state: {e}")
        else:
            # TURBO MODE
            personas_to_run = run_config.get('personas_to_run', [5])
            
            # Define all persona weights upfront for config snapshot and loop
            all_personas = {
                1: {
                    'name': 'SPEED',
                    'weights': {
                        'ugbw': 40.0,
                        'slew_rate': 30.0,
                        'settle_time': 20.0,
                        'pm': 10.0,
                        'gain_ol': 1.0,
                        'power': 1.0,
                        '_pm_target': 62.5,
                        '_pm_range': 2.5
                    }
                },
                2: {
                    'name': 'PRECISION',
                    'weights': {
                        'vos': 30.0,
                        'gain_ol': 25.0,
                        'thd': 10.0,
                        'integrated_noise': 10.0,
                        'cmrr': 7.5,
                        'psrr': 7.5,
                        'output_voltage_swing': 10.0,
                        'ugbw': 2.0,
                        'pm': 1.0
                    }
                },
                3: {
                    'name': 'EFFICIENCY',
                    'weights': {
                        'power': 60.0,
                        'integrated_noise': 10.0,
                        'gain_ol': 2.0,
                        'ugbw': 1.0,
                        'pm': 1.0
                    }
                },
                4: {
                    'name': 'COMPACTNESS',
                    'weights': {
                        'estimated_area': 60.0,
                        'power': 25.0,
                        'gain_ol': 2.0,
                        'ugbw': 1.0,
                        'pm': 2.0,
                        '_pm_target': 67.5,
                        '_pm_range': 7.5
                    }
                },
                5: {
                    'name': 'BALANCED',
                    'weights': {
                        'ugbw': 25.0,
                        'gain_ol': 25.0,
                        'pm': 5.0,
                        'power': 20.0,
                        'vos': 15.0,
                        'output_voltage_swing': 15.0,
                        'integrated_noise': 5.0,
                        'thd': 5.0
                    }
                }
            }
            
            # Save run config snapshot with all personas before starting optimization
            personas_for_config = {all_personas[idx]['name']: all_personas[idx]['weights'] 
                                  for idx in personas_to_run if idx in all_personas}
            save_run_config_snapshot(
                turbo_dir,
                algorithm='turbo_m',
                personas_weights=personas_for_config,
                run_config=run_config,
                netlist_name=netlist_name_base,
                seed_info=seed_info
            )
            
            # Define smooth penalty lambdas (these are the defaults; can be overridden per-persona)
            default_lambdas = {
                '_lam_pm': 2.0e3,
                '_lam_gain': 2.0e3,
                '_lam_ugbw': 2.0e3,
            }
            
            for p_idx in personas_to_run:
                # Retrieve persona definition
                if p_idx not in all_personas:
                    print_error(f"Unknown persona index {p_idx}, skipping.")
                    continue
                
                persona_def = all_personas[p_idx]
                p_name = persona_def['name']
                selected_weights = dict(persona_def['weights'])
                
                # Merge lambdas into weights if persona doesn't specify them (use defaults)
                persona_lambdas = {k: v for k, v in selected_weights.items() if k.startswith('_lam_')}
                if not persona_lambdas:
                    persona_lambdas = default_lambdas
                    
                print(f"\n {Style.DOUBLE_LINE}")
                print(f" {Style.INFO} Starting TuRBO-M Optimization Loop - Persona: {p_name}")
                print(f" {Style.DOUBLE_LINE}")

                persona_seed = normalize_seed(run_config.get(f'turbo_seed_{p_name.lower()}', turbo_seed_base + p_idx))
                print_info(f"Using seeds -> sobol: {sobol_seed}, turbo persona {p_name}: {persona_seed}, numpy: {applied_seeds['numpy_seed']}, torch: {applied_seeds['torch_seed']}")
                
                specs_id = list(selected_weights.keys())
                specs_ideal = [0.0] * len(specs_id) 
                specs_weights_list = list(selected_weights.values())
                
                extractor = Extractor(
                    dim=len(params_id),
                    opt_params=params_id,
                    params_id=params_id,
                    specs_id=specs_id,            
                    specs_ideal=specs_ideal,
                    specs_weights=specs_weights_list,
                    sim_flags=sim_flags,
                    vcm=0,                 
                    vdd=0,                 
                    tempc=27,              
                    ub=full_ub,            
                    lb=full_lb,            
                    config=config_dict,
                    fet_num=0,             
                    results_dir=turbo_dir, 
                    netlist_name=netlist_name_base,
                    size_map=size_map,
                    mode="mass_collection" if use_mass_collection else "test_drive",
                    sim_mode=run_config.get('sim_mode', 'complete')
                )
                
                # re-initialize TuRBO Agent for this persona
                print(f" {Style.CHECK} Initializing TuRBO-M Agent (Dim: {opt_dim}, M: {run_config['num_trust_regions']})")
                turbo_agent = ASPECTOR_TurboM(
                    dim=opt_dim,
                    specs_weights=selected_weights,
                    num_trust_regions=run_config['num_trust_regions'],
                    max_evals=run_config['n_max_evals'],
                    batch_size=run_config['turbo_batch_size'],
                    seed=persona_seed,
                    verbose=True
                )
                
                # reset completion counter for this persona
                total_completed = 0
                turbo_batch_size = run_config['turbo_batch_size']
                
                turbo_state_persona = os.path.join(turbo_dir, f"turbo_state_{p_name.lower()}.pt")
                
                # resume logic
                if os.path.exists(turbo_state_persona):
                    try:
                        turbo_agent.load_state(torch.load(turbo_state_persona))
                        total_completed = len(turbo_agent.X) // 2
                        print_info(f"Resumed TuRBO-M state from {turbo_state_persona} (Completed: {total_completed})")
                    except Exception as e:
                        print_error(f"Failed to load TuRBO-M state: {e}")
                
                # warm start from sobol data
                if len(turbo_agent.X) == 0 and run_config.get('turbo_mode') == 'sight':
                    sobol_parquets = _find_sobol_parquets(netlist_name_base, results_dir)
                    if not sobol_parquets:
                        print_info("Sight mode: No Sobol parquet files found in results directory.")
                    for sobol_pq_path in sobol_parquets:
                        try:
                            print_info(f"Loading Sobol data for warm-start: {sobol_pq_path}")
                            sobol_df = pd.read_parquet(sobol_pq_path)
                            if 'valid' in sobol_df.columns:
                                valid_df = sobol_df[sobol_df['valid'] == True].copy()
                            else:
                                valid_df = sobol_df.copy()
                            
                            if len(valid_df) == 0:
                                print_info("No valid designs in Sobol data, skipping warm-start.")
                            else:
                                X_init, valid_idx = generator.inverse_map(valid_df)
                                
                                if len(X_init) > 0:
                                    valid_df_aligned = valid_df.iloc[valid_idx].reset_index(drop=True)
                                    specs_list = _parquet_to_specs(valid_df_aligned)
                                    
                                    Y_init = turbo_agent.scalarize_specs(specs_list, update_stats=True)
                                    
                                    turbo_agent.load_state(X_init, Y_init)
                                    print_success(f"Warm-started TuRBO-M ({p_name}) with {len(X_init)} designs from {os.path.basename(sobol_pq_path)} (best cost: {Y_init.min().item():.4f})")
                                else:
                                    print_info(f"inverse_map returned 0 valid rows from {os.path.basename(sobol_pq_path)}.")
                        except Exception as e:
                            import traceback
                            print_error(f"Warm-start failed for {os.path.basename(sobol_pq_path)}: {e}")
                            traceback.print_exc()
                            print_info("Continuing without this file.")
                    if len(turbo_agent.X) > 0:
                        print_success(f"Sight warm-start complete: {len(turbo_agent.X)} total points loaded.")
                        
                while total_completed < n_max_evals:
                    # mass runs should not preserve raw Spectre artifacts by default
                    try:
                        os.environ.pop('ASPECTOR_RESULTS_DIR', None)
                    except Exception:
                        pass

                    curr_batch_size = min(turbo_batch_size, n_max_evals - total_completed)
                    print(f"\n {Style.ARROW} TuRBO Asking for {curr_batch_size} candidates...")
                    X_next = turbo_agent.ask(curr_batch_size)
                    X_list = X_next.tolist()
                    samples_nom = generator.generate(curr_batch_size, u_samples=X_list, robust_env=False)
                    samples_rob = generator.generate(curr_batch_size, u_samples=X_list, robust_env=True)
                    full_batch_samples = samples_nom + samples_rob
                    total_sims = len(full_batch_samples)
                    for s in full_batch_samples:
                        s['run_gatekeeper'] = 1
                        s['run_full_char'] = 1
                    print(f" {Style.ARROW} Simulating Robust Batch ({total_sims} sims for {curr_batch_size} candidates)...")
                    sim_results = []
                    for (b_completed, b_total, b_elapsed, data) in run_parallel_simulations(full_batch_samples, extractor, n_workers, adaptive=adaptive_workers):
                        rate = b_completed / b_elapsed if b_elapsed > 0 else 0
                        print(f"   Batch Progress: {b_completed}/{total_sims} | Rate: {rate:.1f} sim/s", end='\r')
                        if data:
                            idx, result_val = data
                            sim_results.append(data)
                            if use_mass_collection and len(result_val) == 3:
                                full_res = result_val[2]
                                flat_config = full_batch_samples[idx]
                                # Determine if this sample is from nominal or robust environment
                                env_type = 'nominal' if idx < curr_batch_size else 'robust'
                                collector.log(
                                    flat_config,
                                    full_res['specs'],
                                    meta={
                                        'sim_id': full_res['id'],
                                        'sim_status': full_res.get('sim_status', -1),
                                        'algorithm': f"turbo_m_{p_name}",
                                        'persona_name': p_name,
                                        'env_type': env_type,
                                        'turbo_seed': persona_seed,
                                        'weights_dict': selected_weights,
                                        'lambdas_dict': persona_lambdas
                                    }
                                )
                    results_map = {}
                    for idx, val_tuple in sim_results:
                        if len(val_tuple) >= 2 and isinstance(val_tuple[1], dict):
                            results_map[idx] = val_tuple[:2]
                    
                    ordered_specs = []
                    valid_indices = []
                    n_candidates = curr_batch_size
                    for i in range(n_candidates):
                        idx_nom = i
                        idx_rob = i + n_candidates
                        res_nom = results_map.get(idx_nom)
                        res_rob = results_map.get(idx_rob)
                        if res_nom and res_rob:
                            specs_nom = res_nom[1]
                            specs_rob = res_rob[1]
                            worst_case_specs = {}
                            all_keys = set(specs_nom.keys()) | set(specs_rob.keys())
                            worst_case_specs['valid'] = specs_nom.get('valid', True) and specs_rob.get('valid', True)
                            for key in all_keys:
                                if key == 'valid': continue
                                val_n = specs_nom.get(key)
                                val_r = specs_rob.get(key)
                                if val_n is None or val_r is None: continue
                                def is_better(v1, v2, k):
                                    if k in ["power", "integrated_noise", "settling_time", "vos", "thd", "ibias"]:
                                        return abs(v1) < abs(v2)
                                    else:
                                        return v1 > v2
                                if is_better(val_n, val_r, key):
                                    worst_case_specs[key] = val_r
                                else:
                                    worst_case_specs[key] = val_n
                            ordered_specs.append(worst_case_specs)
                            valid_indices.append(i)
                    if len(valid_indices) > 0:
                        X_valid = X_next[valid_indices]
                        turbo_agent.tell(X_valid, ordered_specs)
                        total_completed += len(valid_indices)
                        best_v = min([s.best_value for s in turbo_agent.state]) if turbo_agent.state else 0.0
                        print(f"\n {Style.CHECK} Robust Batch (Worst-Case) processed. Total: {total_completed}/{n_max_evals} Best Cost: {best_v:.4f}")
                    # save TuRBO-M state after each batch
                    try:
                        torch.save({
                            'state': turbo_agent.state,
                            'X': turbo_agent.X,
                            'Y': turbo_agent.Y,
                            'spec_stats': turbo_agent.spec_stats,
                            'weights': turbo_agent.weights,
                            'seed': turbo_agent.seed,
                            'sobol_seed_counter': turbo_agent._sobol_seed_counter,
                            'persona_name': p_name,
                            'seed_info': {
                                'global_seed': global_seed,
                                'sobol_seed': sobol_seed,
                                'turbo_seed': persona_seed,
                                'numpy_seed': applied_seeds['numpy_seed'],
                                'torch_seed': applied_seeds['torch_seed'],
                            }
                        }, turbo_state_persona)
                    except Exception as e:
                        print_error(f"Failed to save TuRBO-M state: {e}")

    except KeyboardInterrupt:
        print(f"\n\n {Style.X} Simulation interrupted for {netlist_name_base}.")
        interrupted = True
        
    finally:
        if collector:
            print(f" {Style.INFO} Finalizing Data Collector...")
            try:
                if interrupted:
                    try:
                        collector.flush()
                    except Exception:
                        pass
                    collector.finalize(discard_partial=False, preserve_json=False)
                else:
                    collector.finalize(discard_partial=False, preserve_json=False)
            except Exception as e:
                print_error(f"DataCollector finalize failed: {e}")
        print(f"\n\n {Style.CHECK} Pipeline finished for this netlist.")


# ===== Main ===== #


def main():
    """
    Guides the user through netlist selection, parallel configuration, algorithm choice, 
    and data collection mode. 
    
    Then executes the optimization loop for each selected netlist 
    with the specified settings.
    """
    clear_screen()
    print_header()

    # 1. netlist selection
    netlist_queue = get_netlist_selection()

    # 2. parallel core configuration
    max_cores = mp.cpu_count()
    print_section("Parallel Compute Configuration")
    print(f" {Style.INFO} System has {max_cores} CPU cores available.")
    print(f" {Style.INFO} Worker count is ADAPTIVE — adjusts automatically based on real-time load.")
    
    try:
        load1, _, _ = os.getloadavg()
        print(f" {Style.INFO} Current 1-min CPU load average: {load1:.2f}")
        initial_workers = max(1, int(max_cores - load1) - 2)
        print(f" {Style.INFO} Initial worker estimate: {initial_workers}")
    except AttributeError:
        initial_workers = max(1, max_cores - 2)
    
    print(f"\n Enter 0 for fully adaptive (recommended), or a fixed number.")
    n_workers = get_valid_int(
        "Number of Workers (0 = adaptive)",
        min_val=0,
        max_val=max_cores,
        default=0
    )
    
    adaptive_workers = (n_workers == 0)
    if adaptive_workers:
        n_workers = initial_workers
        print_success(f"Adaptive mode enabled. Starting with {n_workers} workers.")
    else:
        print_success(f"Fixed mode: {n_workers} workers.")

    # 3. algorithm selection
    print_section("Algorithm Selection")
    print(" Select Optimization Algorithm:")
    print("   1. Sobol Sequence (Design Space Exploration)")
    print("   2. TuRBO-M (Integrative Bayesian Optimization)")
    print()
    
    algo_selection = get_valid_int("Enter selection", 1, 2, 1)
    use_turbo = (algo_selection == 2)

    turbo_batch_size = 64
    n_max_evals = 1000
    num_trust_regions = 10
    
    selected_weights = {} 
    
    # defaults
    run_config = {
        "n_workers": n_workers,
        "adaptive_workers": adaptive_workers,
        "use_turbo": use_turbo,
        "n_max_evals": 1000,
        "turbo_batch_size": 64,
        "num_trust_regions": num_trust_regions,
        "selected_weights": {},
        "use_mass_collection": False
    }

    if use_turbo:
        print(f" {Style.CHECK} Selected TuRBO-M")
        
        # persona selection
        print("\n Select Optimization Persona (Defines Primary/Secondary Goals):")
        print("   1. Speed (High UGBW/Slew, strict PM, Power limit)")
        print("   2. Precision (High Gain/CMRR/PSRR, Low Offset/THD)")
        print("   3. Efficiency (Low Power/Noise, min UGBW)")
        print("   4. Compactness (Min Estimated Area, good Swing/Slew)")
        print("   5. Balanced (General Purpose - Demo Default)")
        print("   6. ALL (Pareto Sweep - Runs 1-5 sequentially for Mass Data Collection)")
        
        persona = get_valid_int("Enter persona", 1, 6, 6)
        
        personas_to_run = []
        if persona == 6:
            print(f" {Style.INFO} Persona: PARETO SWEEP (All 5 Personas)")
            personas_to_run = [1, 2, 3, 4, 5]
        else:
            personas_to_run = [persona]
            
        run_config['personas_to_run'] = personas_to_run
        
        turbo_batch_size = get_valid_int(
            "Batch Size (Number of candidates per iteration)",
            min_val=1,
            max_val=1000,
            default=64
        )
        n_max_evals = get_valid_int(
            "Total Maximum Evaluations",
            min_val=turbo_batch_size,
            max_val=10000,
            default=640
        )
        
        run_config['turbo_batch_size'] = turbo_batch_size
        run_config['n_max_evals'] = n_max_evals
        
        run_config['turbo_mode'] = get_turbo_mode(use_turbo)
        
    else:
        print(f" {Style.CHECK} Selected Sobol Explorer")
        n_samples = get_valid_int(
            "Number of Samples to Generate (Per Netlist)",
            min_val=1,
            max_val=1000000,
            default=100
        )
        n_max_evals = n_samples
        run_config['n_max_evals'] = n_max_evals

    # 4. data collection mode
    print_section("Data Collection Mode")
    print(" Select Data Output Strategy:")
    print("   1. Test Drive (JSON per sim)")
    print("   2. Mass Collection (Parquet, Batch JSONs)")
    
    data_mode_sel = get_valid_int("Enter selection", 1, 2, 1)
    use_mass_collection = (data_mode_sel == 2)
    run_config['use_mass_collection'] = use_mass_collection
    
    # 5. simulation mode
    print_section("Simulation Mode")
    print(" Select Simulation Mode:")
    print("   1. Complete Mode (Runs all simulations regardless of DC operating point)")
    print("   2. Efficient Mode (Tier 1: dcOp + STB/XF/Noise, then Tier 2: DC sweep/Transient/PSS only if Tier 1 passes)")
    
    sim_mode_sel = get_valid_int("Enter selection", 1, 2, 1)
    sim_mode = "complete" if sim_mode_sel == 1 else "efficient"
    run_config['sim_mode'] = sim_mode
    
    # 6. output directory
    print_section("Output Directory")
    while True:
        out_dir = input(f" {Style.ARROW} Enter relative path from aspector_core to save results (e.g., results_mtlcad): ").strip()
        if out_dir:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            out_dir = os.path.join(project_root, out_dir)
            print_success(f"Results will be saved to: {out_dir}")
            run_config['output_dir'] = out_dir
            break
        else:
            print_error("You must specify an output directory.")

    # 7. reproducibility / seed mode
    print_section("Seed Configuration")
    print(" Seed Mode:")
    print("   1. Specified Seed (fully reproducible)")
    print("   2. Random Seed (auto-generated, then documented)")
    seed_mode_sel = get_valid_int("Enter selection", 1, 2, 1)

    if seed_mode_sel == 1:
        base_seed = get_valid_int(
            "Global Seed",
            min_val=0,
            max_val=2_147_483_647,
            default=20260319,
        )
        run_config['seed_mode'] = 'specified'
        run_config['global_seed'] = int(base_seed)
        print_success(f"Using specified global seed: {base_seed}")
    else:
        base_seed = random.SystemRandom().randint(0, 2_147_483_647)
        run_config['seed_mode'] = 'random'
        run_config['global_seed'] = int(base_seed)
        print_success(f"Generated random global seed for this launch: {base_seed}")

    print(" Seed Scope:")
    print("   1. Per-Netlist (recommended, unique deterministic offset per topology)")
    print("   2. Shared Across Netlists (all topologies use same base seed)")
    seed_scope_sel = get_valid_int("Enter selection", 1, 2, 1)
    run_config['seed_scope'] = 'per_netlist' if seed_scope_sel == 1 else 'shared'
    print_info(f"Seed scope: {run_config['seed_scope']}")
    
    print()
    print(Style.DOUBLE_LINE)
    print(f"      STARTING BATCH JOB | {len(netlist_queue)} NETLISTS       ".center(70))
    print(Style.DOUBLE_LINE)
    print()
    
    # START BATCH LOOP
    for i, (name, path) in enumerate(netlist_queue):
        print(f"\n[{i+1}/{len(netlist_queue)}] Running Task: {name}")
        run_optimization_task(name, path, run_config)
    
    print(f"\n\n {Style.CHECK} All Tasks Completed.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
