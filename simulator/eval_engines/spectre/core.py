"""
core.py

Author: natelgrw
Last Edited: 01/15/2026

Core Spectre evaluation engine for circuit simulation and design optimization.
Handles netlist generation, simulation execution, result parsing, and 
cost function evaluation for iterative circuit optimization.
"""

import os
import tempfile
from jinja2 import Environment, FileSystemLoader
import shutil
import subprocess
from multiprocessing.dummy import Pool as ThreadPool
# Programmatic configuration dictionaries are used (YAML not required)
import importlib
import json
import hashlib
import random
import numpy as np
import uuid
import time
import gc
from simulator.eval_engines.spectre.parser import SpectreParser

debug = False 


# ===== Spectre Simulation Wrapper ===== #


class SpectreWrapper(object):
    """
    Wrapper for managing Spectre circuit simulations.

    Handles netlist generation from templates, simulation execution via Spectre,
    result parsing, and post-processing. Each instance manages one netlist and
    its associated simulation testbench.
    
    Initialization Parameters:
    --------------------------
    tb_dict : dict
        Testbench configuration dictionary.
    """

    def __init__(self, tb_dict):

        netlist_loc = tb_dict['netlist_template']
        if not os.path.isabs(netlist_loc):
            netlist_loc = os.path.abspath(netlist_loc)
        
        # load post-processing module and class
        pp_module = importlib.import_module(tb_dict['tb_module'])
        pp_class = getattr(pp_module, tb_dict['tb_class'])
        self.post_process = getattr(pp_class, tb_dict['post_process_function'])
        self.tb_params = tb_dict['tb_params']

        self.num_process = 1

        # Calculate project root and lstp path relative to this file
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        self.lstp_path = os.path.join(self.project_root, "lstp")

        # Create scratch directory in system temp — never pollutes results dirs
        _, dsn_netlist_fname = os.path.split(netlist_loc)
        self.base_design_name = os.path.splitext(dsn_netlist_fname)[0] + "_" + uuid.uuid4().hex
        self.gen_dir = tempfile.mkdtemp(prefix="aspector_", suffix="_" + self.base_design_name)

        # setup jinja2 template environment
        file_loader = FileSystemLoader(os.path.dirname(netlist_loc))
        self.jinja_env = Environment(loader=file_loader)
        self.template = self.jinja_env.get_template(dsn_netlist_fname)

    def _get_design_name(self, state):
        """
        Creates a unique identifier filename based on design state parameters.

        Parameters:
        -----------
        state : dict
            Dictionary of parameter names and values for the design.
        
        Returns:
        --------
        fname : str
            Unique filename identifier for the design.
        """
        # Use a short hash of the sorted state to avoid extremely long filenames
        try:
            s = json.dumps(state, sort_keys=True, default=str)
        except Exception:
            s = str(state)
        short_id = hashlib.sha1(s.encode()).hexdigest()[:12]
        fname = f"{self.base_design_name}_{short_id}"
        return fname

    def _create_design(self, state, new_fname):
        """
        Creates sized netlist file from template and design parameters.

        Parameters:
        -----------
        state : dict
            Dictionary of parameter values for rendering the template.
        new_fname : str
            Filename for the generated netlist (without extension).
        
        Returns:
        --------
        design_folder : str
            Path to the folder containing the generated netlist.
        fpath : str
            Full path to the generated netlist file.
        """
        # render template with design parameters
        render_context = state.copy()
        
        # Ensure Critical Environment Variables are set!
        # If they came in via state, they are already there.
        # If not, we should probably set defaults, but ideally 
        # they must be in 'state' from the generator.
        
        # NOTE: Jinja templates fail silently sometimes or create invalid netlists
        # if variables are missing.
        
        render_context['lstp_path'] = self.lstp_path
        
        output = self.template.render(**render_context)

        # Create a short, safe folder/file name derived from the state
        try:
            s = json.dumps(state, sort_keys=True, default=str)
        except Exception:
            s = str(state)
        short_id = hashlib.sha1(s.encode()).hexdigest()[:12]

        safe_base = self.base_design_name
        folder_name = f"{safe_base}_{short_id}"
        design_folder = os.path.join(self.gen_dir, folder_name)
        os.makedirs(design_folder, exist_ok=True)

        fpath = os.path.join(design_folder, f"{safe_base}_{short_id}.scs")

        # Write netlist
        with open(fpath, 'w') as f:
            f.write(output)

        # Write metadata mapping short id -> original requested name and full state
        try:
            meta = {
                'requested_name': str(new_fname),
                'state': state
            }
            meta_path = os.path.join(design_folder, f"{safe_base}_{short_id}.meta.json")
            with open(meta_path, 'w') as mf:
                json.dump(meta, mf, indent=2, default=str)
        except Exception:
            pass

        return design_folder, fpath

    def _simulate(self, fpath):
        """
        Executes Spectre simulation on generated netlist.

        Parameters:
        -----------
        fpath : str
            Full path to the netlist file to simulate.
        
        Returns:
        --------
        info : int
            Error code. 0 indicates successful simulation, 1 indicates error.
        """
        # construct Spectre command
        command = ['nice', '-n', '19', 'spectre', '%s'%fpath, '-format', 'psfbin']
        log_file = os.path.join(os.path.dirname(fpath), 'log.txt')
        err_file = os.path.join(os.path.dirname(fpath), 'err_log.txt')

        # execute simulation and capture output
        with open(log_file, 'w') as file1, open(err_file,'w') as file2:
            exit_code = subprocess.call(command, cwd=os.path.dirname(fpath), stdout=file1, stderr=file2)

        # determine success based on exit code
        info = 0
        if (exit_code % 256):
            info = 1

        return info

    def _create_design_and_simulate(self, state, dsn_name=None, verbose=False):
        """
        Creates a design netlist and runs simulation.

        Parameters:
        -----------
        state : dict
            Dictionary of design parameter values.
        dsn_name : str, optional
            Custom design name. If None, auto-generated from state.
        verbose : bool
            If True, prints design name information.
        
        Returns:
        --------
        state : dict
            The input design state.
        specs : dict
            Dictionary of post-processed simulation results.
        info : int
            Error code from simulation (0 = success).
        """
        # generate design name if not provided
        if dsn_name == None:
            dsn_name = self._get_design_name(state)
        else:
            dsn_name = str(dsn_name)
    
        if verbose:
            print('dsn_name', dsn_name)

        # create netlist from template and run simulation
        design_folder, fpath = self._create_design(state, dsn_name)

        # create netlist from template and run simulation
        try:
            info = self._simulate(fpath)
            results = self._parse_result(design_folder)

            # post process results
            if self.post_process:
                specs = self.post_process(results, state)
                return state, specs, info
            specs = results

            return state, specs, info
        finally:
            self._cleanup(design_folder)

    def _cleanup(self, design_folder):
        """
        Removes the generated design folder and all its contents to save space.
        """
        gc.collect()

        # Prefer moving the generated design folder into a persistent
        # results directory (so raw simulation artifacts are preserved
        # next to the JSON batch files) instead of removing it.
        try:
            results_dir = os.environ.get('ASPECTOR_RESULTS_DIR')
        except Exception:
            results_dir = None

        if os.path.exists(design_folder):
            moved = False
            if results_dir:
                try:
                    raw_store = os.path.join(results_dir, "raw")
                    os.makedirs(raw_store, exist_ok=True)
                    dest = os.path.join(raw_store, os.path.basename(design_folder))
                    # Avoid clobbering an existing folder
                    if os.path.exists(dest):
                        dest = dest + "_" + uuid.uuid4().hex[:8]
                    shutil.move(design_folder, dest)
                    moved = True
                except Exception as e:
                    try:
                        print(f"[!] Failed to move design folder to results dir: {e}")
                    except Exception:
                        pass

            if not moved:
                try:
                    shutil.rmtree(design_folder, ignore_errors=False)
                except Exception as e:
                    # Fallback to stronger delete
                    subprocess.call(['rm', '-rf', design_folder])

        # If gen_dir is now empty, remove it too (no orphan temp dirs)
        try:
            if os.path.exists(self.gen_dir) and not os.listdir(self.gen_dir):
                os.rmdir(self.gen_dir)
        except OSError:
            pass

    def _parse_result(self, design_folder):
        """
        Parses simulation results from Spectre output files.

        Parameters:
        -----------
        design_folder : str
            Path to the design folder containing simulation results.
        
        Returns:
        --------
        res : dict
            Dictionary of parsed simulation results.
        """
        # extract folder name and locate raw simulation output
        _, folder_name = os.path.split(design_folder)
        raw_folder = os.path.join(design_folder, '{}.raw'.format(folder_name))

        # parse results
        res = SpectreParser.parse(raw_folder)       

        return res

    def run(self, states, design_names=None, verbose=False):
        """
        Executes simulations for multiple design states in parallel.

        Uses thread pool to run multiple simulations concurrently based on
        configured number of processes.

        Parameters:
        -----------
        states : list
            List of design state dictionaries to simulate.
        design_names : list, optional
            Custom design names for each state. If None, auto-generated names used.
        verbose : bool
            If True, prints design name information during execution.
        
        Returns:
        --------
        specs : list
            List of (state, specs, info) tuples for each design simulated.
        """
        # execute simulations in parallel using thread pool
        pool = ThreadPool(processes=self.num_process)
        arg_list = [(state, dsn_name, verbose) for (state, dsn_name)in zip(states, design_names)]
        specs = pool.starmap(self._create_design_and_simulate, arg_list)
        pool.close()

        return specs

    def return_path(self):
        """
        Returns the design generation directory path.

        Returns:
        --------
        str
            Path to the directory containing generated designs.
        """
        return self.gen_dir


# ===== Circuit Evaluation Engine ===== #


class EvaluationEngine(object):
    """
    Main evaluation engine for circuit optimization.
    Stripped down for simple data generation.

    Initialization Parameters:
    --------------------------
    config : dict
        Configuration dictionary specifying parameters, specifications, and testbench setup.
    """

    def __init__(self, config):
        # Accept a configuration dictionary. YAML files are deprecated.
        self.design_specs = config
        if isinstance(config, dict):
            self.ver_specs = config
        else:
            raise RuntimeError("EvaluationEngine requires a configuration dictionary. YAML files are deprecated.")

        # setup testbench modules
        self.measurement_specs = self.ver_specs['measurement']
        tbs = self.measurement_specs['testbenches']
        self.netlist_module_dict = {}
        for tb_kw, tb_val in tbs.items():
            self.netlist_module_dict[tb_kw] = SpectreWrapper(tb_val)

    def evaluate(self, design_list, debug=True, parallel_config=None):
        """
        Evaluates designs and returns processed results.

        Parameters:
        -----------
        design_list : list of dicts
            List of design parameter dictionaries.
        debug : bool
        
        Returns:
        --------
        results : list
            List of (state, specs, info) tuples.
        """
        results = []
        
        for state in design_list:
            # run simulations for all testbenches
            sim_results = {}
            for netlist_name, netlist_module in self.netlist_module_dict.items():
                sim_results[netlist_name] = netlist_module._create_design_and_simulate(state, verbose=debug)

            # get specifications from results (using subclass logic)
            # subclass (e.g. OpampMeasMan) must implement get_specs
            if hasattr(self, 'get_specs'):
                specs_dict = self.get_specs(sim_results, state)
            else:
                # If no post-processing mapping, just return the raw specs from wrapper
                # Assuming single testbench for simplicity if no get_specs
                first_res = list(sim_results.values())[0]
                specs_dict = first_res[1]

            # Collect actual info codes from each netlist simulation where available
            info_codes = []
            for net_res in sim_results.values():
                try:
                    # net_res is expected to be a tuple (state, specs, info)
                    if isinstance(net_res, tuple) and len(net_res) > 2:
                        info_codes.append(int(net_res[2]))
                except Exception:
                    pass

            # If any underlying simulation returned non-zero info, propagate failure (1)
            overall_info = 0
            if any(ic for ic in info_codes):
                overall_info = 1

            results.append((state, specs_dict, overall_info))

        return results

