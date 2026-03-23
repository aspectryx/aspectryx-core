"""
config_env.py

Author: natelgrw
Last Edited: 01/15/2026

Environment configuration manager for setting up Spectre simulation parameters,
specification ranges, and YAML configuration files for optimization workflows.
"""

import os


# ===== Environment Configuration Manager ===== #


class EnvironmentConfig(object):
    """
    Configuration manager for setting up optimization environments.

    Handles the creation and management of simulation parameters, specification
    ranges, and YAML configuration files for Spectre-based circuit optimization.

    Initialization Parameters:
    --------------------------
    netlist_path : str
        Path to the circuit netlist file.
    type : str
        Type of measurement (e.g., 'differential', 'single_ended').
    specs : dict
        Target specifications dictionary.
    params : list
        Parameter names to optimize.
    param_lbs : list
        Lower bounds for parameters.
    param_ubs : list
        Upper bounds for parameters.
    """

    def __init__(self, netlist_path, type, specs, params, param_lbs, param_ubs, results_dir="results"):

        self.netlist_path = netlist_path
        self.type = type
        self.specs = specs
        self.params = params
        self.param_lbs = param_lbs
        self.param_ubs = param_ubs
        
        # configuration dictionary with default structure
        self.configs = {
            "database_dir": results_dir,
            "measurement": {
                "meas_params": {},
                "testbenches": {
                    "ac_dc": {
                        "netlist_template": self.netlist_path,
                        "tb_module": f"simulator.eval_engines.spectre.measurements.{self.type}_meas_man",
                        "tb_class": "ACTB",
                        "post_process_function": "process_ac",
                        "tb_params": {}
                    }
                }
            },
            # "params": {},
            # Removed optimization-specific fields
            "target_specs": {}
        }
        
        self.param_dict = {}
        self.yaml_path = ""

    def build_specs(self):
        """
        Build specification dictionary.
        Simpliefied to remove normalization and ranges for optimization.
        """
        # Simply map specs to target_specs
        for spec, val in self.specs.items():
            val = float(val)
            self.configs["target_specs"][spec] = (val,)

    def build_params(self):
        """
        Build parameter bounds dictionary.
        Removed step size calculation for grid optimization.
        Stores (lower_bound, upper_bound) only.
        """
        for i in range(len(self.params)):
            param = self.params[i]
            lb = float(self.param_lbs[i])
            ub = float(self.param_ubs[i])
            
            self.param_dict[param] = (lb, ub)

    def build_configs(self):
        """
        Build complete configuration dictionary.
        """
        self.build_specs()
        # self.build_params() # Disabled per user request to remove 'params' dump
        
        # self.configs["params"] = self.param_dict # Disabled
        # self.configs["params"] = {} # Empty dict to maintain structure if needed

    def write_yaml_configs(self):
        """
        Build and write configuration to YAML file.

        Generates the complete configuration dictionary and writes it to a YAML file
        in the same directory as this Python file. The YAML filename is derived from
        the input netlist filename.

        Returns:
        --------
        str
            Absolute path to the generated YAML configuration file.
        """
        self.build_configs()
        # Deprecated: YAML writing is no longer required; return dict instead
        return self.get_config_dict()

    def get_config_dict(self):
        """
        Return the configuration as a Python dictionary. This is the preferred
        method for providing configuration to the simulation stack instead of
        writing/reading YAML files.
        """
        self.build_configs()
        return dict(self.configs)
    
    def del_yaml(self):
        """
        No-op: YAML files are deprecated. Kept for backward compatibility.
        """
        return
    
