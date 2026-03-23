"""
globalsy.py

Author: natelgrw
Last Edited: 01/15/2026

Global variables for algorithmic exploration modules.
"""

counterrrr = 0

# NOTE: The optimizer will handle spec optimization weights and targets.
# Users should not implement them here manually.
spec_metadata = {}
specs_dict = {}
specs_weights = {}

# Parameter Restrictions

# Basic Parameters
basic_params = {
    'L': {
        'type': 'discrete',
        'lb': 10e-9, # model card lower bound (parsed from lstp/*.pm)
        'ub': 20e-9, # model card upper bound (parsed from lstp/*.pm)
        'unit': 'm',
        'sampling': 'grid',
        'step': 2e-9  # 2 nm grid step
    },
    'Nfin': {
        'type': 'integer',
        'lb': 4,
        'ub': 128,
        'unit': '',
        'sampling': 'integer',
        'step': 1
    },
    'C_internal': {
        'type': 'continuous',
        'lb': 100e-15,  # 100 fF
        'ub': 5e-12,    # 5 pF
        'unit': 'F',
        'sampling': 'logarithmic'
    },
    'R_internal': {
        'type': 'continuous',
        'lb': 500,      # 500 Ohm
        'ub': 100e3,    # 100 kOhm
        'unit': 'Ohm',
        'sampling': 'logarithmic'
    },
    'vbiasn0': {
        'type': 'continuous',
        'lb': '0.45 * vdd_nominal',
        'ub': '0.70 * vdd_nominal',
        'unit': 'V',
        'sampling': 'fractional',
        'step_frac': 0.01
    },
    'vbiasn1': {
        'type': 'continuous',
        'lb': '0.45 * vdd_nominal',
        'ub': '0.70 * vdd_nominal',
        'unit': 'V',
        'sampling': 'fractional',
        'step_frac': 0.01
    },
    'vbiasn2': {
        'type': 'continuous',
        'lb': '0.65 * vdd_nominal',
        'ub': '0.85 * vdd_nominal',
        'unit': 'V',
        'sampling': 'fractional',
        'step_frac': 0.01
    },
    'vbiasp0': {
        'type': 'continuous',
        'lb': '0.40 * vdd_nominal',
        'ub': '0.85 * vdd_nominal',
        'unit': 'V',
        'sampling': 'fractional',
        'step_frac': 0.01
    },
    'vbiasp1': {
        'type': 'continuous',
        'lb': '0.40 * vdd_nominal',
        'ub': '0.85 * vdd_nominal',
        'unit': 'V',
        'sampling': 'fractional',
        'step_frac': 0.01
    },
    'vbiasp2': {
        'type': 'continuous',
        'lb': '0.15 * vdd_nominal',
        'ub': '0.50 * vdd_nominal',
        'unit': 'V',
        'sampling': 'fractional',
        'step_frac': 0.01
    }
}

# Environment Parameters
env_params = {
}

# Testbench Parameters
testbench_params = {
    'Fet_num': [7, 10, 14, 16, 20],
    'VDD': {
        'type': 'continuous',
        'lb': '0.9 * vdd_nominal',
        'ub': '1.1 * vdd_nominal',
        'sampling': 'fractional',
        'step_frac': 0.01
    },
    'VCM': {
        'type': 'continuous',
        # Default VCM range is broad; topology-aware code will refine this when needed
        'lb': '0.10 * vdd_nominal',
        'ub': '0.90 * vdd_nominal',
        'sampling': 'fractional',
        'step_frac': 0.01
    },
    'Tempc': {
        'type': 'integer',
        'lb': -40,
        'ub': 125,
        'step': 1
    }
    'Cload_val': {
        'type': 'continuous',
        'lb': 10e-15,  # 10 fF
        'ub': 5e-12,  # 5 pF
        'sampling': 'logarithmic'
    }
}

# Sampling metadata to make discrete/grid sampling universal across generators
sampling_metadata = {
    # Fractional step of nominal VDD for VDD/VCM/vbias sampling (e.g. 0.01 => 1%)
    'vdd': {'step_frac': 0.01},
    'vcm': {'step_frac': 0.01},
    'vbias': {'step_frac': 0.01},
    # Temperature step in degrees
    'tempc': {'step': 1},
    # Series to use for resistors/capacitors (E-series)
    'R_series': 'E24',
    'C_series': 'E12',
    # Default grid step for channel length if not present in basic_params
    'L_step': 2e-9,
    # Fin count step (integer)
    'Nfin_step': 1
}


def vcm_bounds_for_topology(topology_name, vdd_nominal):
    """Return (lb, ub) fractional multipliers of vdd_nominal for VCM bounds

    topology_name: filename or identifier string for the topology (may be None)
    vdd_nominal: nominal VDD in volts

    Rules:
    - if 'nmos' in topology_name -> use 0.65..0.85 * vdd_nominal
    - elif 'pmos' in topology_name -> use 0.15..0.35 * vdd_nominal
    - else -> use 0.10..0.90 * vdd_nominal
    """
    try:
        name = (topology_name or '').lower()
    except Exception:
        name = ''

    if 'nmos' in name:
        return 0.65 * vdd_nominal, 0.85 * vdd_nominal
    elif 'pmos' in name:
        return 0.15 * vdd_nominal, 0.35 * vdd_nominal
    else:
        return 0.10 * vdd_nominal, 0.90 * vdd_nominal