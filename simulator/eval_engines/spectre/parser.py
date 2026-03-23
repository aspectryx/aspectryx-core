"""
parser.py

Author: natelgrw
Last Edited: 01/15/2026

Spectre simulation result parser for extracting and converting PSF and CSV data
into usable Python dictionaries. Handles transient analysis data export and parsing.
"""

import sys
import os
import scipy.interpolate as interp
import scipy.optimize as sciopt
import libpsf
import fnmatch
import pdb
import IPython
import subprocess
import tempfile
import re
from contextlib import contextmanager

# ===== Context Managers ===== #

@contextmanager
def suppress_output():
    """
    Context manager to redirect stdout and stderr to /dev/null.
    [DISABLED] Useful for silencing C-level libraries (like libpsf) and verbose prints.
    """
    yield
    # with open(os.devnull, "w") as devnull:
    #     old_stdout = os.dup(sys.stdout.fileno())
    #     old_stderr = os.dup(sys.stderr.fileno())
    #     try:
    #         sys.stdout.flush()
    #         sys.stderr.flush()
    #         os.dup2(devnull.fileno(), sys.stdout.fileno())
    #         os.dup2(devnull.fileno(), sys.stderr.fileno())
    #         yield
    #     finally:
    #         os.dup2(old_stdout, sys.stdout.fileno())
    #         os.dup2(old_stderr, sys.stderr.fileno())
    #         os.close(old_stdout)
    #         os.close(old_stderr)

# ===== Constants ===== #


IGNORE_LIST = ['*.primitives', '*.subckts', 'logFile']

OCEAN_TEMPLATE_BOTH = """
resultsDir = \"%(dir_file_path)s\"
openResults(resultsDir)
selectResult('tran)
wave_1 = v("Voutp")
wave_2 = v("Voutn")
ocnPrint(?output \"%(csv_output_path_voutp)s\" wave_1 ?precision 15)
ocnPrint(?output \"%(csv_output_path_voutn)s\" wave_2 ?precision 15)
exit"""

OCEAN_TEMPLATE_SINGLE = """
resultsDir = \"%(dir_file_path)s\"
openResults(resultsDir)
selectResult('tran)
wave_1 = v("Voutp")
ocnPrint(?output \"%(csv_output_path_voutp)s\" wave_1 ?precision 15)
exit"""


# ===== Custom Exceptions ===== #


class FileNotCompatible(Exception):
    """
    Exception raised when a file is not compatible with libpsf parser.
    
    Initialization Parameters:
    --------------------------
    message : str
        Description of the compatibility issue.
    """
    def __init__(self, *args, **kwargs):

        Exception.__init__(self,  args, kwargs)


# ===== Utility Functions ===== #


def is_ignored(string):
    """
    Checks if a filename matches any pattern in the ignore list.

    Parameters:
    -----------
    string : str
        Filename to check.
    
    Returns:
    --------
    bool
        True if filename matches any ignore pattern, False otherwise.
    """
    return any([fnmatch.fnmatch(string, pattern) for pattern in IGNORE_LIST])


def ocean_export_csv(dir_file_path, csv_output_path_voutp, csv_output_path_voutn, include_voutn=True):
    """
    Runs an OCEAN script to export transient signals to CSV files.

    Executes the Cadence OCEAN tool to extract voltage waveforms from PSF results
    and convert them to CSV format for easier parsing.

    Parameters:
    -----------
    dir_file_path : str
        Path to the directory containing PSF simulation results.
    csv_output_path_voutp : str
        Output CSV file path for positive output voltage.
    csv_output_path_voutn : str
        Output CSV file path for negative output voltage.
    include_voutn : bool
        If True, exports both Voutp and Voutn. If False, exports only Voutp.
    
    Returns:
    --------
    None
    
    Raises:
    -------
    RuntimeError
        If OCEAN tool returns non-zero exit code.
    """
    template = OCEAN_TEMPLATE_BOTH if include_voutn else OCEAN_TEMPLATE_SINGLE
    
    # create temporary OCEAN script file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ocn', delete=False) as tmp_script:
        script_path = tmp_script.name
        tmp_script.write(template % {
            "dir_file_path": dir_file_path,
            "csv_output_path_voutp": csv_output_path_voutp,
            "csv_output_path_voutn": csv_output_path_voutn
        })

    try:
        # execute OCEAN tool with script
        result = subprocess.run(
            ["ocean", "-nograph", "-restore", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=100
        )
        if result.returncode != 0:
            raise RuntimeError(f"OCEAN error:\n{result.stderr}")
    finally:
        os.remove(script_path)


def parse_ocean_csv(file_path, key_name, data_dict):
    """
    Parses OCEAN CSV export file and appends data to dictionary.

    Converts time and voltage values to standard units:
    - Time: nanoseconds
    - Voltage: millivolts

    Parameters:
    -----------
    file_path : str
        Path to the OCEAN CSV output file.
    key_name : str
        Dictionary key to store the parsed data under.
    data_dict : dict
        Dictionary to append parsed data to.
    
    Returns:
    --------
    None
        Modifies data_dict in place.
    """
    # unit conversion factors for time
    unit_to_multiplier_time = {
        "s": 1e9,      # seconds → nanoseconds
        "n": 1,        # nanoseconds → nanoseconds
        "p": 1e-3,     # picoseconds → nanoseconds
        "f": 1e-6,     # femtoseconds → nanoseconds
    }

    # unit conversion factors for voltage
    unit_to_multiplier_voltage = {
        "V": 1000,     # volts → millivolts
        "m": 1,        # millivolts → millivolts
        "u": 1e-3,     # microvolts → millivolts
    }

    # parse CSV file line by line
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("time"):
                continue

            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue

            # parse time value and unit
            match_time = re.match(r"([\d\.\-Ee+]+)([a-z]*)", parts[0])
            if match_time:
                time_val = float(match_time.group(1))
                time_unit = match_time.group(2)
                time_ns = round(time_val * unit_to_multiplier_time.get(time_unit, 1), 8)

            # parse voltage value and unit
            match_volt = re.match(r"([\d\.\-Ee+]+)([a-z]*)", parts[1])
            if match_volt:
                volt_val = float(match_volt.group(1))
                volt_unit = match_volt.group(2)
                volt_mV = round(volt_val * unit_to_multiplier_voltage.get(volt_unit, 1), 8)

            data_dict.setdefault(key_name, []).append((time_ns, volt_mV))


# ===== Spectre Parser Class ===== #


class SpectreParser(object):
    """
    Parser for Spectre simulation results.

    Handles parsing of PSF (Post-Simulation Format) files and OCEAN CSV exports
    to extract simulation results into Python dictionaries.
    """

    @classmethod
    def parse(cls, raw_folder):
        """
        Parses all simulation results from a results folder.

        Processes both PSF result files and OCEAN-exported CSV files in the folder.
        Extracts signals and sweep information into a unified dictionary.

        Parameters:
        -----------
        raw_folder : str
            Path to the folder containing simulation results.
        
        Returns:
        --------
        data : dict
            Dictionary mapping result names to signal data and values.
        """
        folder_path = os.path.abspath(raw_folder)
        data = dict()
        try:
            files =  os.listdir(folder_path)
        except FileNotFoundError:
            return data
        
        # iterate through all files in results folder
        for file in files:
            if is_ignored(file):
                continue

            file_path = os.path.join(raw_folder, file)

            # handle transient result files
            if file.endswith(".tran.tran"):
                base_name = file.replace(".tran.tran", "")
                output_csv_voutp = os.path.join(folder_path, f"{base_name}_Voutp.csv")
                output_csv_voutn = os.path.join(folder_path, f"{base_name}_Voutn.csv")

                # export transient results to CSV handling single ended and differential cases
                if not os.path.exists(output_csv_voutp):
                    try:
                        # print(f"Exporting {file_path} ...")
                        try:
                            with suppress_output():
                                ocean_export_csv(file_path, output_csv_voutp, output_csv_voutn, include_voutn=True)
                            has_voutn = True
                        except RuntimeError:
                            # Fallback: Assume failure might be due to missing Voutn
                            with suppress_output():
                                ocean_export_csv(file_path, output_csv_voutp, output_csv_voutn, include_voutn=False)
                            has_voutn = False
                    except Exception as e:
                        # print(f"Failed to export {file}: {e}")
                        continue
                else:
                    has_voutn = os.path.exists(output_csv_voutn)

                # parse ocean csv data
                tran_data = [] # List of (time, voltage) tuples
                parse_ocean_csv(output_csv_voutp, "temp_key", { "temp_key": tran_data })
                
                # Convert list of tuples to dictionary of lists
                tran_dict = {}
                if tran_data:
                    times, vouts = zip(*tran_data)
                    tran_dict['time'] = list(times)
                    tran_dict['Voutp'] = list(vouts)
                else:
                    tran_dict['time'] = []
                    tran_dict['Voutp'] = []

                if has_voutn and os.path.exists(output_csv_voutn):
                     tran_data_n = []
                     parse_ocean_csv(output_csv_voutn, "temp_key", { "temp_key": tran_data_n })
                     
                     if tran_data_n:
                         _, vouts_n = zip(*tran_data_n) # assume times match
                         tran_dict['Voutn'] = list(vouts_n)
                
                data[base_name] = tran_dict

                continue

            # process PSF files
            try:
                with suppress_output():
                    datum = cls.process_file(file_path)
            except (FileNotCompatible, RuntimeError, Exception):
                continue

            # extract signal name from filename
            _, kwrd = os.path.split(file)
            # Keep .fd extension for PSS frequency-domain results so
            # thd_pss.fd and thd_pss.pss get distinct keys (avoids
            # non-deterministic overwrite from os.listdir ordering).
            if not kwrd.endswith('.fd'):
                kwrd = os.path.splitext(kwrd)[0]
            data[kwrd] = datum

        return data

    @classmethod
    def process_file(cls, file):
        """
        Processes a single PSF result file.

        Extracts signal data from PSF file using libpsf library.
        If sweep parameters exist, includes sweep variable names and values.

        Parameters:
        -----------
        file : str
            Path to the PSF file to process.
        
        Returns:
        --------
        datum : dict
            Dictionary containing signal data and sweep information if applicable.
        
        Raises:
        -------
        FileNotCompatible
            If PSF file is not compatible with libpsf parser.
        """
        fpath = os.path.abspath(file)
        try:
            psfobj = libpsf.PSFDataSet(fpath)
        except:
            raise FileNotCompatible('file {} was not compatible with libpsf'.format(file))

        is_swept = psfobj.is_swept()
        datum = dict()
        
        # extract all signal data (skip individual signals that cause libpsf C++ errors)
        for signal in psfobj.get_signal_names():
            try:
                datum[signal] = psfobj.get_signal(signal)
            except (RuntimeError, Exception):
                continue

        # add sweep parameters if present
        if is_swept:
            try:
                datum['sweep_vars'] = psfobj.get_sweep_param_names()
                datum['sweep_values'] = psfobj.get_sweep_values()
            except (RuntimeError, Exception):
                pass

        psfobj.close()
        return datum
