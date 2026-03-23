"""
differential_meas_man.py

Author: natelgrw
Last Edited: 02/06/2026

Measurement manager for processing and calculating performance specs 
for differential op-amp simulations.
"""

from simulator.eval_engines.spectre.core import EvaluationEngine
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt
import scipy.integrate as scint
from simulator import globalsy
from simulator.eval_engines.spectre.measurements.spec_functions import SpecCalc

# ===== Differential Op-Amp Measurement Manager ===== #

class OpampMeasMan(EvaluationEngine):
    """
    Measurement manager for differential op-amp simulations.
    Supports the calculation of performance specs including:
    - Gain
    - UGBW
    - Phase Margin
    - Power Consumption
    - CMRR
    - PSRR
    - Input Offset Voltage (Vos)
    - Linearity Range
    - Output Voltage Swing
    - Integrated Noise
    - Slew Rate
    - Settling Time
    - THD
    """
    
    # Inherit capability to calculate specs directly
    def process_ac(self, results, params):
        return ACTB.process_ac(results, params)

    def __init__(self, config):
        # Accept a configuration dictionary (preferred)
        EvaluationEngine.__init__(self, config)

    def get_specs(self, results_dict, params):
        """
        Constructs a cleaned specs dictionary from an input results dictionary.
        """
        # Flatten results if wrapped in netlist name dict (core.py behavior)
        if results_dict and isinstance(results_dict, dict):
             keys = list(results_dict.keys())
             # Heuristic: if key is a netlist name and value is tuple (state, specs, info)
             if len(keys) == 1 and isinstance(results_dict[keys[0]], tuple):
                  # Extract the actual specs dict from the tuple
                  results_dict = results_dict[keys[0]][1]
        
        # Check if results are already processed (Idempotency)
        if 'gain_ol' in results_dict or 'ugbw' in results_dict:
             return results_dict

        specs_dict = dict()
        # In differential scenarios, results might come differently structured 
        # depending on wrapper. Assuming standard tuple return from core.py
        if 'ac_dc' in results_dict:
             ac_dc_tuple = results_dict['ac_dc']
             specs_dict = ac_dc_tuple[1]
        else:
             # If direct dictionary passed
             return self.process_ac(results_dict, params)
             
        return specs_dict

    def compute_penalty(self, spec_nums, spec_kwrd):
        """
        Computes penalties for given spec numbers based on predefined spec ranges.
        """
        if type(spec_nums) is not list:
            spec_nums = [spec_nums]
        penalties = []
        for spec_num in spec_nums:
            penalty = 0
            if spec_kwrd in self.spec_range:
                spec_min, spec_max, w = self.spec_range[spec_kwrd]
                if spec_max is not None:
                    if spec_num > spec_max:
                        penalty += w * abs(spec_num - spec_max) / abs(spec_num)
                if spec_min is not None:
                    if spec_num < spec_min:
                        penalty += w * abs(spec_num - spec_min) / abs(spec_min)
            penalties.append(penalty)
        return penalties

# ===== AC Analysis Function Class for OpampMeasMan ===== #

class ACTB(object):
    """
    AC Analysis Trait Base for OpampMeasMan.
    """

    @classmethod
    def process_ac(self, results, params):
        """
        Processes AC analysis results to compute performance specifications.
        """
        # --- Safely Extract Results ---
        # Debugging hook for result type issues
        if not isinstance(results, dict):
             return {}

        # AC Analysis (Differential)
        # Prioritize STB loop for stability/gain analysis.
        # AC Analysis (Differential)
        # Open Loop Analysis (prioritize stb_ol)
        ac_result_ol = results.get('stb_ol') or results.get('acswp-000_ac') or results.get('stb_loop') or results.get('stb_sim')
        ac_result_cm = results.get('acswp-001_ac')
        dc_results = results.get('dcOp_sim') or results.get('dcswp-500_dcOp') or results.get('vos_sim')
        noise_results = results.get('noise_sim') or results.get('noise')
        xf_resultsdict = results.get('xf_sim')
        thd_results = results.get('thd_pss.fd') or results.get('thd_pss') or results.get('thd_extract') or results.get('thd_sim')
        slew_results = results.get('slew_large_tran') or results.get('slew_sim')
        settle_results = results.get('settle_small_tran') or results.get('settle_tran') or results.get('settle_sim')
        xf_cm_results = results.get('xf_cm_sim')
        xf_psrr_results = results.get('xf_psrr') 

        # --- Initialize Specs to None ---
        vos = None
        gain_ol = None
        gain_ol_lin = None
        ugbw = None
        phm = None
        estimated_area = None
        power = None
        cmrr = None
        psrr = None

        # linearity = None # linearity is now reported as THD
        output_voltage_swing = None
        integrated_noise = None
        slew_rate = None
        settle_time = None
        thd = None
        
        valid = False

        # --- Metric Extraction ---
        # Pre-populate all transistor op dicts with 0.0 so every row has numeric values
        num_transistors = sum(1 for k in (params or {}) if k.startswith('nB') and k[2:].isdigit())
        mm_names = [f'MM{i}' for i in range(num_transistors)]
        ids_MM = {mm: 0.0 for mm in mm_names}
        gm_MM = {mm: 0.0 for mm in mm_names}
        gds_MM = {mm: 0.0 for mm in mm_names}
        # gmbs (body effect transconductance) removed — always zero in our flows
        vth_MM = {mm: 0.0 for mm in mm_names}
        vdsat_MM = {mm: 0.0 for mm in mm_names}
        vgs_MM = {mm: 0.0 for mm in mm_names}
        vds_MM = {mm: 0.0 for mm in mm_names}
        cgg_MM = {mm: 0.0 for mm in mm_names}
        cgs_MM = {mm: 0.0 for mm in mm_names}
        cdd_MM = {mm: 0.0 for mm in mm_names}
        cgd_MM = {mm: 0.0 for mm in mm_names}
        css_MM = {mm: 0.0 for mm in mm_names}
        region_MM = {mm: 0.0 for mm in mm_names}

        # 1. DC Processing
        if dc_results:
            vcm = 0.0
            if params and 'vcm' in params:
                 vcm = float(params['vcm'])
            elif results.get('vcm'):
                 vcm = float(results['vcm'])
            else:
                 vcm = dc_results.get("cm", 0.0)
            
            # Identify VDD if possible (check params or DC results)
            vdd_val = 0.0
            if params and 'vdd' in params:
                 vdd_val = float(params['vdd'])
            elif results.get('vdd'):
                 vdd_val = float(results['vdd'])
            else:
                 # Last resort: Try to find V0 DC value if stored, or assume typical 1.0/1.8 given user context
                 # For foundation model accuracy, we relying on 'vdd' existing in params is best.
                 vdd_val = 0.0
            
            # Extract Supply Current and Compute Power
            # Standard: V0 is supply. V0:p is current.
            if 'V0:p' in dc_results:
                 i_supply = np.abs(dc_results['V0:p']) 
                 # If VDD is known, Power = V * I. 
                 if vdd_val > 0:
                      power = i_supply * vdd_val
                 else:
                      # Fallback if VDD unknown: return Current (Amps) but label effectively incorrect
                      # However, for 1000% accuracy, we need VDD.
                      # We will use the 'param' lookup which should be valid.
                      power = i_supply # Warning: This is just current if VDD=0
            
            for comp, val in dc_results.items():
                if comp.startswith("MM"):
                    base = comp.split(':')[0]
                    try:
                        if comp.endswith("ids"):
                            ids_MM[base] = float(np.abs(val))
                        elif comp.endswith("gm"):
                            gm_MM[base] = float(np.abs(val))
                        elif comp.endswith("gds") or comp.endswith("gmds"):
                            gds_MM[base] = float(np.abs(val))
                        # gmbs omitted
                        elif comp.endswith("vth"):
                            vth_MM[base] = float(np.abs(val))
                        elif comp.endswith("vdsat"):
                            vdsat_MM[base] = float(np.abs(val))
                        elif comp.endswith("vgs"):
                            vgs_MM[base] = float(np.abs(val))
                        elif comp.endswith("vds"):
                            vds_MM[base] = float(np.abs(val))
                        elif comp.endswith("cgg"):
                            cgg_MM[base] = float(np.abs(val))
                        elif comp.endswith("cgs"):
                            cgs_MM[base] = float(np.abs(val))
                        elif comp.endswith("cdd"):
                            cdd_MM[base] = float(np.abs(val))
                        elif comp.endswith("cgd"):
                            cgd_MM[base] = float(np.abs(val))
                        elif comp.endswith("css"):
                            css_MM[base] = float(np.abs(val))
                        elif comp.endswith("region"):
                            region_MM[base] = float(np.abs(val))
                    except Exception:
                        # Preserve original raw value on any unexpected type
                        if comp.endswith("ids"):
                            ids_MM[base] = val
                        elif comp.endswith("gm"):
                            gm_MM[base] = val
                        elif comp.endswith("gds") or comp.endswith("gmds"):
                            gds_MM[base] = val
                        # gmbs omitted
                        elif comp.endswith("vth"):
                            vth_MM[base] = val
                        elif comp.endswith("vdsat"):
                            vdsat_MM[base] = val
                        elif comp.endswith("vgs"):
                            vgs_MM[base] = val
                        elif comp.endswith("vds"):
                            vds_MM[base] = val
                        elif comp.endswith("cgg"):
                            cgg_MM[base] = val
                        elif comp.endswith("cgs"):
                            cgs_MM[base] = val
                        elif comp.endswith("cdd"):
                            cdd_MM[base] = val
                        elif comp.endswith("cgd"):
                            cgd_MM[base] = val
                        elif comp.endswith("css"):
                            css_MM[base] = val
                        elif comp.endswith("region"):
                            region_MM[base] = val

            vos = SpecCalc.find_vos(results)          # target=0: Voutp-Voutn crosses zero
            output_voltage_swing = SpecCalc.find_output_voltage_swing(results)
        
        # 2. AC Processing (Open Loop)
        if ac_result_ol:
            # Check for consolidated loopGain (STB analysis)
            # Search keys safely for loopGain
            keys = list(ac_result_ol.keys())
            loop_key = next((k for k in keys if "loopGain" in k and "dB" not in k), None)
            
            if loop_key:
                vout_diff = ac_result_ol[loop_key]
            else:
                vout_diff = None

            # Fallback: Check for loopGain_dB and phase if complex loopGain missing
            if vout_diff is None:
                 db_key = next((k for k in keys if "loopGain" in k and "dB" in k), None)
                 ph_key = next((k for k in keys if "phase" in k or "Phase" in k), None)
                 
                 if db_key and ph_key:
                      lg_db = np.array(ac_result_ol[db_key])
                      lg_ph = np.array(ac_result_ol[ph_key])
                      # Reconstruct complex form: 10^(dB/20) * exp(j * deg2rad(phase))
                      lg_mag = 10**(lg_db/20.0)
                      vout_diff = lg_mag * np.exp(1j * np.deg2rad(lg_ph))

            # If no loopGain, check for differential Voutp - Voutn (Only if standard AC sweep)
            if vout_diff is None:
                 if 'Voutp' in ac_result_ol and 'Voutn' in ac_result_ol:
                     vout_p = np.array(ac_result_ol['Voutp'])
                     vout_n = np.array(ac_result_ol['Voutn'])
                     vout_diff = vout_p - vout_n
                 elif 'Voutp' in ac_result_ol:
                     vout_diff = np.array(ac_result_ol['Voutp'])
                 else:
                     vout_diff = np.array([])
            else:
                 vout_diff = np.array(vout_diff) # Ensure array

            freq = ac_result_ol.get('sweep_values')
            if freq is None:
                # Try finding 'freq' search key
                freq_key = next((k for k in keys if "freq" in k.lower()), None)
                if freq_key: freq = ac_result_ol[freq_key]

            if len(vout_diff) > 0 and freq is not None and len(freq) > 0:
                gain_ol_lin = SpecCalc.find_dc_gain(vout_diff)
                ugbw, valid = SpecCalc.find_ugbw(freq, vout_diff)
                phm = SpecCalc.find_phm(freq, vout_diff, ugbw, valid)
                
                # Check Linearity using DC Gain
                # linearity = self.find_linearity(results, vout_diff)
            else:
                 # If we have Vout but no Freq, we can at least get DC gain
                 if len(vout_diff) > 0:
                      gain_ol_lin = SpecCalc.find_dc_gain(vout_diff)
                 # Add Debug info if failures persist
                 # print(f"DEBUG: Keys in ac_result: {keys}")
        
        # 2b. Area Calculation (Technology-aware estimate)
        if params:
             estimated_area = SpecCalc.find_estimated_area(params)
        
        # 3. CMRR / PSRR Processing
        # CMRR from dedicated shim or main XF
        if xf_cm_results and gain_ol_lin is not None:
            cmrr = SpecCalc.find_cmrr_xf(xf_cm_results, np.abs(gain_ol_lin), source_names=('Vsig', 'V1', 'VCM'))
        elif xf_resultsdict and gain_ol_lin is not None:
            cmrr = SpecCalc.find_cmrr_xf(xf_resultsdict, np.abs(gain_ol_lin))

        # PSRR from dedicated shim or main XF
        if xf_psrr_results and gain_ol_lin is not None:
            psrr = SpecCalc.find_psrr(xf_psrr_results, np.abs(gain_ol_lin))
        elif xf_resultsdict and gain_ol_lin is not None:
            psrr = SpecCalc.find_psrr(xf_resultsdict, np.abs(gain_ol_lin))

        
        # Convert Open Loop Gain to dB
        if gain_ol_lin is not None and gain_ol_lin != 0:
            # Take absolute value to handle inverted gain (180 phase)
            gain_ol = 20 * np.log10(np.abs(gain_ol_lin))


        # 4. Noise Processing
        if noise_results:
             integrated_noise = SpecCalc.find_integrated_noise(noise_results)

        # 5. Transient Processing (Slew / Settling)
        if slew_results:
             time = slew_results.get('time', [])
             t_val_p = slew_results.get('Voutp')
             t_val_n = slew_results.get('Voutn')
             
             if (time is None or len(time) == 0) and 'sweep_values' in slew_results:
                  time = slew_results['sweep_values']

             if t_val_p is not None and len(t_val_p) > 0:
                  if t_val_n is not None and len(t_val_n) == len(t_val_p):
                       diff_tran = np.array(t_val_p) - np.array(t_val_n)
                  else:
                       diff_tran = np.array(t_val_p) 
                  
                  min_len = min(len(time), len(diff_tran))
                  tran_data = list(zip(time[:min_len], diff_tran[:min_len]))
                  
                  slew_rate = SpecCalc.find_slew_rate(tran_data)
        
        if settle_results:
             time = settle_results.get('time', [])
             t_val_p = settle_results.get('Voutp')
             t_val_n = settle_results.get('Voutn')
             
             if (time is None or len(time) == 0) and 'sweep_values' in settle_results:
                  time = settle_results['sweep_values']

             if t_val_p is not None and len(t_val_p) > 0:
                  if t_val_n is not None and len(t_val_n) == len(t_val_p):
                       diff_tran = np.array(t_val_p) - np.array(t_val_n)
                  else:
                       diff_tran = np.array(t_val_p) 
                  
                  min_len = min(len(time), len(diff_tran))
                  tran_data = list(zip(time[:min_len], diff_tran[:min_len]))
                  
                  settle_time = SpecCalc.find_settle_time(tran_data)

        # 6. THD
        if thd_results:
            thd = SpecCalc.find_thd(thd_results)

        # Return Results (Nulls replaced with extreme penalties)
        results = dict(
            gain_ol = gain_ol if gain_ol is not None else -1000.0,
            ugbw = ugbw if ugbw is not None else 1.0,
            pm = phm if phm is not None else -180.0,
            # gain margin removed as a reported spec
            estimated_area = estimated_area if estimated_area is not None else 1.0,
            power = power if power is not None else 1.0,
            vos = vos if vos is not None else 10.0,
            cmrr = cmrr if cmrr is not None else None,
            psrr = psrr if psrr is not None else None,
            thd = thd if thd is not None else 1000.0,
            output_voltage_swing = output_voltage_swing if output_voltage_swing is not None else 0.0,
            integrated_noise = integrated_noise if integrated_noise is not None else None,
            # Map unmeasurable sentinel (-1.0) to numeric defaults for downstream consumers
            slew_rate = (slew_rate if (slew_rate is not None and slew_rate >= 0) else 0.0),
            settle_time = (settle_time if (settle_time is not None and settle_time >= 0) else 1000.0),
            valid = valid,
            zregion_of_operation_MM = region_MM,
            zzids_MM = ids_MM,
            zzvds_MM = vds_MM,
            zzvgs_MM = vgs_MM,
            zzgm_MM = gm_MM,
            zzgds_MM = gds_MM,
            # gmbs removed
            zzvth_MM = vth_MM,
            zzvdsat_MM = vdsat_MM,
            zzcgg_MM = cgg_MM,
            zzcgs_MM = cgs_MM,
            zzcdd_MM = cdd_MM,
            zzcgd_MM = cgd_MM,
            zzcss_MM = css_MM
        )
        return results

    # @classmethod
    # def find_linearity(self, results, vout_diff, allowed_deviation_pct=2.0):
    #     # Similar to Diff logic but uses single ended params
    #     gain = SpecCalc.find_dc_gain(vout_diff)
    #     # Allow linearity calculation even if gain is low/None, to report the range
    #     # if gain is None or gain < 1: return None

    #     dc_offsets, vouts = self.extract_dc_sweep(results)
    #     if len(dc_offsets) < 4: return None

    #     spline = interp.UnivariateSpline(dc_offsets, vouts, s=0)
    #     slope_spline = spline.derivative(n=1)
    #     fine_x = np.linspace(dc_offsets.min(), dc_offsets.max(), 2000)
    #     fine_slope = slope_spline(fine_x)
        
    #     # Safe finding of zero crossing for linearity center
    #     zero_idxs = np.where(np.isclose(fine_x, 0, atol=1e-3))[0]
    #     if len(zero_idxs) > 0:
    #          zero_idx = zero_idxs[0]
    #     else:
    #          zero_idx = np.argmin(np.abs(fine_x))

    #     slope_at_zero = fine_slope[zero_idx]
    #     allowed_dev = abs(slope_at_zero) * (allowed_deviation_pct / 100.0)

    #     left_idx = zero_idx
    #     while left_idx > 0 and abs(fine_slope[left_idx] - slope_at_zero) <= allowed_dev:
    #         left_idx -= 1
    #     right_idx = zero_idx
    #     while right_idx < len(fine_x) - 1 and abs(fine_slope[right_idx] - slope_at_zero) <= allowed_dev:
    #         right_idx += 1

    #     return (fine_x[left_idx], fine_x[right_idx])


        # Output swing calculation moved to SpecCalc in spec_functions.py
    # Redundant methods removed/replaced by SpecCalc calls:
    # find_dc_gain, find_ugbw, find_phm, find_integrated_noise, find_slew_rate, find_settle_time, _get_best_crossing
    # find_psrr, find_cmrr_xf - moved up


