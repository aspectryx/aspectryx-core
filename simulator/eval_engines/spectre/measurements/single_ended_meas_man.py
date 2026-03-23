"""single_ended_meas_man.py

Author: natelgrw
Last Edited: 03/18/2026

Measurement manager for processing and calculating performance specs
for single-ended op-amp simulations.
"""

from simulator.eval_engines.spectre.core import EvaluationEngine
import numpy as np
from simulator import globalsy
from simulator.eval_engines.spectre.measurements.spec_functions import SpecCalc

# ===== Single-Ended Op-Amp Measurement Manager ===== #

class OpampMeasMan(EvaluationEngine):
    """
    Measurement manager for single-ended op-amp simulations.
    Supports the calculation of performance specs including:
    - Gain (OL)
    - UGBW
    - Phase Margin
    - Power Consumption
    - CMRR
    - PSRR
    - Input Offset Voltage (Vos)
    - Output Voltage Swing
    - Integrated Noise
    - Slew Rate
    - Settling Time
    - THD
    """

    def process_ac(self, results, params):
        return ACTB.process_ac(results, params)

    def __init__(self, config):
        EvaluationEngine.__init__(self, config)

    def get_specs(self, results_dict, params):
        """
        Constructs a cleaned specs dictionary from an input results dictionary.
        """
        # Flatten results if wrapped in netlist name dict (core.py behavior)
        if results_dict and isinstance(results_dict, dict):
            keys = list(results_dict.keys())
            if len(keys) == 1 and isinstance(results_dict[keys[0]], tuple):
                results_dict = results_dict[keys[0]][1]

        # Idempotency check
        if 'gain_ol' in results_dict or 'ugbw' in results_dict:
            return results_dict

        if 'ac_dc' in results_dict:
            ac_dc_tuple = results_dict['ac_dc']
            return ac_dc_tuple[1]
        else:
            return self.process_ac(results_dict, params)

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
    AC Analysis Trait Base for single-ended OpampMeasMan.
    """

    @classmethod
    def process_ac(cls, results, params):
        """
        Processes simulation results to compute single-ended op-amp performance specs.
        """
        if not isinstance(results, dict):
            return {}

        # --- Simulation result lookup ---
        ac_result_se    = (results.get('stb_ol') or results.get('acswp-000_ac') or
                           results.get('stb_loop') or results.get('ac') or results.get('stb_sim'))
        dc_results      = (results.get('dcOp_sim') or results.get('dcswp-500_dcOp') or
                           results.get('vos_sim') or results.get('dcOp') or results.get('dc'))
        noise_results   = results.get('noise_sim') or results.get('noise')
        xf_resultsdict  = results.get('xf_sim')
        thd_results     = (results.get('thd_pss.fd') or results.get('thd_pss') or
                           results.get('thd_extract') or results.get('thd_sim'))
        slew_results    = results.get('slew_large_tran') or results.get('slew_sim')
        settle_results  = (results.get('settle_small_tran') or results.get('settle_tran') or
                           results.get('settle_sim'))
        xf_cm_results   = results.get('xf_cm_sim')
        xf_psrr_results = results.get('xf_psrr')

        # --- Spec init ---
        vos = gain_ol = gain_ol_lin = ugbw = phm = None
        estimated_area = power = cmrr = psrr = output_voltage_swing = None
        integrated_noise = slew_rate = settle_time = thd = None
        valid = False

        # --- Pre-zero op-point dicts (MM0–MM6 — matches topology) ---
        num_transistors = sum(1 for k in (params or {}) if k.startswith('nB') and k[2:].isdigit())
        mm_names = [f'MM{i}' for i in range(num_transistors)]
        ids_MM    = {mm: 0.0 for mm in mm_names}
        gm_MM     = {mm: 0.0 for mm in mm_names}
        gds_MM    = {mm: 0.0 for mm in mm_names}
        vth_MM    = {mm: 0.0 for mm in mm_names}
        vdsat_MM  = {mm: 0.0 for mm in mm_names}
        vgs_MM    = {mm: 0.0 for mm in mm_names}
        vds_MM    = {mm: 0.0 for mm in mm_names}
        cgg_MM    = {mm: 0.0 for mm in mm_names}
        cgs_MM    = {mm: 0.0 for mm in mm_names}
        cdd_MM    = {mm: 0.0 for mm in mm_names}
        cgd_MM    = {mm: 0.0 for mm in mm_names}
        css_MM    = {mm: 0.0 for mm in mm_names}
        region_MM = {mm: 0.0 for mm in mm_names}

        # 1. DC Processing
        if dc_results:
            vcm = 0.0
            if params and 'vcm' in params:
                vcm = float(params['vcm'])
            elif results.get('vcm'):
                vcm = float(results['vcm'])
            else:
                vcm = dc_results.get('cm', 0.0)

            vdd_val = 0.0
            if params and 'vdd' in params:
                vdd_val = float(params['vdd'])
            elif results.get('vdd'):
                vdd_val = float(results['vdd'])

            if 'V0:p' in dc_results:
                i_supply = np.abs(dc_results['V0:p'])
                power = i_supply * vdd_val if vdd_val > 0 else i_supply

            for comp, val in dc_results.items():
                if not comp.startswith('MM'):
                    continue
                base = comp.split(':')[0]
                suffix = comp.split(':')[1] if ':' in comp else ''
                try:
                    fval = float(np.abs(val))
                except Exception:
                    fval = val
                if suffix == 'ids':         ids_MM[base]    = fval
                elif suffix == 'gm':        gm_MM[base]     = fval
                elif suffix in ('gds', 'gmds'): gds_MM[base] = fval
                elif suffix == 'vth':       vth_MM[base]    = fval
                elif suffix == 'vdsat':     vdsat_MM[base]  = fval
                elif suffix == 'vgs':       vgs_MM[base]    = fval
                elif suffix == 'vds':       vds_MM[base]    = fval
                elif suffix == 'cgg':       cgg_MM[base]    = fval
                elif suffix == 'cgs':       cgs_MM[base]    = fval
                elif suffix == 'cdd':       cdd_MM[base]    = fval
                elif suffix == 'cgd':       cgd_MM[base]    = fval
                elif suffix == 'css':       css_MM[base]    = fval
                elif suffix == 'region':    region_MM[base] = fval

            vos = SpecCalc.find_vos(results, vcm)
            output_voltage_swing = SpecCalc.find_output_voltage_swing(results, vcm)

        # 2. AC Processing (Single-Ended — Voutp / Vout)
        if ac_result_se:
            keys = list(ac_result_se.keys())
            loop_key = next((k for k in keys if 'loopGain' in k and 'dB' not in k), None)
            vout = ac_result_se[loop_key] if loop_key else None

            if vout is None:
                db_key = next((k for k in keys if 'loopGain' in k and 'dB' in k), None)
                ph_key = next((k for k in keys if 'phase' in k.lower()), None)
                if db_key and ph_key:
                    lg_mag = 10 ** (np.array(ac_result_se[db_key]) / 20.0)
                    vout = lg_mag * np.exp(1j * np.deg2rad(np.array(ac_result_se[ph_key])))

            if vout is None:
                vout = np.array(
                    ac_result_se.get('Voutp') or ac_result_se.get('Vout') or []
                )
            else:
                vout = np.array(vout)

            freq = ac_result_se.get('sweep_values')
            if freq is None:
                freq_key = next((k for k in keys if 'freq' in k.lower()), None)
                if freq_key:
                    freq = ac_result_se[freq_key]

            if len(vout) > 0 and freq is not None and len(freq) > 0:
                gain_ol_lin = SpecCalc.find_dc_gain(vout)
                ugbw, valid = SpecCalc.find_ugbw(freq, vout)
                phm = SpecCalc.find_phm(freq, vout, ugbw, valid)
            elif len(vout) > 0:
                gain_ol_lin = SpecCalc.find_dc_gain(vout)

        # 2b. Area Calculation (Technology-aware — mirrors differential)
        if params:
            estimated_area = SpecCalc.find_estimated_area(params)

        # 3. CMRR / PSRR
        if xf_cm_results and gain_ol_lin is not None:
            cmrr = SpecCalc.find_cmrr_xf(xf_cm_results, np.abs(gain_ol_lin))
        elif xf_resultsdict and gain_ol_lin is not None:
            cmrr = SpecCalc.find_cmrr_xf(xf_resultsdict, np.abs(gain_ol_lin))

        if xf_psrr_results and gain_ol_lin is not None:
            psrr = SpecCalc.find_psrr(xf_psrr_results, np.abs(gain_ol_lin))
        elif xf_resultsdict and gain_ol_lin is not None:
            psrr = SpecCalc.find_psrr(xf_resultsdict, np.abs(gain_ol_lin))

        if gain_ol_lin is not None and gain_ol_lin != 0:
            gain_ol = 20 * np.log10(np.abs(gain_ol_lin))

        # 4. Noise
        if noise_results:
            integrated_noise = SpecCalc.find_integrated_noise(noise_results)

        # 5. Transient — Single-Ended output (Voutp or Vout)
        if slew_results:
            time = slew_results.get('time', [])
            if (time is None or len(time) == 0) and 'sweep_values' in slew_results:
                time = slew_results['sweep_values']
            t_val = slew_results.get('Voutp') or slew_results.get('Vout')
            if t_val is not None and len(t_val) > 0:
                min_len = min(len(time), len(t_val))
                slew_rate = SpecCalc.find_slew_rate(
                    list(zip(time[:min_len], np.array(t_val[:min_len])))
                )

        if settle_results:
            time = settle_results.get('time', [])
            if (time is None or len(time) == 0) and 'sweep_values' in settle_results:
                time = settle_results['sweep_values']
            t_val = settle_results.get('Voutp') or settle_results.get('Vout')
            if t_val is not None and len(t_val) > 0:
                min_len = min(len(time), len(t_val))
                settle_time = SpecCalc.find_settle_time(
                    list(zip(time[:min_len], np.array(t_val[:min_len])))
                )

        # 6. THD
        if thd_results:
            thd = SpecCalc.find_thd(thd_results)

        # --- Return ---
        return dict(
            gain_ol  = gain_ol  if gain_ol  is not None else -1000.0,
            ugbw     = ugbw     if ugbw     is not None else 1.0,
            pm       = phm      if phm      is not None else -180.0,
            estimated_area = estimated_area if estimated_area is not None else 1.0,
            power    = power    if power    is not None else 1.0,
            vos      = vos      if vos      is not None else 10.0,
            cmrr     = cmrr     if cmrr     is not None else None,
            psrr     = psrr     if psrr     is not None else None,
            thd      = thd      if thd      is not None else 1000.0,
            output_voltage_swing = output_voltage_swing if output_voltage_swing is not None else 0.0,
            integrated_noise     = integrated_noise     if integrated_noise     is not None else None,
            slew_rate   = slew_rate   if (slew_rate   is not None and slew_rate   >= 0) else 0.0,
            settle_time = settle_time if (settle_time is not None and settle_time >= 0) else 1000.0,
            valid = valid,
            zregion_of_operation_MM = region_MM,
            zzids_MM    = ids_MM,
            zzvds_MM    = vds_MM,
            zzvgs_MM    = vgs_MM,
            zzgm_MM     = gm_MM,
            zzgds_MM    = gds_MM,
            zzvth_MM    = vth_MM,
            zzvdsat_MM  = vdsat_MM,
            zzcgg_MM    = cgg_MM,
            zzcgs_MM    = cgs_MM,
            zzcdd_MM    = cdd_MM,
            zzcgd_MM    = cgd_MM,
            zzcss_MM    = css_MM,
        )

