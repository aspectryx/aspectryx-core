"""
spec_functions.py

Author: natelgrw
Last Edited: 02/06/2026

Shared utility functions for calculating op-amp specifications
from simulation results. Used by both SingleEnded and Differential
measurement managers.
"""

import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt
import scipy.integrate as scint


class SpecCalc(object):
    """
    Collection of static methods for specification calculation.
    """

    @staticmethod
    def find_dc_gain(vout):
        """
        Finds the DC gain from output voltage array (index 0).
        """
        if vout is None or len(vout) == 0:
            return None
        return float(np.abs(vout)[0])

    @staticmethod
    def _get_best_crossing(xvec, yvec, val):
        """
        Finds the best crossing point where yvec crosses val.
        Returns (crossing_x, valid_bool)
        """
        if len(xvec) < 2: 
             return None, False
             
        try:
            interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)
            def fzero(x): return interp_fun(x) - val
            
            # Check if crossing is possible in range
            y_min, y_max = np.min(yvec), np.max(yvec)
            if val < y_min or val > y_max:
                return None, False

            return sciopt.brentq(fzero, xvec[0], xvec[-1]), True
        except:
            return None, False

    @staticmethod
    def find_ugbw(freq, vout):
        """
        Finds Unity Gain Bandwidth using Log-Log interpolation for high accuracy.
        Assumes gain decreases monotonically near 0dB.
        """
        if freq is None or vout is None or len(freq) != len(vout): 
            return None, False
            
        gain = np.abs(vout)
        
        # 1. Find the first time gain crosses 1.0 (0dB) from above
        # Valid UGBW requires gain starting > 1
        if gain[0] < 1.0:
            return None, False
            
        # Find indices where gain transitions from >=1 to <1
        # This approach avoids oscillating spline solutions
        crossings = np.where((gain[:-1] >= 1.0) & (gain[1:] < 1.0))[0]
        
        if len(crossings) == 0:
            return None, False
            
        # Take the first crossing (standard definition)
        idx = crossings[0]
        
        # 2. Log-Log Interpolation
        # log(gain) vs log(freq) is linear for dominant pole systems
        # y = log10(gain), x = log10(freq)
        # We want x where y = log10(1) = 0
        
        f1, f2 = freq[idx], freq[idx+1]
        g1, g2 = gain[idx], gain[idx+1]
        
        # Avoid log(0)
        if f1 <= 0 or f2 <= 0 or g1 <= 0 or g2 <= 0:
            return None, False
            
        x1, x2 = np.log10(f1), np.log10(f2)
        y1, y2 = np.log10(g1), np.log10(g2)
        
        if y1 == y2: # Horizontal segment?
             return None, False
             
        # Interpolate for y=0
        # (0 - y1) = (y2 - y1) / (x2 - x1) * (x_target - x1)
        # x_target = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
        
        x_target = x1 - y1 * (x2 - x1) / (y2 - y1)
        ugbw = 10**x_target
        
        return float(ugbw), True

    @staticmethod
    def find_phm(freq, vout, ugbw, valid_ugbw):
        """
        Finds Phase Margin at UGBW.
        Dynamically handles probe polarity by calculating relative phase degradation.
        """
        if not valid_ugbw or freq is None or vout is None or ugbw is None:
            return None
            
        phase = np.angle(vout, deg=True)
        # 1. Unwrap Phase safely
        phase_rad = np.deg2rad(phase)
        phase_unwrapped = np.rad2deg(np.unwrap(phase_rad))
        
        # 2. Linear Interpolation in Log-Freq domain
        log_freq = np.log10(np.clip(freq, 1e-12, None))
        log_ugbw = np.log10(ugbw)
        
        try:
            phase_fun = interp.interp1d(log_freq, phase_unwrapped, kind='linear', fill_value="extrapolate")
            phase_at_ugbw = float(phase_fun(log_ugbw))
            
            # 3. Calculate True Phase Margin
            starting_phase = phase_unwrapped[0]
            
            # Normalize starting phase to either 180 or 0
            if starting_phase > 90:
                dc_phase = 180.0
            elif starting_phase < -90:
                dc_phase = -180.0
            else:
                dc_phase = 0.0
                
            # How much did the phase drop from DC to UGBW?
            phase_drop = dc_phase - phase_at_ugbw
            
            # Phase margin is how far we are from a 180 degree drop
            pm = 180.0 - phase_drop
                
            return float(pm)
        except:
            return None


    @staticmethod
    def find_integrated_noise(noise_results, f_start=1e6, f_stop=5e8, num_points=55):
        """
        Integrates total output noise safely.
        """
        if noise_results is None or len(noise_results) == 0:
             return None

        # Case 1: Total Output Noise
        preferred_out_keys = ('out', 'Vout', 'vout', 'out_diff', 'Vout_diff')
        out_like_keys = [k for k in noise_results.keys() if 'out' in str(k).lower()]
        target_key = next((k for k in preferred_out_keys if k in noise_results), None)
        if target_key is None and len(out_like_keys) == 1:
            target_key = out_like_keys[0]

        if target_key:
             # SPECTRE CHECK: 'out' is usually V/sqrt(Hz) (Amplitude Spectral Density)
             # We must square it to get V^2/Hz for integration.
             noise_asd = np.ravel(np.abs(np.asarray(noise_results[target_key], dtype=float)))
             freqs = np.ravel(np.asarray(noise_results.get('sweep_values', []), dtype=float))
             n = min(len(freqs), len(noise_asd))
             if n > 1:
                  # simps expects y, x. Result is in V^2
                  total_power = scint.simps(noise_asd[:n]**2, freqs[:n])
                  return float(np.sqrt(total_power)) # Result in V_rms

        # Case 2: Component Noise (MM keys)
        # Note: Component noise is often already exported as PSD (V^2/Hz)
        total_integrated_v2 = 0.0
        mm_keys = [k for k in noise_results.keys() if str(k).startswith("MM")]
        
        if mm_keys:
            freqs = np.ravel(np.asarray(noise_results.get('sweep_values', []), dtype=float))
            if len(freqs) <= 1:
                freqs = np.logspace(np.log10(f_start), np.log10(f_stop), num_points)
            for key in mm_keys:
                # If these are V^2/Hz, do NOT square them again.
                try:
                    psd_vals = np.ravel(np.asarray(noise_results[key], dtype=float))
                except Exception:
                    continue
                n = min(len(psd_vals), len(freqs))
                if n > 1:
                    total_integrated_v2 += scint.simps(np.maximum(psd_vals[:n], 0.0), freqs[:n])
            if total_integrated_v2 > 0.0:
                return float(np.sqrt(total_integrated_v2))

        return None

    @staticmethod
    def find_slew_rate(tran_data, delay=5.0, lo_pct=0.1, hi_pct=0.9, min_swing=10.0):
        """
        Calculates Slew Rate in V/us, assuming input is in [ns] and [mV].
        Note: 1 mV/ns is exactly 1 V/us.
        - delay=5.0: Matches 5ns step in the .scs
        - min_swing=10.0: Requires at least 10mV of movement
        """
        if not tran_data or len(tran_data) < 5:
            return 0.0

        time = np.array([t for t, _ in tran_data])
        vout = np.array([v for _, v in tran_data])

        # Isolate post-step region (ns)
        mask = time >= delay
        if np.sum(mask) < 5: return 0.0
        t_s, v_s = time[mask], vout[mask]

        # Calculate achieved swing in mV
        v_init = v_s[0]
        v_final = np.mean(v_s[-max(2, int(len(v_s)*0.05)):])
        delta_v = v_final - v_init
        
        if np.abs(delta_v) < min_swing:
            return 0.0 # Circuit didn't move

        # Thresholds (mV)
        v10 = v_init + lo_pct * delta_v
        v90 = v_init + hi_pct * delta_v

        # Build a shape-preserving interpolator for sub-time-step crossings.
        try:
            v_interp = interp.PchipInterpolator(t_s, v_s, extrapolate=False)
        except Exception:
            return 0.0

        def _first_crossing_time(threshold, rising):
            if rising:
                bracket = np.where((v_s[:-1] < threshold) & (v_s[1:] >= threshold))[0]
            else:
                bracket = np.where((v_s[:-1] > threshold) & (v_s[1:] <= threshold))[0]

            if len(bracket) == 0:
                idx = np.where(v_s >= threshold)[0] if rising else np.where(v_s <= threshold)[0]
                if len(idx) == 0:
                    return None
                return float(t_s[int(idx[0])])

            i = int(bracket[0])
            t0, t1 = float(t_s[i]), float(t_s[i + 1])

            # Primary: root on PCHIP within bracket.
            try:
                return float(sciopt.brentq(lambda tt: float(v_interp(tt) - threshold), t0, t1))
            except Exception:
                # Fallback: linear interpolation between bracket endpoints.
                v0, v1 = float(v_s[i]), float(v_s[i + 1])
                if np.isclose(v1, v0):
                    return t1
                return float(t0 + (t1 - t0) * (threshold - v0) / (v1 - v0))

        is_rising = bool(delta_v > 0)
        t10 = _first_crossing_time(v10, rising=is_rising)
        t90 = _first_crossing_time(v90, rising=is_rising)
        if t10 is None or t90 is None:
            return 0.0

        # Ensure we have a valid time delta (ns)
        dt = float(t90 - t10)
        dv = float(v90 - v10)
        if dt < 1e-12:
            return 0.0

        # Math: (mV / ns) = (V*1e-3 / s*1e-9) = V/s * 1e6
        # Dividing V/s by 1e6 gives V/us.
        # Therefore: (mV / ns) == (V / us)
        slew_rate_v_us = np.abs(dv / dt)
        
        return float(slew_rate_v_us)

    @staticmethod
    def find_settle_time(tran_data, tol=0.01, delay=5.0, t_stop=200.0, noise_floor_mv=0.5):
        """
        Calculates 1% Settling Time for automated ML dataset extraction.
        
        Args:
            tran_data: List of tuples (time_ns, vout_mV).
            tol: Fractional tolerance band (default 0.01 for 1%).
            delay: Time in ns when the step pulse begins.
            t_stop: Total simulation time in ns.
            noise_floor_mv: Minimum absolute tolerance band to prevent 
                            infinite settling times on microscopic steps.
                            
        Returns:
            settle_time_ns (float): Time taken to settle within the band.
                                    Returns t_stop (penalty) if it never settles.
        """
        # 1. Catch empty or crashed simulations
        if not tran_data or len(tran_data) < 10:
            return float(t_stop)

        # 2. Convert to fast NumPy arrays
        data = np.array(tran_data)
        t = data[:, 0]
        v = data[:, 1]

        # Isolate the post-step response
        post_step_mask = t >= delay
        if np.sum(post_step_mask) < 10:
            return float(t_stop)

        t_post = t[post_step_mask]
        v_post = v[post_step_mask]

        # 3. Establish Baseline & Target (Robust to high Vos)
        # V_initial: Mean of the points immediately preceding the step
        pre_step_mask = (t < delay) & (t > delay - 1.0) 
        v_initial = np.mean(v[pre_step_mask]) if np.sum(pre_step_mask) > 0 else v_post[0]
        
        # V_final: Mean of the last 10% of the window (filters high-frequency ringing)
        tail_idx = max(5, int(len(v_post) * 0.1))
        v_final = np.mean(v_post[-tail_idx:])

        # 4. Calculate True Step Size & Dynamic Band
        step_size = np.abs(v_final - v_initial)
        
        # If the amplifier didn't move (dead design), it fails to settle
        if step_size < noise_floor_mv:
            return float(t_stop)

        # Tolerance band is 1% of the TRUE movement, strictly bounded by the noise floor
        band = max(step_size * tol, noise_floor_mv)
        
        # 5. Vectorized Error Calculation (Massively faster than a Python for-loop)
        err = np.abs(v_post - v_final)
        
        # Find all indices where the error exceeds the tolerance band
        unsettled_indices = np.where(err > band)[0]
        
        if len(unsettled_indices) == 0:
            # Settled instantly
            return 0.0
            
        # The last time the signal was outside the band
        last_unsettled_idx = unsettled_indices[-1]
        
        # If the last unsettled point is at the very end of the simulation, it never settled
        if last_unsettled_idx >= len(t_post) - 2:
            return float(t_stop)
            
        # Refine settle boundary with a shape-preserving interpolator.
        i0 = int(last_unsettled_idx)
        i1 = int(last_unsettled_idx + 1)
        t0, t1 = float(t_post[i0]), float(t_post[i1])
        e0 = float(err[i0] - band)
        e1 = float(err[i1] - band)

        t_settled = t1
        try:
            v_interp = interp.PchipInterpolator(t_post, v_post, extrapolate=False)

            # Root where |v(t) - v_final| - band = 0 in the last unsettled bracket.
            def ferr(tt):
                return abs(float(v_interp(tt)) - float(v_final)) - float(band)

            if e0 > 0.0 and e1 <= 0.0:
                t_settled = float(sciopt.brentq(ferr, t0, t1))
        except Exception:
            # Fallback: linear interpolation on the error crossing.
            if e0 > 0.0 and e1 <= 0.0 and not np.isclose(e0, e1):
                t_settled = float(t0 + (t1 - t0) * (-e0) / (e1 - e0))
        
        return float(max(0.0, t_settled - delay))

    @staticmethod
    def extract_dc_sweep(results):
        """
        Extract DC sweep data as sorted 1D arrays (dc_offset, output).

        Supports consolidated sweeps (`swing_sweep`, `dc_swing`) and
        split sweeps (`dcswp-*`). Differential output is preferred when
        both `Voutp` and `Voutn` are available.
        """
        def _to_1d_float(vec):
            arr = np.asarray(vec)
            if arr.ndim > 1:
                arr = np.ravel(arr)
            return arr.astype(float)

        def _sanitize_xy(x, y):
            try:
                x = _to_1d_float(x)
                y = _to_1d_float(y)
            except Exception:
                return np.array([]), np.array([])

            n = min(len(x), len(y))
            if n < 2:
                return np.array([]), np.array([])
            x = x[:n]
            y = y[:n]

            finite = np.isfinite(x) & np.isfinite(y)
            x = x[finite]
            y = y[finite]
            if len(x) < 2:
                return np.array([]), np.array([])

            order = np.argsort(x)
            x = x[order]
            y = y[order]

            # Merge duplicate x-values by averaging y-values.
            uniq_x, inv = np.unique(x, return_inverse=True)
            if len(uniq_x) != len(x):
                y_accum = np.bincount(inv, weights=y)
                y_count = np.bincount(inv)
                y = y_accum / np.maximum(y_count, 1)
                x = uniq_x

            return x, y

        if not isinstance(results, dict):
            return np.array([]), np.array([])

        # 1) Consolidated sweep blocks
        for swing_key in ('swing_sweep', 'dc_swing'):
            if swing_key not in results:
                continue
            swing_res = results[swing_key]
            if not isinstance(swing_res, dict) or 'sweep_values' not in swing_res:
                continue

            x_raw = swing_res['sweep_values']
            if 'Voutp' in swing_res and 'Voutn' in swing_res:
                y_raw = np.asarray(swing_res['Voutp']) - np.asarray(swing_res['Voutn'])
            elif 'Vout' in swing_res:
                y_raw = swing_res['Vout']
            elif 'Voutp' in swing_res:
                y_raw = swing_res['Voutp']
            else:
                continue

            x, y = _sanitize_xy(x_raw, y_raw)
            if len(x) >= 2:
                return x, y

        # 2) Split dcswp-* blocks
        dc_offsets = []
        vouts = []
        sweep_keys = [k for k in results.keys() if k.startswith('dcswp-')]
        sweep_keys.sort()

        for k in sweep_keys:
            res_dict = results.get(k)
            if not isinstance(res_dict, dict):
                continue

            if 'Voutp' in res_dict and 'Voutn' in res_dict:
                y_val = np.asarray(res_dict['Voutp']) - np.asarray(res_dict['Voutn'])
            elif 'Vout' in res_dict:
                y_val = np.asarray(res_dict['Vout'])
            elif 'Voutp' in res_dict:
                y_val = np.asarray(res_dict['Voutp'])
            else:
                continue

            y_flat = np.ravel(y_val)
            if len(y_flat) == 0:
                continue
            y_scalar = float(np.real(y_flat[0]))

            x_scalar = None
            if 'dc_offset' in res_dict:
                try:
                    x_scalar = float(np.ravel(res_dict['dc_offset'])[0])
                except Exception:
                    x_scalar = None
            if x_scalar is None and 'sweep_values' in res_dict:
                try:
                    x_scalar = float(np.ravel(res_dict['sweep_values'])[0])
                except Exception:
                    x_scalar = None
            if x_scalar is None:
                # Last-resort fallback keeps legacy behavior for existing datasets.
                try:
                    parts = k.split('_')[0].split('-')
                    num = int(parts[1]) if len(parts) > 1 else None
                    if num is not None:
                        x_scalar = -0.1 + (num * 0.001)
                except Exception:
                    x_scalar = None

            if x_scalar is not None:
                dc_offsets.append(x_scalar)
                vouts.append(y_scalar)

        if len(dc_offsets) >= 2:
            x, y = _sanitize_xy(dc_offsets, vouts)
            if len(x) >= 2:
                return x, y

        return np.array([]), np.array([])

    @staticmethod
    def find_estimated_area(params):
        """
        Estimate total on-chip area (um^2) from sizing parameters.

        This is a first-order proxy combining:
        - transistor footprint from gate length/fins and node pitch,
        - capacitor area from capacitance density,
        - resistor area from area-per-ohm coefficient.
        """
        if not isinstance(params, dict) or len(params) == 0:
            return None

        node_map = {
            7:  {'cpp': 54e-9,  'pitch': 22e-9, 'cap_density': 25e-15 / 1e-12, 'res_coeff': (50e-9)**2  / 600},
            10: {'cpp': 66e-9,  'pitch': 28e-9, 'cap_density': 18e-15 / 1e-12, 'res_coeff': (80e-9)**2  / 400},
            14: {'cpp': 78e-9,  'pitch': 32e-9, 'cap_density': 12e-15 / 1e-12, 'res_coeff': (100e-9)**2 / 250},
            16: {'cpp': 90e-9,  'pitch': 42e-9, 'cap_density': 10e-15 / 1e-12, 'res_coeff': (130e-9)**2 / 200},
            20: {'cpp': 110e-9, 'pitch': 60e-9, 'cap_density':  8e-15 / 1e-12, 'res_coeff': (200e-9)**2 / 150},
        }
        scale_factor = 3.0

        try:
            tech_node = int(params.get('fet_num', 10))
        except Exception:
            tech_node = 10
        tech = node_map.get(tech_node, node_map[10])

        area_m2 = 0.0

        # Active device footprint.
        for key, val in params.items():
            if not key.startswith('nA'):
                continue
            width_key = 'nB' + key[2:]
            if width_key not in params:
                continue
            try:
                l_eff = max(float(val), tech['cpp'])
                width = float(params[width_key]) * tech['pitch']
                area_m2 += l_eff * width
            except Exception:
                continue

        area_m2 *= scale_factor

        # Passive devices.
        cap_density = tech['cap_density']
        res_area_coeff = tech['res_coeff']
        for key, val in params.items():
            if key.startswith('nC') and val > 0:
                try:
                    area_m2 += float(val) / cap_density
                except Exception:
                    pass
            elif key.startswith('nR') and val > 0:
                try:
                    area_m2 += float(val) * res_area_coeff
                except Exception:
                    pass

        return float(area_m2 * 1e12)

    @staticmethod
    def find_vos(results, vcm=None):
        """
        Find input offset (Vos) from DC sweep by locating where the output
        crosses the target level.
        - Differential: target = 0  (Voutp - Voutn should cross zero)
        - Single-ended: target = vcm (Vout should cross vcm at the balance point)
        Returns Vos (float, the dc_offset input value) or None.
        """
        dc_offsets, vouts = SpecCalc.extract_dc_sweep(results)
        if len(dc_offsets) < 4:
            return None

        target = float(vcm) if vcm is not None else 0.0
        try:
            y_shift = np.asarray(vouts) - target

            # Exact/near-exact crossing on sampled points.
            best_idx = int(np.argmin(np.abs(y_shift)))
            if np.isclose(y_shift[best_idx], 0.0, atol=1e-12):
                return float(dc_offsets[best_idx])

            # Build all sign-change brackets, then pick the one nearest dc_offset=0.
            bracket_idxs = np.where(y_shift[:-1] * y_shift[1:] < 0)[0]
            if len(bracket_idxs) == 0:
                return None

            mids = 0.5 * (dc_offsets[bracket_idxs] + dc_offsets[bracket_idxs + 1])
            chosen = int(bracket_idxs[int(np.argmin(np.abs(mids)))])
            x_lo, x_hi = float(dc_offsets[chosen]), float(dc_offsets[chosen + 1])

            pchip = interp.PchipInterpolator(dc_offsets, y_shift, extrapolate=False)
            return float(sciopt.brentq(lambda x: float(pchip(x)), x_lo, x_hi))
        except Exception:
            # Robust fallback: linear interpolation on nearest bracket.
            try:
                y_shift = np.asarray(vouts) - target
                bracket_idxs = np.where(y_shift[:-1] * y_shift[1:] < 0)[0]
                if len(bracket_idxs) == 0:
                    return None
                mids = 0.5 * (dc_offsets[bracket_idxs] + dc_offsets[bracket_idxs + 1])
                chosen = int(bracket_idxs[int(np.argmin(np.abs(mids)))])
                x1, x2 = float(dc_offsets[chosen]), float(dc_offsets[chosen + 1])
                y1, y2 = float(y_shift[chosen]), float(y_shift[chosen + 1])
                if np.isclose(y2, y1):
                    return None
                return float(x1 + (x2 - x1) * (-y1) / (y2 - y1))
            except Exception:
                return None

    @staticmethod
    def find_output_voltage_swing(results, vcm=None, allowed_deviation_pct=10.0):
        dc_offsets, vouts = SpecCalc.extract_dc_sweep(results)
        if len(dc_offsets) < 6:
            return None

        try:
            x = np.asarray(dc_offsets, dtype=float)
            y = np.asarray(vouts, dtype=float)
            pchip = interp.PchipInterpolator(x, y, extrapolate=False)
            slope = pchip.derivative()

            # Anchor at the measured operating point when available.
            x_center = SpecCalc.find_vos(results, vcm=vcm)
            if x_center is None or x_center < x[0] or x_center > x[-1]:
                x_center = float(x[int(np.argmin(np.abs(x)))])

            slope_center = float(slope(x_center))
            tol = abs(slope_center) * (float(allowed_deviation_pct) / 100.0)
            if tol <= 0:
                grad = np.gradient(y, x)
                tol = max(1e-12, 0.1 * np.max(np.abs(grad)))

            # Dense sampling keeps bounds stable against sweep step size.
            x_fine = np.linspace(x[0], x[-1], max(2000, len(x) * 20))
            slope_fine = slope(x_fine)
            err = np.abs(slope_fine - slope_center) - tol
            in_band = err <= 0.0

            center_idx = int(np.argmin(np.abs(x_fine - x_center)))
            if not in_band[center_idx]:
                return 0.0

            left_idx = center_idx
            while left_idx > 0 and in_band[left_idx - 1]:
                left_idx -= 1
            right_idx = center_idx
            while right_idx < len(x_fine) - 1 and in_band[right_idx + 1]:
                right_idx += 1

            # Refine each boundary by linear interpolation on err=0 crossing.
            x_left = x_fine[left_idx]
            if left_idx > 0:
                e0, e1 = err[left_idx - 1], err[left_idx]
                x0, x1 = x_fine[left_idx - 1], x_fine[left_idx]
                if e0 > 0 and e1 <= 0 and not np.isclose(e0, e1):
                    x_left = x0 + (x1 - x0) * (-e0) / (e1 - e0)

            x_right = x_fine[right_idx]
            if right_idx < len(x_fine) - 1:
                e0, e1 = err[right_idx], err[right_idx + 1]
                x0, x1 = x_fine[right_idx], x_fine[right_idx + 1]
                if e0 <= 0 and e1 > 0 and not np.isclose(e0, e1):
                    x_right = x0 + (x1 - x0) * (-e0) / (e1 - e0)

            y_left = float(pchip(x_left))
            y_right = float(pchip(x_right))
            return float(max(0.0, abs(y_right - y_left)))
        except Exception:
            return None

    @staticmethod
    def _find_ugbw_freq_idx(xf_results):
        """
        Find the frequency index where the main AC response crosses 0dB (gain = 1.0).
        Uses linear interpolation to find the exact crossing point.
        Returns fractional index (float), or 0 (fallback to DC) if not found.
        """
        if not isinstance(xf_results, dict):
            return 0.0

        # Extract main AC gain (differential output or single-ended)
        gain_response = None
        for key in ('Vout', 'Voutp', 'out', 'V0'):
            if key in xf_results:
                candidate = xf_results[key]
                if not isinstance(candidate, dict) and hasattr(candidate, '__len__'):
                    if len(candidate) > 4:  # Sanity check
                        gain_response = candidate
                        break

        if gain_response is None:
            return 0.0  # Fallback to DC

        try:
            # Convert to magnitude array and find 0dB crossing
            gain_mag = np.abs(np.asarray(gain_response, dtype=complex))
            # Find first crossing where gain drops from >= 1.0 to < 1.0
            crossings = np.where((gain_mag[:-1] >= 1.0) & (gain_mag[1:] < 1.0))[0]
            if len(crossings) == 0:
                return 0.0
            
            i = int(crossings[0])
            g0, g1 = float(gain_mag[i]), float(gain_mag[i + 1])
            
            # Linear interpolation: find fractional position where gain = 1.0
            # frac = 0 means at index i, frac = 1 means at index i+1
            if not np.isclose(g0, g1):
                frac = (1.0 - g0) / (g1 - g0)
                # Clamp to [0, 1] to avoid extrapolation
                frac = max(0.0, min(1.0, frac))
                return float(i + frac)
            else:
                return float(i)
        except Exception:
            return 0.0

    @staticmethod
    def find_psrr(xf_results, dc_gain_lin):
        if dc_gain_lin is None or dc_gain_lin == 0 or xf_results is None:
            return None

        def _find_source_tf(xf, source_name):
            # Direct mapping
            if source_name in xf:
                return xf[source_name]
            # Nested mapping
            for v in xf.values():
                if isinstance(v, dict) and source_name in v:
                    return v[source_name]
            return None

        tf_vdd = _find_source_tf(xf_results, 'V0')
        if tf_vdd is None:
            # Try common alternative names
            tf_vdd = _find_source_tf(xf_results, 'VDD')

        if tf_vdd is not None:
            try:
                # Convert transfer function to array
                tf_vdd_arr = np.asarray(tf_vdd, dtype=complex)
                if len(tf_vdd_arr) == 0:
                    return None

                # Get the main AC gain array to find UGBW point
                gain_response = None
                for key in ('Vout', 'Voutp', 'out', 'V0'):
                    if key in xf_results and not isinstance(xf_results[key], dict):
                        candidate = xf_results[key]
                        if len(candidate) == len(tf_vdd_arr):
                            gain_response = candidate
                            break

                # Evaluate at UGBW if we can find it, otherwise fallback to DC
                eval_idx = 0.0
                gain_at_ugbw = dc_gain_lin
                
                if gain_response is not None:
                    try:
                        gain_mag = np.abs(np.asarray(gain_response, dtype=complex))
                        crossings = np.where((gain_mag[:-1] >= 1.0) & (gain_mag[1:] < 1.0))[0]
                        if len(crossings) > 0:
                            i = int(crossings[0])
                            g0, g1 = float(gain_mag[i]), float(gain_mag[i + 1])
                            
                            # Linear interpolation for fractional index
                            if not np.isclose(g0, g1):
                                frac = (1.0 - g0) / (g1 - g0)
                                frac = max(0.0, min(1.0, frac))
                                eval_idx = i + frac
                            else:
                                eval_idx = float(i)
                            
                            # Interpolate gain at exact crossing point
                            # For uniformly-sampled frequency sweep, linear interp is appropriate
                            gain_at_ugbw = 1.0  # At crossing point, gain = 1.0 by definition
                        else:
                            gain_at_ugbw = float(np.abs(gain_response[0]))
                    except Exception:
                        gain_at_ugbw = dc_gain_lin

                # Round fractional index to nearest integer for array access
                eval_idx_int = int(np.round(eval_idx))
                # Ensure index is valid
                if eval_idx_int >= len(tf_vdd_arr):
                    eval_idx_int = 0

                # Interpolate transfer function at fractional index if available
                if eval_idx > 0 and eval_idx < len(tf_vdd_arr) - 1 and not np.isclose(eval_idx, np.round(eval_idx)):
                    # Linear interpolation for fractional index
                    i_lo = int(np.floor(eval_idx))
                    i_hi = int(np.ceil(eval_idx))
                    frac = eval_idx - i_lo
                    tf_vdd_val = float(np.abs(
                        (1.0 - frac) * tf_vdd_arr[i_lo] + frac * tf_vdd_arr[i_hi]
                    ))
                else:
                    tf_vdd_val = float(np.abs(tf_vdd_arr[eval_idx_int]))

                if tf_vdd_val > 1e-12:
                    return float(20 * np.log10(gain_at_ugbw / tf_vdd_val))
                else:
                    return 0.0
            except Exception:
                return None
        return None

    @staticmethod
    def find_cmrr_xf(xf_results, dc_gain_lin, source_names=None):
        if dc_gain_lin is None or dc_gain_lin == 0 or xf_results is None:
            return None

        def _find_source_tf(xf, source_name):
            if source_name in xf:
                return xf[source_name]
            for v in xf.values():
                if isinstance(v, dict) and source_name in v:
                    return v[source_name]
            return None

        if source_names is None:
            source_names = ('V1', 'VCM')

        tf_cm = None
        for source_name in source_names:
            tf_cm = _find_source_tf(xf_results, source_name)
            if tf_cm is not None:
                break

        if tf_cm is not None:
            try:
                # Convert transfer function to array
                tf_cm_arr = np.asarray(tf_cm, dtype=complex)
                if len(tf_cm_arr) == 0:
                    return None

                # Get the main AC gain array to find UGBW point
                gain_response = None
                for key in ('Vout', 'Voutp', 'out', 'V0'):
                    if key in xf_results and not isinstance(xf_results[key], dict):
                        candidate = xf_results[key]
                        if len(candidate) == len(tf_cm_arr):
                            gain_response = candidate
                            break

                # Evaluate at UGBW if we can find it, otherwise fallback to DC
                eval_idx = 0.0
                gain_at_ugbw = dc_gain_lin
                
                if gain_response is not None:
                    try:
                        gain_mag = np.abs(np.asarray(gain_response, dtype=complex))
                        crossings = np.where((gain_mag[:-1] >= 1.0) & (gain_mag[1:] < 1.0))[0]
                        if len(crossings) > 0:
                            i = int(crossings[0])
                            g0, g1 = float(gain_mag[i]), float(gain_mag[i + 1])
                            
                            # Linear interpolation for fractional index
                            if not np.isclose(g0, g1):
                                frac = (1.0 - g0) / (g1 - g0)
                                frac = max(0.0, min(1.0, frac))
                                eval_idx = i + frac
                            else:
                                eval_idx = float(i)
                            
                            # Interpolate gain at exact crossing point
                            # For uniformly-sampled frequency sweep, at crossing point gain = 1.0
                            gain_at_ugbw = 1.0  # At crossing point, gain = 1.0 by definition
                        else:
                            gain_at_ugbw = float(np.abs(gain_response[0]))
                    except Exception:
                        gain_at_ugbw = dc_gain_lin

                # Round fractional index to nearest integer for array access
                eval_idx_int = int(np.round(eval_idx))
                # Ensure index is valid
                if eval_idx_int >= len(tf_cm_arr):
                    eval_idx_int = 0

                # Interpolate transfer function at fractional index if available
                if eval_idx > 0 and eval_idx < len(tf_cm_arr) - 1 and not np.isclose(eval_idx, np.round(eval_idx)):
                    # Linear interpolation for fractional index
                    i_lo = int(np.floor(eval_idx))
                    i_hi = int(np.ceil(eval_idx))
                    frac = eval_idx - i_lo
                    tf_cm_val = float(np.abs(
                        (1.0 - frac) * tf_cm_arr[i_lo] + frac * tf_cm_arr[i_hi]
                    ))
                else:
                    tf_cm_val = float(np.abs(tf_cm_arr[eval_idx_int]))

                if tf_cm_val > 1e-12:
                    return float(20 * np.log10(gain_at_ugbw / tf_cm_val))
                else:
                    return 0.0
            except Exception:
                return None
        return None

    @staticmethod
    def find_thd(thd_results):
        if thd_results is None:
            return None
        sweep_vars = thd_results.get('sweep_vars', [])
        sweep_values = thd_results.get('sweep_values', [])
        is_pss = False
        if 'harmonic' in sweep_vars or 'freq' in sweep_vars:
            is_pss = True
        elif len(sweep_values) > 1 and sweep_values[1] > 1e6:
            is_pss = True

        # Choose differential or single-ended signal
        v_p = thd_results.get('Voutp')
        v_n = thd_results.get('Voutn')
        v_sig = thd_results.get('Vout')
        if v_p is not None:
            v_p = np.array(v_p)
            if v_n is not None and len(v_n) == len(v_p):
                v_sig = v_p - np.array(v_n)
            else:
                v_sig = v_p
        elif v_sig is not None:
            v_sig = np.array(v_sig)
        else:
            return None

        if is_pss:
            v_mag = np.abs(v_sig)
            freqs = np.array(sweep_values)
            v_mag[0] = 0.0
            fund_idx = np.argmax(v_mag)
            fund_mag = v_mag[fund_idx]
            fund_freq = freqs[fund_idx]
            if fund_mag < 1e-12:
                return None
            harm_power = 0.0
            for i in range(2, 11):
                target_freq = fund_freq * i
                matches = np.where(np.isclose(freqs, target_freq, rtol=1e-3))[0]
                if len(matches) > 0:
                    harm_power += v_mag[matches[0]]**2
        else:
            time = thd_results.get('time', [])
            if (time is None or len(time) == 0) and 'sweep_values' in thd_results:
                time = thd_results['sweep_values']
            time = np.array(time)
            if len(time) != len(v_sig):
                min_len = min(len(time), len(v_sig))
                time = time[:min_len]
                v_sig = v_sig[:min_len]
            time_clean, unique_indices = np.unique(time, return_index=True)
            v_clean = v_sig[unique_indices]
            n = len(v_clean)
            if n < 10:
                return None
            t_uniform = np.linspace(time_clean[0], time_clean[-1], num=n)
            dt = t_uniform[1] - t_uniform[0]
            interpolator = interp.interp1d(time_clean, v_clean, kind='cubic', fill_value="extrapolate")
            v_uniform = interpolator(t_uniform)
            window = np.hanning(n)
            v_windowed = v_uniform * window
            fft_vals = np.fft.rfft(v_windowed)
            fft_mag = np.abs(fft_vals) / (n / 2.0)
            freqs = np.fft.rfftfreq(n, d=dt)
            fft_mag[0] = 0.0
            fund_idx = np.argmax(fft_mag)
            fund_mag = fft_mag[fund_idx]
            fund_freq = freqs[fund_idx]
            if fund_mag < 1e-12:
                return None
            harm_power = 0.0
            for i in range(2, 11):
                target_freq = fund_freq * i
                window_mask = (freqs >= target_freq * 0.98) & (freqs <= target_freq * 1.02)
                if np.any(window_mask):
                    harm_power += np.max(fft_mag[window_mask])**2

        if harm_power <= 0:
            return -120.0
        thd_lin = np.sqrt(harm_power) / fund_mag
        if thd_lin <= 1e-6:
            return -120.0
        return float(20 * np.log10(thd_lin))
