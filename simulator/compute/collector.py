"""
collector.py

Data Collection module for Mass Data Generation.
Buffers simulation results into intermediate JSON files and merges them into a final Parquet file.
"""

import os
import json
import uuid
import pandas as pd
import time
import glob
import re
from collections import defaultdict

class DataCollector:
    def __init__(self, output_dir, buffer_size=1000, parquet_name="dataset.parquet", max_rows_per_file=50000):
        """
        Initialize DataCollector.
        
        Args:
            output_dir (str): Directory to save intermediate files and final parquet.
            buffer_size (int): Number of records to hold in memory before flush.
            parquet_name (str): Name of the final parquet file.
        """
        self.output_dir = output_dir
        self.buffer_size = buffer_size
        self.buffer = []
        self.parquet_name = parquet_name
        self.dataset_path = os.path.join(output_dir, parquet_name)
        # Maximum number of rows per output parquet file (per-topology). When reached,
        # finalize will emit a numbered parquet part and may delete consumed JSONs.
        self.max_rows_per_file = max_rows_per_file
        
        os.makedirs(output_dir, exist_ok=True)
        
    def log(self, config, specs, meta=None, operating_points=None):
        """
        Log a single simulation result.
        
        Args:
            config (dict): Input parameters (Sizing + Env).
            specs (dict): Output specifications.
            meta (dict, optional): Metadata (sim_id, timestamp).
            operating_points (dict, optional): Operating points data.
        """
        record = {}
        
        # Add Input Config
        for k, v in config.items():
            record[f"in_{k}"] = v

        # Ensure common input parameters exist for differential netlists
        try:
            is_diff = record.get('is_diff', False)
        except Exception:
            is_diff = False

        if is_diff:
            required_inputs = {
                'fet_num': 0,
                'vdd': 0.0,
                'vcm': 0.0,
                'tempc': 25,
                'cload_val': 0.0,
                'vbiasp0': 0.0,
                'vbiasn0': 0.0,
                'nA1': 0.0,
                'nA2': 0.0,
                'nA3': 0.0,
                'nB1': 0,
                'nB2': 0,
                'nB3': 0,
                'run_gatekeeper': 0,
                'run_full_char': 0
            }
            for k, default in required_inputs.items():
                in_key = f'in_{k}'
                if in_key not in record:
                    record[in_key] = default
            
        # Add Output Specs
        if specs:
            valid = specs.get('valid', False)
            record['valid'] = valid
            for k, v in specs.items():
                if k == 'valid': continue
                # Handle tuples (like swing [min, max]) by converting to scalar or string
                if isinstance(v, (list, tuple)):
                    record[f"out_{k}_min"] = v[0]
                    record[f"out_{k}_max"] = v[1]
                    record[f"out_{k}_val"] = abs(v[1] - v[0])
                else:
                    record[f"out_{k}"] = v
        else:
            record['valid'] = False
            
        # Add Operating Points
        if operating_points:
            for comp, params in operating_points.items():
                for param_name, val in params.items():
                    record[f"op_{comp}_{param_name}"] = val
                    
        if meta:
            # Remove 'env' if present to avoid redundancy/errors
            meta_copy = dict(meta)
            meta_copy.pop('env', None)
            
            # JSON-serialize dict fields (weights_dict, lambdas_dict) for JSON/Parquet compatibility
            for dict_key in ['weights_dict', 'lambdas_dict']:
                if dict_key in meta_copy and isinstance(meta_copy[dict_key], dict):
                    try:
                        meta_copy[dict_key] = json.dumps(meta_copy[dict_key])
                    except Exception:
                        pass  # If serialization fails, leave as-is and let json.dump handle it
            
            record.update(meta_copy)
            # Add is_diff flag based on netlist name: True if 'differential' in name
            try:
                netname = meta.get('netlist_name') or record.get('netlist_name') or ''
                record['is_diff'] = True if isinstance(netname, str) and 'differential' in netname.lower() else False
            except Exception:
                record['is_diff'] = False
            
        record['timestamp'] = time.time()
        # Always add algorithm and netlist_name if present in meta
        if meta:
            if 'algorithm' in meta:
                record['algorithm'] = meta['algorithm']
            if 'netlist_name' in meta:
                record['netlist_name'] = meta['netlist_name']
        # Ensure differential netlists include a complete set of spec keys and OP fields
        try:
            is_diff = record.get('is_diff', False)
        except Exception:
            is_diff = False

        if is_diff:
            # Required spec keys and conservative defaults for failed sims
            required_specs = {
                'estimated_area': 0.0,
                'cmrr': -1000,
                'gain_ol': -1000,
                'integrated_noise': 0.0,
                'output_voltage_swing': 0.0,
                'pm': -180,
                'power': 0.0,
                'psrr': -1000,
                'settle_time': 1000,
                'slew_rate': 0.0,
                'thd': 1000,
                'ugbw': 1.0,
                # Use the same numeric fallback used by measurement managers.
                'vos': 10.0
            }

            for k, default in required_specs.items():
                out_key = f'out_{k}'
                if out_key not in record:
                    record[out_key] = default

            # Ensure operating point and capacitance keys exist for a small set of FETs
            # (MM0..MM4) to keep schema consistent even when OP data is missing.
            op_params = ['region_of_operation', 'ids', 'vds', 'vgs', 'gm', 'gds', 'vth', 'vdsat']
            cap_params = ['cgg', 'cgs', 'cdd', 'cgd', 'css']
            for mm in range(5):
                for p in op_params:
                    key = f'op_MM{mm}_{p}'
                    if key not in record:
                        record[key] = None
                for p in cap_params:
                    key = f'op_MM{mm}_{p}'
                    if key not in record:
                        record[key] = None

        self.buffer.append(record)
        
        if len(self.buffer) >= self.buffer_size:
            self.flush()
            
    def flush(self):
        """
        Write buffer to an intermediate JSON file.
        """
        if not self.buffer:
            return
            
        batch_id = str(uuid.uuid4())
        batch_filename = f"batch_{batch_id}.json"
        batch_path = os.path.join(self.output_dir, batch_filename)
        
        try:
            # Preserve numeric fidelity: do not round or coerce floats here.
            with open(batch_path, 'w') as f:
                json.dump(self.buffer, f, indent=2)
        except Exception as e:
            print(f"[!] Failed to write batch {batch_filename}: {e}")
            
        self.buffer = [] # Clear buffer

    def finalize(self, discard_partial=True, preserve_json=True):
        """
        Merges all intermediate JSON files into a single Parquet file and cleans up.
        
        Args:
            discard_partial (bool): If True, discard any unflushed buffer (< buffer_size).
                                    This prevents duplicates on resume since the Sobol/TuRBO
                                    state checkpoint only advances on full-batch boundaries.
        """
        # Handle partial buffer
        if discard_partial:
            if self.buffer:
                print(f"[i] Discarding {len(self.buffer)} buffered records (partial batch).")
            self.buffer = []
        else:
            self.flush()

        # Find only intermediate batch files under output_dir (recursive).
        # This intentionally excludes run metadata files (run_config*.json, metadata*.json)
        # so they are never ingested as simulation records or deleted during cleanup.
        json_files = []
        json_files.extend(sorted(glob.glob(os.path.join(self.output_dir, "**", "batch_*.json"), recursive=True)))
        json_files.extend(sorted(glob.glob(os.path.join(self.output_dir, "**", "batch_*.jsonl"), recursive=True)))
        # remove duplicates and sort
        json_files = sorted(list(dict.fromkeys(json_files)))

        if not json_files:
            print("No JSON files found to aggregate into Parquet.")
            return

        # Streaming aggregation: accumulate records per-topology and flush to numbered parquet parts
        buffers = defaultdict(list)      # topo -> list of (record_dict, source_json_path)
        seen_ids = defaultdict(set)      # topo -> set(sim_id)
        jf_remaining = {}                # jf -> remaining record count not yet flushed
        part_idx = defaultdict(lambda: 1)  # topo -> next part index

        # Inspect existing parquet parts in output_dir and initialize part indices
        # and seen_ids so we continue numbering and avoid duplicates across runs.
        try:
            parquet_files = sorted(glob.glob(os.path.join(self.output_dir, "*.parquet")))
            part_re = re.compile(r"(?P<topo>.+)_(?P<idx>\d+)\.parquet$")
            for p in parquet_files:
                bn = os.path.basename(p)
                m = part_re.match(bn)
                if not m:
                    continue
                topo = m.group('topo')
                try:
                    idx = int(m.group('idx'))
                    if idx >= part_idx[topo]:
                        part_idx[topo] = idx + 1
                except Exception:
                    pass
                # Load existing sim_id column to seed seen_ids for this topology
                try:
                    existing_ids = pd.read_parquet(p, columns=['sim_id'])
                    if 'sim_id' in existing_ids.columns:
                        for sid in existing_ids['sim_id'].dropna().unique():
                            seen_ids[topo].add(sid)
                except Exception:
                    # ignore read errors; dedupe will still work for new sim_ids
                    pass
        except Exception:
            pass

        # Initialize seen_ids from existing parquet parts (so we don't duplicate across runs)
        for jf in json_files:
            # Skip markers/metadata in marker dirs
            if os.path.sep + 'markers' + os.path.sep in jf:
                continue
            try:
                # Count records in jf for bookkeeping
                if jf.lower().endswith('.jsonl'):
                    cnt = 0
                    with open(jf, 'r') as f:
                        for line in f:
                            if line.strip():
                                cnt += 1
                    jf_remaining[jf] = cnt
                else:
                    with open(jf, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            jf_remaining[jf] = len(data)
                        else:
                            jf_remaining[jf] = 1
            except Exception:
                jf_remaining[jf] = jf_remaining.get(jf, 0)

        # Helper to write a parquet part for a topology
        def _write_part(topo, items):
            # items: list of tuples (record_dict, source_jf)
            nonlocal part_idx
            if not items:
                return True, None
            records = [rec for (rec, _jf) in items]
            try:
                df_part = pd.DataFrame(records)
            except Exception as e:
                return False, f"build_df_failed: {e}"

            # sanitize topology name
            safe_topo = str(topo).replace(' ', '_').replace('/', '_')
            idx = part_idx[topo]
            out_name = f"{safe_topo}_{idx}.parquet"
            out_path = os.path.join(self.output_dir, out_name)
            tmp_path = out_path + ".tmp"
            try:
                df_part = df_part.dropna(axis=1, how='all')
                df_part.to_parquet(tmp_path)
                os.replace(tmp_path, out_path)
                print(f"Successfully wrote {len(df_part)} records to {out_path}")
                part_idx[topo] += 1
                # decrement jf_remaining for source files and remove files fully consumed
                for (_rec, src) in items:
                    if src in jf_remaining:
                        jf_remaining[src] = max(0, jf_remaining[src] - 1)
                        if jf_remaining[src] == 0 and not preserve_json:
                            try:
                                os.remove(src)
                            except Exception:
                                pass
                return True, None
            except Exception as e:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                return False, str(e)

        # Stream through JSON files and route records into per-topo buffers
        for jf in json_files:
            if os.path.sep + 'markers' + os.path.sep in jf:
                continue
            try:
                if jf.lower().endswith('.jsonl'):
                    with open(jf, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                item = json.loads(line)
                            except Exception:
                                continue
                            if isinstance(item, dict) and set(item.keys()) <= {'id','sim_id','sim_status','netlist','timestamp'}:
                                continue
                            if isinstance(item, dict) and set(item.keys()) <= {'requested_name','state'}:
                                continue
                            topo = item.get('netlist_name') or item.get('requested_name') or 'unknown'
                            sid = item.get('sim_id') or item.get('id')
                            if sid is not None and sid in seen_ids[topo]:
                                # already seen
                                continue
                            if sid is not None:
                                seen_ids[topo].add(sid)
                            buffers[topo].append((item, jf))
                else:
                    with open(jf, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    if set(item.keys()) <= {'id','sim_id','sim_status','netlist','timestamp'}:
                                        continue
                                    if set(item.keys()) <= {'requested_name','state'}:
                                        continue
                                topo = item.get('netlist_name') or item.get('requested_name') or 'unknown'
                                sid = item.get('sim_id') or item.get('id')
                                if sid is not None and sid in seen_ids[topo]:
                                    continue
                                if sid is not None:
                                    seen_ids[topo].add(sid)
                                buffers[topo].append((item, jf))
                        elif isinstance(data, dict):
                            item = data
                            if 'specs' not in item and 'netlist_name' not in item and set(item.keys()) <= {'id','sim_id','sim_status','netlist','timestamp'}:
                                continue
                            if 'requested_name' in item and 'state' in item and 'netlist_name' not in item:
                                continue
                            topo = item.get('netlist_name') or item.get('requested_name') or 'unknown'
                            sid = item.get('sim_id') or item.get('id')
                            if sid is not None and sid in seen_ids[topo]:
                                continue
                            if sid is not None:
                                seen_ids[topo].add(sid)
                            buffers[topo].append((item, jf))
            except Exception as e:
                print(f"[!] Error reading {jf}: {e}")

            # After processing this JSON file, check if any topo reached the threshold and flush parts
            for topo, items in list(buffers.items()):
                if len(items) >= self.max_rows_per_file:
                    to_write = items[:self.max_rows_per_file]
                    ok, err = _write_part(topo, to_write)
                    if not ok:
                        print(f"[!] Failed to write part for {topo}: {err}")
                    # remove written items from buffer
                    buffers[topo] = items[self.max_rows_per_file:]

        # After all JSON files processed, flush remaining buffers to final parts
        for topo, items in list(buffers.items()):
            if not items:
                continue
            # write in chunks of max_rows_per_file
            i = 0
            while i < len(items):
                chunk = items[i:i + self.max_rows_per_file]
                ok, err = _write_part(topo, chunk)
                if not ok:
                    print(f"[!] Failed to write final part for {topo}: {err}")
                    break
                i += self.max_rows_per_file

        # If caller requested preserving JSON files, leave them; otherwise clean up any leftover JSONs
        if preserve_json:
            print("Preserving intermediate JSON files for debugging (preserve_json=True).")
            return

        # Remove any remaining json files that weren't removed during part writes
        for jf, rem in list(jf_remaining.items()):
            if rem == 0:
                # already removed
                continue
            try:
                os.remove(jf)
            except Exception:
                pass
        print("Cleaned up intermediate JSON files.")
