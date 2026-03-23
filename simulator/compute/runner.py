"""
runner.py

Handles parallel execution of circuit simulations.
Separates compute logic from CLI user interface.
"""

import multiprocessing as mp
import time
import os

def worker_task(args):
    """
    Worker function executed by multiprocessing Pool.
    Receives (index, config, tokenizer (extractor)).
    """
    idx, config, extractor = args
    try:
        # Run extraction
        val = extractor(config, sim_id=idx)
        return True, (idx, val)
    except Exception as e:
        import traceback
        err_msg = f"Sim {idx} Failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\n[WORKER ERROR] {err_msg}", flush=True)
        return False, (idx, err_msg)

# --- Adaptive Worker Utilities --- #

MIN_FREE_CORES = 2  # Always leave this many cores for OS / other users

def get_adaptive_worker_count(max_cores=None):
    """
    Calculate a nice worker count based on real-time system load.
    Uses all available cores minus current load and a small safety buffer.
    """
    if max_cores is None:
        max_cores = mp.cpu_count()
    try:
        load1, _, _ = os.getloadavg()
    except AttributeError:
        load1 = 0
    free = max(1, int(max_cores - load1) - MIN_FREE_CORES)
    return free


def run_parallel_simulations(samples, extractor, n_workers, adaptive=False):
    """
    Orchestrates the parallel execution of the simulation batch.
    Yields progress updates to the caller.
    
    Parameters:
    -----------
    samples : list
        List of configuration dictionaries (Sobol samples).
    extractor : Extractor
        Configured Extractor instance.
    n_workers : int
        Number of parallel processes (initial). Ignored if adaptive=True on re-pool.
    adaptive : bool
        If True, re-checks system load and adjusts pool size between micro-batches.
        
    Yields:
    -------
    tuple
        (completed_count, total_count, elapsed_time, result_data)
        result_data is (index, (reward, specs)) or None if failed
    """
    
    # Pre-flight: compute sim_keys and ensure uniqueness and freshness
    sim_keys = []
    for config in samples:
        try:
            key = extractor.sim_key_for_params(config)
        except Exception:
            key = None
        sim_keys.append(key)

    # Check for duplicate keys within the batch
    unique_keys = set(k for k in sim_keys if k is not None)
    if len(unique_keys) != len(sim_keys):
        raise RuntimeError("Pre-flight check failed: duplicate parametrizations detected in samples.")

    # Check for already-existing results in extractor.results_dir
    existing = set()
    if hasattr(extractor, 'results_dir') and extractor.results_dir:
        for k in unique_keys:
            if k is None:
                continue
            if os.path.exists(os.path.join(extractor.results_dir, f"{k}.json")):
                existing.add(k)
    if existing:
        raise RuntimeError(f"Pre-flight check failed: {len(existing)} samples already have results in {extractor.results_dir}")

    task_args = []
    for i, config in enumerate(samples):
        task_args.append((i, config, extractor))

    total = len(samples)
    completed = 0
    start_time = time.time()
    
    if not adaptive:
        # Fixed mode: single pool for the entire batch
        chunk_size = max(1, total // (n_workers * 4))
        try:
            with mp.Pool(processes=n_workers) as pool:
                for result in pool.imap_unordered(worker_task, task_args, chunksize=chunk_size):
                    success, payload = result
                    idx, data = payload
                    if not success:
                        print(f"\n[!] SIMULATION FAILED - ID {idx}: {data}", flush=True)
                        final_data = (idx, (0.0, {'valid': False}))
                    else:
                        final_data = (idx, data)
                    completed += 1
                    elapsed = time.time() - start_time
                    yield (completed, total, elapsed, final_data)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
    else:
        # Adaptive mode: process in micro-batches, re-check load between each
        MICRO_BATCH_INTERVAL = 50  # Re-evaluate load every N sims
        remaining = list(task_args)
        
        try:
            while remaining:
                # Re-check load and pick worker count
                current_workers = get_adaptive_worker_count()
                # Don't log every time, just when it changes meaningfully
                micro_size = min(MICRO_BATCH_INTERVAL, len(remaining))
                micro_batch = remaining[:micro_size]
                remaining = remaining[micro_size:]
                
                chunk_size = max(1, micro_size // (current_workers * 2))
                
                with mp.Pool(processes=current_workers) as pool:
                    for result in pool.imap_unordered(worker_task, micro_batch, chunksize=chunk_size):
                        success, payload = result
                        idx, data = payload
                        if not success:
                            print(f"\n[!] SIMULATION FAILED - ID {idx}: {data}", flush=True)
                            final_data = (idx, (0.0, {'valid': False}))
                        else:
                            final_data = (idx, data)
                        completed += 1
                        elapsed = time.time() - start_time
                        yield (completed, total, elapsed, final_data)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
