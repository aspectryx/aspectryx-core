# CORVINA

A Cadence Spectre optimization and data-generation pipeline for OTA netlists.

Current Version: **1.5.1**

## ⚡ About

CORVINA contains an interactive CLI capable of running closed-loop design space exploration with Sobol and TuRBO-M over OTA design parameters. Each simulation point is parsed into structured records for:

- Input sizing and environment parameters
- Device-level operating-point features
- Performance specification outputs

These records are exported as JSON batches and merged into a final dataset.


### Key Output Specs

The collector uses a canonical output key set (see `simulator/compute/collector.py`). Core specs include:

- `estimated_area_um2`
- `cmrr_dc_db`
- `gain_ol_dc_db`
- `integrated_noise_vrms`
- `output_voltage_swing_range_v`
- `output_voltage_swing_min_v`
- `output_voltage_swing_max_v`
- `pm_deg`
- `power_w`
- `v_cm_ctrl`
- `psrr_dc_db`
- `settle_time_small_ns`
- `settle_time_large_ns`
- `slew_rate_v_us`
- `thd_db`
- `ugbw_hz`
- `vos_v`

### Personas (TuRBO-M)

The CLI exposes 10 optimization personas:

1. SPEED
2. PRECISION
3. EFFICIENCY
4. COMPACTNESS
5. BALANCED
6. ROBUSTNESS
7. LINEARITY
8. LOW_HEADROOM
9. STARTUP_RELIABILITY
10. DRIVE_LOAD

Persona definitions are in `simulator/cli/cli.py`, and scalarization logic is in `algorithms/turbo_m/turbo_m.py`.

## ⚙️ Requirements

- Linux/Unix environment
- Cadence Spectre, Virtuoso, and ViVA
- Conda with dependencies in `rlenv38.yml`

### Quick Start

1. Enter repository root.

2. Create and activate the Conda environment:

```bash
conda env create -f rlenv38.yml
conda activate rlenv38
```

3. Run pipeline:

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```
  
