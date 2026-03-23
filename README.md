# ASPECTOR Core

A Cadence Spectre simulation pipeline for op-amp netlists. Used as part of the ASPECTOR analog design suite to extract performance data for dataset construction.

Current Version: **1.5.0**

## ⚡ Current Features

ASPECTOR Core boasts a sleek, comprehensive terminal CLI to specify means of simulation, optimization, and extraction of performance data.

A TuRBO algorithm is used to optimize sizing, bias, and environment parameters of op-amp Spectre netlists to meet up to 15 target performance specs:

When running the pipeline, each netlist is converted to a `.json` graph before subsequent simulations produce data that is extracted and converted into sizing, operating point, and performance spec `.json` objects. 

## 📖 How to Use

**Requirements:** 
- Linux/Unix environment
- Cadence Spectre, Virtuoso, and ViVA software
- Conda environment with all dependencies specified in `rlenv38.yml`

**1. Clone Repository:**

  ```
  git clone https://github.com/natelgrw/titan_foundation_model.git
  cd titan_foundation_model
  ```
   
**2. Create & Activate Conda Environment:**

  ```
  conda env create -f rlenv38.yml
  conda activate rlenv38
  ```

**3. Run Optimization Pipeline:**

  ```
  chmod +x run_pipeline.sh
  ./run_pipeline.sh
  ```
  