Version **1.5.0**

Date Released: **03/22/2026**

- Pipeline upgrades and major refactoring to spec functions and sampling algorithms

Version **1.4.2**

Date Released: **02/22/2026**

- Pipeline upgrades to accomodate new testbench and netlist format

Version **1.4.1**

Date Released: **02/14/2026**

- Test drive (.json) and mass collection (.parquet) modes for data collection
- Automatic progress saving for Sobol and TuRBO-M search with ability to load previous simulation results into current algorithm memory
- Processing of multiple topologies in a single pipeline request

Version **1.4.0**

Date Released: **02/06/2026**

- Complete redesign of op-amp netlist testbenches
- Corrected spec measurement and extraction functions
- New Sobol and TuRBO-M algorithms implemented to explore design space
- Updated simulation and data extraction pipeline accomodating new testbench

Version **1.3.0**

Date Released: **01/26/2026**

- Customization of specs extracted from simulation
- Customization of optimization specs and weights

Version **1.2.0**

Date Released: **01/24/2026**

- Newly formatted CLI
- Complete reorganization of codebase
- Automated removal of heavy leftover simulation files after data extraction
- Removal of absolute file paths for filesystem configs

Version **1.1.0**

Date Released: **01/22/2026**

- Netlist -> Graph script integration
- Revamped data collection pipeline with formatted `json` output
- Cleaned up codebase

Version **1.0.0**

Date Released: **11/24/2025**

- Full TuRBO-powered Spectre netlist simulation pipeline
- Optimization across 4 parameters
- Calculation of 11 performance specs from sized configurations