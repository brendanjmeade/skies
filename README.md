# Stochastic Kinematically Informed Earthquake Sequences
![](https://user-images.githubusercontent.com/4225359/229966215-1c40b94d-2748-441f-9298-74bd48618e50.png)

# Kinematic / statistical earthquake sequence generation
`skies` is a python-based package designed to model three-dimensional earthquake sequences in time across geometrically complex fault systems.

# Getting started
To set up a development conda environment, run the following commands in the `skies` folder.
```
conda config --prepend channels conda-forge
conda env create
conda activate skies
pip install --no-use-pep517 -e .
```

A model run can be started with something like:
```
python ./../skies/skies_generate_sequence.py ./data/cascadia_params.json --n_time_steps 100000 --omori_rate_perturbation_exponent 2.0 --repl 1
```

Input parameters:
| parameter name | example value | use |
| --- | --- | --- |
| `area_scaling` | 1.25 | |
| `b_value` | -1 | |
| `default_omori_decay_time` | 10 | |
| `geometric_moment_rate_scale_factor` | 5 | |
| `initial_mesh_slip_deficit_scaling` | 0 | |
| `initial_slip_deficit_rate_file` | "rates.npy" | |
| `location_probability_amplitude_scale_factor` | 1 | |
| `location_probability_data_scale_factor` | 0.00001 | |
| `max_latitude` | 52 | |
| `max_longitude` | 231 | |
| `maximum_event_moment_magnitude` | 9.5 | |
| `mesh_index` | 0 | |
| `mesh_parameters_file_name` | "mesh_parameters.json" | |
| `min_contour_value` | 0.1 | |
| `min_latitude` | 38 | |
| `min_longitude` | 239 | |
| `minimum_event_moment_magnitude` | 5.0 | |
| `minimum_probability` | 1e-10 | |
| `n_binary` | 2 | |
| `n_contour_levels` | 10 | |
| `n_events_omori_history_effect` | 100 | |
| `n_grid_latitude` | 500 | |
| `n_grid_longitude` | 500 | |
| `n_samples` | 1 | |
| `n_time_steps` | 1000000 | |
| `omori_amplitude_scale_factor` | 1e-8 | |
| `omori_rate_perturbation_scale_factor` | 1e-1 | |
| `omori_rate_perturbation_exponent` | 1.0 | |
| `plot_events_in_loop` | False | |
| `shear_modulus` | 30000000000 | |
| `time_probability_amplitude_scale_factor` | 0.15 | |
| `time_probability_data_scale_factor` | 1e-12 | |
| `time_probability_history_scale_factor` | 1000000000000 | |
| `time_step` | 5e-5 | |
| `write_event_pickle_files` | 0 | |
| `repl` | 0 | |
| `base_runs_folder` | "./runs" | |
| `geometric_moment_nucleation_probability` | "low" | |


### Folder structure and file locations for applications
We assume that a project is arranged using the following folder structure:
```
project_name/
|
├── data/
|   ├── mesh_parameters.json
│   ├── mesh_001.msh
│   ├── mesh_002.msh
│   └── mesh_NNN.msh
|
└── runs/
    └── 2022_12_11_20_38_21/
       ├── 2022_12_11_20_38_21.hdf
       ├── probability_magnitude.png
       ├── probability_magnitude.pdf
       ├── random_state.pickle
       ├── mesh.pickle
       ├── time_series.pickle
       ├── initial_mesh_data.png
       ├── initial_mesh_data.pdf
       ├── 2022_12_11_20_38_21_mesh_geometry.vtk
       ├── params.json
       ├── 2022_12_11_20_38_21.log    
       ├── model_segment.csv
       ├── model_block.csv
       └── model_station.csv
```

[![DOI](https://zenodo.org/badge/535758677.svg)](https://zenodo.org/badge/latestdoi/535758677)


