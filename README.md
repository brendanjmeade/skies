# Stochastic Kinematically Informed Earthquake Sequences
![](https://user-images.githubusercontent.com/4225359/229966215-1c40b94d-2748-441f-9298-74bd48618e50.png)

# Kinematic / statistical earthquake sequence generation
`skies` is a python-based package designed to model three-dimensional earthquake sequences in time across geometrically complex fault systems.  It's the code advertised by the paper, ["Meade, B. J., (2024) A kinematic method for generating earthquake sequences, Computers and Geosciences"](https://www.sciencedirect.com/science/article/abs/pii/S0098300423002212)

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
| parameter name | example value | description |
| --- | --- | --- |
| `area_scaling` | 1.25 | Factor that expands rupture area from empirical rupture area |
| `b_value` | -1 | Gutenberg-Ricter $b$-value |
| `default_omori_decay_time` | 10 | Controls length of Omori decay following earthquakes (**smaller is longer** Omori decay time) |
| `geometric_moment_rate_scale_factor` | 1 | Scales rates of moment accumulation.  Should always be 1 except for debugging |
| `initial_mesh_slip_deficit_scaling` | 0 | UNSURE |
| `initial_slip_deficit_rate_file` | "rates.npy" | Geometric moment accumulation rates on a single mesh.  This is just a single numpy array with one geometric moment rate component, generally dip-slip or strike-slip |
| `location_probability_amplitude_scale_factor` | 1 | Location probability amplitude scale factor (leading coefficient in front of $\tanh$ for location probability, $\gamma^h_a$ in the paper) |
| `location_probability_data_scale_factor` | 0.00001 | Location probability amplitude scale factor (coefficient that multiplies arguments to $\tanh$ for location probability, $\gamma^h_d$ in the paper) |
| `max_latitude` | 52 | Plotting |
| `max_longitude` | 231 | Plotting |
| `maximum_event_moment_magnitude` | 9.5 | Maximum event magnitude (could also be limited by total mesh area) |
| `mesh_index` | 0 | If the `mesh_parameters_file_name` file points to more than one mesh select the one specified by the index here |
| `mesh_parameters_file_name` | "mesh_parameters.json" | `celeri` style mesh parameters file |
| `min_contour_value` | 0.1 | Plotting |
| `min_latitude` | 38 | Plotting |
| `min_longitude` | 239 | Plotting |
| `minimum_event_moment_magnitude` | 5.0 | Minimum event magnitude (could also be limited by minimum mesh element area) |
| `minimum_probability` | 1e-10 | Minimum event probability in time |
| `n_binary` | 2 | WTF |
| `n_contour_levels` | 10 | Plotting |
| `n_events_omori_history_effect` | 100 | Number of more recent events that contribute to cumulative Omori effect (UNUSED?) |
| `n_grid_latitude` | 500 | Plotting |
| `n_grid_longitude` | 500 | Plotting |
| `n_samples` | 1 | UNSURE |
| `n_time_steps` | 1000000 | Number of time steps (not real time) |
| `omori_amplitude_scale_factor` | 1e-8 | Omori numerator amplitude, $\beta_j$ in the paper |
| `omori_rate_perturbation_scale_factor` | 1e-1 | Divides Omori time difference, $\tau_j$ in the paper |
| `omori_rate_perturbation_exponent` | 1.0 | Omori time difference exponent, $p_j$ in the paper  |
| `plot_events_in_loop` | False | Plotting |
| `shear_modulus` | 3e10 | Shear modulus |
| `time_probability_amplitude_scale_factor` | 0.15 | Time probability amplitude scale factor (leading coefficient in front of $\tanh$ for time probability), $\gamma^t_a$ in the paper |
| `time_probability_data_scale_factor` | 1e-12 | Time probability amplitude scale factor (coefficient that multiplies arguments to $\tanh$ for time probability), $\gamma^t_d$ in the paper |
| `time_probability_history_scale_factor` | 1e12 | UNSURE |
| `time_step` | 5e-5 | Time step duration (not real time)|
| `write_event_pickle_files` | 0 | Write a pickle file for each earthquake|
| `repl` | 0 | Drop into iPython REPL at end of run |
| `base_runs_folder` | "./runs" | Base output folder |
| `geometric_moment_nucleation_probability` | "low" | Should earthquake nucleate in regions of "high" or "low" geometric moment (NEED TO REVISE) |

- Time probability equation:
     - $\gamma_a^t$: `NNN`
     - $\gamma_d^t$: `NNN`

$$
p^t = \gamma_a^t \tanh \left( \gamma_d^t \left[r^\mathrm{a} + \sum\nolimits_j^{n(t_j \leq t)} \{ r^\mathrm{o} + r^\mathrm{r} \} + \mathcal{A} \right] \right).
$$

- Location probability equation:
     - $\gamma_a^h$: `NNN`
     - $\gamma_d^h$: `NNN`

$$
p_i^h(t_k) = \gamma_a^h \tanh ( \gamma_d^h \left[m^\mathrm{a}_i - m^\mathrm{r}_i\right] )
$$

- Omori time decay equation:
     - $\beta$: `NNN`
     - $p$: `NNN`
     - $\tau$: `NNN`
  
$$
r^\mathrm{o}_j(t) = \frac{\beta_j}{1 + \frac{(t - t_j)^{p_j}}{\tau_j}}
$$

- Moment reduction following slip events
     - $\omega'$: `NNN`
     - $\beta'$: `NNN`
     - $\psi'$: `NNN`

$$
r^\mathrm{r} = -\omega' \beta' \left[ \sum\nolimits_i m_i(t_j) \right] ^{\psi'}
$$


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


