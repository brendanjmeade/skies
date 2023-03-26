# Stochastic Kinematically Informed Earthquake Sequences

# Next generation earthquake cycle imaging
`skies` is a python based package designed to model three-dimensional earthquake sequences in time across geometrically complex fault systems.

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


