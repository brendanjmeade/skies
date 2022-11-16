import datetime
import json
import os
import pickle

import addict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import skies

plt.close("all")

run_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
base_runs_folder = "./runs/"
output_folder = os.path.join(base_runs_folder, run_name)

# TODO: read this from command line/params file
mesh_parameters_file_name = "western_north_america_mesh_parameters.json"
skies.create_output_folder(base_runs_folder, output_folder)
meshes = skies.read_meshes(mesh_parameters_file_name)
skies.print_magnitude_overview(meshes)


np.random.seed(2)

# Parameters for model run
params = addict.Dict()
params.n_time_steps = 400
params.time_step = 5e-7
params.b_value = -1.0
params.shear_modulus = 3e10
params.n_samples = 1
params.n_binary = 2
params.minimum_event_moment_magnitude = 6.0
params.maximum_event_moment_magnitude = 9.0
params.time_probability_amplitude_scale_factor = 5e-2
params.time_probability_data_scale_factor = 1e-12
params.area_scaling = 1.25
params.default_omori_decay_time = 100.0
params.minimum_probability = 1e-10
params.time_probability_history_scale_factor = 1e11
params.location_probability_amplitude_scale_factor = 1.0
params.location_probability_data_scale_factor = 1e-5
params.omori_amplitude_scale_factor = 3e-9
params.omori_rate_perturbation_scale_factor = 1e-1
params.mesh_index = 0
params.initial_mesh_slip_deficit_scaling = 0.0
params.geometic_moment_rate_scale_factor = 5e1
params.plot_events_in_loop = True
params.n_events_omori_history_effect = 100
params.n_grid_longitude = 500
params.n_grid_latitude = 500
params.min_longitude = 239.0
params.max_longitude = 231.0
params.min_latitude = 38.0
params.max_latitude = 52.0
params.n_contour_levels = 10
params.min_contour_value = 0.1  # (m)

# Save params dictionary to .json file in run folder
with open(output_folder + "/params.json", "w") as params_output_file:
    json.dump(params, params_output_file)


# Load initial slip defict and multiply by time cascadia_low_resolution_tde_dip_slip_rates.npy
# TODO: Make this something that is loaded from a file specified in params
mesh_initial_dip_slip_deficit = np.load(
    "cascadia_low_resolution_tde_dip_slip_rates.npy"
)
skies.plot_initial_data(meshes, mesh_initial_dip_slip_deficit)
plt.show()


# Storage
time_series = addict.Dict()
time_series.time = np.linspace(0, params.n_time_steps, params.n_time_steps)
time_series.probability_weight = np.zeros_like(time_series.time)
time_series.probability = np.zeros_like(time_series.time)
time_series.event_magnitude = np.zeros_like(time_series.time)
time_series.event_trigger_flag = np.zeros_like(time_series.time)
time_series.last_event_time = 0

# TODO: #22 Change earthquake to event. Make a dictionary?  Can I eliminate `earthquake_index_list`, `earthquake_magnitude_list`, `earthquake_probability_list`
earthquake_probability_list = []
time_series.omori_history_effect = np.zeros(
    (params.n_time_steps, params.n_events_omori_history_effect)
)

# Initial geometric moment and storage
# TODO: #23 Convert to mesh_geometric_moment dictionary?
# Select mesh if multiple have been loaded
# TODO: Move down to other mesh statements and make this mesh.mesh
mesh = meshes[params.mesh_index]
mesh_geometric_moment = np.zeros(mesh.n_tde)
mesh_last_event_slip = np.zeros(mesh.n_tde)
mesh_total_slip = np.zeros(mesh.n_tde)
mesh_geometric_moment_pre_event = np.copy(mesh_geometric_moment)
mesh_geometric_moment_post_event = np.zeros_like(mesh_geometric_moment)
mesh_geometric_moment_scalar = np.zeros_like(time_series.time)
mesh_geometric_moment_scalar_non_zero = np.zeros_like(time_series.time)
mesh_geometric_moment_scalar[0] = np.sum(mesh_geometric_moment)

mesh_interseismic_loading_rate = (
    params.geometic_moment_rate_scale_factor * mesh_initial_dip_slip_deficit
)

# Main time loop
start_time = datetime.datetime.now()
for i in tqdm(range(params.n_time_steps - 1), colour="cyan"):
    # Update mesh_geometric_moment
    mesh_geometric_moment += (
        params.time_step * mesh_interseismic_loading_rate * mesh.areas
    )
    # temp[i, :] = mesh_geometric_moment
    mesh_geometric_moment_scalar[i + 1] = np.sum(mesh_geometric_moment)
    mesh_geometric_moment_scalar_non_zero[i + 1] = np.sum(
        mesh_geometric_moment[np.where(mesh_geometric_moment > 0.0)]
    )

    # Determine whether there is an event at this time step
    time_series.probability_weight[i] = skies.get_tanh_probability(
        time_series.probability[i],
        params.time_probability_amplitude_scale_factor,
        params.time_probability_data_scale_factor,
    )
    time_series.event_trigger_flag[i] = np.random.choice(
        params.n_binary,
        params.n_samples,
        p=[1 - time_series.probability_weight[i], time_series.probability_weight[i]],
    )

    if bool(time_series.event_trigger_flag[i]):
        time_series.last_event_time = i
        event = addict.Dict()
        event.shear_modulus = np.array([params.shear_modulus])
        event.area_scaling = params.area_scaling
        event.moment_magnitude = skies.get_gutenberg_richter_magnitude(
            params.b_value,
            params.minimum_event_moment_magnitude,
            params.maximum_event_moment_magnitude,
        )
        event.moment = skies.moment_magnitude_to_moment(event.moment_magnitude)
        event.geometric_moment = event.moment / event.shear_modulus
        time_series.event_magnitude[i] = event.moment_magnitude[0]

        # Find event hypocentral triangle
        event.location_probability = skies.get_tanh_probability_vector(
            mesh_geometric_moment_pre_event,
            params.location_probability_amplitude_scale_factor,
            params.location_probability_data_scale_factor,
        )
        event.hypocenter_triangle_index = np.random.choice(
            mesh.n_tde, params.n_samples, p=event.location_probability
        )[0]

        # Generate coseismic slip area and slip distribution
        event = skies.get_event_area_slip_triangle_index(mesh, event)
        event.mesh_geometric_moment_pre_event = np.copy(mesh_geometric_moment_pre_event)
        event.mesh_geometric_moment_post_event = np.copy(
            mesh_geometric_moment_pre_event - (event.slip_all_elements * mesh.areas)
        )

        # Generate Omori rate decay
        event.omori_amplitude = (
            params.omori_amplitude_scale_factor * event.geometric_moment_scalar
        )
        event.omori_decay_time = params.default_omori_decay_time
        omori_rate_perturbation = skies.get_omori_decay_probability(
            time_series.time,
            time_series.time[i],
            event.omori_amplitude,
            decay_time=event.omori_decay_time,
        )

        # Coseismic offset to Omori rate effect
        omori_rate_perturbation[np.where(time_series.time > time_series.time[i])] -= (
            event.omori_amplitude * params.omori_rate_perturbation_scale_factor
        )

        # Store Omori rate decay
        earthquake_probability_list.append(omori_rate_perturbation)
        time_series.omori_history_effect[
            :, 0
        ] = omori_rate_perturbation  # Still need to implement below.

        # Update spatially variable mesh parameters
        mesh_geometric_moment -= event.slip_all_elements * mesh.areas
        mesh_last_event_slip = event.slip_all_elements
        mesh_total_slip += event.slip_all_elements
        event.mesh_last_event_slip = event.slip_all_elements
        event.mesh_total_slip = mesh_total_slip

    else:
        # Create dummy event dictionary because no event occured
        event = skies.create_non_event(mesh.n_tde)
        event.mesh_geometric_moment_pre_event = np.copy(mesh_geometric_moment_pre_event)
        event.mesh_geometric_moment_post_event = mesh_geometric_moment_pre_event + (
            params.time_step * mesh_interseismic_loading_rate * mesh.areas
        )
        event.mesh_last_event_slip = mesh_last_event_slip
        event.mesh_total_slip = mesh_total_slip

    # TODO: Check this???
    event.mesh_initial_dip_slip_deficit = mesh_initial_dip_slip_deficit

    # Save event dictionary as pickle file
    event_pickle_file_name = f"{output_folder}/events/event_{i:010.0f}.pickle"
    with open(event_pickle_file_name, "wb") as pickle_file:
        pickle.dump(event, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    # Save time step parameters as .vtk files for plotting with Paraview and pyvista

    # Pre-event moment for next time step
    mesh_geometric_moment_pre_event = np.copy(event.mesh_geometric_moment_post_event)

    # Update probability
    time_series.probability[i + 1] = mesh_geometric_moment_scalar_non_zero[i + 1]

    # Sum contribution from all past earthquakes
    for j in range(len(earthquake_probability_list)):
        time_series.probability[i + 1] += (
            params.time_probability_history_scale_factor
            * earthquake_probability_list[j][i + 1]
        )
end_time = datetime.datetime.now()
print(f"\nSequence generation run time: {str(end_time - start_time)}\n")

# Plot time probability and event moment magnitude time series
start_idx = 0
end_idx = time_series.time.size
skies.plot_probability_and_events_time_series(
    params,
    output_folder,
    time_series,
    start_idx,
    end_idx,
)
