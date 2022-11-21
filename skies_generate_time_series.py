import datetime
import json
import os
import pickle
import rich

import addict
import matplotlib.pyplot as plt
import numpy as np
from rich.progress import track

import skies

plt.close("all")

run_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
base_runs_folder = "./runs/"
output_folder = os.path.join(base_runs_folder, run_name)
skies.create_output_folder(base_runs_folder, output_folder)


np.random.seed(2)


# TODO: Option for truncating eigenvalues (1000?)

# params dictionary with model run parameters
# TODO: Read from command line and allow overloading like `celeri`
params = addict.Dict()
params.n_time_steps = 4000
params.time_step = 5e-7
params.b_value = -1.0
params.shear_modulus = 3e10
params.n_samples = 1
params.n_binary = 2
params.minimum_event_moment_magnitude = 5.5
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
params.mesh_parameters_file_name = "./data/western_north_america_mesh_parameters.json"
params.initial_slip_deficit_rate_file = (
    "./data/cascadia_low_resolution_tde_dip_slip_rates.npy"
)

# Save params dictionary to .json file in output_folder
with open(output_folder + "/params.json", "w") as params_output_file:
    json.dump(params, params_output_file)

# Time-series storage
time_series = addict.Dict()
time_series.time = np.linspace(0, params.n_time_steps, params.n_time_steps)
time_series.probability_weight = np.zeros_like(time_series.time)
time_series.probability = np.zeros_like(time_series.time)
time_series.event_magnitude = np.zeros_like(time_series.time)
time_series.event_trigger_flag = np.zeros_like(time_series.time)
time_series.cumulate_omori_effect = np.zeros_like(time_series.time)
time_series.last_event_time = 0


# Mesh storage
mesh = addict.Dict()
meshes = skies.read_meshes(params.mesh_parameters_file_name)
mesh.mesh = meshes[params.mesh_index]
mesh.mesh_geometric_moment = np.zeros(mesh.mesh.n_tde)
mesh.mesh_last_event_slip = np.zeros(mesh.mesh.n_tde)
mesh.mesh_total_slip = np.zeros(mesh.mesh.n_tde)
mesh.mesh_geometric_moment_pre_event = np.copy(mesh.mesh_geometric_moment)
mesh.mesh_geometric_moment_post_event = np.zeros_like(mesh.mesh_geometric_moment)
mesh.mesh_geometric_moment_scalar = np.zeros_like(time_series.time)
mesh.mesh_geometric_moment_scalar_non_zero = np.zeros_like(time_series.time)
mesh.mesh_geometric_moment_scalar[0] = np.sum(mesh.mesh_geometric_moment)
# TODO: This should be generalized so that strike- or -dip slip
# or both can be specified
mesh.mesh_initial_dip_slip_deficit = np.load(params.initial_slip_deficit_rate_file)
mesh.mesh_interseismic_loading_rate = (
    params.geometic_moment_rate_scale_factor * mesh.mesh_initial_dip_slip_deficit
)

# Display information about initial mesh and slip deficit rates
skies.print_magnitude_overview(mesh.mesh)
skies.plot_initial_data(mesh.mesh, mesh.mesh_initial_dip_slip_deficit, output_folder)

# Main time loop
start_time = datetime.datetime.now()
for i in track(range(params.n_time_steps - 1), description="Event generation"):
    # Update mesh_geometric_moment
    mesh.mesh_geometric_moment += (
        params.time_step * mesh.mesh_interseismic_loading_rate * mesh.mesh.areas
    )
    mesh.mesh_geometric_moment_scalar[i + 1] = np.sum(mesh.mesh_geometric_moment)
    mesh.mesh_geometric_moment_scalar_non_zero[i + 1] = np.sum(
        mesh.mesh_geometric_moment[np.where(mesh.mesh_geometric_moment > 0.0)]
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
            mesh.mesh_geometric_moment_pre_event,
            params.location_probability_amplitude_scale_factor,
            params.location_probability_data_scale_factor,
        )
        event.hypocenter_triangle_index = np.random.choice(
            mesh.mesh.n_tde, params.n_samples, p=event.location_probability
        )[0]

        # Generate coseismic slip area and slip distribution
        event = skies.get_event_area_slip_triangle_index(mesh.mesh, event)
        event.mesh_geometric_moment_pre_event = np.copy(
            mesh.mesh_geometric_moment_pre_event
        )
        event.mesh_geometric_moment_post_event = np.copy(
            mesh.mesh_geometric_moment_pre_event
            - (event.slip_all_elements * mesh.mesh.areas)
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
        time_series.cumulate_omori_effect += (
            params.time_probability_history_scale_factor * omori_rate_perturbation
        )

        # Update spatially variable mesh parameters
        mesh.mesh_geometric_moment -= event.slip_all_elements * mesh.mesh.areas
        mesh.mesh_last_event_slip = event.slip_all_elements
        mesh.mesh_total_slip += event.slip_all_elements
        event.mesh_last_event_slip = event.slip_all_elements
        event.mesh_total_slip = mesh.mesh_total_slip

    else:
        # Create dummy event dictionary because no event occured
        event = skies.create_non_event(mesh.mesh.n_tde)
        event.mesh_geometric_moment_pre_event = np.copy(
            mesh.mesh_geometric_moment_pre_event
        )
        event.mesh_geometric_moment_post_event = (
            mesh.mesh_geometric_moment_pre_event
            + (params.time_step * mesh.mesh_interseismic_loading_rate * mesh.mesh.areas)
        )
        event.mesh_last_event_slip = mesh.mesh_last_event_slip
        event.mesh_total_slip = mesh.mesh_total_slip

    # TODO: Check this???
    event.mesh_initial_dip_slip_deficit = mesh.mesh_initial_dip_slip_deficit

    # Save event dictionary as pickle file
    event_pickle_file_name = f"{output_folder}/events/event_{i:010.0f}.pickle"
    with open(event_pickle_file_name, "wb") as pickle_file:
        pickle.dump(event, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    # Save time step parameters as .vtk files for plotting with Paraview and pyvista

    # Pre-event moment for next time step
    mesh.mesh_geometric_moment_pre_event = np.copy(
        event.mesh_geometric_moment_post_event
    )

    # Update probability
    time_series.probability[i + 1] = (
        time_series.cumulate_omori_effect[i]
        + mesh.mesh_geometric_moment_scalar_non_zero[i]
    )


print(np.where(time_series.event_magnitude > 0)[0])
plt.figure()
plt.plot(
    time_series.cumulate_omori_effect + mesh.mesh_geometric_moment_scalar_non_zero, "r+"
)
plt.plot(time_series.probability, "b.")

# plt.semilogy(
#     time_series.cumulate_omori_effect
#     + mesh.mesh_geometric_moment_scalar_non_zero
#     - time_series.probability,
#     "r+",
# )
plt.show(block=False)


end_time = datetime.datetime.now()
print(f"Event sequence generation run time: {str(end_time - start_time)}")

# Save time_series dictionary to .pickle file in output_folder
with open(output_folder + "/time_series.pickle", "wb") as pickle_file:
    pickle.dump(time_series, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

# Save mesh dictionary to .pickle file in output_folder
with open(output_folder + "/mesh.pickle", "wb") as pickle_file:
    pickle.dump(mesh, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

# Save random state to .pickle file in output_folder
with open(output_folder + "/random_state.pickle", "wb") as pickle_file:
    pickle.dump(np.random.get_state(), pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

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
