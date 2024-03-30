import glob
import pickle
import sys

import addict
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm

import skies

plt.close("all")

# Dumb copy and past of params from skies_time_05.py
params = addict.Dict()
params.n_time_steps = 40000
params.time_step = 5e-7
params.b_value = -1.0
params.shear_modulus = 3e10
params.n_samples = 1
params.n_binary = 2
params.minimum_event_moment_magnitude = 5.0
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

# Obtain list of pickled event files in run folder
run_folder = "./runs/2022_11_11_20_21_51/"
event_file_names = glob.glob(run_folder + "event_*.pickle")
event_file_names.sort()
n_event_files = len(event_file_names)
all_event_file_indices = range(n_event_files)

# Read event time series and extract mangitudes
time_series_moment_magnitude = np.zeros(n_event_files)
print(f"\nReading event_*.pickle files from {run_folder}")
for i in tqdm(range(n_event_files), colour="cyan"):
    event = pickle.load(open(event_file_names[i], "rb"))
    time_series_moment_magnitude[i] = event.moment_magnitude
print(f"Done reading event_*.pickle files from {run_folder}")

# Print magnitudes
event_indices = np.where(time_series_moment_magnitude > 0.0)[0]
if len(event_indices) > 0:
    print("\nEvents found:\n")
    for i in range(len(event_indices)):
        print(
            f"Moment magnitude = {time_series_moment_magnitude[event_indices[i]]:0.2f}"
            f" at time step {event_indices[i]}"
        )


plt.figure()
plt.plot(time_series_moment_magnitude, "rx")
plt.show(block=False)
# sys.exit()

# Plot a single event
event_file_index = 0


fontsize = 16
KM2_TO_M2 = 1e6


def plot_event_for_animation(
    params,
    event,
    meshes,
    pre_event_slip_deficit,
    last_event_slip,
    total_slip,
    iteration_step,
):
    """
    1. Slip deficit rate
    2. Current moment distribution
    3. Last earthquake
    4. Total slip
    """

    plt.figure(figsize=(10, 4))

    # Plot spatially variable temporally constant slip deficit rate
    print("Plotting slip deficit rate")
    plt.subplot(1, 4, 1)
    fill_value = event.mesh_initial_dip_slip_deficit
    x_vec = np.linspace(
        params.min_longitude, params.max_longitude, params.n_grid_longitude
    )
    y_vec = np.linspace(
        params.min_latitude, params.max_latitude, params.n_grid_latitude
    )
    x_mat, y_mat = np.meshgrid(x_vec, y_vec)
    centroids_lon = meshes[0].centroids[:, 0]
    centroids_lat = meshes[0].centroids[:, 1]
    # centroids_val = fill_value
    fill_value_mat = griddata(
        (centroids_lon, centroids_lat), fill_value, (x_mat, y_mat), method="cubic"
    )
    # Set values outside of mesh polygon to nan so they don't plot
    inpolygon_vals = skies.inpolygon(
        x_mat, y_mat, meshes[0].x_perimeter, meshes[0].y_perimeter
    )
    inpolygon_vals = np.reshape(
        inpolygon_vals, (params.n_grid_longitude, params.n_grid_latitude)
    )
    fill_value_mat[~inpolygon_vals] = np.nan
    cmap = cc.cm.bmy_r
    levels = np.linspace(0, 30, 11)
    plt.contourf(x_mat, y_mat, fill_value_mat, cmap=cmap, levels=levels, extend="both")
    plt.contour(
        x_mat,
        y_mat,
        fill_value_mat,
        colors="k",
        linestyles="solid",
        linewidths=0.25,
        levels=levels,
        extend="both",
    )
    plt.plot(meshes[0].x_perimeter, meshes[0].y_perimeter, "-k", linewidth=1.0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().set_facecolor("gainsboro")
    plt.plot(meshes[0].x_perimeter, meshes[0].y_perimeter, "-k")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("$v_{sd}$", fontsize=fontsize)

    # Plot spatially variable pre event moment
    print("Plotting pre-event moment")
    plt.subplot(1, 4, 2)
    fill_value = pre_event_slip_deficit
    x_vec = np.linspace(
        params.min_longitude, params.max_longitude, params.n_grid_longitude
    )
    y_vec = np.linspace(
        params.min_latitude, params.max_latitude, params.n_grid_latitude
    )
    x_mat, y_mat = np.meshgrid(x_vec, y_vec)
    centroids_lon = meshes[0].centroids[:, 0]
    centroids_lat = meshes[0].centroids[:, 1]
    # centroids_val = fill_value
    fill_value_mat = griddata(
        (centroids_lon, centroids_lat), fill_value, (x_mat, y_mat), method="cubic"
    )
    # Set values outside of mesh polygon to nan so they don't plot
    inpolygon_vals = skies.inpolygon(
        x_mat, y_mat, meshes[0].x_perimeter, meshes[0].y_perimeter
    )
    inpolygon_vals = np.reshape(
        inpolygon_vals, (params.n_grid_longitude, params.n_grid_latitude)
    )
    fill_value_mat[~inpolygon_vals] = np.nan
    cmap = cc.cm.CET_L19
    cmap = cc.cm.coolwarm
    levels = np.linspace(-5e9, 5e9, 11)
    plt.contourf(x_mat, y_mat, fill_value_mat, cmap=cmap, levels=levels, extend="both")
    plt.contour(
        x_mat,
        y_mat,
        fill_value_mat,
        colors="k",
        linestyles="solid",
        linewidths=0.25,
        levels=levels,
    )
    plt.plot(meshes[0].x_perimeter, meshes[0].y_perimeter, "-k", linewidth=1.0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().set_facecolor("gainsboro")
    plt.plot(meshes[0].x_perimeter, meshes[0].y_perimeter, "-k")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("$m$", fontsize=fontsize)

    # Plot most recent coseismic slip distribution
    print("Plotting last event")
    plt.subplot(1, 4, 3)
    fill_value = np.zeros(meshes[0].n_tde)
    fill_value = last_event_slip
    x_vec = np.linspace(
        params.min_longitude, params.max_longitude, params.n_grid_longitude
    )
    y_vec = np.linspace(
        params.min_latitude, params.max_latitude, params.n_grid_latitude
    )
    x_mat, y_mat = np.meshgrid(x_vec, y_vec)
    centroids_lon = meshes[0].centroids[:, 0]
    centroids_lat = meshes[0].centroids[:, 1]
    centroids_val = fill_value
    fill_value_mat = griddata(
        (centroids_lon, centroids_lat), fill_value, (x_mat, y_mat), method="cubic"
    )
    # Set values outside of mesh polygon to nan so they don't plot
    inpolygon_vals = skies.inpolygon(
        x_mat, y_mat, meshes[0].x_perimeter, meshes[0].y_perimeter
    )
    inpolygon_vals = np.reshape(
        inpolygon_vals, (params.n_grid_longitude, params.n_grid_latitude)
    )
    fill_value_mat[~inpolygon_vals] = np.nan
    cmap = cc.cm.CET_L19
    levels = np.linspace(0.1, 15, 11)
    plt.contourf(x_mat, y_mat, fill_value_mat, cmap=cmap, levels=levels, extend="both")
    plt.contour(
        x_mat,
        y_mat,
        fill_value_mat,
        colors="k",
        linestyles="solid",
        linewidths=0.25,
        levels=levels,
    )
    plt.plot(meshes[0].x_perimeter, meshes[0].y_perimeter, "-k", linewidth=1.0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().set_facecolor("gainsboro")
    plt.plot(meshes[0].x_perimeter, meshes[0].y_perimeter, "-k")
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel(
    #     f"$t=${last_event_time}, $M_W$={event.moment_magnitude[0]:0.1f}",
    #     fontsize=fontsize,
    # )

    plt.xlabel(
        f"$M_W$={event.moment_magnitude[0]:0.1f}",
        fontsize=fontsize,
    )
    # plt.xlabel(f"$t=${last_event_time}", fontsize=fontsize)

    # Plot total coseismic slip distribution
    print("Plotting total slip")
    plt.subplot(1, 4, 4)
    fill_value = np.zeros(meshes[0].n_tde)
    fill_value = total_slip
    x_vec = np.linspace(
        params.min_longitude, params.max_longitude, params.n_grid_longitude
    )
    y_vec = np.linspace(
        params.min_latitude, params.max_latitude, params.n_grid_latitude
    )
    x_mat, y_mat = np.meshgrid(x_vec, y_vec)
    centroids_lon = meshes[0].centroids[:, 0]
    centroids_lat = meshes[0].centroids[:, 1]
    # centroids_val = fill_value
    fill_value_mat = griddata(
        (centroids_lon, centroids_lat), fill_value, (x_mat, y_mat), method="cubic"
    )
    # Set values outside of mesh polygon to nan so they don't plot
    inpolygon_vals = skies.inpolygon(
        x_mat, y_mat, meshes[0].x_perimeter, meshes[0].y_perimeter
    )
    inpolygon_vals = np.reshape(
        inpolygon_vals, (params.n_grid_longitude, params.n_grid_latitude)
    )
    fill_value_mat[~inpolygon_vals] = np.nan
    cmap = cc.cm.bmy_r
    levels = np.linspace(0.1, 15, 11)
    plt.contourf(x_mat, y_mat, fill_value_mat, cmap=cmap, levels=levels, extend="both")
    plt.contour(
        x_mat,
        y_mat,
        fill_value_mat,
        colors="k",
        linestyles="solid",
        linewidths=0.25,
        levels=levels,
    )
    plt.plot(meshes[0].x_perimeter, meshes[0].y_perimeter, "-k", linewidth=1.0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().set_facecolor("gainsboro")
    plt.plot(meshes[0].x_perimeter, meshes[0].y_perimeter, "-k")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("$\sum s$", fontsize=fontsize)

    plt.suptitle(f"$t$={iteration_step}", fontsize=fontsize)

    # Save figure to file
    base_file_name = f"{iteration_step:010d}"
    plt.savefig(base_file_name + ".png", dpi=500)
    # plt.close("all")
    plt.show(block=False)


# Hacky read mesh file
mesh_parameters_file_name = "western_north_america_mesh_parameters.json"
meshes = skies.read_meshes(mesh_parameters_file_name)

event_indices = [
    2786,
    3877,
    4765,
    4778,
    5020,
    7472,
    7559,
    7683,
    7912,
    7993,
    8100,
    8233,
    8357,
    8670,
    9123,
]

event_file_index = 8357

event = pickle.load(open(event_file_names[event_file_index], "rb"))
print(f"Read: {event_file_names[event_file_index]}")
plot_event_for_animation(
    params,
    event,
    meshes,
    event.mesh_geometric_moment_pre_event,
    event.mesh_last_event_slip,
    event.mesh_total_slip,
    event_file_index,
)

# Write vtk file for visualization with paraview or pyvista
mesh_index = 0
vtk_file_name = skies.get_vtk_file_name(
    run_folder, mesh_parameters_file_name, mesh_index, event_file_index
)
skies.write_vtk_file(meshes[mesh_index], event.mesh_total_slip, "slip", vtk_file_name)
print(f"Wrote: {vtk_file_name}")
