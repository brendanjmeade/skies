import argparse
import datetime
import json
import logging
import os
import pickle
import sys
import uuid
import warnings
from contextlib import contextmanager

import addict
import colorcet as cc
import h5py
import matplotlib
import matplotlib.pyplot as plt
import meshio
import numpy as np
import scipy
from ismember import ismember
from rich.logging import RichHandler
from rich.progress import track

logger = logging.getLogger(__name__)

# Constants and parameters
N_BINARY = 2
N_GRID_X = 500
N_GRID_Y = 500
MAKE_EIGENVECTOR_EXAMPLE_PLOT = False
N_CONTOUR_LEVELS = 10
AREA_SCALING = (
    1.25  # Increases rupture area to partially compensate for sigmoid filtering
)
RADIUS_EARTH = 6371e3  # (m)
KM2_TO_M2 = 1e6  # kilometers squared to meters squared
KM2M = 1.0e3
DYNECM_TO_NM = 1e-7  # dynes centimeters to Newton meters
SHEAR_MODULUS = 3e10  # Shear modulus (Pa)
MINIMUM_EVENT_MOMENT_MAGNITUDE = 8.0
MAXIMUM_EVENT_MOMENT_MAGNITUDE = 8.1


def create_output_folder(base_runs_folder, output_path):
    # Check to see if "runs" folder exists and if not create it
    if not os.path.exists(base_runs_folder):
        os.mkdir(base_runs_folder)

    # Make output folder for current run
    # logger.info(f"Output folder: {output_path}")
    os.mkdir(output_path)
    os.mkdir(output_path + "/events")


def get_mesh_perimeter(meshes):
    for i in range(len(meshes)):
        x_coords = meshes[i].meshio_object.points[:, 0]
        y_coords = meshes[i].meshio_object.points[:, 1]
        meshes[i].x_perimeter = x_coords[meshes[i].ordered_edge_nodes[:, 0]]
        meshes[i].y_perimeter = y_coords[meshes[i].ordered_edge_nodes[:, 0]]
        meshes[i].x_perimeter = np.append(
            meshes[i].x_perimeter, x_coords[meshes[i].ordered_edge_nodes[0, 0]]
        )
        meshes[i].y_perimeter = np.append(
            meshes[i].y_perimeter, y_coords[meshes[i].ordered_edge_nodes[0, 0]]
        )


def triangle_normal(triangles):
    # The cross product of two sides is a normal vector
    # https://stackoverflow.com/questions/71346322/numpy-area-of-triangle-and-equation-of-a-plane-on-which-triangle-lies-on
    return np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], axis=1
    )


def triangle_area(triangles):
    # The norm of the cross product of two sides is twice the area
    # https://stackoverflow.com/questions/71346322/numpy-area-of-triangle-and-equation-of-a-plane-on-which-triangle-lies-on
    return np.linalg.norm(triangle_normal(triangles), axis=1) / 2.0


@contextmanager
def suppress_stderr():
    """
    https://stackoverflow.com/questions/4178614/suppressing-output-of-module-calling-outside-library
    """
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


@contextmanager
def suppress_stdout():
    """
    https://stackoverflow.com/questions/4178614/suppressing-output-of-module-calling-outside-library
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# Read mesh data - List of dictionaries version
def read_meshes(mesh_parameters_file_name):
    meshes = []
    if mesh_parameters_file_name != "":
        with open(mesh_parameters_file_name) as f:
            mesh_param = json.load(f)
            logger.info(f"Read: {mesh_parameters_file_name}")

        if len(mesh_param) > 0:
            for i in range(len(mesh_param)):
                meshes.append(addict.Dict())
                with suppress_stdout():
                    with suppress_stderr():
                        meshes[i].meshio_object = meshio.read(
                            mesh_param[i]["mesh_filename"]
                        )

                meshes[i].file_name = mesh_param[i]["mesh_filename"]
                meshes[i].verts = meshes[i].meshio_object.get_cells_type("triangle")

                # Expand mesh coordinates
                meshes[i].lon1 = meshes[i].meshio_object.points[
                    meshes[i].verts[:, 0], 0
                ]
                meshes[i].lon2 = meshes[i].meshio_object.points[
                    meshes[i].verts[:, 1], 0
                ]
                meshes[i].lon3 = meshes[i].meshio_object.points[
                    meshes[i].verts[:, 2], 0
                ]
                meshes[i].lat1 = meshes[i].meshio_object.points[
                    meshes[i].verts[:, 0], 1
                ]
                meshes[i].lat2 = meshes[i].meshio_object.points[
                    meshes[i].verts[:, 1], 1
                ]
                meshes[i].lat3 = meshes[i].meshio_object.points[
                    meshes[i].verts[:, 2], 1
                ]
                meshes[i].dep1 = meshes[i].meshio_object.points[
                    meshes[i].verts[:, 0], 2
                ]
                meshes[i].dep2 = meshes[i].meshio_object.points[
                    meshes[i].verts[:, 1], 2
                ]
                meshes[i].dep3 = meshes[i].meshio_object.points[
                    meshes[i].verts[:, 2], 2
                ]
                meshes[i].centroids = np.mean(
                    meshes[i].meshio_object.points[meshes[i].verts, :], axis=1
                )

                # Cartesian coordinates in meters
                meshes[i].x1, meshes[i].y1, meshes[i].z1 = sph2cart(
                    meshes[i].lon1,
                    meshes[i].lat1,
                    RADIUS_EARTH + KM2M * meshes[i].dep1,
                )
                meshes[i].x2, meshes[i].y2, meshes[i].z2 = sph2cart(
                    meshes[i].lon2,
                    meshes[i].lat2,
                    RADIUS_EARTH + KM2M * meshes[i].dep2,
                )
                meshes[i].x3, meshes[i].y3, meshes[i].z3 = sph2cart(
                    meshes[i].lon3,
                    meshes[i].lat3,
                    RADIUS_EARTH + KM2M * meshes[i].dep3,
                )

                # Cartesian triangle centroids
                meshes[i].x_centroid = (
                    meshes[i].x1 + meshes[i].x2 + meshes[i].x3
                ) / 3.0
                meshes[i].y_centroid = (
                    meshes[i].y1 + meshes[i].y2 + meshes[i].y3
                ) / 3.0
                meshes[i].z_centroid = (
                    meshes[i].z1 + meshes[i].z2 + meshes[i].z3
                ) / 3.0

                # Cross products for orientations
                tri_leg1 = np.transpose(
                    [
                        np.deg2rad(meshes[i].lon2 - meshes[i].lon1),
                        np.deg2rad(meshes[i].lat2 - meshes[i].lat1),
                        (1 + KM2M * meshes[i].dep2 / RADIUS_EARTH)
                        - (1 + KM2M * meshes[i].dep1 / RADIUS_EARTH),
                    ]
                )
                tri_leg2 = np.transpose(
                    [
                        np.deg2rad(meshes[i].lon3 - meshes[i].lon1),
                        np.deg2rad(meshes[i].lat3 - meshes[i].lat1),
                        (1 + KM2M * meshes[i].dep3 / RADIUS_EARTH)
                        - (1 + KM2M * meshes[i].dep1 / RADIUS_EARTH),
                    ]
                )

                def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                    """This exists only because of a Pylance reporting complication
                    associated with a numpy stub bug for np.cross:
                    https://github.com/microsoft/pylance-release/issues/3277#issuecomment-1237782014

                    This function simply exists to eliminate the cosmetics of a
                    incorrect "code unreachable".

                    If numpy fixes this bug we can go back to the sensible:
                    meshes[i].nv = np.cross(tri_leg1, tri_leg2)
                    """
                    return np.cross(a, b)

                meshes[i].nv = cross2(tri_leg1, tri_leg2)
                azimuth, elevation, r = cart2sph(
                    meshes[i].nv[:, 0], meshes[i].nv[:, 1], meshes[i].nv[:, 2]
                )
                meshes[i].strike = wrap2360(-np.rad2deg(azimuth))
                meshes[i].dip = 90 - np.rad2deg(elevation)
                meshes[i].dip_flag = meshes[i].dip != 90
                meshes[i].smoothing_weight = mesh_param[i]["smoothing_weight"]
                meshes[i].top_slip_rate_constraint = mesh_param[i][
                    "top_slip_rate_constraint"
                ]
                meshes[i].bot_slip_rate_constraint = mesh_param[i][
                    "bot_slip_rate_constraint"
                ]
                meshes[i].side_slip_rate_constraint = mesh_param[i][
                    "side_slip_rate_constraint"
                ]
                meshes[i].n_tde = meshes[i].lon1.size

                # Calcuate areas of each triangle in mesh
                triangle_vertex_array = np.zeros((meshes[i].n_tde, 3, 3))
                triangle_vertex_array[:, 0, 0] = meshes[i].x1
                triangle_vertex_array[:, 1, 0] = meshes[i].x2
                triangle_vertex_array[:, 2, 0] = meshes[i].x3
                triangle_vertex_array[:, 0, 1] = meshes[i].y1
                triangle_vertex_array[:, 1, 1] = meshes[i].y2
                triangle_vertex_array[:, 2, 1] = meshes[i].y3
                triangle_vertex_array[:, 0, 2] = meshes[i].z1
                triangle_vertex_array[:, 1, 2] = meshes[i].z2
                triangle_vertex_array[:, 2, 2] = meshes[i].z3
                meshes[i].areas = triangle_area(triangle_vertex_array)
                get_mesh_edge_elements(meshes)
                logger.info(f"Read: {mesh_param[i]['mesh_filename']}")
            get_mesh_perimeter(meshes)
    return meshes


def print_event(event, meshes):
    logger.info("Event data:")
    logger.info(
        f"Hypocenter longitude = {meshes[0].centroids[event.hypocenter_triangle_index, 0][0]:0.4f} (deg)"
    )
    logger.info(
        f"Hypocenter latitude = {meshes[0].centroids[event.hypocenter_triangle_index, 1][0]:0.4f} (deg)"
    )
    logger.info(
        f"Hypocenter depth = {meshes[0].centroids[event.hypocenter_triangle_index, 2][0]:0.4f} (km)"
    )
    logger.info(f"Hypocenter triangle index = {event.hypocenter_triangle_index[0]}")
    logger.info(f"Mean slip = {np.mean(event.slip):0.2f} (m)")
    logger.info(f"Minimum slip = {np.min(event.slip):0.2f} (m)")
    logger.info(f"Maximum slip = {np.max(event.slip):0.2f} (m)")
    logger.info(f"Moment magnitude = {event.moment_magnitude[0]}")
    logger.info(f"Moment = {event.moment[0]:0.3} (N m)")
    logger.info(f"Number of eigenvalues = {event.n_eigenvalues}")
    logger.info(f"Rupture area = {event.actual_area / 1e6:0.2f} (km^2)")
    logger.info(f"Scaling law rupture area = {event.target_area[0] / 1e6:0.2f} (km^2)")


def print_magnitude_overview(mesh):
    minimum_single_triangle_moment_magnitude = area_to_moment_magnitude_allen_and_hayes(
        np.min(mesh.areas)
    )
    maximum_single_triangle_moment_magnitude = area_to_moment_magnitude_allen_and_hayes(
        np.max(mesh.areas)
    )
    maximum_moment_magnitude = area_to_moment_magnitude_allen_and_hayes(
        np.sum(mesh.areas)
    )

    logger.info("Magnitude analysis of mesh and area scaling relationships:")
    logger.info(
        f"Maximum moment magnitude of entire mesh = {maximum_moment_magnitude:0.2f}"
    )
    logger.info(
        f"Maximum moment magnitude of single mesh element = {maximum_single_triangle_moment_magnitude:0.2f}"
    )
    logger.info(
        f"Minimum moment magnitude of single mesh element = {minimum_single_triangle_moment_magnitude:0.2f}"
    )


def create_event(meshes, probability):
    # Dictionary for storing `event` information for a single earthquake
    event = addict.Dict()

    # Select random event magnitude from GR distribution
    b_value = -1.0
    event.moment_magnitude = get_gutenberg_richter_magnitude(
        b_value, MINIMUM_EVENT_MOMENT_MAGNITUDE, MAXIMUM_EVENT_MOMENT_MAGNITUDE
    )

    event.moment = 10 ** (1.5 * (event.moment_magnitude + 10.7) - 7.0)
    event.target_area = AREA_SCALING * moment_magnitude_to_area_allen_and_hayes(
        event.moment_magnitude
    )
    event.total_mesh_area = np.sum(meshes[0].areas)

    # Select random triangle, should be based on coupling or slip deficit rate/history
    triangle_probabilities = probability
    event.hypocenter_triangle_index = np.random.choice(
        meshes[0].n_tde, 1, p=triangle_probabilities
    )

    # Calculate distance from hypocenter triangle toto all other triangles
    event.hypocenter_triangle_to_all_triangles_distances = (
        get_hypocenter_triangle_to_all_triangles_distances(meshes, event)
    )

    # Find the triangles close to the hypocenter that accumulate enough area to be a
    # part of the event rupture
    sorted_distance_index = np.argsort(
        event.hypocenter_triangle_to_all_triangles_distances
    )
    cumulative_area = np.cumsum(meshes[0].areas[sorted_distance_index])
    event.triangle_index = sorted_distance_index[
        np.where(cumulative_area < event.target_area)[0]
    ]
    event.actual_area = np.sum(meshes[0].areas[event.triangle_index])
    event.mean_slip = event.moment / (SHEAR_MODULUS * event.actual_area)

    # Calculate eigenvalues and eigenvectors for the current event area
    # Were not storing these in event because that would start to consume a
    # lot of RAM as we build histories of multiple events
    event.n_eigenvalues = event.triangle_index.size
    eigenvalues, eigenvectors = get_eigenvalues_and_eigenvectors(
        event.n_eigenvalues,
        (meshes[0].x_centroid[event.hypocenter_triangle_index] - meshes[0].x_centroid)[
            event.triangle_index
        ],
        (meshes[0].y_centroid[event.hypocenter_triangle_index] - meshes[0].y_centroid)[
            event.triangle_index
        ],
        (meshes[0].z_centroid[event.hypocenter_triangle_index] - meshes[0].z_centroid)[
            event.triangle_index
        ],
    )

    # Generate random slip distribution for current rupture patch
    event = get_event_slip(meshes, event, eigenvalues, eigenvectors)
    return event


def get_location_probability(slip_deficit):
    # Map slip defict to earthquake probability
    temp_slip_deficit = np.copy(slip_deficit)
    temp_slip_deficit[temp_slip_deficit < 0.0] = 0.0
    if np.min(temp_slip_deficit) == np.max(temp_slip_deficit):
        probability = np.zeros_like(temp_slip_deficit)
    else:
        probability = 1 - normalized_sigmoid(1e-5, 1e-1, temp_slip_deficit)
        probability = probability - np.min(probability)
        probability = probability / np.max(probability)
        probability = probability / np.sum(probability)
    return probability


def sph2cart(lon, lat, radius):
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z


def cart2sph(x, y, z):
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


def wrap2360(lon):
    lon[np.where(lon < 0.0)] += 360.0
    return lon


def get_mesh_edge_elements(meshes):
    # Find indices of elements lining top, bottom, and sides of each mesh

    get_ordered_edge_nodes(meshes)

    for i in range(len(meshes)):
        coords = meshes[i].meshio_object.points
        vertices = meshes[i].verts

        # Arrays of all element side node pairs
        side_1 = np.sort(np.vstack((vertices[:, 0], vertices[:, 1])).T, 1)
        side_2 = np.sort(np.vstack((vertices[:, 1], vertices[:, 2])).T, 1)
        side_3 = np.sort(np.vstack((vertices[:, 2], vertices[:, 0])).T, 1)

        # Sort edge node array
        sorted_edge_nodes = np.sort(meshes[i].ordered_edge_nodes, 1)

        # Indices of element sides that are in edge node array
        side_1_in_edge, side_1_in_edge_idx = ismember(sorted_edge_nodes, side_1, "rows")
        side_2_in_edge, side_2_in_edge_idx = ismember(sorted_edge_nodes, side_2, "rows")
        side_3_in_edge, side_3_in_edge_idx = ismember(sorted_edge_nodes, side_3, "rows")

        # Depths of nodes
        side_1_depths = np.abs(
            coords[
                np.column_stack(
                    (side_1[side_1_in_edge_idx, :], vertices[side_1_in_edge_idx, 2])
                ),
                2,
            ]
        )
        side_2_depths = np.abs(
            coords[
                np.column_stack(
                    (side_2[side_2_in_edge_idx, :], vertices[side_2_in_edge_idx, 0])
                ),
                2,
            ]
        )
        side_3_depths = np.abs(
            coords[
                np.column_stack(
                    (side_3[side_3_in_edge_idx, :], vertices[side_3_in_edge_idx, 1])
                ),
                2,
            ]
        )
        # Top elements are those where the depth difference between the non-edge node
        # and the mean of the edge nodes is greater than the depth difference between
        # the edge nodes themselves
        top1 = (side_1_depths[:, 2] - np.mean(side_1_depths[:, 0:2], 1)) > (
            np.abs(side_1_depths[:, 0] - side_1_depths[:, 1])
        )
        top2 = (side_2_depths[:, 2] - np.mean(side_2_depths[:, 0:2], 1)) > (
            np.abs(side_2_depths[:, 0] - side_2_depths[:, 1])
        )
        top3 = (side_3_depths[:, 2] - np.mean(side_3_depths[:, 0:2], 1)) > (
            np.abs(side_3_depths[:, 0] - side_3_depths[:, 1])
        )
        tops = np.full(len(vertices), False, dtype=bool)
        tops[side_1_in_edge_idx[top1]] = True
        tops[side_2_in_edge_idx[top2]] = True
        tops[side_3_in_edge_idx[top3]] = True
        meshes[i].top_elements = tops

        # Bottom elements are those where the depth difference between the non-edge node
        # and the mean of the edge nodes is more negative than the depth difference between
        # the edge nodes themselves
        bot1 = side_1_depths[:, 2] - np.mean(side_1_depths[:, 0:2], 1) < -np.abs(
            side_1_depths[:, 0] - side_1_depths[:, 1]
        )
        bot2 = side_2_depths[:, 2] - np.mean(side_2_depths[:, 0:2], 1) < -np.abs(
            side_2_depths[:, 0] - side_2_depths[:, 1]
        )
        bot3 = side_3_depths[:, 2] - np.mean(side_3_depths[:, 0:2], 1) < -np.abs(
            side_3_depths[:, 0] - side_3_depths[:, 1]
        )
        bots = np.full(len(vertices), False, dtype=bool)
        bots[side_1_in_edge_idx[bot1]] = True
        bots[side_2_in_edge_idx[bot2]] = True
        bots[side_3_in_edge_idx[bot3]] = True
        meshes[i].bot_elements = bots

        # Side elements are a set difference between all edges and tops, bottoms
        sides = np.full(len(vertices), False, dtype=bool)
        sides[side_1_in_edge_idx] = True
        sides[side_2_in_edge_idx] = True
        sides[side_3_in_edge_idx] = True
        sides[np.where(tops != 0)] = False
        sides[np.where(bots != 0)] = False
        meshes[i].side_elements = sides


def get_ordered_edge_nodes(meshes):
    """Find exterior edges of each mesh and return them in the dictionary
    for each mesh.

    Args:
        meshes (List): list of mesh dictionaries
    """

    for i in range(len(meshes)):
        # Make side arrays containing vertex indices of sides
        vertices = meshes[i].verts
        side_1 = np.sort(np.vstack((vertices[:, 0], vertices[:, 1])).T, 1)
        side_2 = np.sort(np.vstack((vertices[:, 1], vertices[:, 2])).T, 1)
        side_3 = np.sort(np.vstack((vertices[:, 2], vertices[:, 0])).T, 1)
        all_sides = np.vstack((side_1, side_2, side_3))
        unique_sides, sides_count = np.unique(all_sides, return_counts=True, axis=0)
        edge_nodes = unique_sides[np.where(sides_count == 1)]

        meshes[i].ordered_edge_nodes = np.zeros_like(edge_nodes)
        meshes[i].ordered_edge_nodes[0, :] = edge_nodes[0, :]
        last_row = 0
        for j in range(1, len(edge_nodes)):
            idx = np.where(
                (edge_nodes == meshes[i].ordered_edge_nodes[j - 1, 1])
            )  # Edge node indices the same as previous row, second column
            next_idx = np.where(
                idx[0][:] != last_row
            )  # One of those indices is the last row itself. Find the other row index
            next_row = idx[0][next_idx]  # Index of the next ordered row
            next_col = idx[1][next_idx]  # Index of the next ordered column (1 or 2)
            if next_col == 1:
                next_col_ord = [1, 0]  # Flip edge ordering
            else:
                next_col_ord = [0, 1]
            meshes[i].ordered_edge_nodes[j, :] = edge_nodes[next_row, next_col_ord]
            last_row = (
                next_row  # Update last_row so that it's excluded in the next iteration
            )


def inpolygon(xq, yq, xv, yv):
    """From: https://stackoverflow.com/questions/31542843/inpolygon-examples-of-matplotlib-path-path-contains-points-method

    Args:
        xq : x coordinates of points to test
        yq : y coordinates of points to test
        xv : x coordinates of polygon vertices
        yv : y coordinates of polygon vertices

    Returns:
        _type_: Boolean like for in or out of polygon
    """
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = matplotlib.path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)


def rbf_interpolate(fill_value):
    # Observation coordinates and data
    x_vec = np.linspace(231, 239, N_GRID_X)
    y_vec = np.linspace(38, 52, N_GRID_Y)
    x_mat, y_mat = np.meshgrid(x_vec, y_vec)
    y_mat = y_mat
    centroids_lon = meshes[0].centroids[:, 0]
    centroids_lat = meshes[0].centroids[:, 1]
    centroids_val = fill_value

    # Package for RBFInterpolator
    xgrid = np.stack((x_mat, y_mat))
    xflat = xgrid.reshape(2, -1).T
    xobs = np.vstack((centroids_lon, centroids_lat)).T
    yobs = centroids_val
    yflat = scipy.interpolate.RBFInterpolator(
        xobs, yobs, kernel="cubic", smoothing=0.01, epsilon=1.5
    )(xflat)
    ygrid = yflat.reshape(N_GRID_X, N_GRID_Y)
    return xgrid, ygrid


def rbf_interpolate_single_mesh(mesh, params, fill_value):
    # Observation coordinates and data
    x_vec = np.linspace(
        params.min_longitude, params.max_longitude, params.n_grid_longitude
    )
    y_vec = np.linspace(
        params.min_latitude, params.max_latitude, params.n_grid_latitude
    )
    x_mat, y_mat = np.meshgrid(x_vec, y_vec)
    # y_mat = y_mat
    centroids_lon = mesh.centroids[:, 0]
    centroids_lat = mesh.centroids[:, 1]
    centroids_val = fill_value

    # Package for RBFInterpolator
    xgrid = np.stack((x_mat, y_mat))
    xflat = xgrid.reshape(2, -1).T
    xobs = np.vstack((centroids_lon, centroids_lat)).T
    yobs = centroids_val
    yflat = scipy.interpolate.RBFInterpolator(
        xobs, yobs, kernel="cubic", smoothing=0.0000, epsilon=1.0
    )(xflat)
    ygrid = yflat.reshape(params.n_grid_longitude, params.n_grid_latitude)
    return xgrid, ygrid


def get_synthetic_displacements(mesh, tri_linear_operator):
    """
    Prescribe dip-slip in a Gaussian pattern
    """
    tri_centroid_to_mesh_lon = mesh.centroids[:, 0] - np.mean(mesh.centroids[:, 0])
    tri_centroid_to_mesh_lat = mesh.centroids[:, 1] - np.mean(mesh.centroids[:, 1])

    # Hardcoded northern Cascadia example that Jack suggested.
    tri_centroid_to_mesh_lon = mesh.centroids[:, 0] - 234.5
    tri_centroid_to_mesh_lat = mesh.centroids[:, 1] - 48.5

    # Southern Cascadia example
    tri_centroid_to_mesh_lon = mesh.centroids[:, 0] - np.mean(mesh.centroids[:, 0]) - 2
    tri_centroid_to_mesh_lat = mesh.centroids[:, 1] - np.mean(mesh.centroids[:, 1])

    tri_centroid_to_mesh_centroid_distance = np.sqrt(
        tri_centroid_to_mesh_lon**2 + tri_centroid_to_mesh_lat**2
    )
    dip_slip_distribution = np.exp(
        -((tri_centroid_to_mesh_centroid_distance / 1.0) ** 2.0)
    )
    slip_distribution = np.zeros(2 * dip_slip_distribution.size)
    slip_distribution[1::2] = dip_slip_distribution  # Dip slip only
    slip_distribution[0::2] = 1e-4 * np.random.randn(
        dip_slip_distribution.size
    )  # Adding a teeny amount of non-zero noise here just so contouring works...ugh
    synthetic_displacements = tri_linear_operator @ slip_distribution
    return slip_distribution, synthetic_displacements


def plot_slip_distributions(
    slip_distribution_input, slip_distribution_estimated, suptitle_string
):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.title("input strike-slip")
    interpolate_and_plot(slip_distribution_input[0::2])

    plt.subplot(1, 4, 2)
    plt.title("estimated strike-slip")
    interpolate_and_plot(slip_distribution_estimated[0::2])

    plt.subplot(1, 4, 3)
    plt.title("input dip-slip")
    interpolate_and_plot(slip_distribution_input[1::2])

    plt.subplot(1, 4, 4)
    plt.title("estimated dip-slip")
    interpolate_and_plot(slip_distribution_estimated[1::2])

    plt.suptitle(suptitle_string)
    # plt.show()


def interpolate_and_plot(fill_value):
    # Interpolate values onto a regular grid for plotting
    # If the fill value has very little variation so it can be contoured
    if fill_value.ptp() < 1e-4:
        fill_value = 1e-4 * np.ones_like(fill_value)

    xgrid, ygrid = rbf_interpolate(fill_value)
    xflat = xgrid.reshape(2, -1).T
    inpolygon_vals = inpolygon(
        xflat[:, 0], xflat[:, 1], meshes[0].x_perimeter, meshes[0].y_perimeter
    )
    inpolygon_vals = np.reshape(inpolygon_vals, (N_GRID_X, N_GRID_Y))
    ygrid[~inpolygon_vals] = np.nan

    # Plot
    levels = np.linspace(-1.0, 1.0, N_CONTOUR_LEVELS)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="No contour levels were found within the data range."
        )
        plt.contourf(*xgrid, ygrid, cmap=cmap, levels=levels, extend="both")
        plt.contour(
            *xgrid,
            ygrid,
            colors="k",
            linestyles="solid",
            linewidths=0.25,
            levels=levels,
        )
    plt.plot(meshes[0].x_perimeter, meshes[0].y_perimeter, "-k", linewidth=1.0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xticks([])
    plt.yticks([])


def get_eigenvalues_and_eigenvectors(n_eigenvalues, x, y, z):
    n_tde = x.size

    # Calculate Cartesian distances between triangle centroids
    centroid_coordinates = np.array([x, y, z]).T
    distance_matrix = scipy.spatial.distance.cdist(
        centroid_coordinates, centroid_coordinates, "euclidean"
    )

    # Rescale distance matrix to the range 0-1
    distance_matrix = (distance_matrix - np.min(distance_matrix)) / np.ptp(
        distance_matrix
    )

    # Calculate correlation matrix
    correlation_matrix = np.exp(-distance_matrix)

    # https://stackoverflow.com/questions/12167654/fastest-way-to-compute-k-largest-eigenvalues-and-corresponding-eigenvectors-with
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        correlation_matrix,
        subset_by_index=[n_tde - n_eigenvalues, n_tde - 1],
    )
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    ordered_index = np.flip(np.argsort(eigenvalues))
    eigenvalues = eigenvalues[ordered_index]
    eigenvectors = eigenvectors[:, ordered_index]
    return eigenvalues, eigenvectors


def get_synthetic_accumulated_slip(mesh, sources):
    slip_distribution = np.zeros(2 * mesh.n_tde)
    strike_slip_distribution = np.zeros(mesh.n_tde)
    dip_slip_distribution = np.zeros(mesh.n_tde)

    for i in range(sources.lon.size):
        source_to_mesh_centroid_lon = mesh.centroids[:, 0] - sources.lon[i]
        source_to_mesh_centroid_lat = mesh.centroids[:, 1] - sources.lat[i]

        source_to_mesh_centroid_distance = np.sqrt(
            source_to_mesh_centroid_lon**2.0 + source_to_mesh_centroid_lat**2.0
        )

        # Guassian slip pattern
        if sources.slip_type[i] == "strike_slip":
            strike_slip_distribution += sources.magnitude[i] * np.exp(
                -((source_to_mesh_centroid_distance / 1.0) ** 2.0)
            )
        elif sources.slip_type[i] == "dip_slip":
            dip_slip_distribution += sources.magnitude[i] * np.exp(
                -((source_to_mesh_centroid_distance / 1.0) ** 2.0)
            )

    slip_distribution[0::2] = strike_slip_distribution  # Strike slip only
    slip_distribution[1::2] = dip_slip_distribution  # Dip slip only
    return slip_distribution


def plot_meshes(mesh, fill_value, ax, cmap_string):
    x_coords = mesh.meshio_object.points[:, 0]
    y_coords = mesh.meshio_object.points[:, 1]
    vertex_array = np.asarray(mesh.verts)

    if not ax:
        ax = plt.gca()
    xy = np.c_[x_coords, y_coords]
    verts = xy[vertex_array]
    pc = matplotlib.collections.PolyCollection(
        verts,
        edgecolor="k",
        cmap=cmap_string,
        linewidth=0.1,
        alpha=1.0,
    )
    pc.set_array(fill_value)
    ax.add_collection(pc)
    ax.autoscale()
    plt.gca().set_aspect("equal")
    return pc


def moment_magnitude_to_area_allen_and_hayes(moment_magnitude):
    """Calculate emperically estimated slip areas using
    relationships from:

    Allen and Hayes (2017), Alternative Rupture-Scaling
    Relationships for Subduction Interface and Other Offshore
    Environments, Bulletin of the Seismological Society of America,
    Vol. 107, No. 3, pp. 1240–1253, June 2017, doi: 10.1785/0120160255

    All values taken from their table 2

    Note: $S_2$ in the paper's notation is what we use for rupture_area

    Args:
        moment_magnitude: Array of moment magnitudes

    Returns:
        area: rupture area in meters squared
    """
    hinge_moment_magnitude = 8.63
    if moment_magnitude <= hinge_moment_magnitude:
        a = -5.62
        b = 1.22
    elif moment_magnitude > hinge_moment_magnitude:
        a = 2.23
        b = 0.31
    area = KM2_TO_M2 * 10 ** (a + b * moment_magnitude)
    return area


def get_gutenberg_richter_magnitude(b_value, minimum_magnitude, maximum_magnitude):
    """
    Return a random magnitude from the Gutenberg-Ricter distribution with
    slope b_value and bounded by minimum_magnitude and maximum_magnitude

    """

    # Set an initial magnitude that is much larger than is anticipated.
    # The purpose of this is to trigger entry into the while loop which
    # will iterate to make sure that no evetns larger than
    # `maximum_magnitude` are returned.
    magnitude = 1e6

    rng = np.random.RandomState()
    while magnitude > maximum_magnitude:
        magnitude = minimum_magnitude + rng.exponential(
            1.0 / (-b_value / np.log10(np.e)), 1
        )
    return magnitude


def normalized_sigmoid(a, b, x):
    """
    Returns array of a horizontal mirrored normalized sigmoid function
    output between 0 and 1
    Function parameters a = center; b = width
    https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
    """
    s = 1 / (1 + np.exp(b * -(x - a)))
    s = 1 * (s - np.min(s)) / (np.max(s) - np.min(s))  # normalize function to 0-1
    return s


# Visualize eigenvectors
def interpolate_and_plot(fill_value):
    # Interpolate values onto a regular grid for plotting
    # If the fill value has very little variation so it can be contoured
    if fill_value.ptp() < 1e-4:
        fill_value = 1e-4 * np.ones_like(fill_value)

    xgrid, ygrid = rbf_interpolate(fill_value)
    xflat = xgrid.reshape(2, -1).T
    inpolygon_vals = inpolygon(
        xflat[:, 0], xflat[:, 1], meshes[0].x_perimeter, meshes[0].y_perimeter
    )
    inpolygon_vals = np.reshape(inpolygon_vals, (N_GRID_X, N_GRID_Y))
    ygrid[~inpolygon_vals] = np.nan

    # Plot
    levels = np.linspace(-1.0, 1.0, N_CONTOUR_LEVELS)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="No contour levels were found within the data range."
        )
        plt.contourf(*xgrid, ygrid, cmap=cmap, levels=levels, extend="both")
        plt.contour(
            *xgrid,
            ygrid,
            colors="k",
            linestyles="solid",
            linewidths=0.25,
            levels=levels,
        )
    plt.plot(meshes[0].x_perimeter, meshes[0].y_perimeter, "-k", linewidth=1.0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xticks([])
    plt.yticks([])


def get_hypocenter_triangle_to_all_triangles_distances(meshes, event):
    # Find distance between current index mesh triangle all others
    x_centroid = (
        meshes[0].x_centroid[event.hypocenter_triangle_index] - meshes[0].x_centroid
    )
    y_centroid = (
        meshes[0].y_centroid[event.hypocenter_triangle_index] - meshes[0].y_centroid
    )
    z_centroid = (
        meshes[0].z_centroid[event.hypocenter_triangle_index] - meshes[0].z_centroid
    )

    all_triangle_cartesian_centroid_coordinates = np.vstack(
        (x_centroid, y_centroid, z_centroid)
    ).T

    hypocenter_triangle_cartesian_centroid_coordinates = np.array(
        [
            x_centroid[event.hypocenter_triangle_index],
            y_centroid[event.hypocenter_triangle_index],
            z_centroid[event.hypocenter_triangle_index],
        ]
    ).T

    hypocenter_triangle_to_all_triangles_distances = scipy.spatial.distance.cdist(
        hypocenter_triangle_cartesian_centroid_coordinates,
        all_triangle_cartesian_centroid_coordinates,
        "euclidean",
    )

    hypocenter_triangle_to_all_triangles_distances = np.squeeze(
        hypocenter_triangle_to_all_triangles_distances
    )

    return hypocenter_triangle_to_all_triangles_distances


def get_hypocenter_triangle_to_all_triangles_distances_single_mesh(
    mesh, hypocenter_triangle_index
):
    # Find distance between current index mesh triangle all others
    x_centroid = mesh.x_centroid[hypocenter_triangle_index] - mesh.x_centroid
    y_centroid = mesh.y_centroid[hypocenter_triangle_index] - mesh.y_centroid
    z_centroid = mesh.z_centroid[hypocenter_triangle_index] - mesh.z_centroid

    all_triangle_cartesian_centroid_coordinates = np.vstack(
        (x_centroid, y_centroid, z_centroid)
    ).T

    hypocenter_triangle_cartesian_centroid_coordinates = np.array(
        [
            x_centroid[hypocenter_triangle_index],
            y_centroid[hypocenter_triangle_index],
            z_centroid[hypocenter_triangle_index],
        ]
    )[None, :]

    hypocenter_triangle_to_all_triangles_distances = scipy.spatial.distance.cdist(
        hypocenter_triangle_cartesian_centroid_coordinates,
        all_triangle_cartesian_centroid_coordinates,
        "euclidean",
    )

    hypocenter_triangle_to_all_triangles_distances = np.squeeze(
        hypocenter_triangle_to_all_triangles_distances
    )

    return hypocenter_triangle_to_all_triangles_distances


def get_event_slip(meshes, event, eigenvalues, eigenvectors):
    event.slip = np.zeros(event.triangle_index.size)
    weights = np.random.randn(eigenvalues.size)
    for k in range(1, weights.size):
        event.slip += weights[k] * np.sqrt(eigenvalues[k]) * eigenvectors[:, k]
    event.slip = np.exp(event.slip)

    # Apply taper to slip.  This is ad hoc and may need revision
    distances = event.hypocenter_triangle_to_all_triangles_distances[
        event.triangle_index
    ]
    taper_transition = 1.5 * np.mean(distances)
    taper_width = 10 / taper_transition  # m

    slip_taper = normalized_sigmoid(taper_transition, taper_width, distances)
    event.slip = event.slip * slip_taper

    # After taper is applied rescale slip magnitudes to get the correct moment
    event.pre_scaled_moment = SHEAR_MODULUS * np.sum(
        event.slip * meshes[0].areas[event.triangle_index]
    )
    event.slip_scaling_factor = event.moment / event.pre_scaled_moment
    event.slip = event.slip * event.slip_scaling_factor

    event.slip_all_elements = np.zeros(meshes[0].n_tde)
    event.slip_all_elements[event.triangle_index] = event.slip

    return event


def plot_initial_data(mesh, initial_slip_deficit_rate, output_folder):
    # Plot all mesh data and initial slip deficit condition
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 4, 1)
    pc = plot_meshes(mesh, np.zeros(mesh.areas.size), plt.gca(), "Blues")
    plt.plot(mesh.x_perimeter, mesh.y_perimeter, "-k")
    plt.title(f"{mesh.n_tde} triangles")

    plt.subplot(1, 4, 2)
    pc = plot_meshes(mesh, mesh.areas / 1e6, plt.gca(), "plasma")
    plt.colorbar(pc, label="triangle areas (km^2)")
    plt.plot(mesh.x_perimeter, mesh.y_perimeter, "-k")
    plt.title(f"{np.sum(mesh.areas) / KM2_TO_M2:0.2f} (km^2)")

    plt.subplot(1, 4, 3)
    pc = plot_meshes(mesh, initial_slip_deficit_rate, plt.gca(), "inferno_r")
    plt.colorbar(pc, label="slip deficit rate (mm/yr)")
    plt.plot(mesh.x_perimeter, mesh.y_perimeter, "-k")
    plt.title(f"{np.max(initial_slip_deficit_rate):0.2f} (mm/yr)")

    plt.savefig(output_folder + "/initial_mesh_data.pdf")
    plt.savefig(output_folder + "/initial_mesh_data.png", dpi=500)
    plt.close("all")


def plot_event(
    event,
    meshes,
    pre_event_slip_deficit,
    probability,
    post_event_slip_deficit,
    t,
    iteration_step,
):
    # Plot distances from current event hypocenter triangle
    plt.figure(figsize=(15, 2))

    # Plot pre-earthquake slip deficit
    plt.subplot(1, 6, 1)
    pc = plot_meshes(meshes, pre_event_slip_deficit, plt.gca(), "spring")
    plt.colorbar(pc, format="%.0e")
    # plt.colorbar(pc, label="initial slip deficit (m)")
    plt.plot(meshes[0].x_perimeter, meshes[0].y_perimeter, "-k")
    plt.xticks([])
    plt.yticks([])
    plt.title("pre-eq $\hat{m}$")

    # Plot current probability
    plt.subplot(1, 6, 2)
    pc = plot_meshes(meshes, probability, plt.gca(), "viridis")
    plt.colorbar(pc, format="%.0e")
    plt.plot(meshes[0].x_perimeter, meshes[0].y_perimeter, "-k")
    plt.xticks([])
    plt.yticks([])
    plt.title(f"weights")

    plt.subplot(1, 6, 3)
    pc = plot_meshes(
        meshes,
        event.hypocenter_triangle_to_all_triangles_distances / 1e3,
        plt.gca(),
        "Reds",
    )
    plt.colorbar(pc, format="%.0e")
    plt.plot(
        meshes[0].centroids[event.hypocenter_triangle_index, 0],
        meshes[0].centroids[event.hypocenter_triangle_index, 1],
        "*k",
        markersize=15,
    )
    plt.plot(meshes[0].x_perimeter, meshes[0].y_perimeter, "-k")
    plt.xticks([])
    plt.yticks([])
    plt.title(f"i = {iteration_step}")
    # plt.title(f"index = {event.hypocenter_triangle_index[0]}")

    # Plot triangles involved in current event
    plt.subplot(1, 6, 4)
    fill_value = np.zeros(meshes[0].n_tde)
    fill_value[event.triangle_index] = 1
    pc = plot_meshes(meshes, fill_value, plt.gca(), "Blues")
    plt.colorbar(pc, format="%.0e")
    plt.plot(meshes[0].x_perimeter, meshes[0].y_perimeter, "-k")
    plt.xticks([])
    plt.yticks([])
    plt.title(f"{event.actual_area / KM2_TO_M2:0.2f} (km^2)")

    # Plot slip distribution
    plt.subplot(1, 6, 5)
    fill_value = np.zeros(meshes[0].n_tde)
    fill_value[event.triangle_index] = event.slip
    x_coords = meshes[0].meshio_object.points[:, 0]
    y_coords = meshes[0].meshio_object.points[:, 1]
    vertex_array = np.asarray(meshes[0].verts)
    ax = plt.gca()
    xy = np.c_[x_coords, y_coords]
    verts = xy[vertex_array]
    pc = matplotlib.collections.PolyCollection(
        verts,
        edgecolor=None,
        cmap="gnuplot2_r",
        linewidth=0,
        alpha=1.0,
    )
    pc.set_array(fill_value)
    ax.add_collection(pc)
    ax.autoscale()
    plt.gca().set_aspect("equal")
    plt.plot(meshes[0].x_perimeter, meshes[0].y_perimeter, "-k")
    plt.colorbar(pc, format="%.0e")
    plt.xticks([])
    plt.yticks([])
    plt.title(f"$M_W$ = {event.moment_magnitude[0]:0.2f}")

    plt.subplot(1, 6, 6)
    pc = plot_meshes(meshes, post_event_slip_deficit, plt.gca(), "spring")
    plt.colorbar(pc, format="%.0e")
    plt.plot(meshes[0].x_perimeter, meshes[0].y_perimeter, "-k")
    plt.xticks([])
    plt.yticks([])
    plt.title("post-eq $\hat{m}$")

    # plt.suptitle(f"i = {iteration_step}")
    # plt.show()


def area_to_moment_magnitude_allen_and_hayes(area):
    """Calculate emperically estimated slip areas using
    relationships from:

    Allen and Hayes (2017), Alternative Rupture-Scaling
    Relationships for Subduction Interface and Other Offshore
    Environments, Bulletin of the Seismological Society of America,
    Vol. 107, No. 3, pp. 1240–1253, June 2017, doi: 10.1785/0120160255

    All values taken from their table 2

    Note: $S_2$ in the paper's notation is what we use for rupture_area

    Args:
        area: rupture area in meters squared

    Returns:
        moment_magnitude: Array of moment magnitudes
    """
    hinge_area = 80714792455.11925  # (m)
    if area <= hinge_area:
        a = -5.62
        b = 1.22
    elif area > hinge_area:
        a = 2.23
        b = 0.31
    moment_magnitude = (np.log10(area / KM2_TO_M2) - a) / b
    return moment_magnitude


def get_triangle_index_closest_to_hypocenter(
    mesh, hypocenter_longitude, hypocenter_latitude, hypocenter_depth
):
    km2m = 1e3
    radius_earth = 6371e3

    # Convert hypocenter longitude, latitude, and depth to Cartesian coordinates
    hypocenter_x, hypocenter_y, hypocenter_z = sph2cart(
        hypocenter_longitude,
        hypocenter_latitude,
        radius_earth + km2m * hypocenter_depth,
    )

    # Find triangle index for triangle with centroid coordinates closest to hypocenter coordinates
    distances = scipy.spatial.distance.cdist(
        np.array([hypocenter_x, hypocenter_y, hypocenter_z])[None, :],
        np.vstack((mesh.x_centroid, mesh.y_centroid, mesh.z_centroid)).T,
        "euclidean",
    )

    # Index of mesh triangle smallest distance away
    hypocenter_triangle_index = np.argsort(distances)[0][0]
    return hypocenter_triangle_index


# def moment_magnitude_to_moment(moment_magnitude):
#     moment = 10 ** (1.5 * (moment_magnitude + 10.7) - 7.0)
#     return moment


def moment_to_moment_magnitude(moment):
    """
    Convert moment to moment magnitude
    Assumes MKS units
    """
    moment_magnitude = 2.0 / 3.0 * (np.log10(moment) - 9.05)
    return moment_magnitude


def moment_magnitude_to_moment(moment_magnitude):
    """
    Convert moment magnitude to moment
    Assumes MKS units
    """
    moment = 10 ** (3.0 / 2.0 * moment_magnitude + 9.05)
    return moment


def get_event_area_and_mean_slip(mesh, event):
    # In the case where event area larger than the area of the hypocentral triangle
    # then just have uniform slip on the single hypocentral triangle
    if event.target_area <= event.hypocenter_triangle_area:
        # logger.warning("Target area is less than hypocenter triangle area")
        event.actual_area = event.hypocenter_triangle_area
        event.triangle_index = event.hypocenter_triangle_index
        event.hypocenter_triangle_to_all_triangles_distances = np.array([])

    elif event.target_area > event.hypocenter_triangle_area:
        # Calculate distance from hypocenter triangle to all other triangles
        event.hypocenter_triangle_to_all_triangles_distances = (
            get_hypocenter_triangle_to_all_triangles_distances_single_mesh(
                mesh, event.hypocenter_triangle_index
            )
        )

        # Find the triangles close to the hypocenter that accumulate enough area to be a
        # part of the event rupture
        sorted_distance_index = np.argsort(
            event.hypocenter_triangle_to_all_triangles_distances
        )
        cumulative_area = np.cumsum(mesh.areas[sorted_distance_index])
        event.triangle_index = sorted_distance_index[
            np.where(cumulative_area < event.target_area)[0]
        ]
        event.actual_area = np.sum(mesh.areas[event.triangle_index])

    # Rescale slip
    event.mean_slip = event.moment / (event.shear_modulus * event.actual_area)
    return event


def get_event_slip_single_mesh(mesh, event):
    if event.triangle_index.size == 1:
        event.n_eigenvalues = 0
        event.slip = event.moment / (
            event.shear_modulus * event.hypocenter_triangle_area
        )
        event.pre_scaled_moment = np.copy(event.moment)

    elif event.triangle_index.size > 1:
        event.n_eigenvalues = event.triangle_index.size
        eigenvalues, eigenvectors = get_eigenvalues_and_eigenvectors(
            event.n_eigenvalues,
            (mesh.x_centroid[event.hypocenter_triangle_index] - mesh.x_centroid)[
                event.triangle_index
            ],
            (mesh.y_centroid[event.hypocenter_triangle_index] - mesh.y_centroid)[
                event.triangle_index
            ],
            (mesh.z_centroid[event.hypocenter_triangle_index] - mesh.z_centroid)[
                event.triangle_index
            ],
        )

        event.slip = np.zeros(event.triangle_index.size)
        weights = np.random.randn(eigenvalues.size)
        for i in range(1, weights.size):
            event.slip += weights[i] * np.sqrt(eigenvalues[i]) * eigenvectors[:, i]
        event.slip = np.exp(event.slip)

        # Apply taper to slip.  This is ad hoc and may need revision
        distances = event.hypocenter_triangle_to_all_triangles_distances[
            event.triangle_index
        ]

        # Sigmoid taper: Requires careful tuning
        # taper_transition = 1.5 * np.mean(distances)
        # taper_width = 10 / taper_transition  # (m)
        # slip_taper = normalized_sigmoid(taper_transition, taper_width, distances)

        # Cosine taper
        slip_taper = np.cos(np.pi / 2 * distances / np.max(distances))
        event.slip = event.slip * slip_taper

    # After taper is applied rescale slip magnitudes to get the correct moment
    event.pre_scaled_moment = event.shear_modulus * np.sum(
        event.slip * mesh.areas[event.triangle_index]
    )
    event.slip_scaling_factor = event.moment / event.pre_scaled_moment
    event.slip = event.slip * event.slip_scaling_factor
    event.slip_all_elements = np.zeros(mesh.n_tde)
    event.slip_all_elements[event.triangle_index] = event.slip
    event.geometric_moment = mesh.areas * event.slip_all_elements
    event.scalar_geometric_moment = np.sum(event.geometric_moment)
    return event


def get_event_area_slip_hypocenter(mesh, event):
    event.hypocenter_triangle_index = get_triangle_index_closest_to_hypocenter(
        mesh,
        event.hypocenter_longitude,
        event.hypocenter_latitude,
        event.hypocenter_depth,
    )
    event = get_event_area_slip_triangle_index(mesh, event)
    return event


def get_event_area_slip_triangle_index(mesh, event):
    event.moment = moment_magnitude_to_moment(event.moment_magnitude)
    event.geometric_moment_scalar = event.moment / event.shear_modulus
    event.target_area = event.area_scaling * moment_magnitude_to_area_allen_and_hayes(
        event.moment_magnitude
    )
    event.hypocenter_triangle_area = mesh.areas[event.hypocenter_triangle_index]

    event = get_event_area_and_mean_slip(mesh, event)
    event = get_event_slip_single_mesh(mesh, event)
    return event


def get_datetime_uuid_string():
    return (
        str(
            (
                datetime.datetime.now() - datetime.datetime.fromtimestamp(0)
            ).total_seconds()
        ).replace(".", "")
        + uuid.uuid4().hex
    )


def quick_plot_slip(mesh, event, params):
    plt.figure(figsize=(8, 8))
    fill_value = event.slip_all_elements
    xgrid, ygrid = rbf_interpolate_single_mesh(mesh, params, fill_value)
    xflat = xgrid.reshape(2, -1).T
    inpolygon_vals = inpolygon(
        xflat[:, 0], xflat[:, 1], mesh.x_perimeter, mesh.y_perimeter
    )
    inpolygon_vals = np.reshape(
        inpolygon_vals, (params.n_grid_longitude, params.n_grid_latitude)
    )
    ygrid[~inpolygon_vals] = np.nan

    # Plot
    cmap = cc.cm.CET_L19
    levels = np.linspace(
        params.min_contour_value, np.nanmax(ygrid), params.n_contour_levels
    )

    plt.contourf(*xgrid, ygrid, cmap=cmap, levels=levels, extend="both")
    cb = plt.colorbar(
        cax=plt.gca().inset_axes((0.03, 0.60, 0.02, 0.35)), label="slip (m)"
    )
    cb.set_label(label="slip (m)", size=10)
    cb.ax.tick_params(labelsize=10)
    plt.contour(
        *xgrid,
        ygrid,
        colors="k",
        linestyles="solid",
        linewidths=0.25,
        levels=levels,
    )
    plt.plot(mesh.x_perimeter, mesh.y_perimeter, "-k", linewidth=1.0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().set_facecolor("gainsboro")
    plt.title(f"$M_W$ = {event.moment_magnitude:0.3}")
    if params.savefig:
        base_file_name = "./runs/" + params.run_name + "/" + get_datetime_uuid_string()
        plt.savefig(base_file_name + ".pdf")
        plt.savefig(base_file_name + ".png", dpi=500)
    # plt.show()


def plot_event_select_eigenmodes(mesh, event, params):
    # Show eigenmodes for this event
    event.n_eigenvalues = event.triangle_index.size
    logger.info(f"Number of triangle mesh elements = {mesh.n_tde}")
    logger.info(f"Number of eigenvalues = {event.n_eigenvalues}")

    # Use Karhunen-Loeve to compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = get_eigenvalues_and_eigenvectors(
        event.n_eigenvalues,
        (mesh.x_centroid[event.hypocenter_triangle_index] - mesh.x_centroid)[
            event.triangle_index
        ],
        (mesh.y_centroid[event.hypocenter_triangle_index] - mesh.y_centroid)[
            event.triangle_index
        ],
        (mesh.z_centroid[event.hypocenter_triangle_index] - mesh.z_centroid)[
            event.triangle_index
        ],
    )

    # def quick_plot_mode(mesh, fill_value, params):
    #     xgrid, ygrid = rbf_interpolate_single_mesh(mesh, params, fill_value)
    #     xflat = xgrid.reshape(2, -1).T
    #     inpolygon_vals = inpolygon(
    #         xflat[:, 0], xflat[:, 1], mesh.x_perimeter, mesh.y_perimeter
    #     )
    #     inpolygon_vals = np.reshape(
    #         inpolygon_vals, (params.n_grid_longitude, params.n_grid_latitude)
    #     )
    #     ygrid[~inpolygon_vals] = np.nan

    #     # Filled contour plot
    #     cmap = cc.cm.CET_D10
    #     levels = np.linspace(-1.0, 1.0, 10)
    #     plt.contourf(*xgrid, ygrid, cmap=cmap, levels=levels, extend="both")
    #     plt.contour(
    #         *xgrid,
    #         ygrid,
    #         colors="k",
    #         linestyles="solid",
    #         linewidths=0.25,
    #         levels=levels,
    #     )
    #     plt.plot(mesh.x_perimeter, mesh.y_perimeter, "-k", linewidth=1.0)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.gca().set_facecolor("gainsboro")
    #     plt.gca().set_aspect("equal", adjustable="box")

    # Plot select eigenmodes
    plt.figure(figsize=(12, 4))
    for i in range(0, 10):
        ax = plt.subplot(2, 5, i + 1)

        # Select eigenmode for current index
        n_eigenvalues = eigenvalues.size
        if n_eigenvalues > 55:
            if i >= 5:
                i = i + 50 - 5
        elif n_eigenvalues >= 10 and n_eigenvalues < 55:
            if i >= 5:
                i = n_eigenvalues - 5 + (i - 5)
        elif n_eigenvalues < 10:
            if i >= n_eigenvalues:
                break
        logger.info(f"Plotting mode {i}")

        # Select eigenmode to contour
        fill_value = np.zeros(mesh.n_tde)
        fill_value[event.triangle_index] = eigenvectors[:, i]

        # Normalize eigenmode for interpretable plotting
        min_value = np.min(fill_value)
        max_value = np.max(fill_value)
        if np.abs(max_value) > np.abs(min_value):
            fill_value = fill_value / max_value
        else:
            fill_value = fill_value / np.abs(min_value)

        # Sign convetion for special case of first eigenmode
        if i == 0 and np.nanmean(fill_value) < 0:
            fill_value = -1 * fill_value

        # Plot and title
        quick_plot_mode(mesh, fill_value, params)
        plt.title(f"mode {i} of {n_eigenvalues}")

        # Save as .pdf and .png
        if params.savefig:
            base_file_name = (
                "./runs/" + params.run_name + "/" + get_datetime_uuid_string()
            )
            plt.savefig(base_file_name + ".pdf")
            plt.savefig(base_file_name + ".png", dpi=500)

    # plt.show()


def quick_plot_mode(mesh, fill_value, params):
    xgrid, ygrid = rbf_interpolate_single_mesh(mesh, params, fill_value)
    xflat = xgrid.reshape(2, -1).T
    inpolygon_vals = inpolygon(
        xflat[:, 0], xflat[:, 1], mesh.x_perimeter, mesh.y_perimeter
    )
    inpolygon_vals = np.reshape(
        inpolygon_vals, (params.n_grid_longitude, params.n_grid_latitude)
    )
    ygrid[~inpolygon_vals] = np.nan

    # Filled contour plot
    cmap = cc.cm.CET_D10
    levels = np.linspace(-1.0, 1.0, 10)
    plt.contourf(*xgrid, ygrid, cmap=cmap, levels=levels, extend="both")
    plt.contour(
        *xgrid,
        ygrid,
        colors="k",
        linestyles="solid",
        linewidths=0.25,
        levels=levels,
    )
    plt.plot(mesh.x_perimeter, mesh.y_perimeter, "-k", linewidth=1.0)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_facecolor("gainsboro")
    plt.gca().set_aspect("equal", adjustable="box")


# def get_geometric_moment_condition(event_geometric_moment, mesh_geometric_moment):
#     # Case 1: Average geometric moment is grearter than zero
#     if np.sum(mesh_geometric_moment) > 0:
#         geometric_moment_condition = True
#     else:
#         geometric_moment_condition = False

# Case 2: Accumulated geometric moment greater than or equal to

# if event_geometric_moment < np.sum(
#     mesh_geometric_moment[np.where(mesh_geometric_moment > 0.0)]
# ):
#     geometric_moment_condition = True
#     # print(f"{event_geometric_moment[0]:0.2e}")
#     # print(f"{np.sum(mesh_geometric_moment[np.where(mesh_geometric_moment > 0.0)]):0.2e}")
# else:
#     geometric_moment_condition = False
# return geometric_moment_condition


def get_tanh_probability(x, amplitude_scale_factor, data_scale_factor):
    tanh_probability = amplitude_scale_factor * np.tanh(data_scale_factor * x)
    if tanh_probability < 0:
        tanh_probability = 0
    if tanh_probability > 1:
        tanh_probability = 1
    return tanh_probability


def get_tanh_probability_vector(x, amplitude_scale_factor, data_scale_factor):
    tanh_probability = amplitude_scale_factor * np.tanh(data_scale_factor * x)
    tanh_probability[np.where(x < 0.0)] = 0.0
    tanh_probability -= np.min(tanh_probability)
    tanh_probability[np.isnan(tanh_probability)] = 0.0
    tanh_probability = tanh_probability / np.sum(tanh_probability)
    return tanh_probability


def get_omori_decay_probability(time_vector, time_of_earthquake, amplitude, decay_time):
    omori_decay_probability = amplitude / (
        1 + (1 / decay_time) * (time_vector - time_of_earthquake)
    )
    omori_decay_probability[time_vector < time_of_earthquake] = 0.0
    return omori_decay_probability


def create_non_event(n_tde):
    event = addict.Dict()
    event.shear_modulus = 0.0
    event.area_scaling = 0.0
    event.moment_magnitude = np.array([0.0])
    event.moment = 0.0
    event.geometric_moment = 0.0
    event.location_probability = 0.0
    event.hypocenter_triangle_index = np.nan
    event.target_area = 0.0
    event.hypocenter_triangle_area = 0.0
    event.actual_area = np.nan
    event.triangle_index = 0
    event.hypocenter_triangle_to_all_triangles_distances
    event.mean_slip = np.nan
    event.n_eigenvalues = np.nan
    event.slip = np.nan
    event.pre_scaled_moment = np.nan
    event.slip_scaling_factor = np.nan
    event.slip_all_elements = np.zeros(n_tde)  # This is the important one
    event.geometric_moment_scalar = np.nan
    event.mesh_geometric_moment_pre_event = np.zeros(n_tde)  # This is important
    event.mesh_geometric_moment_post_event = np.zeros(n_tde)  # This is important
    event.omori_amplitude = np.nan
    event.omori_decay_time = np.nan
    return event


def write_vtk_file(mesh, cell_data, cell_data_label, vtk_file_name):
    """
    See: https://github.com/nschloe/meshio
    """
    pass
    # points = mesh.meshio_object.points.tolist()
    # cells = [("triangle", mesh.meshio_object.cells[2].data.tolist())]
    # cell_data = {cell_data_label: [cell_data.tolist()]}
    # vtk_mesh = meshio.Mesh(points, cells, cell_data=cell_data)
    # vtk_mesh.write(vtk_file_name, file_format="vtk")


def get_vtk_file_name(run_folder, mesh_parameters_file_name, mesh_index, event_index):
    vtk_file_name = (
        run_folder
        + mesh_parameters_file_name.split(".")[0]
        + "_"
        + str(mesh_index)
        + "_"
        + str(event_index)
        + ".vtk"
    )
    return vtk_file_name


def plot_probability_and_events_time_series(
    params,
    output_folder,
    time_series,
    start_idx,
    end_idx,
):
    event_idx = np.where(time_series.event_trigger_flag == 1)[0]
    figsize = (10, 5)
    plt.figure(figsize=figsize)

    # Plot probability time series
    plt.subplot(2, 1, 1)
    plt.plot(
        time_series.time[start_idx:end_idx],
        time_series.probability_weight[start_idx:end_idx],
        "-k",
        linewidth=0.5,
    )
    plt.plot(
        [time_series.time[start_idx], time_series.time[end_idx - 1]],
        [0, 0],
        "-k",
        linewidth=0.5,
    )
    plt.fill_between(
        time_series.time[start_idx:end_idx],
        time_series.probability_weight[start_idx:end_idx],
        np.zeros_like(time_series.time[start_idx:end_idx]),
        time_series.probability_weight[start_idx:end_idx] > 0,
        color="gold",
        alpha=1.0,
        edgecolor=None,
    )

    plt.fill_between(
        time_series.time[start_idx:end_idx],
        time_series.probability_weight[start_idx:end_idx],
        np.zeros_like(time_series.time[start_idx:end_idx]),
        time_series.probability_weight[start_idx:end_idx] < 0,
        color="dodgerblue",
        alpha=1.0,
        edgecolor=None,
    )
    plt.xlim([start_idx, end_idx])
    plt.gca().set_ylim(bottom=0.0)
    plt.ylabel("$p^t$")

    # Plot earthquake magnitude stem plot
    plt.subplot(2, 1, 2)
    for i in range(event_idx.size):
        plt.plot(
            [
                time_series.time[event_idx[i]],
                time_series.time[event_idx[i]],
            ],
            [
                params.minimum_event_moment_magnitude,
                time_series.event_magnitude[event_idx[i]],
            ],
            "-",
            linewidth=0.1,
            zorder=10,
            color="k",
        )

    cmap = cc.cm.CET_L17
    magnitude_plot_size = 1e-5 * 10 ** time_series.event_magnitude[event_idx]
    plt.scatter(
        time_series.time[event_idx],
        time_series.event_magnitude[event_idx],
        s=magnitude_plot_size,
        c=time_series.event_magnitude[event_idx],
        zorder=20,
        alpha=1.0,
        edgecolors="k",
        linewidths=0.5,
        cmap=cmap,
        vmin=6.0,
        vmax=9.0,
    )

    plt.xlabel("time")
    plt.ylabel("$M_W$")
    plt.xlim([start_idx, end_idx])
    plt.gca().set_ylim(bottom=params.minimum_event_moment_magnitude)
    plt.savefig(output_folder + "/probability_magnitude" + ".pdf")
    plt.savefig(output_folder + "/probability_magnitude" + ".png", dpi=500)


def get_logger(log_level, params):
    params.log_file_name = params.output_folder + "/" + params.run_name + ".log"

    logger = logging.getLogger(__name__)
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    logger.setLevel(log_level)
    shell_handler = RichHandler()
    file_handler = logging.FileHandler(params.log_file_name)
    shell_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler.setFormatter(
        logging.Formatter(
            "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
        )
    )
    logger.addHandler(shell_handler)
    logger.addHandler(file_handler)
    logger.info(f"Output folder: {params.output_folder}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_file_name", type=str, help="Name of *_params.json file")
    parser.add_argument(
        "--mesh_parameters_file_name",
        type=str,
        default=None,
        required=False,
        help="Name of *_mesh_parameters.json file",
    )
    parser.add_argument(
        "--initial_slip_deficit_rate_file",
        type=str,
        default=None,
        required=False,
        help="Name of *.npy file with slip deficit rates",
    )
    parser.add_argument(
        "--write_event_pickle_files",
        type=str,
        default=None,
        required=False,
        help="Flag for saving each event as an individual .pickle file (0 or 1)",
    )
    parser.add_argument(
        "--repl",
        type=int,
        default=None,
        required=False,
        help="Flag for dropping into REPL (0 or 1)",
    )
    parser.add_argument(
        "--pickle_save",
        type=int,
        default=None,
        required=False,
        help="Flag for saving major data structures in pickle file (0 | 1)",
    )
    parser.add_argument(
        "--n_time_steps",
        type=int,
        default=None,
        required=False,
        help="Number of time steps",
    )
    parser.add_argument(
        "--time_step",
        type=float,
        default=None,
        required=False,
        help="Time step duration",
    )
    parser.add_argument(
        "--b_value",
        type=float,
        default=None,
        required=False,
        help="Gutenberg-Richter b-value",
    )
    parser.add_argument(
        "--shear_modulus",
        type=float,
        default=None,
        required=False,
        help="Shear modulus (Pa)",
    )
    parser.add_argument(
        "--minimum_event_moment_magnitude",
        type=float,
        default=None,
        required=False,
        help="Minimum event size (moment magnitude)",
    )
    parser.add_argument(
        "--maximum_event_moment_magnitude",
        type=float,
        default=None,
        required=False,
        help="Maximum event size (moment magnitude)",
    )
    parser.add_argument(
        "--time_probability_amplitude_scale_factor",
        type=float,
        default=None,
        required=False,
        help="Time probability amplitude scale factor",
    )
    parser.add_argument(
        "--time_probability_data_scale_factor",
        type=float,
        default=None,
        required=False,
        help="Time probability data scale factor",
    )
    parser.add_argument(
        "--area_scaling",
        type=float,
        default=None,
        required=False,
        help="Event area scaling factor",
    )
    parser.add_argument(
        "--default_omori_decay_time",
        type=int,
        default=None,
        required=False,
        help="Default (reference) Omori decay time",
    )
    parser.add_argument(
        "--minimum_probability",
        type=float,
        default=None,
        required=False,
        help="Minimum time probability",
    )
    parser.add_argument(
        "--time_probability_history_scale_factor",
        type=float,
        default=None,
        required=False,
        help="Time probability history scale factor",
    )
    parser.add_argument(
        "--location_probability_amplitude_scale_factor",
        type=float,
        default=None,
        required=False,
        help="Location probability amplitude scale factor",
    )
    parser.add_argument(
        "--location_probability_data_scale_factor",
        type=float,
        default=None,
        required=False,
        help="Location probability data scale factor",
    )
    parser.add_argument(
        "--omori_amplitude_scale_factor",
        type=float,
        default=None,
        required=False,
        help="Omori amplitude scale factor",
    )
    parser.add_argument(
        "--omori_rate_perturbation_scale_factor",
        type=float,
        default=None,
        required=False,
        help="Omori rate perturbation scale factor",
    )
    parser.add_argument(
        "--omori_rate_perturbation_exponent",
        type=float,
        default=None,
        required=False,
        help="Omori rate perturbation exponent",
    )
    parser.add_argument(
        "--mesh_index",
        type=int,
        default=None,
        required=False,
        help="Mesh index",
    )
    parser.add_argument(
        "--initial_mesh_slip_deficit_scaling",
        type=float,
        default=None,
        required=False,
        help="initial_mesh_slip_deficit_scaling",
    )
    parser.add_argument(
        "--geometic_moment_rate_scale_factor",
        type=float,
        default=None,
        required=False,
        help="geometic_moment_rate_scale_factor",
    )
    args = addict.Dict(vars(parser.parse_args()))
    return args


def get_params(params_file_name):
    """Read *params.json file and return contents as a dictionary

    Args:
        params_file_name (string): Path to params file

    Returns:
        params (Dict): Dictionary with content of params file
    """
    with open(params_file_name, "r") as f:
        params = json.load(f)
    params = addict.Dict(params)  # Convert to dot notation dictionary
    params.file_name = params_file_name

    # Add run_name and output_path
    params.run_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    params.output_path = os.path.join(params.base_runs_folder, params.run_name)
    params.output_folder = os.path.join(params.base_runs_folder, params.run_name)

    # Sort params keys alphabetically for readability
    params = addict.Dict(sorted(params.items()))
    return params


def process_args(params, args):
    for key in params:
        if key in args:
            if args[key] is not None:
                logger.warning(f"ORIGINAL: params.{key}: {params[key]}")
                params[key] = args[key]
                logger.warning(f"REPLACED: params.{key}: {params[key]}")
            else:
                logger.info(f"params.{key}: {params[key]}")


def initialize_time_series(params):
    """
    Time-series storage
    """
    time_series = addict.Dict()
    time_series.time = np.linspace(0, params.n_time_steps, params.n_time_steps)
    time_series.probability_weight = np.zeros_like(time_series.time)
    time_series.probability = np.zeros_like(time_series.time)
    time_series.event_magnitude = np.zeros_like(time_series.time)
    time_series.event_trigger_flag = np.zeros_like(time_series.time)
    time_series.cumulate_omori_effect = np.zeros_like(time_series.time)
    time_series.event_longitude = np.zeros_like(time_series.time)
    time_series.event_latitude = np.zeros_like(time_series.time)
    time_series.event_depth = np.zeros_like(time_series.time)
    time_series.event_x = np.zeros_like(time_series.time)
    time_series.event_y = np.zeros_like(time_series.time)
    time_series.event_z = np.zeros_like(time_series.time)
    time_series.last_event_time = 0
    return time_series


def initialize_mesh(params):
    """
    Mesh storage
    """
    mesh = addict.Dict()
    meshes = read_meshes(params.mesh_parameters_file_name)
    mesh.mesh = meshes[params.mesh_index]
    mesh.mesh_geometric_moment = np.zeros(mesh.mesh.n_tde)
    mesh.mesh_last_event_slip = np.zeros(mesh.mesh.n_tde)
    mesh.mesh_total_slip = np.zeros(mesh.mesh.n_tde)
    mesh.mesh_geometric_moment_pre_event = np.copy(mesh.mesh_geometric_moment)
    mesh.mesh_geometric_moment_post_event = np.zeros_like(mesh.mesh_geometric_moment)
    mesh.mesh_geometric_moment_scalar = np.zeros(params.n_time_steps)
    mesh.mesh_geometric_moment_scalar_non_zero = np.zeros(params.n_time_steps)
    mesh.mesh_geometric_moment_scalar[0] = np.sum(mesh.mesh_geometric_moment)
    # TODO: This should be generalized so that strike- or -dip slip
    # or both can be specified
    mesh.mesh_initial_dip_slip_deficit = np.load(params.initial_slip_deficit_rate_file)
    mesh.mesh_interseismic_loading_rate = (
        params.geometic_moment_rate_scale_factor * mesh.mesh_initial_dip_slip_deficit
    )
    return mesh


def initialize_hdf(params, mesh):
    """
    Open HDF file and create groups for saving data
    """
    hdf_file_datasets = addict.Dict()
    hdf_file_name = params.output_folder + "/" + params.run_name + ".hdf"
    hdf_file = h5py.File(hdf_file_name, "w")
    hdf_file_datasets.cumulative_event_slip = hdf_file.create_dataset(
        "cumulative_slip",
        shape=(params.n_time_steps, mesh.mesh.n_tde),
        dtype=float,
    )
    hdf_file_datasets.geometric_moment = hdf_file.create_dataset(
        "geometric_moment",
        shape=(params.n_time_steps, mesh.mesh.n_tde),
        dtype=float,
    )
    # hdf_file_datasets.last_event_slip = hdf_file.create_dataset(
    #     "last_event_slip", shape=(params.n_time_steps, mesh.mesh.n_tde), dtype=float
    # )
    hdf_file_datasets.loading_rate = hdf_file.create_dataset(
        "loading_rate", shape=(mesh.mesh.n_tde), dtype=float
    )
    return hdf_file, hdf_file_datasets


def save_all(params, mesh, time_series):
    """
    Saving params, mesh, and time series.
    HDF and event pickle files are saved by separate processes
    """

    # Save params dictionary to .json file in output_folder
    with open(params.output_folder + "/params.json", "w") as params_output_file:
        json.dump(params, params_output_file)

    # Write vtk file with geometry only
    vtk_file_name = params.output_folder + "/" + params.run_name + "_mesh_geometry.vtk"
    write_vtk_file(mesh.mesh, np.zeros(mesh.mesh.n_tde), "none", vtk_file_name)

    # Plot initial state and mesh
    plot_initial_data(
        mesh.mesh, mesh.mesh_initial_dip_slip_deficit, params.output_folder
    )

    # Save time_series dictionary to .pickle file in output_folder
    with open(params.output_folder + "/time_series.pickle", "wb") as pickle_file:
        pickle.dump(time_series, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    # Save mesh dictionary to .pickle file in output_folder
    with open(params.output_folder + "/mesh.pickle", "wb") as pickle_file:
        pickle.dump(mesh, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    # Save random state to .pickle file in output_folder
    with open(params.output_folder + "/random_state.pickle", "wb") as pickle_file:
        pickle.dump(
            np.random.get_state(), pickle_file, protocol=pickle.HIGHEST_PROTOCOL
        )

    # Plot time probability and event moment magnitude time series
    plot_probability_and_events_time_series(
        params,
        params.output_folder,
        time_series,
        0,
        params.n_time_steps,
    )


def time_step_loop(params, time_series, mesh):
    # Main time loop
    hdf_file, hdf_file_datasets = initialize_hdf(params, mesh)
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
        time_series.probability_weight[i] = get_tanh_probability(
            time_series.probability[i],
            params.time_probability_amplitude_scale_factor,
            params.time_probability_data_scale_factor,
        )
        time_series.event_trigger_flag[i] = np.random.choice(
            N_BINARY,
            1,
            p=[
                1 - time_series.probability_weight[i],
                time_series.probability_weight[i],
            ],
        )

        if bool(time_series.event_trigger_flag[i]):
            time_series.last_event_time = i
            event = addict.Dict()
            event.shear_modulus = np.array([params.shear_modulus])
            event.area_scaling = params.area_scaling
            event.moment_magnitude = get_gutenberg_richter_magnitude(
                params.b_value,
                params.minimum_event_moment_magnitude,
                params.maximum_event_moment_magnitude,
            )
            event.moment = moment_magnitude_to_moment(event.moment_magnitude)
            event.geometric_moment = event.moment / event.shear_modulus
            time_series.event_magnitude[i] = event.moment_magnitude[0]

            # Find event hypocentral triangle
            event.location_probability = get_tanh_probability_vector(
                mesh.mesh_geometric_moment_pre_event,
                params.location_probability_amplitude_scale_factor,
                params.location_probability_data_scale_factor,
            )
            event.hypocenter_triangle_index = np.random.choice(
                mesh.mesh.n_tde, 1, p=event.location_probability
            )[0]

            # Store coordinates of central mesh element
            time_series.event_longitude[i] = mesh.mesh.centroids[:, 0][
                event.hypocenter_triangle_index
            ]
            time_series.event_latitude[i] = mesh.mesh.centroids[:, 1][
                event.hypocenter_triangle_index
            ]
            time_series.event_depth[i] = mesh.mesh.centroids[:, 2][
                event.hypocenter_triangle_index
            ]
            time_series.event_x[i] = mesh.mesh.x_centroid[
                event.hypocenter_triangle_index
            ]
            time_series.event_y[i] = mesh.mesh.y_centroid[
                event.hypocenter_triangle_index
            ]
            time_series.event_z[i] = mesh.mesh.z_centroid[
                event.hypocenter_triangle_index
            ]

            # Generate coseismic slip area and slip distribution
            event = get_event_area_slip_triangle_index(mesh.mesh, event)
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
            omori_rate_perturbation = get_omori_decay_probability(
                time_series.time,
                time_series.time[i],
                event.omori_amplitude,
                decay_time=event.omori_decay_time,
            )

            # Coseismic offset to Omori rate effect
            omori_rate_perturbation[
                np.where(time_series.time > time_series.time[i])
            ] -= (
                params.omori_rate_perturbation_scale_factor
                * event.omori_amplitude**params.omori_rate_perturbation_exponent
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

            # Save event dictionary as pickle file
            if params.write_event_pickle_files:
                event_pickle_file_name = (
                    f"{params.output_folder}/events/event_{i:010.0f}.pickle"
                )
                with open(event_pickle_file_name, "wb") as pickle_file:
                    pickle.dump(event, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            # Create dummy event dictionary because no event occured
            event = create_non_event(mesh.mesh.n_tde)
            event.mesh_geometric_moment_pre_event = np.copy(
                mesh.mesh_geometric_moment_pre_event
            )
            event.mesh_geometric_moment_post_event = (
                mesh.mesh_geometric_moment_pre_event
                + (
                    params.time_step
                    * mesh.mesh_interseismic_loading_rate
                    * mesh.mesh.areas
                )
            )
            event.mesh_last_event_slip = mesh.mesh_last_event_slip
            event.mesh_total_slip = mesh.mesh_total_slip

        # TODO: Check this???
        event.mesh_initial_dip_slip_deficit = mesh.mesh_initial_dip_slip_deficit

        # Save mesh values to HDF file
        hdf_file_datasets.cumulative_event_slip[i, :] = mesh.mesh_total_slip
        hdf_file_datasets.geometric_moment[i, :] = mesh.mesh_geometric_moment
        # hdf_file_datasets.loading_rate[i, :] = mesh.mesh_initial_dip_slip_deficit

        # Pre-event moment for next time step
        mesh.mesh_geometric_moment_pre_event = np.copy(
            event.mesh_geometric_moment_post_event
        )

        # Update probability
        time_series.probability[i + 1] = (
            time_series.cumulate_omori_effect[i]
            + mesh.mesh_geometric_moment_scalar_non_zero[i]
        )

    hdf_file_datasets.loading_rate = mesh.mesh_initial_dip_slip_deficit
    hdf_file.close()
    end_time = datetime.datetime.now()
    logger.info(f"Event sequence generation run time: {(end_time - start_time)}")
    logger.info(
        f"Generated {np.count_nonzero(time_series.event_magnitude)} events in {params.n_time_steps} time steps"
    )
