import addict
import copy
import datetime
import json
import meshio
import os
import warnings
import scipy
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from pprint import pprint
from ismember import ismember
import matplotlib


def print_magnitude_overview(meshes):
    minimum_single_triangle_moment_magnitude = area_to_moment_magnitude_allen_and_hayes(
        np.min(meshes[0].areas)
    )
    maximum_single_triangle_moment_magnitude = area_to_moment_magnitude_allen_and_hayes(
        np.max(meshes[0].areas)
    )
    maximum_moment_magnitude = area_to_moment_magnitude_allen_and_hayes(
        np.sum(meshes[0].areas)
    )

    print("Magnitude overview:")
    print(f"Maximum moment magnitude of entire mesh = {maximum_moment_magnitude:0.2f}")
    print(
        f"Maximum moment magnitude of single mesh element = {maximum_single_triangle_moment_magnitude:0.2f}"
    )
    print(
        f"Minimum moment magnitude of single mesh element = {minimum_single_triangle_moment_magnitude:0.2f}"
    )
    print(f"Maximum allowed moment magnitude = {MAXIMUM_EVENT_MOMENT_MAGNITUDE:0.2f}")
    print(f"Minimum allowed moment magnitude = {MINIMUM_EVENT_MOMENT_MAGNITUDE:0.2f}")

    if MINIMUM_EVENT_MOMENT_MAGNITUDE < minimum_single_triangle_moment_magnitude:
        print(
            "MINIMUM_EVENT_MOMENT_MAGNITUDE is less than minimum moment magnitude of single mesh"
        )
        print(
            "WARNING: To avoid subgrid scale events increase MINIMUM_EVENT_MOMENT_MAGNITUDE"
        )

    if maximum_moment_magnitude > MAXIMUM_EVENT_MOMENT_MAGNITUDE:
        print(
            f"Maximum moment magnitude of entire mesh ({maximum_moment_magnitude:0.2f}) exceeds MAXIMUM_EVENT_MOMENT_MAGNITUDE"
        )
        print(
            f"WARNING: Events larger than {MAXIMUM_EVENT_MOMENT_MAGNITUDE:0.2f} will be clipped to {MAXIMUM_EVENT_MOMENT_MAGNITUDE:0.2f}"
        )
