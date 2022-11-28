import pkg_resources

from .skies import (
    cart2sph,
    create_output_folder,
    print_magnitude_overview,
    triangle_area,
    triangle_normal,
    read_meshes,
    print_event,
    create_event,
    get_location_probability,
    cart2sph,
    sph2cart,
    wrap2360,
    get_mesh_edge_elements,
    get_ordered_edge_nodes,
    inpolygon,
    rbf_interpolate,
    get_synthetic_displacements,
    plot_slip_distributions,
    get_eigenvalues_and_eigenvectors,
    get_synthetic_accumulated_slip,
    plot_meshes,
    moment_magnitude_to_area_allen_and_hayes,
    get_gutenberg_richter_magnitude,
    normalized_sigmoid,
    interpolate_and_plot,
    get_hypocenter_triangle_to_all_triangles_distances,
    get_event_slip,
    plot_initial_data,
    plot_event,
    area_to_moment_magnitude_allen_and_hayes,
    get_triangle_index_closest_to_hypocenter,
    moment_magnitude_to_moment,
    get_hypocenter_triangle_to_all_triangles_distances_single_mesh,
    get_event_area_and_mean_slip,
    get_event_slip_single_mesh,
    get_event_area_slip_hypocenter,
    get_event_area_slip_triangle_index,
    rbf_interpolate_single_mesh,
    get_datetime_uuid_string,
    quick_plot_slip,
    plot_event_select_eigenmodes,
    quick_plot_mode,
    get_tanh_probability_vector,
    get_tanh_probability,
    get_omori_decay_probability,
    create_non_event,
    write_vtk_file,
    get_vtk_file_name,
    plot_probability_and_events_time_series,
    get_logger,
    parse_args,
)


try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"

__all__ = []
