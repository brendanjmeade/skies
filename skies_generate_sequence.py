import logging

import IPython
from rich.pretty import pprint

import skies


def main(args):
    params = skies.get_params(args.params_file_name)
    skies.create_output_folder(params.base_runs_folder, params.output_folder)
    skies.get_logger(logging.INFO, params)
    skies.process_args(params, args)
    time_series = skies.initialize_time_series(params)
    mesh = skies.initialize_mesh(params)
    skies.print_magnitude_overview(mesh.mesh)
    skies.time_step_loop(params, time_series, mesh)
    skies.save_all(params, mesh, time_series)

    if bool(params.repl):
        IPython.embed(banner1="")


if __name__ == "__main__":
    args = skies.parse_args()
    main(args)
