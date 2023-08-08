__all__ = [
    "parse_args",
]

from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

from bound import config
import bound.dirs
from bound import logger
import bound.utils


def parse_args() -> Namespace:
    r""" Parse, checks, and refactors the input arguments"""
    args = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # noinspection PyTypeChecker
    args.add_argument("config_file", help="Path to the configuration file", type=Path)

    args.add_argument("-d", help="Debug mode -- Disable non-determinism", action="store_true")
    args.add_argument("-q", help="Enable quiet mode", action="store_true")
    args.add_argument("--nmodels", help="Number of ensemble models", type=int, default=None)
    args.add_argument("--deg", help="SSL degree", type=int, default=None)
    args.add_argument("--walk", help="Switch feature split to walking split", action="store_true")
    args.add_argument("--spiral", help="Switch to spiral feature split", action="store_true")
    args = args.parse_args()

    if not args.config_file.exists() or not args.config_file.is_file():
        raise ValueError(f"Unknown configuration file \"{args.config_file}\"")

    # Need to update directories first as other commands rely on these directories
    logger.setup()

    config.parse(args.config_file)
    if args.d:
        config.enable_debug_mode()
        bound.utils.set_debug_mode(seed=1)
    if args.nmodels:
        config.override_num_models(n_models=args.nmodels)
    if args.deg:
        config.override_ssl_degree(degree=args.deg)
    if args.q:
        config.set_quiet()
    if args.walk and args.spiral:
        raise ValueError("Cannot enable walking and spiral feature splits")
    if args.walk:
        config.override_to_walk_split()

    # Configure the random seeds based on torch's seed
    bound.utils.set_random_seeds()
    # Generates the data for learning
    config.print_configuration()
    args.tg = bound.utils.configure_dataset_args()
    return args
