from argparse import Namespace
import logging
from typing import NoReturn

import bound
from bound import config
import bound.dirs
import bound.results
import bound.utils
from cmd_args import parse_args


def _main(args: Namespace) -> NoReturn:
    r""" Tests a single model (alternate) type """
    model = bound.train_ensemble(tg=args.tg, opt_params=config.MODEL_PARAMS)
    bound.results.calc(model=model, tg=args.tg)


if __name__ == '__main__':
    _main(parse_args())
    logging.info("All Feature Robustness Results Calculation Completed")
