__all__ = [
    "ALT_TYPE",
    "BATCH_SIZE",
    "COLUMN_FEAT_SPLIT",
    "DATASET",
    "DEBUG",
    "FORWARD_BATCH_SIZE",
    "IS_BOUND_PERCENT",
    "IS_CLASSIFICATION",
    "IS_REGRESSION",
    "LEARNING_RATE",
    "MODEL_NAME",
    "MODEL_PARAMS",
    "NUM_CLASSES",
    "NUM_EPOCH",
    "NUM_TEST_SPLITS",
    "N_DISJOINT_MODELS",
    "PARTITION_TRAIN",
    "PATCH_FEAT_SPLIT",
    "QUIET",
    "RANDOM_FEAT_SPLIT",
    "SSL_DEGREE",
    "SSL_PARAMS",
    "TOP_K_VALS",
    "VALIDATION_SPLIT_RATIO",
    "WALKING_FEAT_SPLIT",
    "WEIGHT_DECAY",
    "enable_debug_mode",
    "get_n_model",
    "is_alt_submodel",
    "is_ssl",
    "override_num_models",
    "override_ssl_degree",
    "override_to_walk_split",
    "parse",
    "print_configuration",
    "set_num_classes",
    "set_num_feats",
    "use_validation",
]

import logging
import math
from pathlib import Path
from typing import Callable, NoReturn, Optional, Union

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarint import ScalarInt

from .types import AltSubType
from .datasets import SplitDataset

DATASET = None  # type: Optional[SplitDataset]
DATASET_KEY = "dataset"

ALT_TYPE = None  # type: Optional[AltSubType]
MODEL_NAME = None

TREE_TYPE_KEY = "alt_type"
MODEL_PARAMS = dict()
MODEL_PARAMS_KEY = "model_params"

SSL_PARAMS = dict()
SSL_PARAMS_KEY = "ssl_params"

NUM_CLASSES = None

DEBUG = False

BATCH_SIZE = None
FORWARD_BATCH_SIZE = None
NUM_EPOCH = None
LEARNING_RATE = None
WEIGHT_DECAY = None

IS_CLASSIFICATION = False

IS_REGRESSION = False
BOUND_DIST = None
IS_BOUND_PERCENT = False

PARTITION_TRAIN = False
COLUMN_FEAT_SPLIT = False
RANDOM_FEAT_SPLIT = False
PATCH_FEAT_SPLIT = False
WALKING_FEAT_SPLIT = False
DIM_X = None

TOP_K_VALS = None

N_DISJOINT_MODELS = 51
SSL_DEGREE = -1

# Fraction of training samples used for
VALIDATION_SPLIT_RATIO = 0.05
NUM_TEST_SPLITS = None

QUIET = False

LEARNER_CONFIGS = dict()


def parse(config_file: Union[Path, str]) -> NoReturn:
    r""" Parses the configuration """
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Unable to find config file {config_file}")
    if not config_file.is_file():
        raise FileExistsError(f"Configuration {config_file} does not appear to be a file")

    with open(str(config_file), 'r') as f_in:
        all_yaml_docs = YAML().load_all(f_in)

        base_config = next(all_yaml_docs)
        _parse_general_settings(base_config)

    # Sanity checks nothing is out of whack
    _verify_configuration()


def _parse_general_settings(config) -> NoReturn:
    r"""
    Parses general settings for the learning configuration including the dataset, priors, positive
    & negative class information.  It also extracts general learner information
    """
    module_dict = _get_module_dict()
    for key, val in config.items():
        key = key.upper()
        if key.lower() == DATASET_KEY:
            ds_name = val.upper()
            try:
                module_dict[key] = SplitDataset[ds_name]
            except KeyError:
                raise ValueError(f"Unknown dataset {ds_name}")
        elif key.lower() == TREE_TYPE_KEY:
            tree_type = val.upper()
            module_dict[key] = AltSubType[tree_type]
        elif key.lower() == MODEL_PARAMS_KEY:
            module_dict[key] = _convert_commented_map(val)
        elif key.lower() == SSL_PARAMS_KEY:
            module_dict[key] = _convert_commented_map(val)
        # Drop in replacement field
        else:
            if key not in module_dict:
                raise ValueError(f"Unknown configuration field \"{key}\"")
            module_dict[key] = val


def _convert_commented_map(com_map: CommentedMap) -> dict:
    r"""
    Ruamel returns variables of type \p CommentedMap. This function converts the \p CommentedMap
    object into a standard dictionary.
    """
    def _convert_val(val):
        if val == "None":
            return None
        if isinstance(val, (float, int, str)):
            return val
        if isinstance(val, ScalarFloat):
            return float(val)
        if isinstance(val, ScalarInt):
            return int(val)
        raise ValueError("Unknown value type in converting CommentedMap")

    return {key: _convert_val(val) for key, val in com_map.items()}


def _get_module_dict() -> dict:
    r""" Standardizes construction of the module dictionary """
    return globals()


def _verify_configuration() -> NoReturn:
    r""" Sanity checks the configuration """
    if DATASET is None:
        raise ValueError("A dataset must be specified")

    pos_params = (
        (LEARNING_RATE, "Learning rate"),
        (NUM_EPOCH, "Number of training epochs"),
        (N_DISJOINT_MODELS, "Number of disjoint ensemble models"),
    )
    for param, name in pos_params:
        if param is not None and param <= 0:
            raise ValueError(f"{name} must be positive")

    nn_params = (
        (WEIGHT_DECAY, "Weight decay"),
    )
    for param, name in nn_params:
        # noinspection PyTypeChecker
        if param is not None and param < 0:
            raise ValueError(f"{name} must be non-negative")

    global FORWARD_BATCH_SIZE
    if FORWARD_BATCH_SIZE is None:
        FORWARD_BATCH_SIZE = BATCH_SIZE

    # Only a single feature split version is supported
    feats_versions = [
        COLUMN_FEAT_SPLIT,
        PATCH_FEAT_SPLIT,
        RANDOM_FEAT_SPLIT,
        WALKING_FEAT_SPLIT,
    ]
    n_set = sum([feats for feats in feats_versions])
    if n_set != 1:
        raise ValueError("Exactly one feature split is required")
    n_split = N_DISJOINT_MODELS
    sqrt = int(round(math.sqrt(n_split)))
    if PATCH_FEAT_SPLIT and sqrt * sqrt != n_split:
        raise ValueError(f"Dataset split count {n_split} is not a perfect square")


def print_configuration(log: Callable = logging.info) -> NoReturn:
    r""" Print the configuration settings """

    def _none_format(_val: Optional[float], format_str: str) -> str:
        if _val is None:
            return "None"
        return f"{_val:{format_str}}"

    log(f"Dataset: {DATASET.value.name}")
    # log(f"Is Rotate: {IS_ROTATE}")
    log(f"Is Classification: {IS_CLASSIFICATION}")
    if IS_CLASSIFICATION:
        log(f"Top K Vals: {TOP_K_VALS}")

    log(f"Is Regression: {IS_REGRESSION}")
    if IS_REGRESSION:
        assert BOUND_DIST is not None, "Bound distance required for regression"
        log(f"Bound Distance: {BOUND_DIST}")
        log(f"Is Bound Distance Percentage: {IS_BOUND_PERCENT}")

    log(f"Random Feature Split: {RANDOM_FEAT_SPLIT}")
    log(f"Column Feature Split: {COLUMN_FEAT_SPLIT}")
    log(f"Patch Feature Split: {PATCH_FEAT_SPLIT}")
    log(f"Walking Feature Split: {WALKING_FEAT_SPLIT}")
    log(f"Partition Training Set: {PARTITION_TRAIN}")

    log(f"Batch Size: {BATCH_SIZE}")
    log(f"# Epoch: {NUM_EPOCH}")
    log(f"Learning Rate: {_none_format(LEARNING_RATE, '.0E')}")
    log(f"Weight Decay: {_none_format(WEIGHT_DECAY, '.0E')}")

    submodel_str = ALT_TYPE.value if is_alt_submodel() else "NA"
    log(f"Submodel Type: {submodel_str}")
    if MODEL_NAME is not None:
        log(f"Model Name: {MODEL_NAME}")
    log(f"Submodel Constructor Parameters: {MODEL_PARAMS}")
    log(f"Submodel SSL Parameters: {SSL_PARAMS}")
    log(f"# Disjoint Models: {N_DISJOINT_MODELS}")
    log(f"SSL Degree: {SSL_DEGREE}")
    log(f"Validation Split Ratio: {VALIDATION_SPLIT_RATIO:.2f}")

    log(f"Quiet Mode: {QUIET}")


def reset_learner_settings() -> NoReturn:
    r""" DEBUG ONLY.  Reset the settings specific to individual learners/loss functions """
    global LEARNER_CONFIGS
    LEARNER_CONFIGS = dict()


def set_quiet() -> NoReturn:
    r""" Enables quiet mode """
    global QUIET
    QUIET = True


def enable_debug_mode() -> NoReturn:
    r""" Enables debug mode for the learner """
    global DEBUG
    DEBUG = True


def set_num_classes(num_classes: int) -> NoReturn:
    r""" Updates the number of classes """
    global NUM_CLASSES
    NUM_CLASSES = num_classes


def override_num_models(n_models: int) -> NoReturn:
    r""" Overrides the number of training models """
    global N_DISJOINT_MODELS
    N_DISJOINT_MODELS = n_models
    # _verify_num_models_odd()
    logging.info(f"Overriding number of disjoint models to \"{N_DISJOINT_MODELS}\"")


def is_alt_submodel() -> bool:
    r""" Returns \p True if the specified model is an alternate type """
    return not ALT_TYPE.is_torch()


def is_ssl() -> bool:
    r""" Returns \p True if the experiments are semi-supervised learning """
    return SSL_DEGREE > 1


def override_ssl_degree(degree: int) -> NoReturn:
    assert degree > 0, "SSL degree must be positive"
    global SSL_DEGREE
    logging.debug(f"Overriding SSL degree to {degree}")
    SSL_DEGREE = degree


def set_num_feats(n_feats: int) -> NoReturn:
    r""" Override the number of features """
    global DIM_X
    DIM_X = n_feats


def get_n_model() -> int:
    r""" Returns the total number of models """
    return N_DISJOINT_MODELS * SSL_DEGREE


def override_to_walk_split() -> NoReturn:
    r""" Overrides the feature partition to walking split """
    global RANDOM_FEAT_SPLIT
    global PATCH_FEAT_SPLIT
    RANDOM_FEAT_SPLIT = PATCH_FEAT_SPLIT = False

    global WALKING_FEAT_SPLIT
    WALKING_FEAT_SPLIT = True


def use_validation() -> bool:
    r""" Returns \p True if validation data is used """
    return VALIDATION_SPLIT_RATIO > 0
