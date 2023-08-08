__all__ = [
    "construct_tfms",
    "load_data",
]

import dill as pk
import logging
from pathlib import Path

import torch
import torchvision.transforms as tfms

from .types import SplitDataset
from . import utils
from .. import _config as config
from ..types import TensorGroup
from .. import utils as parent_utils


GD_FILE_IDS = {
    SplitDataset.AMES: "1Nx9YLK3UtrOkfJ4sIv7UjNznIXna0JlX",
    SplitDataset.WEATHER: "18Oo_QjjxAq4mIOPfzB_joIGR9gC75e_Y",
}


def load_data(data_dir: Path) -> TensorGroup:
    r""" Load the tabular data """
    tg_pkl_path = parent_utils.construct_filename("bk-tg", out_dir=data_dir,
                                                  file_ext="pk", add_ds_to_path=False,
                                                  add_label_fields=False)
    logging.info(f"Tabular Data Path: {tg_pkl_path}")
    if not tg_pkl_path.exists():
        base_name = config.DATASET.value.name.lower().replace("_", "-")
        gd_name = base_name + "-raw.tar.gz"
        gd_path = data_dir / gd_name

        file_id = GD_FILE_IDS[config.DATASET]
        file_url = "https://drive.google.com/uc?id={}".format(file_id)
        # Preprocessed tensors stored on Google Drive
        utils.download_from_google_drive(dest=data_dir, gd_url=file_url,
                                         file_name=str(gd_path), decompress=True)
        assert gd_path.exists(), f"Downloaded dataset not found at \"{gd_path}\""

        # Standardized location of the decompressed tensors location
        tensor_path = data_dir / "tensors" / (base_name + ".pt")
        assert tensor_path.exists(), f"Decompressed tensors file not found \"{tensor_path}\""
        data_dict = torch.load(tensor_path)  # type: dict

        tg = TensorGroup()
        # Extract train -- Formed by combining train and dev sets
        tr_x_lst, tr_y_lst = [data_dict[utils.TR_X_KEY]], [data_dict[utils.TR_Y_KEY]]
        if data_dict[utils.VAL_X_KEY] is not None:
            tr_x_lst.append(data_dict[utils.VAL_X_KEY])
            tr_y_lst.append(data_dict[utils.VAL_Y_KEY])
        tg.tr_x, tg.tr_y = torch.cat(tr_x_lst, dim=0), torch.cat(tr_y_lst, dim=0)

        # Test set is used directly
        tg.test_x, tg.test_y = data_dict[utils.TEST_X_KEY], data_dict[utils.TEST_Y_KEY]

        # Copy the labels
        tg.tr_lbls, tg.test_lbls = tg.tr_y, tg.test_y

        tg.calc_tr_hash()
        tg.build_ids()

        with open(tg_pkl_path, "wb+") as f_out:
            pk.dump(tg, f_out)

    with open(tg_pkl_path, "rb") as f_in:
        tg = pk.load(f_in)

    utils.print_stats(tg=tg)

    return tg


def construct_tfms():
    r""" Tuple of train and test transforms respectively """
    return tfms.Compose([]), tfms.Compose([])
