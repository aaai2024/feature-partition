__all__ = [
    "DisjointEnsemble",
    "create_fit_dataloader",
    "get_validation_mask",
    "train_ensemble",
]

import collections

import dill as pk
import io
import logging
import math
from pathlib import Path
from pickle import UnpicklingError
import sys
from typing import Any, List, NoReturn, Optional, Tuple
import warnings

from lightgbm import LGBMRegressor
import pycm
from sklearn.linear_model import \
    HuberRegressor, \
    Lasso
from sklearn.semi_supervised import SelfTrainingClassifier
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch import BoolTensor, LongTensor, Tensor
import torch.nn.functional as F  # noqa

from . import _config as config
from . import dirs
from ._fixed_predictor import MedianRegressor
from .learner_submodel import SubLearner
from . import robustness
from .types import CustomTensorDataset, TensorGroup

from . import utils

BoundReturnType = Tuple[LongTensor, Optional[Any]]

UNSUPERVISED_LABEL = -1

REG_LOSS = F.l1_loss


class DisjointEnsemble:
    r""" Ensemble regression learner """
    def __init__(self, x: Tensor, opt_params: Optional[dict]):
        super().__init__()

        flds = []
        if config.is_alt_submodel():
            flds.append(config.ALT_TYPE.name.lower())
        flds += ["disjoint"]
        self._prefix = "-".join(flds)

        self._assign_submodel_feats(x=x)

        self._model_paths = []

        # Stores the models which each dataset part participated in training
        self._ds_map = None

        self._opt_params = opt_params if opt_params is not None else dict()

    def _assign_submodel_feats(self, x: Tensor) -> NoReturn:
        r""" Assign the submodel features """
        ds_parts_path = utils.construct_filename(prefix=self._prefix + "-model-feats",
                                                 file_ext="pk",
                                                 out_dir=self.build_models_dir(),
                                                 model_num=None,
                                                 add_ds_to_path=False,
                                                 add_label_fields=False)
        if not ds_parts_path.exists():
            logging.debug("Reseeding the random number generator for consistent part splitting")
            utils.set_random_seeds()

            if config.RANDOM_FEAT_SPLIT:
                model_feats = self._assign_feats_uar()
            elif config.PATCH_FEAT_SPLIT:
                model_feats = self._assign_patch_feats()
            elif config.WALKING_FEAT_SPLIT:
                model_feats = self._assign_walking_feats()
            else:
                raise NotImplementedError("Feature split mode not implemented")

            self._submodel_feats = model_feats
            # Stores the dataset parts used by each model
            with open(ds_parts_path, "wb+") as f_out:
                pk.dump(self._submodel_feats, f_out)
        else:
            # logging.warning(f"Loading submodel feats from file \"{ds_parts_path}\"")
            with open(ds_parts_path, "rb") as f_in:
                self._submodel_feats = pk.load(f_in)

        # Calculate the spread degree
        flat_feats = [feat for sub_feats in self._submodel_feats for feat in sub_feats]
        deg = collections.Counter(flat_feats)
        least_common, most_common = deg.most_common()[-1], deg.most_common(n=1)[0]
        # Most common returns a tuple of the value and the count. we just want the count
        least_common, most_common = least_common[1], most_common[1]
        assert least_common == most_common, "Inconsistent spread degree"
        self._spread_degree = most_common

    def _assign_feats_uar(self) -> List[List[int]]:
        r"""
        Assigns features to each model uniformly at random

        The implementation does not directly rely on a hash function.  Instead it performs
        a series of random permutations.  This ensures that each model is trained on the same
        number of dataset partitions.

        :return: Assignment of dataset parts to the models
        """
        assert config.N_DISJOINT_MODELS <= config.DIM_X, "Feat dim must exceed disjoint model count"
        n_feat_split = config.N_DISJOINT_MODELS

        # Reset the random seeds before feature split to ensure a clear seed for the splits
        utils.set_random_seeds()

        model_feats = [[] for _ in range(n_feat_split)]
        feat_lst = torch.arange(config.DIM_X, dtype=torch.long)
        feat_lst = feat_lst[torch.randperm(feat_lst.numel())]
        # Select the dataset parts assigned to each of the ensemble models
        i_model = 0
        for feat in feat_lst.tolist():
            # noinspection PyTypeChecker
            model_feats[i_model].append(feat)
            i_model = (i_model + 1) % n_feat_split

        self._check_feats_list_valid(model_feats=model_feats)
        return model_feats

    def _assign_walking_feats(self) -> List[List[int]]:
        r"""
        Assigns features using a walking pattern

        :return: Assignment of dataset parts to the models
        """
        assert not config.PATCH_FEAT_SPLIT and not config.RANDOM_FEAT_SPLIT

        assert config.N_DISJOINT_MODELS <= config.DIM_X, "Feat dim must exceed disjoint model count"
        n_feat_split = config.N_DISJOINT_MODELS

        # # Reset the random seeds before feature split to ensure a clear seed for the splits
        # utils.set_random_seeds()

        model_feats = []
        feat_lst = torch.arange(config.DIM_X, dtype=torch.long)
        feat_lst_mod = feat_lst % n_feat_split
        for feat_idx in range(n_feat_split):
            mask = feat_lst_mod == feat_idx
            # noinspection PyTypeChecker
            model_feats.append(feat_lst[mask].tolist())

        self._check_feats_list_valid(model_feats=model_feats)
        return model_feats

    @staticmethod
    def _check_feats_list_valid(model_feats: List[List[int]]) -> NoReturn:
        r""" Verify nothing invalid in the features list """
        tot_len = sum([len(feats) for feats in model_feats])
        assert tot_len == config.DIM_X, "Unexpected feature count"
        # Verify no features dropped or added
        all_kept_feats = set(feat for feats in model_feats for feat in feats)
        assert len(all_kept_feats) == config.DIM_X, "Duplicate feature found"
        assert min(all_kept_feats) == 0 and max(all_kept_feats) == config.DIM_X - 1, \
            "Unexpected feature index"

    @staticmethod
    def _assign_patch_feats() -> List[List[int]]:
        r"""
        Assigns features to each model uniformly at random

        The implementation does not directly rely on a hash function.  Instead it performs
        a series of random permutations.  This ensures that each model is trained on the same
        number of dataset partitions.

        :return: Assignment of dataset parts to the models
        """
        assert not config.WALKING_FEAT_SPLIT and not config.RANDOM_FEAT_SPLIT
        if config.SSL_DEGREE > 1:
            raise NotImplemented("Spread random spread degree above 1 not implemented")

        model_feats = [[mod_id] for mod_id in range(0, config.N_DISJOINT_MODELS)]
        return model_feats

    @staticmethod
    def _assign_column_feats(x: Tensor) -> List[List[int]]:
        r"""
        Assigns features to each model uniformly at random

        The implementation does not directly rely on a hash function.  Instead it performs
        a series of random permutations.  This ensures that each model is trained on the same
        number of dataset partitions.

        :return: Assignment of dataset parts to the models
        """
        assert config.N_DISJOINT_MODELS <= x.shape[2], "Feat dim must exceed disjoint model count"
        n_row, n_col = x.shape[1], x.shape[2]
        model_feats = []
        for i in range(config.N_DISJOINT_MODELS):
            feats = []
            start_col = i * n_col // config.N_DISJOINT_MODELS
            end_col = min(n_col, (i + 1) * n_col // config.N_DISJOINT_MODELS)
            for col in range(start_col, end_col):
                for row in range(n_row):
                    feats.append(row * n_col + col)
            model_feats.append(feats)
        return model_feats

    def name(self) -> str:
        r""" Standardizes name of the ensemble model """
        return self._prefix

    def build_models_dir(self) -> Path:
        r""" Builds the directory to build the models folder """
        return dirs.MODELS_DIR / config.DATASET.value.name.lower() / self._prefix.lower()

    # @property
    # @abc.abstractmethod
    # def cover_type(self) -> str:
    #     r""" Define the cover type of the model """

    def get_submodel_type(self):
        r""" Gets the class of the submodels """
        assert self._model_paths, "No models to load"
        with open(self._model_paths[0], "rb") as f_in:
            model = pk.load(f_in)
        return model.__class__

    @staticmethod
    def _get_submodel_feats_idx(model_id) -> int:
        r""" Gets the submodel features ID number """
        return model_id // config.SSL_DEGREE

    def get_submodel_feats(self, model_id: int) -> List[int]:
        r""" Gets the features of the submodel """
        feats_idx = self._get_submodel_feats_idx(model_id=model_id)
        return self._submodel_feats[feats_idx]

    # def _log_submodel_info(self, model_id: int, sizes: List[int]) -> NoReturn:
    def _log_submodel_info(self, model_id: int, tr_y: Tensor) -> NoReturn:
        r""" Log information about the submodel dataset sizes """
        # logging.info(f"Model {model_id}: Dataset Parts Sizes: {sizes}")
        # # Log info about the model
        # size = sum(sizes)
        # ratio = size / self._tr_size
        submodel_feats = self.get_submodel_feats(model_id=model_id)
        n_sub_feats = len(submodel_feats)
        n_feats = config.DIM_X
        rate = n_sub_feats / n_feats

        header = f"Model {model_id}:"
        logging.info(f"{header} # Features: {n_sub_feats} / {n_feats} ({rate:.2%})")
        n_tr = tr_y.numel()
        logging.info(f"{header} # Train Instance: {n_tr}")
        if config.IS_CLASSIFICATION and config.is_ssl():
            n_unlabel = (tr_y == UNSUPERVISED_LABEL).sum().item()
            rate = n_unlabel / n_tr
            logging.info(f"{header} # Unlabeled Train: {n_unlabel} / {n_tr} ({rate:.2%})")

    def fit(self, tg: TensorGroup):
        r"""
        Fit all models
        :param tg: Training & testing tensors
        :return: Training \p DataLoader
        """
        # self._train_start = time.time()
        # for model_id, submodel_feats in enumerate(self._submodel_feats):
        n_model = config.get_n_model()
        for model_id in range(n_model):
            submodel_feats = self.get_submodel_feats(model_id=model_id)

            model_path = utils.construct_filename(prefix=self._prefix, file_ext="pk",
                                                  out_dir=self.build_models_dir(),
                                                  model_num=model_id, add_ds_to_path=False)
            self._model_paths.append(model_path)

            if model_path.exists():
                continue

            # Log info about the model
            submodel_feats = sorted(submodel_feats)
            logging.debug(f"Model ID (out of {self.n_models}): {model_id}")
            logging.info(f"Model {model_id}: Submodel Features: {submodel_feats}")
            feat_idx = self._get_submodel_feats_idx(model_id=model_id)
            logging.info(f"Model {model_id}: Submodel Feature Index: {feat_idx}")

            if config.is_alt_submodel():
                model = self._train_alt_submodel(tg=tg, model_id=model_id,
                                                 submodel_feats=submodel_feats)
            elif config.ALT_TYPE.is_custom_nn():
                model = self._train_new_model_module(model_id=model_id, tg=tg,
                                                     submodel_feats=submodel_feats)
            else:
                raise ValueError("Unknown how to train specified model type")

            # Logging test error after each model significantly slows down hyperparameter
            # tuning so only log the accuracy/error during normal training.
            if config.IS_CLASSIFICATION:
                self.calc_acc(model=model, model_id=model_id, tg=tg, is_test=False)
                self.calc_acc(model=model, model_id=model_id, tg=tg, is_test=True)
            else:
                self.calc_test_err(model=model, model_id=model_id, tg=tg)

            if isinstance(model, nn.Module):
                model.cpu()
            with open(model_path, "wb+") as f_out:
                pk.dump(model, f_out)

    def _train_new_model_module(self, tg: TensorGroup, model_id: int, submodel_feats: List[int]):
        r""" Trains a standard torch neural module """
        x_base = tg.tr_x[0:1]

        model = SubLearner(model_id=model_id, submodel_feats=submodel_feats, prefix=self._prefix,
                           x=x_base, opt_params=self._opt_params)
        model.to(utils.TORCH_DEVICE)

        bs = config.BATCH_SIZE if "bs" not in self._opt_params else int(self._opt_params["bs"])
        train, valid = create_fit_dataloader(model_id=model_id, tg=tg,
                                             submodel_feats=submodel_feats, bs=bs)

        # noinspection PyUnresolvedReferences
        tr_y = train.dataset.tensors[1]
        self._log_submodel_info(model_id=model_id, tr_y=tr_y)

        with utils.TrainTimer(model_name=self.name(), model_id=model_id):
            model.fit(train_dl=train, valid_dl=valid)
        model.eval()
        model.clear_models()

        return model

    def _train_alt_submodel(self, tg: TensorGroup, model_id: int, submodel_feats: List[int]):
        r""" Trains a TabNet learner as the submodel """
        # x_tr, y_tr, x_val, y_val, sizes = _split_train_val(tg=tg, n_ds_parts=self._n_ds_parts,
        #                                                    ds_part_lst=ds_parts)
        x_tr, y_tr, x_val, y_val = _split_train_val(model_id=model_id, tg=tg,
                                                    submodel_feats=submodel_feats)

        self._log_submodel_info(model_id=model_id, tr_y=y_tr)
        # Flatten x since tree assumes the data is a vector
        if len(x_tr.shape) > 2:
            x_tr = x_tr.view([x_tr.shape[0], -1])

        # noinspection PyTypeChecker
        if config.IS_CLASSIFICATION:
            raise ValueError(f"Unknown alternate cls. submodel type {config.ALT_TYPE.name}")
        else:
            try:
                self._opt_params["max_iter"] = int(self._opt_params["max_iter"])
            except KeyError:
                pass
            if config.ALT_TYPE.is_lgbm():
                model = LGBMRegressor(**self._opt_params)
            elif config.ALT_TYPE.is_huber():
                model = HuberRegressor(**self._opt_params)
            elif config.ALT_TYPE.is_lasso():
                model = Lasso(**self._opt_params)
            elif config.ALT_TYPE.is_median():
                model = MedianRegressor(**self._opt_params)
            else:
                raise ValueError(f"Unknown alternate reg. submodel type {config.ALT_TYPE.name}")

        # Mark that the model has no test transforms
        if config.is_ssl():
            assert config.IS_CLASSIFICATION, "Only classification supported for SSL"
            model = SelfTrainingClassifier(base_estimator=model, **config.SSL_PARAMS)
        model.test_tfms = None

        with utils.TrainTimer(model_id=model_id, model_name=self.name()):
            model.fit(X=x_tr.numpy(), y=y_tr.numpy())

        if config.is_ssl():
            term_cond = model.termination_condition_
            logging.info(f"Model {model_id}: SSL Terminate Condition: {term_cond}")
            tr_size = model.transduction_.shape[0]
            logging.info(f"Model {model_id}: SSL Final Train Size: {tr_size}")

        return model

    def calc_acc(self, model, model_id: int, tg: TensorGroup, is_test: bool) -> NoReturn:
        r""" Log the test accuracy for a single model when performing classification """
        assert config.IS_CLASSIFICATION, "Accuracy only applicable for classification"

        x = tg.test_x if is_test else tg.tr_x
        test_x = _select_x_features(x=x, feats=self.get_submodel_feats(model_id=model_id))

        ds = CustomTensorDataset((test_x,), transform=model.test_tfms)
        dl = DataLoader(ds, batch_size=config.FORWARD_BATCH_SIZE, shuffle=False,
                        drop_last=False, num_workers=utils.NUM_WORKERS)

        all_y_hat = []
        with torch.no_grad():
            for xs, in dl:
                if config.ALT_TYPE.is_torch():
                    xs = xs.to(utils.TORCH_DEVICE)
                y_hat = self._model_predict(model=model, xs=xs)
                # y_hat = torch.argmax(scores, dim=1)
                # noinspection PyUnresolvedReferences
                all_y_hat.append(y_hat)
        y_hat = torch.cat(all_y_hat, dim=0).cpu()
        if len(y_hat.shape) == 2:
            y_hat = y_hat.squeeze(dim=1)

        # Pre-calculate fields needed in other calculations
        y = tg.test_y if is_test else tg.tr_y
        conf_matrix = pycm.ConfusionMatrix(actual_vector=y.numpy(), predict_vector=y_hat.numpy())

        str_prefix = f"Model {model_id}"
        # noinspection PyUnresolvedReferences
        ds_name = "Test" if is_test else "Train"
        logging.debug(f"{str_prefix} {ds_name} Size: {y.numel()}")
        logging.debug(f"{str_prefix} {ds_name} Accuracy: {100. * conf_matrix.Overall_ACC:.3}%")  # noqa

        sys.stdout = cm_out = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            # Write confusion matrix to a string so it can be logged
            conf_matrix.print_matrix()
        sys.stdout = sys.__stdout__
        # Log the confusion matrix
        cm_str = cm_out.getvalue()
        logging.debug(f"{str_prefix} {ds_name} Confusion Matrix: \n{cm_str}")

    def calc_test_err(self, model, model_id: int, tg: TensorGroup) -> NoReturn:
        r"""
        Estimate the test error for the submodel \p model.
        :param model: Submodel of interest
        :param model_id: Identification number of the submodel
        :param tg: Tensor information
        """
        assert not config.IS_CLASSIFICATION, "Test error for direct regression not classification"

        # x, y = tg.test_x, tg.test_y
        # if config.IS_ROTATE:
        #     y = utils.calc_rand_rotation(n_ele=y.numel())
        #     x = utils.rotate_tensor(xs=x, angle=y)
        test_x = _select_x_features(x=tg.test_x, feats=self.get_submodel_feats(model_id=model_id))
        ds = CustomTensorDataset((test_x, tg.test_y), transform=model.test_tfms)
        dl = DataLoader(ds, batch_size=config.FORWARD_BATCH_SIZE, shuffle=False,
                        drop_last=False, num_workers=utils.NUM_WORKERS)

        # Get loss estimate for the whole test set
        all_y, all_yhat = [], []
        for batch_tensors in dl:
            try:
                batch = model.organize_batch(batch_tensors)
                xs, ys = batch.xs, batch.ys
            except AttributeError:
                xs, ys = batch_tensors[0], batch_tensors[1]

            with torch.no_grad():
                y_hat = self._model_forward(model=model, xs=xs)
            all_yhat.append(y_hat)
            all_y.append(ys)

        # Combine all the results and ensure a valid shape for calculating the loss
        y_hat, y = torch.cat(all_yhat, dim=0).cpu(), torch.cat(all_y, dim=0).cpu()
        if len(y_hat.shape) == 2:
            y_hat = y_hat.squeeze(dim=1)
        if len(y.shape) == 2:
            y = y.squeeze(dim=1)

        err = REG_LOSS(input=y_hat, target=y)
        logging.info(f"Model {model_id} Mean Test Loss: {err:.4E}")

    def forward(self, x: Tensor) -> Tensor:
        r""" Make the prediction across all ensemble submodels """
        # noinspection PyUnresolvedReferences
        preds = []
        for model_path in self._model_paths:
            with open(model_path, "rb") as f_in:
                model = pk.load(f_in)

            model.to(utils.TORCH_DEVICE)
            model.eval()

            scores = self._model_forward(model=model, xs=x)

            preds.append(scores)
        # Aggregate across the ensemble
        preds = torch.cat(preds, dim=1)
        return preds

    def predict_wide(self, x: Tensor) -> Tensor:
        r"""
        Special version of the forward method where it uses an underlying \p DataLoader to
        limit the amount each model needs to be reloaded from disk.
        """
        return self._forward_wide(x=x, use_predict=True)

    def forward_wide(self, x: Tensor) -> Tensor:
        r"""
        Special version of the forward method where it uses an underlying \p DataLoader to
        limit the amount each model needs to be reloaded from disk.
        """
        return self._forward_wide(x=x, use_predict=False)

    def _forward_wide(self, x: Tensor, use_predict: bool) -> Tensor:
        r"""
        Implementation of the wide results method used by \p predict_wide and \p forward_wide.
        """
        ensemble_preds = []

        for i_model, model_path in enumerate(self._model_paths):
            # logging.debug(f"Loading Submodel File: {model_path}")
            try:
                model = self._load_submodel(model_path=model_path)
            except EOFError as e:
                raise ValueError(f"Unable to load submodel file \"{model_path}\" with res. {e}")

            if isinstance(model, nn.Module):
                model.to(utils.TORCH_DEVICE)
                model.eval()

            bs = config.FORWARD_BATCH_SIZE
            if bs is None:
                bs = x.shape[0]
            try:
                tfms = model.test_tfms
            except AttributeError:
                tfms = None

            # Need to customize by model since each model may have a different input transform
            x_tmp = _select_x_features(x=x, feats=self.get_submodel_feats(model_id=i_model))
            if config.DATASET.is_ames():
                dl = [(x_tmp,)]
            else:
                ds = CustomTensorDataset((x_tmp,), transform=tfms)
                dl = DataLoader(ds, batch_size=bs, shuffle=False, drop_last=False,
                                num_workers=utils.NUM_WORKERS)

            model_preds = []
            with torch.no_grad():
                for xs, in dl:
                    if config.ALT_TYPE.is_torch():
                        xs = xs.to(utils.TORCH_DEVICE)
                    if use_predict:
                        ys = self._model_predict(model=model, xs=xs).view([-1, 1])
                    else:
                        ys = self._model_forward(model=model, xs=xs)
                        if config.IS_CLASSIFICATION:
                            ys.unsqueeze_(dim=1)
                    model_preds.append(ys.cpu())

            # All of the example predictions combined
            model_preds = torch.cat(model_preds, dim=0)  # .view([-1, 1])
            ensemble_preds.append(model_preds)

        # Aggregate
        ensemble_pred = torch.cat(ensemble_preds, dim=1)
        return ensemble_pred

    def _forward_detail(self, x: Tensor) -> List[List[Tensor]]:
        r"""
        Special version of the forward method where it uses an underlying \p DataLoader to
        limit the amount each model needs to be reloaded from disk.
        """
        ex_preds = [[] for _ in range(x.shape[0])]

        for model_path in self._model_paths:
            with open(model_path, "rb") as f_in:
                model = pk.load(f_in)

            bs = config.FORWARD_BATCH_SIZE
            tfms = model.test_tfms if not config.DATASET.is_tabular() else None
            # Need to customize by model since each model may have a different input transform
            ds = CustomTensorDataset((x,), transform=tfms)
            dl = DataLoader(ds, batch_size=bs, shuffle=False, drop_last=False,
                            num_workers=utils.NUM_WORKERS)

            with torch.no_grad():
                i_ele = 0
                for xs, in dl:
                    if config.ALT_TYPE.is_torch():
                        xs = xs.to(utils.TORCH_DEVICE)
                    ys = model.predict_detail(xs)
                    for idx in range(xs.shape[0]):
                        ex_preds[i_ele].append(ys[idx].view([1, -1]))
                        i_ele += 1

        assert len(ex_preds) == x.shape[0], "Mismatch in example counts"
        assert all(len(vals) == self.n_models for vals in ex_preds), "Mismatch with model count"
        return ex_preds

    @staticmethod
    def _load_submodel(model_path: Path):
        r""" Standardizes loading a submodel path """
        try:
            assert model_path.exists(), f"Load submodule \"{model_path}\" but file does not exist"
            with open(model_path, "rb") as f_in:
                return pk.load(f_in)
        except UnpicklingError as e:
            logging.error(f"Error opening file {model_path}")
            raise e

    @staticmethod
    def _model_forward(model, xs: Tensor) -> Tensor:
        r""" Standardizes the forward method for submodels as it can differ by setup """
        if config.ALT_TYPE.is_custom_nn():
            yhat = model.forward(xs)
            # if config.IS_CLASSIFICATION:
            #     if yhat.shape[1] > 1:
            #         yhat = torch.argmax(yhat, dim=1)
            return yhat

        device = xs.device
        # # xs = xs.view([xs.shape[0], -1]).cpu().numpy()
        # xs = _select_x_features(x=xs, feats=feats)
        if not config.ALT_TYPE.is_torch():
            xs = xs.numpy()
        # Custom for models using the sklearn API
        if config.IS_CLASSIFICATION:
            y_hat = model.predict_proba(xs)
        elif config.IS_REGRESSION:
            y_hat = model.predict(xs)
        else:
            raise ValueError("Unknown how to predict with model")
        y_hat = torch.from_numpy(y_hat)
        if config.IS_REGRESSION:
            y_hat = y_hat.view([-1, 1])
        return y_hat.to(device)

    def _model_predict(self, model, xs: Tensor) -> Tensor:
        r""" Return the model prediction """
        out = self._model_forward(model=model, xs=xs)
        if len(out.shape) > 1 and out.shape[1] > 1:
            out = torch.argmax(out, dim=1)
        return out

    @property
    def n_models(self) -> int:
        r""" Number of submodels used in the ensemble """
        return config.get_n_model()

    def calc_prediction(self, ys: Tensor) -> Tensor:
        r"""
        Calculate the predicted output. By default, \p use_median is True and the median is used
        to combine the votes.
        """
        assert ys.shape[1] == self.n_models, "Mismatch between number of models and the dimension"
        if config.IS_CLASSIFICATION:
            ys, _ = torch.sort(ys, dim=1)
            lbl = []
            # Iterate through all instances and return the most popular label for each instance
            # Breaking ties using the smallest label
            for i in range(ys.shape[0]):
                row = ys[i]
                counter = collections.Counter(row.tolist())
                most_popular = counter.most_common(n=1)
                most_popular_lbl, _ = most_popular[0]
                lbl.append(most_popular_lbl)

            vals = torch.LongTensor(lbl)
        elif config.IS_REGRESSION:
            assert len(ys.shape) == 2, "Unexpected shape for the regression prediction"
            vals, _ = torch.median(ys, dim=1)
        else:
            raise NotImplementedError
        return vals

    @staticmethod
    def calc_classification_bound(n_cls: int, full_yhat: Tensor,
                                  y_lbl: LongTensor) -> LongTensor:
        r""" Calculates the top-1 bound """
        bound = robustness.certifier.calc_classification_bound(n_cls, full_yhat=full_yhat,
                                                               y=y_lbl)
        return bound

    @staticmethod
    def calc_topk_bound(k: int, n_cls: int, full_yhat: Tensor,
                        y: LongTensor) -> LongTensor:
        r"""
        Modifies the script to calculate the top-k bound
        :param k:
        :param n_cls:
        :param full_yhat:
        :param y:
        :return:
        """
        bound = robustness.certifier.calc_topk_bound(k=k, n_cls=n_cls, full_yhat=full_yhat, y=y)
        return bound

    @staticmethod
    def calc_runoff_bound(full_yhat: Tensor) -> Tuple[LongTensor, LongTensor]:
        r""" Calculates the top-1 bound """
        res = robustness.runoff.calc(all_votes=full_yhat)
        return res


def create_fit_dataloader(model_id: int, tg: TensorGroup, submodel_feats: List[int], bs: int) \
        -> Tuple[DataLoader, DataLoader]:
    r"""
    Simple method that splits the positive and unlabeled sets into stratified training and
    validation \p DataLoader objects

    :param model_id:
    :param tg: TensorGroup of vectors
    :param submodel_feats: List of the submodel features
    :param bs: Training batch size
    :return: Training and validation \p DataLoader objects respectively then the size of the
             dataset parts
    """
    x_tr, y_tr, x_val, y_val = _split_train_val(model_id=model_id, tg=tg,
                                                submodel_feats=submodel_feats)
    tr_tfms, val_tfms = utils.get_tfms(x=x_tr)

    # Construct the train dataloader
    tr_ds = CustomTensorDataset([x_tr, y_tr], transform=tr_tfms)
    tr = DataLoader(tr_ds, shuffle=True, drop_last=True, batch_size=bs,
                    num_workers=utils.NUM_WORKERS)

    # construct the validation dataloader
    val_ds = CustomTensorDataset([x_val, y_val], transform=val_tfms)
    val = DataLoader(val_ds, shuffle=False, drop_last=False,
                     batch_size=config.FORWARD_BATCH_SIZE, num_workers=utils.NUM_WORKERS)

    return tr, val


def _split_train_val(model_id: int, tg: TensorGroup, submodel_feats: List[int]) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r""" Groups X and y into train and validation based on the dataset parts """

    tr_x, tr_y = tg.tr_x, tg.tr_y
    if config.PARTITION_TRAIN:
        assert config.SSL_DEGREE == 1, "Multiple spread degree not supported"
        mask = tg.tr_hash % config.get_n_model() == model_id
        tr_x, tr_y = tg.tr_x[mask], tg.tr_y[mask]
    elif config.SSL_DEGREE > 1:
        assert not config.PARTITION_TRAIN, "Partition train not supported with SSL degree > 1"
        tr_x, tr_y = tg.tr_x.clone(), tg.tr_y.clone()
        tr_hash = tg.tr_hash % config.SSL_DEGREE
        mask = (tr_hash % config.SSL_DEGREE) != (model_id % config.SSL_DEGREE)
        assert config.is_ssl() or mask.sum() == tr_y.numel(), \
            "All examples should be on when not in SSL mode"
        tr_y[mask] = UNSUPERVISED_LABEL

    if config.use_validation():
        val_mask = get_validation_mask(n_tr=tr_x.shape[0])
        x_val, y_val = tr_x[val_mask], tr_y[val_mask]
        tr_mask = ~val_mask
        x_tr, y_tr = tr_x[tr_mask], tr_y[tr_mask]
    else:
        # Record the test error for data collection
        x_tr, y_tr = tr_x, tr_y
        x_val, y_val = tg.test_x, tg.test_y

    # Downsample the features
    x_tr = _select_x_features(x=x_tr, feats=submodel_feats)
    x_val = _select_x_features(x=x_val, feats=submodel_feats)

    return x_tr, y_tr, x_val, y_val


def _select_x_features(x: Tensor, feats: List[int]) -> Tensor:
    r""" Select and return the x features """
    # Single case for the empty tensor
    if x.numel() == 0:
        return x
    if config.RANDOM_FEAT_SPLIT or config.WALKING_FEAT_SPLIT:
        return _select_feats_from_list(x=x, feats=feats)
    if config.PATCH_FEAT_SPLIT:
        return _select_patch_feature(x=x, feats=feats)
    raise ValueError("Unknown way to select x features")


def _select_feats_from_list(x: Tensor, feats: List[int]) -> Tensor:
    r""" Encapsulates the selection of random features """
    assert not config.PATCH_FEAT_SPLIT, "Patch feature split cannot be enabled in random mode"

    if len(x.shape) == 2:
        feats = torch.LongTensor(sorted(feats))
        x = x[:, feats]
        return x

    if len(x.shape) in (3, 4):
        # Get the list of features to zero out
        all_feats = torch.arange(config.DIM_X, dtype=torch.long)
        feats = torch.LongTensor(sorted(feats))
        zero_feats = torch.isin(all_feats, feats, invert=True)
        # Get the indices of the features to drop
        zero_feats = all_feats[zero_feats]

        # Convert to 1D to make it efficient to zero out
        out = x.clone().reshape(list(x.shape[:2]) + [-1])
        out[:, :, zero_feats] = 0
        if len(x.shape) == 4:
            out = out.view(x.shape)
        else:
            out = out.view([x.shape[0], -1])
        return out

    raise NotImplementedError(f"Random feature selection not implemented for size {x.shape}")


def _select_patch_feature(x: Tensor, feats: List[int]):
    r""" Select the features as a patch """
    assert not config.RANDOM_FEAT_SPLIT, "Random feature split cannot be enabled in patch mode"
    assert config.SSL_DEGREE == 1, "Patch feature split not supported with SSL degree > 1"

    assert len(x.shape) == 4, "Expected X tensor to be 4D"
    if config.SSL_DEGREE != 1:
        raise NotImplemented("Only a single spread degree currently supported")

    n_col = int(round(math.sqrt(config.DIM_X)))
    # Calculate the row and column of the patch
    row, col = feats[0] // n_col, feats[0] % n_col

    def _calc_dim(idx: int, feat_dim: int) -> Tuple[int, int]:
        r""" Calculate the patch dimensions """
        first = int(round(idx * feat_dim / n_col))
        if idx - 1 == n_col:
            last = feat_dim
        else:
            last = 1 + int(round((idx + 1) * feat_dim / n_col))
        return first, last

    # Extract the patch
    first_row, last_row = _calc_dim(idx=row, feat_dim=x.shape[-2])
    first_col, last_col = _calc_dim(idx=col, feat_dim=x.shape[-1])
    x = x[:, :, first_row:last_row, first_col:last_col]
    return x


def get_validation_mask(n_tr: int) -> BoolTensor:
    r""" Returns \p True for each element in \p hash_vals if that value is used for validation """
    n_val = int(n_tr * config.VALIDATION_SPLIT_RATIO)
    perm = torch.randperm(n_tr)[:n_val]

    mask = torch.zeros((n_tr,), dtype=torch.bool)  # type: BoolTensor  # noqa
    mask[perm] = True
    return mask


def in1d(ar1, ar2) -> BoolTensor:
    r""" Returns \p True if each element in \p ar1 is in \p ar2 """
    mask = ar2.new_zeros((max(ar1.max(), ar2.max()) + 1,), dtype=torch.bool)
    mask[ar2.unique()] = True
    return mask[ar1]


def train_ensemble(tg: TensorGroup, opt_params: Optional[dict] = None) -> DisjointEnsemble:
    r"""
    Train an ensemble where each submodel is trained on a disjoint dataset

    :param tg: \p TensorGroup
    :param opt_params: Optional model parameters. Primarily used for hyperparameter tuning
    :return: Collection of trained classifiers
    """
    # Prefix used for the serialized backup
    prefix_flds = []
    if config.is_alt_submodel():
        prefix_flds.append(config.ALT_TYPE.name.lower())
    prefix_flds += [
        "disjoint",
        f"{config.N_DISJOINT_MODELS:04d}",
    ]
    if config.is_ssl():
        prefix_flds.append(f"ssl{config.SSL_DEGREE:03d}")
    prefix_flds.append("fin")

    model_dir = dirs.MODELS_DIR / config.DATASET.value.name.lower() / "fin"
    train_net_path = utils.construct_filename("-".join(prefix_flds), out_dir=model_dir,
                                              file_ext="pk", add_ds_to_path=False)

    # Model description only used for logging serialization info
    model_str = f"{config.get_n_model()}"
    if config.is_alt_submodel():
        model_str = f"{model_str} {config.ALT_TYPE.value}"
    model_desc = f"Disjoint ensemble with {model_str} total models"

    if not train_net_path.exists():
        learner = DisjointEnsemble(opt_params=opt_params, x=tg.tr_x[0])
        learner.fit(tg=tg)

        logging.info(f"Saving final {model_desc}...")
        with open(str(train_net_path), "wb+") as f_out:
            pk.dump(learner, f_out)

    # Load the saved module
    logging.info(f"Loading final {model_desc}...")
    with open(str(train_net_path), "rb") as f_in:
        learner = pk.load(f_in)  # CombinedLearner
    return learner
