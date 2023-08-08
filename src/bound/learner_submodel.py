__all__ = [
    "SubLearner",
]

import dataclasses
import logging
from pathlib import Path
import time
from typing import List, NoReturn, Tuple, Union
import warnings

import numpy as np

import torch
from torch import LongTensor, Tensor
from torch import nn as nn
# noinspection PyUnresolvedReferences
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader

from . import _config as config
from . import _losses as losses
from . import dirs
from .logger import TrainingLogger
from . import utils


@dataclasses.dataclass
class Batch:
    xs: Tensor
    ys: Union[LongTensor, Tensor]
    lbls: LongTensor

    def check(self) -> NoReturn:
        r""" Encompasses checking the validity of the batch """
        assert self.xs.shape[0] == self.ys.numel() == self.lbls.numel(), "Mismatch in length"

    def skip(self) -> bool:
        r""" If \p True, skip processing this batch """
        return self.xs.shape[0] == 0

    def __len__(self) -> int:
        r""" Returns the number of elements in the batch """
        return self.lbls.numel()

    def cuda(self) -> NoReturn:
        r""" Move the batch to the GPU """
        for f in dataclasses.fields(self):
            f_val = self.__getattribute__(f.name)
            if isinstance(f_val, torch.Tensor):
                self.__setattr__(f.name, f_val.to(utils.TORCH_DEVICE))


class SubLearner(nn.Module):
    def __init__(self, prefix: str, model_id: int, submodel_feats: List[int], x: Tensor,
                 opt_params: dict):
        super().__init__()

        self._prefix = prefix
        self._logger = self._train_start = None

        self._model_id = model_id
        self._submodel_feats = submodel_feats

        self.loss = losses.Loss()

        self.optim = None
        self.sched = None
        self._time_str = None

        self._train_loss = self._num_batch = self._valid_loss = None
        self._best_loss, self._best_ep = np.inf, None

        self._tr_tfms = self._test_tfms = None

        self._opt_params = opt_params
        self._module = utils.get_new_model(x=x, opt_params=self._opt_params)

    def _get_name(self) -> str:
        r""" Get the name of the model """
        flds = [
            self._prefix,
            f"model-id={self._model_id:04}"
        ]
        return "-".join(flds)

    def forward(self, xs: Tensor) -> Tensor:
        out = self._module.forward(xs)
        return out

    def fit(self, train_dl: DataLoader, valid_dl: DataLoader) -> NoReturn:
        r""" Fits \p modules' learners to the training and validation \p DataLoader objects """
        # Extract the transforms.  Needed for management later on
        self._tr_tfms = train_dl.dataset.transform  # noqa
        self._test_tfms = valid_dl.dataset.transform  # noqa

        self._configure_fit_vars(train_dl=train_dl)

        # Handle epochs with actual updates
        for ep in range(self.get_num_epochs() + 1):
            self.epoch_start()

            if ep > 0:
                # tmp_dl = train_dl if not config.IS_ROTATE else self.rotate_dataloader(dl=train_dl)
                for batch_tensors in train_dl:
                    batch = self.organize_batch(batch_tensors)
                    if batch.skip():
                        continue
                    self.process_batch(batch)

            self.calc_valid_loss(epoch=ep, valid=valid_dl)

            self._log_epoch(ep)

        self._restore_best_model()
        self.eval()

    def _get_optim_params(self, wd: float):
        r"""
        Special function to disable weight decay of the bias (and other terms)
        See:
        https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially\
        -with-batchnorm/16994/5
        and
        https://discuss.pytorch.org/t/changing-the-weight-decay-on-bias-using-named\
        -parameters/19132/3
        """
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        l2_val = wd
        return [{'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': l2_val}]

    def _restore_best_model(self):
        r""" Restores the best model (i.e., with the minimum validation error) """
        msg = f"Restoring {self._get_name()}'s best trained model"
        logging.debug(f"Starting: {msg}")
        load_module(self, self._build_serialize_name(ep=self._best_ep))
        logging.debug(f"COMPLETED: {msg}")
        self.eval()

    def clear_models(self):
        r""" Restores the best trained model from disk """
        for ep in range(0, self.get_num_epochs() + 1):
            file_path = self._build_serialize_name(ep=ep)
            # Since no longer saving the model file after each epoch have to actually check
            # whether the file path exists.
            if file_path.exists():
                file_path.unlink()  # actually performs the deletion

    def get_num_epochs(self) -> int:
        r""" Returns the number of training epochs """
        if "n_epoch" in self._opt_params:
            return int(self._opt_params["n_epoch"])
        return config.NUM_EPOCH

    def _build_serialize_name(self, ep: int) -> Path:
        r""" Constructs the serialized model's name """
        serialize_dir = dirs.MODELS_DIR / self._prefix.lower()
        prefix = [self._prefix, f"ep={ep:03d}"]
        return utils.construct_filename("_".join(prefix), out_dir=serialize_dir,
                                        model_num=self._model_id,
                                        file_ext="pth", add_timestamp=False)

    def _configure_fit_vars(self, train_dl: DataLoader) -> NoReturn:
        r""" Set initial values/construct all variables used in a fit method """
        learner_names = [self._get_name()]
        loss_names = ["LR", "L-Train", "L-Val", "B"]

        l_name_len = len(learner_names[0]) // 2
        lr_width = 7
        sizes = [lr_width, max(l_name_len, 11), max(l_name_len, 11), 1]

        # Configure the linear dimension automatically
        self.eval()
        for batch_tensors in train_dl:
            batch = self.organize_batch(batch_tensors)
            # Need to transfer to the CPU to ensure everything on the same device
            self.cpu()
            self.forward(batch.xs.cpu())
            self.to(utils.TORCH_DEVICE)
            break

        def _get_param(_name: str, val):
            r""" Optionally get overriding optional parameters """
            return val if not _name in self._opt_params else self._opt_params[_name]

        lr = float(_get_param("lr", config.LEARNING_RATE))
        wd = float(_get_param("wd", config.WEIGHT_DECAY))
        params = self._get_optim_params(wd=wd)

        optim = _get_param("optim", "adamw").lower()
        if optim == "adamw":
            self.optim = AdamW(params, lr=lr, weight_decay=wd)
        elif optim == "adam":
            self.optim = Adam(params, lr=lr, weight_decay=wd)
        elif optim == "sgd":
            self.optim = SGD(params, lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
        else:
            raise ValueError(f"Unknown optimized \"{optim}\"")

        len_dl = len(train_dl)
        if config.DATASET.is_mnist():
            t_max = self.get_num_epochs() * len_dl
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=t_max)
        elif config.DATASET.is_cifar():
            milestones = [
                (0, 1e-8),
                (len_dl * config.NUM_EPOCH // 5, 1),  # Matches more general version of source code
                (len_dl * config.NUM_EPOCH, 1e-8),
            ]
            schedulers, milestone_vals = [], []
            for start_lr, end_lr in zip(milestones[:-1], milestones[1:]):
                piece_sched = torch.optim.lr_scheduler.LinearLR(self.optim, start_factor=start_lr[1],
                                                                end_factor=end_lr[1],
                                                                total_iters=end_lr[0] - start_lr[0])
                schedulers.append(piece_sched)
                milestone_vals.append(end_lr[0])
            # Drop last milestone value since milestones one less than the schedulers
            sched = torch.optim.lr_scheduler.SequentialLR(self.optim, schedulers=schedulers,
                                                          milestones=milestone_vals[:-1])
        else:
            raise NotImplementedError(f"No LR scheduler for dataset {config.DATASET.value.name}")
        self.sched = sched

        # Always log the time in number of seconds
        learner_names.append("")
        loss_names.append("Time")
        sizes.append(10)
        self._logger = TrainingLogger(learner_names, loss_names, sizes)

        self._train_start = time.time()

    def epoch_start(self):
        r""" Configures the module for the start of an epoch """
        self._train_loss, self._num_batch = torch.zeros((), device=utils.TORCH_DEVICE), 0

        self._valid_loss = np.inf

        self.train()

    def process_batch(self, batch: Batch) -> NoReturn:
        r""" Process a batch including tracking the loss and pushing the gradients """
        self.optim.zero_grad()

        loss = self.loss.calc_train_loss(self.forward(batch.xs), batch.ys)
        loss.backward()
        self.optim.step()

        self._train_loss += loss.detach()
        self._num_batch += 1

        # Must be called before scheduler step to ensure relevant LR is stored
        if self.sched is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.sched.step()

    @staticmethod
    def organize_batch(batch_tensors: Tuple[Tensor, ...]) -> Batch:
        r"""
        Organize the batch tensors
        :param batch_tensors: Tuple of tensors returned by the dataloader
        :return: Organized batch
        """
        assert len(batch_tensors) == 2, "Unexpected batch length"
        xs, lbls = batch_tensors  # type: Tensor, LongTensor

        batch = Batch(xs=xs, ys=lbls, lbls=lbls)
        batch.cuda()
        return batch

    def calc_valid_loss(self, epoch: int, valid: DataLoader):
        r""" Calculates and stores the validation loss """
        all_scores, all_ys = [], []

        self.eval()
        for batch_tensors in valid:
            batch = self.organize_batch(batch_tensors)
            if batch.skip():
                continue

            all_ys.append(batch.ys.cpu())
            with torch.no_grad():
                all_scores.append(self.forward(batch.xs).cpu())

        dec_scores, ys = torch.cat(all_scores, dim=0), torch.cat(all_ys, dim=0)
        val_loss = self.loss.calc_validation_loss(dec_scores, ys)
        self._valid_loss = abs(float(val_loss.item()))

        if self._valid_loss < self._best_loss or not config.use_validation():
            self._best_loss = self._valid_loss
            self._best_ep = epoch
            # Since may be a lot of epochs, only save if the epoch is one of the best
            self._save_epoch_module(ep=epoch)

    def _save_epoch_module(self, ep: int):
        r""" Serializes the (sub)epoch parameters """
        # Update the best loss if appropriate
        save_module(self, self._build_serialize_name(ep=ep))

    def _log_epoch(self, ep: int) -> NoReturn:
        r"""
        Log the results of the epoch
        :param ep: Epoch number
        """
        flds = [
            f"{self.sched.get_last_lr()[0]:.1E}",
            self._train_loss / self._num_batch,
            self._valid_loss,
            self._best_ep == ep,
            time.time() - self._train_start,
        ]
        self._logger.log(ep, flds)

    @property
    def test_tfms(self):
        r""" Accessor for the test transforms information """
        return self._test_tfms


def save_module(module: nn.Module, filepath: Path) -> NoReturn:
    r""" Save the specified \p model to disk """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "module-state-dict": module.state_dict(),
    }, str(filepath))


def load_module(module: nn.Module, filepath: Path):
    r"""
    Loads the specified model in file \p filepath into \p module and then returns \p module.

    :param module: \p Module where the module on disk will be loaded
    :param filepath: File where the \p Module is stored
    :return: Loaded model
    """
    state_dicts = torch.load(str(filepath), map_location=utils.TORCH_DEVICE)
    # Map location allows for mapping model trained on any device to be loaded
    module.load_state_dict(state_dicts["module-state-dict"])
    module.eval()
    return module
