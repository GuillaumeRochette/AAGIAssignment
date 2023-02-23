from math import cos, pi
import warnings

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def warm_up_cosine_factor(
    epoch: int,
    warm_up_epochs: int,
    warm_up_factor: float,
    cosine_epochs: int,
    cosine_factor: float,
) -> float:
    """
    Function that computes the current learning rate factor given the current epoch, and initial parameters.

    :param epoch:
    :param warm_up_epochs:
    :param warm_up_factor:
    :param cosine_epochs:
    :param cosine_factor:
    :return:
    """
    if epoch < 0:
        factor = 1.0
    elif 0 <= epoch < warm_up_epochs:
        factor = warm_up_factor + (1e0 - warm_up_factor) * epoch / warm_up_epochs
    elif warm_up_epochs <= epoch < warm_up_epochs + cosine_epochs:
        t = (epoch - warm_up_epochs) / cosine_epochs
        factor = cosine_factor + 0.5 * (1.0 - cosine_factor) * (1.0 + cos(t * pi))
    else:
        factor = cosine_factor
    return factor


class CosineLR(_LRScheduler):
    """
    Scheduler which updates the learning rates of the optimizer given the above-mentioned function.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warm_up_epochs: int,
        warm_up_factor: float,
        cosine_epochs: int,
        cosine_factor: float,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warm_up_epochs = warm_up_epochs
        self.warm_up_factor = warm_up_factor
        self.cosine_epochs = cosine_epochs
        self.cosine_factor = cosine_factor
        super().__init__(
            optimizer=optimizer,
            last_epoch=last_epoch,
            verbose=verbose,
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        num = warm_up_cosine_factor(
            epoch=self.last_epoch,
            warm_up_epochs=self.warm_up_epochs,
            warm_up_factor=self.warm_up_factor,
            cosine_epochs=self.cosine_epochs,
            cosine_factor=self.cosine_factor,
        )
        den = warm_up_cosine_factor(
            epoch=self.last_epoch - 1,
            warm_up_epochs=self.warm_up_epochs,
            warm_up_factor=self.warm_up_factor,
            cosine_epochs=self.cosine_epochs,
            cosine_factor=self.cosine_factor,
        )
        factor = num / max(den, 1e-8)
        return [factor * group["lr"] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        factor = warm_up_cosine_factor(
            epoch=self.last_epoch,
            warm_up_epochs=self.warm_up_epochs,
            warm_up_factor=self.warm_up_factor,
            cosine_epochs=self.cosine_epochs,
            cosine_factor=self.cosine_factor,
        )
        return [factor * base_lr for base_lr in self.base_lrs]


if __name__ == "__main__":
    import torch

    max_epochs = 110
    m = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(m.parameters(), lr=1.0)
    sch = CosineLR(
        optimizer=opt,
        warm_up_epochs=5,
        warm_up_factor=1e-2,
        cosine_epochs=95,
        cosine_factor=1e-2,
        verbose=True,
    )
    for epoch in range(max_epochs):
        factor = warm_up_cosine_factor(
            epoch,
            warm_up_epochs=5,
            warm_up_factor=1e-2,
            cosine_epochs=95,
            cosine_factor=1e-2,
        )
        print(epoch, factor)
        opt.zero_grad()
        opt.step()
        sch.step()
