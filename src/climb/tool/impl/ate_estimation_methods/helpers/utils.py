# losses.py contains a pytorch implementation of the counterfactual balancing loss introduced
# by Shalit et al 2017.


import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
from torch.nn import MSELoss
from torch.nn.modules.loss import _Loss


class CFRLoss(_Loss):
    """
    Counterfactual regression loss as proposed by Shalit et al. (2017)

    Arguments
    --------------
    alpha: float, regularization hyperparameter for integral probability metric (IPM), default to 1e-3
    """

    ipm_metric = {
        "W1": SamplesLoss(loss="sinkhorn", p=1, backend="tensorized"),
        "W2": SamplesLoss(loss="sinkhorn", p=2, backend="tensorized"),
        "MMD": SamplesLoss(loss="energy", backend="tensorized"),
    }

    def __init__(
        self,
        size_average: bool = None,
        reduce: bool = None,
        reduction: str = "mean",
        alpha: float = 1e-3,
        metric: str = "W2",
    ) -> None:
        assert (
            metric in self.ipm_metric.keys()
        ), "The metric must be one of the following: {}".format(self.ipm_metric.keys())
        self.alpha = alpha
        self.metric = metric
        super(CFRLoss, self).__init__(size_average, reduce, reduction)

    def forward(
        self,
        prediction1: torch.Tensor,
        prediction0: torch.Tensor,
        target1: torch.Tensor,
        target0: torch.Tensor,
        Treatment: torch.Tensor,
        phi_output: torch.Tensor,
    ) -> torch.Tensor:
        # Treatment = torch.tensor(Treatment, dtype=torch.float32)
        Treatment = Treatment.clone().detach()
        Treatment = Treatment.float()
        w1 = 1.0 / (2 * torch.mean(Treatment))
        w0 = 1.0 / (2 * (1 - torch.mean(Treatment)))
        mse = MSELoss()
        phi0, phi1 = phi_output[Treatment == 0], phi_output[Treatment == 1]
        factual_err = w0 * mse(prediction0[Treatment == 0], target0) + w1 * mse(
            prediction1[Treatment == 1], target1
        )

        imbalance_term = (
            self.ipm_metric[self.metric](phi0, phi1)
            if len(phi0) > 0 and len(phi1) > 0
            else torch.tensor(0.0)
        )
        return factual_err + self.alpha * imbalance_term