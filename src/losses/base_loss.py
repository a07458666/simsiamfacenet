import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.bilinear_sampler import apply_disparity

class BaseLoss(nn.modules.Module):

    def __init__(self, num_losses, device):
        super(BaseLoss, self).__init__()
        self.device = device
        self.train = True

        # Record the weights.
        self.num_losses = num_losses
        self.alphas = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)

    @staticmethod
    def invariance_loss(
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """Computes mse loss given batch of projected features z1 from
        view 1 and
        projected features z2 from view 2.
        Args:
            z1 (.Tensor): NxD Tensor containing projected features from view 1.
            z2 (.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: invariance loss (mean squared error).
        """

        return F.mse_loss(
            z1,
            z2,
        )

    @staticmethod
    def variance_loss(
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """Computes variance loss given batch of projected features
        z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (.Tensor): NxD Tensor containing projected features from view 1.
            z2 (.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: variance regularization loss.
        """

        eps = 1e-4
        std_z1 = torch.sqrt(z1.var(dim=0) + eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + eps)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
        return std_loss

    @staticmethod
    def covariance_loss(
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """Computes covariance loss given batch of projected features
         z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (.Tensor): NxD Tensor containing projected features from view 1.
            z2 (.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: covariance regularization loss.
        """

        (
            N,
            D,
        ) = z1.size()

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (N - 1)
        cov_z2 = (z2.T @ z2) / (N - 1)

        diag = torch.eye(
            D,
            device=z1.device,
        )
        cov_loss = (
            cov_z1[~diag.bool()].pow_(2).sum() / D
            + cov_z2[~diag.bool()].pow_(2).sum() / D
        )
        return cov_loss

    @staticmethod
    def cosine_similarity_loss(p, z,):  # distance
        z = z.detach()  # stop gradient
        return -(F.cosine_similarity(p, z,).mean())

    @staticmethod
    def simsiam_vicreg_loss_func(z1: torch.Tensor, z2: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, sim_loss_weight: float = 1.0, var_loss_weight: float = 1.0, cov_loss_weight: float = 1e-2) -> torch.Tensor:
        sim_loss = (
            self.cosine_similarity_loss(
                p1,
                z2.detach(),
            )
            + cosine_similarity_loss(
                p2,
                z1.detach(),
            )
        ) / 2
        #     sim_loss = invariance_loss(z1, z2)
        var_loss = self.variance_loss(
            z1,
            z2,
        )
        cov_loss = self.covariance_loss(
            z1,
            z2,
        )

        loss = (
            sim_loss_weight * sim_loss
            + var_loss_weight * var_loss
            + cov_loss_weight * cov_loss
        )
        return loss


    def to_eval(self):
        self.train = False

    def to_train(self):
        self.train = True

    def forward(self, pred, target):
        """ 
        Args:
            pred    [z1, z2, p1, p2, ps1, ps2]
            target  y

        Return:
            (float): The loss
        """
        z1, z2, p1, p2, ps1, ps2 = pred
        crossEntropyLoss = torch.nn.CrossEntropyLoss()(p1, target)
        
#         sim_loss = (self.cosine_similarity_loss(ps1,z2.detach(),) + self.cosine_similarity_loss(ps2,z1.detach(),)) / 2
#         var_loss = self.variance_loss(z1,z2,)
#         cov_loss = self.covariance_loss(z1,z2,)
        
        ssl_loss = self.simsiam_vicreg_loss_func(z1, z2, ps1, ps2)

        loss = [crossEntropyLoss, ssl_loss]
        return loss
