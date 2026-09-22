"""
NC-Plasticity regularization (Ours).

Augments the SEC-Prompt base-session objective with:
    L_plastic = max(0, tau - V(B))

where V(B) is the sample-weighted within-class variability of L2-normalized
features in the mini-batch. This imposes a soft lower bound on intra-class
spread, preventing excessive Neural Collapse during base training.

Config keys:
    tau_var        (float, default 0.5):  variability floor tau
    lambda_plastic (float, default 1.0):  weight of L_plastic
"""

import logging
import torch
from torch.nn import functional as F

from models.sec import Learner as SecLearner


def _batch_within_var(features, labels):
    """Sample-weighted within-class variability V(B) on L2-normalized features."""
    feat_n = F.normalize(features, dim=1)
    classes = labels.unique()
    total_sq = feat_n.new_tensor(0.0)
    n = 0
    for c in classes:
        mask = labels == c
        if mask.sum() < 2:
            continue
        f_c = feat_n[mask]
        mu_c = f_c.mean(0, keepdim=True)
        total_sq = total_sq + ((f_c - mu_c) ** 2).sum()
        n += mask.sum().item()
    return total_sq / n if n > 0 else feat_n.new_tensor(0.0)


class Learner(SecLearner):

    def _init_train(self, train_loader, test_loader, train_loader_for_protonet, optimizer, scheduler):
        if self._cur_task > 0:
            return super()._init_train(
                train_loader, test_loader, train_loader_for_protonet, optimizer, scheduler
            )

        tau       = float(self.args.get("tau_var", 0.5))
        lam       = float(self.args.get("lambda_plastic", 1.0))
        n_epochs  = self.args["tuned_epoch"]

        logging.info("[NCPlasticity] tau=%.3f | lambda=%.3f", tau, lam)

        for epoch in range(n_epochs):
            self._network.train()
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                out      = self._network(inputs, targets=targets)
                logits   = out["logits"]
                features = out["features"]

                loss_ce      = F.cross_entropy(logits, targets)
                loss_match   = out["loss_match"]
                v_b          = _batch_within_var(features, targets)
                loss_plastic = torch.clamp(tau - v_b, min=0.0)

                loss = loss_ce + self.args["beta"] * loss_match + lam * loss_plastic

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
