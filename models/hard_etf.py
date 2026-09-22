"""
Ablation: Hard ETF assignment (Table 7).

Each session initializes new class weights to fixed ETF vertices and applies
a Gram-matrix loss to old class weights. A single ETF of size
nc2_etf_max_classes is built once and reused across all sessions so the
target subspace does not shift.

This intervention achieves geometric alignment (lower NC2) but severs the
connection between classifier weights and the actual feature distribution,
collapsing new-class accuracy to near zero.

Config keys:
    lambda_nc2          (float, default 0.1):   weight of the Gram loss
    nc2_etf_max_classes (int,   default 200):   total ETF size (built once)
    etf_seed            (int,   default 42):    random seed for ETF construction
    nc2_etf_expansion   (bool,  default True):  initialize new weights to ETF vertices
"""

import logging
import torch
import torch.nn.functional as F
from torch import optim

from models.sec import Learner as SecLearner
from utils.neural_collapse import NeuralCollapseMetrics


class Learner(SecLearner):

    def __init__(self, args):
        super().__init__(args)
        self._fixed_etf = None

    def _network_fc_weight(self):
        if hasattr(self._network, "module"):
            return self._network.module.fc.weight
        return self._network.fc.weight

    def _get_fixed_etf(self):
        if self._fixed_etf is None:
            max_cls = int(self.args.get("nc2_etf_max_classes", 200))
            d = self._network_fc_weight().shape[1]
            etf = NeuralCollapseMetrics.build_etf_vectors(
                max_cls, d, device=self._device,
                seed=int(self.args.get("etf_seed", 42)),
            )
            self._fixed_etf = F.normalize(etf, dim=1)
        return self._fixed_etf

    def _etf_expansion(self, etf):
        with torch.no_grad():
            self._network_fc_weight()[self._known_classes:self._total_classes] = \
                etf[self._known_classes:self._total_classes]

    def _compute_l_geo(self, etf):
        if self._known_classes < 2:
            return self._network_fc_weight().new_zeros(())
        w_old = F.normalize(self._network_fc_weight()[:self._known_classes], dim=1)
        v_old = etf[:self._known_classes].detach()
        return torch.norm(w_old @ w_old.T - v_old @ v_old.T, p="fro")

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        if self._cur_task == 0:
            super()._train(train_loader, test_loader, train_loader_for_protonet)
            return

        self._network.to(self._device)
        self._network.backbone.Freeze_new()

        self.replace_fc(train_loader_for_protonet, self._network, None)
        etf = self._get_fixed_etf()
        if self.args.get("nc2_etf_expansion", True):
            self._etf_expansion(etf)

        optimizer = optim.SGD(
            self._network.parameters(),
            momentum=0.9, lr=self.fs_lr, weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args["fs_epoch"], eta_min=self.min_lr
        )
        lambda_nc2 = float(self.args.get("lambda_nc2", 0.1))

        for epoch in range(self.args["fs_epoch"]):
            self._network.train()
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                out = self._network(inputs, train=True, targets=targets)
                logits = out["logits"]
                targets_exp = targets.repeat(int(logits.shape[0] / targets.shape[0]))
                loss = (F.cross_entropy(logits, targets_exp)
                        + self.args["beta"] * out["loss_match"]
                        + lambda_nc2 * self._compute_l_geo(etf))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        self._finalize_classifier(train_loader_for_protonet)
