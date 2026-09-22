"""
Ablation: Soft NC2 regularization (Table 7).

Adds lambda_nc2 * L_geo to the base-session objective, where L_geo penalizes
deviation of the base classifier weights from the ETF Gram structure.
Classifier weights are overwritten by class-mean prototypes at each session
boundary, so this constraint has no lasting effect across sessions.

Config keys:
    lambda_nc2 (float, default 0.1): weight of the NC2 loss
"""

import torch.nn.functional as F
from models.sec import Learner as SecLearner
from utils.neural_collapse import NeuralCollapseMetrics


class Learner(SecLearner):

    def _fc_weight(self):
        import torch.nn as nn
        net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        return net.fc.weight

    def _nc2_loss(self):
        fc_w = self._fc_weight()[:self._total_classes]
        return NeuralCollapseMetrics.compute_nc2_loss(fc_w)

    def _init_train(self, train_loader, test_loader, train_loader_for_protonet, optimizer, scheduler):
        import torch.nn.functional as F

        lambda_nc2 = float(self.args.get("lambda_nc2", 0.1))
        n_epochs = self.args["fs_epoch"] if (
            isinstance(self.args["kshot"], int) and self._known_classes > 0
        ) else self.args["tuned_epoch"]

        for epoch in range(n_epochs):
            self._network.train()
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                if self._cur_task == 0:
                    out = self._network(inputs, targets=targets)
                    logits = out["logits"]
                    loss = (F.cross_entropy(logits, targets)
                            + out["loss_match"] * self.args["beta"]
                            + lambda_nc2 * self._nc2_loss())
                else:
                    out = self._network(inputs, train=True, targets=targets)
                    logits = out["logits"]
                    targets = targets.repeat(int(logits.shape[0] / targets.shape[0]))
                    loss = F.cross_entropy(logits, targets) + out["loss_match"] * self.args["beta"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
