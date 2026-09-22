# watermark version

import logging
import os
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from backbone.vpt_backbone import SimpleVitNet
from models.base import BaseLearner
from utils.neural_collapse import NeuralCollapseMetrics

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 1


def cos_loss(cosine, label):
    loss = 0
    for i, y in enumerate(label):
        loss += 1 - cosine[i, y]
    return loss / len(label)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.fs_lr = args['fs_lr']
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args
        self.midfeature = None

    def _loader_workers(self):
        return 0 if self._device.type == "cpu" else num_workers

    def _use_final_replace_fc(self):
        return bool(self.args.get("final_replace_fc", True))

    def _final_proto_blend(self):
        blend = float(self.args.get("final_proto_blend", 1.0))
        return max(0.0, min(1.0, blend))

    def after_task(self):
        self._known_classes = self._total_classes

    def replace_fc(self, trainloader, model, args):
        # use class prototype as classifier weights.
        model = model.eval()
        embedding_list = []
        label_list = []
        if self._cur_task > 0:
            with torch.no_grad():
                for i, batch in enumerate(trainloader):
                    (_, data, label) = batch
                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model(data)['features']
                    embedding_list.append(embedding.cpu())
                    label = label.repeat(int(embedding.shape[0] / label.shape[0]))
                    label_list.append(label.cpu())
            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)
        else:
            with torch.no_grad():
                for i, batch in enumerate(trainloader):
                    (_, data, label) = batch
                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model(data)['features']
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())
            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto

        return model


    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # print(self._total_classes)
        self._network.update_fc(self._total_classes)
        self._network.backbone.TSP.process_task_count(self._total_classes)
        self._network.backbone.RSP.process_task_count()
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        if isinstance(self.args['kshot'], int) and self._known_classes > 0:
            train_bs = self.args['fs_batch_size']
        else:
            train_bs = self.batch_size
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train", kshot=self.args["kshot"])
        self.train_dataset = train_dataset

        loader_workers = self._loader_workers()
        self.train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=loader_workers)

        self.data_manager = data_manager

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=loader_workers)
        test_curr_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="test",
                                                     mode="test")
        self.test_curr_loader = DataLoader(test_curr_dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=loader_workers)
        # Keep base-class test loader for forgetting analysis with a fixed class set.
        base_indices = np.arange(0, self.args["init_cls"])
        test_base_dataset = data_manager.get_dataset(base_indices, source="test", mode="test")
        self.test_base_loader = DataLoader(
            test_base_dataset, batch_size=self.batch_size, shuffle=False, num_workers=loader_workers
        )
        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                              source="train", mode="test", kshot=self.args["kshot"])

        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=int(self.batch_size / 2),
                                                    shuffle=False, num_workers=loader_workers)

        logging.info("training set size: {}, fc construct set size: {}".format(len(train_dataset),
                                                                               len(train_dataset_for_protonet)))

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _get_fc_weight_tensor(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.fc.weight.data
        return self._network.fc.weight.data

    def _finalize_classifier(self, train_loader_for_protonet):
        if self._cur_task == 0:
            self.replace_fc(train_loader_for_protonet, self._network, None)
            return

        if not self._use_final_replace_fc():
            logging.info("[FinalReplaceFC] skipped for incremental session %d", self._cur_task)
            return

        learned_new = None
        if self._known_classes < self._total_classes:
            learned_new = self._get_fc_weight_tensor()[
                self._known_classes : self._total_classes
            ].clone()

        self.replace_fc(train_loader_for_protonet, self._network, None)

        if learned_new is None:
            return

        blend = self._final_proto_blend()
        if blend >= 1.0:
            return

        proto_new = self._get_fc_weight_tensor()[
            self._known_classes : self._total_classes
        ].clone()
        mixed = (1.0 - blend) * learned_new + blend * proto_new
        self._get_fc_weight_tensor()[
            self._known_classes : self._total_classes
        ] = mixed
        logging.info(
            "[FinalReplaceFC] blended learned/proto new-class weights with alpha=%.3f",
            blend,
        )

    def _train(self, train_loader, test_loader, train_loader_for_protonet):

        self._network.to(self._device)
        if self._cur_task > 0:
            self._network.backbone.Freeze_new()
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info('total parameters: {}'.format(total_params))
        total_trainable_params = sum(
            p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info('trainable parameters: {}'.format(total_trainable_params))

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())

        if os.path.exists(self.args["base_model_path"]) and self._cur_task == 0:
            logging.info(
                '================= load base model from: {} ================='.format(self.args["base_model_path"]))
            print(self._cur_task)
            self._network.load_state_dict(torch.load(self.args["base_model_path"], map_location='cpu'))
            # self.replace_midfeature(train_loader_for_protonet, self._network, None)
            # self.replace_fc(train_loader_for_protonet, self._network, None)

        else:
            if self._cur_task > 0:
                self.replace_fc(train_loader_for_protonet, self._network, None)

            if self._cur_task == 0:
                if self.args['optimizer'] == 'sgd':
                    optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,
                                          weight_decay=self.weight_decay)
                elif self.args['optimizer'] == 'adam':
                    optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'],
                                                                 eta_min=self.min_lr)
                self._init_train(train_loader, test_loader, train_loader_for_protonet, optimizer, scheduler)
            else:
                if self.args['optimizer'] == 'sgd':
                    optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.fs_lr,
                                          weight_decay=self.weight_decay)
                elif self.args['optimizer'] == 'adam':
                    optimizer = optim.AdamW(self._network.parameters(), lr=self.fs_lr, weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['fs_epoch'],
                                                                 eta_min=self.min_lr)
                self._init_train(train_loader, test_loader, train_loader_for_protonet, optimizer, scheduler)
            self._finalize_classifier(train_loader_for_protonet)
            # self.replace_midfeature(train_loader_for_protonet, self._network, None)
            if self._cur_task == 0:
                torch.save(self._network.state_dict(), self.args["base_model_path"])

    def eval_task(self):
        y_pred, y_true, all_embedding, all_labels = self._eval_acc(self.test_loader, return_embeddings=True)
        accy = self._evaluate(y_pred, y_true)
        base_embedding, base_labels = None, None

        if hasattr(self, "test_base_loader"):
            base_pred, base_true, base_embedding, base_labels = self._eval_acc(
                self.test_base_loader, return_embeddings=True
            )
            base_top1 = np.around((base_pred.T[0] == base_true).sum() * 100 / len(base_true), decimals=2)
            accy["base_top1"] = float(base_top1)

        if self.args.get("measure_nc", True):
            classifier_weights = self._get_classifier_weights().cpu()

            # test set NC (전 세션 — collapse vs forgetting 상관관계용)
            nc = NeuralCollapseMetrics.measure_all(
                features=all_embedding,
                labels=all_labels,
                classifier_weights=classifier_weights,
                session=self._cur_task,
            )
            accy["nc"] = nc
            logging.info(
                "[Session %d] NC1=%.4f | NC2=%.4f | NC3=%.4f | NC4=%.4f",
                self._cur_task,
                nc["NC1"],
                nc["NC2"],
                nc["NC3"],
                nc["NC4"],
            )

            # incremental session에서 NC2를 base/new 관점으로 분해
            if self._cur_task > 0:
                nc2_decomp = NeuralCollapseMetrics.decompose_nc2(
                    classifier_weights, self.args["init_cls"]
                )
                accy["nc2_decomp"] = nc2_decomp
                logging.info(
                    "[Session %d | NC2-DECOMP] bb=%.4f | nn=%.4f | bn=%.4f | new_standalone=%.4f",
                    self._cur_task,
                    nc2_decomp.get("nc2_bb", float("nan")),
                    nc2_decomp.get("nc2_nn", float("nan")),
                    nc2_decomp.get("nc2_bn", float("nan")),
                    nc2_decomp.get("nc2_new_standalone", float("nan")),
                )

            if base_embedding is not None and base_labels is not None:
                base_classifier_weights = classifier_weights[: self.args["init_cls"]]
                nc_base = NeuralCollapseMetrics.measure_all(
                    features=base_embedding,
                    labels=base_labels,
                    classifier_weights=base_classifier_weights,
                    session=self._cur_task,
                )
                accy["nc_base"] = nc_base
                logging.info(
                    "[Session %d | BASE-TEST] NC1=%.4f | NC2=%.4f | NC3=%.4f | NC4=%.4f",
                    self._cur_task,
                    nc_base["NC1"],
                    nc_base["NC2"],
                    nc_base["NC3"],
                    nc_base["NC4"],
                )

            # base session에서만 train set NC 추가 측정 (NC 형성 증명용)
            if self._cur_task == 0:
                train_full_dataset = self.data_manager.get_dataset(
                    np.arange(0, self._total_classes), source="train", mode="test"
                )
                train_full_loader = DataLoader(
                    train_full_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self._loader_workers()
                )
                _, _, train_embedding, train_labels = self._eval_acc(train_full_loader, return_embeddings=True)
                nc_train = NeuralCollapseMetrics.measure_all(
                    features=train_embedding,
                    labels=train_labels,
                    classifier_weights=classifier_weights,
                    session=self._cur_task,
                )
                accy["nc_train"] = nc_train
                logging.info(
                    "[Session %d | TRAIN] NC1=%.4f | NC2=%.4f | NC3=%.4f | NC4=%.4f",
                    self._cur_task,
                    nc_train["NC1"],
                    nc_train["NC2"],
                    nc_train["NC3"],
                    nc_train["NC4"],
                )

        return accy

    def _get_classifier_weights(self):
        return self._get_fc_weight_tensor()[: self._total_classes].detach()

    def _eval_acc(self, loader, return_embeddings=False):
        self._network.eval()
        y_pred, y_true = [], []
        all_embedding, all_labels = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                out = self._network(inputs)
                outputs = out["logits"]
                embedding = out["features"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            if return_embeddings:
                all_embedding.append(embedding.cpu())
                all_labels.append(targets.cpu())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)

        if return_embeddings:
            all_embedding = torch.cat(all_embedding, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            return y_pred, y_true, all_embedding, all_labels

        return y_pred, y_true  # [N, topk]

    # naive train
    def _init_train(self, train_loader, test_loader, train_loader_for_protonet, optimizer, scheduler):
        if isinstance(self.args['kshot'], int) and self._known_classes > 0:
            total_epoch = self.args['fs_epoch']
        else:
            total_epoch = self.args['tuned_epoch']
        for _, epoch in enumerate(range(total_epoch)):
            self._network.train()
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # print(inputs.shape)
                if self._cur_task == 0:

                    out = self._network(inputs, targets=targets)
                    logits = out["logits"]
                    loss_pc = out['loss_match']
                    loss = F.cross_entropy(logits, targets) + loss_pc * self.args["beta"]

                else:

                    out = self._network(inputs, train=True, targets=targets)
                    logits = out["logits"]
                    loss_pc = out['loss_match']
                    targets = targets.repeat(int(logits.shape[0] / targets.shape[0]))
                    loss = F.cross_entropy(logits, targets) + loss_pc * self.args["beta"]
                    # print(loss_pc)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

            scheduler.step()

            # Probe checkpoint saving — no forward pass, no RNG effect
            if self._cur_task == 0:
                probe_epochs = set(self.args.get("probe_ckpt_epochs", []))
                if (epoch + 1) in probe_epochs:
                    probe_path = self.args["base_model_path"].replace(
                        ".pth", f"_probe_ep{epoch+1:02d}.pth"
                    )
                    net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
                    torch.save(net.state_dict(), probe_path)
                    logging.info("[PROBE] Saved epoch %d checkpoint → %s", epoch + 1, probe_path)

            # Epoch-wise NC2 logging for S0 bifurcation analysis
            if self._cur_task == 0 and self.args.get("log_epoch_nc", False):
                self._network.eval()
                feats, labs = [], []
                with torch.no_grad():
                    for _, inputs, targets in train_loader:
                        f = self._network.extract_vector(inputs.to(self._device))
                        feats.append(f.cpu())
                        labs.append(targets)
                feats = torch.cat(feats)
                labs = torch.cat(labs)
                cw = self._get_classifier_weights().cpu()
                nc_ep = NeuralCollapseMetrics.measure_all(
                    features=feats, labels=labs,
                    classifier_weights=cw, session=0,
                )
                logging.info(
                    "[S0 Epoch %d] NC1=%.4f | NC2=%.4f | NC3=%.4f | NC4=%.4f",
                    epoch + 1, nc_ep["NC1"], nc_ep["NC2"], nc_ep["NC3"], nc_ep["NC4"],
                )
                # Save per-epoch checkpoint for geometry-aware selection
                if self.args.get("save_epoch_ckpt", False):
                    epoch_ckpt_path = self.args["base_model_path"].replace(
                        ".pth", f"_ep{epoch+1:02d}.pth"
                    )
                    torch.save(self._network.state_dict(), epoch_ckpt_path)
                self._network.train()
