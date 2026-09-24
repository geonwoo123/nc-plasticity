"""
Multi-seed evaluation entry point for the paired protocol (Table 4-6).

Run a configuration across multiple random seeds and report per-seed and
mean accuracy. Designed for the 15-seed protocol used in the paper.

Usage:
  python run_multiseed.py --config configs/cub/nc_plasticity.json
"""

import os
import sys
import argparse
import json
import copy

import torch
from trainer import _set_device, print_args, Harmonic_Accuracy
from utils.data_manager import DataManager
from utils import factory
import logging
import random
import numpy as np


def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cli = parser.parse_args()

    with open(cli.config) as f:
        args = json.load(f)

    logfilename = f"logs/repeat_{args.get('prefix', 'run')}"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("=== MULTI-SEED EVALUATION ===")

    seed_list = copy.deepcopy(args["seed"]) if isinstance(args["seed"], list) else [args["seed"]]
    device = copy.deepcopy(args["device"])
    seed_avg_accs = []

    for run_idx, seed in enumerate(seed_list):
        logging.info(f"=== Run {run_idx + 1}/{len(seed_list)} | Seed {seed} ===")
        args["seed"] = seed
        args["device"] = device

        set_random(seed)
        _set_device(args)

        if "nb_tasks" not in args:
            args["nb_tasks"] = 11

        saved_path = "saved_model/{}/{}/{}_{}/{}_{}".format(
            "sec_tr", args["dataset"], args["tuned_epoch"], args["init_lr"],
            args["prompt_token_num"], args["prompt_pool_num"],
        )
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        args["base_model_path"] = "saved_model/{}/{}/{}_{}/{}_{}/{}_{}_{}_{}.pth".format(
            "sec_tr", args["dataset"], args["tuned_epoch"], args["init_lr"],
            args["prompt_token_num"], args["prompt_pool_num"], args.get("prefix", "default"),
            args["tuned_epoch"], args["seed"], args["batch_size"],
        )

        print_args(args)

        data_manager = DataManager(
            args["dataset"],
            args["shuffle"],
            seed,
            args["init_cls"],
            args["increment"],
            args,
        )
        args["nb_classes"] = data_manager.nb_classes
        args["nb_tasks"] = data_manager.nb_tasks

        model = factory.get_model(args["model_name"], args)

        top1_curve = []
        for task in range(data_manager.nb_tasks):
            model.incremental_train(data_manager)
            accy = model.eval_task()
            model.after_task()

            top1 = accy["top1"]
            top1_curve.append(top1)

            Hacc, old_acc, new_acc = Harmonic_Accuracy(accy["grouped"], args["init_cls"])
            if new_acc is not None and not isinstance(new_acc, list):
                new_str = f" | New={new_acc:.2f} | Old={old_acc:.2f} | H={Hacc:.2f}"
            else:
                new_str = f" | Old={old_acc:.2f}" if not isinstance(old_acc, list) else ""
            logging.info(f"[Run {run_idx+1} | Seed {seed} | Task {task}] Top1={top1:.2f}{new_str} | Curve={[round(x,2) for x in top1_curve]}")

        avg_acc = sum(top1_curve) / len(top1_curve)
        logging.info(f"=== Run {run_idx+1} done | Seed {seed} | Avg ACC={avg_acc:.2f} | Curve={[round(x,2) for x in top1_curve]} ===")
        seed_avg_accs.append(avg_acc)

    mean_acc = float(np.mean(seed_avg_accs))
    std_acc = float(np.std(seed_avg_accs, ddof=1)) if len(seed_avg_accs) > 1 else 0.0
    logging.info(
        f"=== FINAL | {len(seed_avg_accs)}-seed AA = {mean_acc:.2f} ± {std_acc:.2f} "
        f"(sample std, ddof=1) | Per-seed={[round(x, 2) for x in seed_avg_accs]} ==="
    )


if __name__ == "__main__":
    main()
