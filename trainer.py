import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import random
import numpy as np
import json
import csv
from utils.neural_collapse import pearson_corr


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = args["init_cls"]
    if "measure_nc" not in args:
        args["measure_nc"] = True
    logs_name = "logs/{}/{}/{}/{}_{}/{}".format("sec_tr",args["dataset"],args['tuned_epoch'], args['init_lr'], args["kshot"], args["beta"])
    saved_path = "saved_model/{}/{}/{}_{}/{}_{}".format("sec_tr", args["dataset"], args['tuned_epoch'], args['init_lr'], args["prompt_token_num"],args["prompt_pool_num"])

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    logfilename = "logs/{}/{}/{}/{}_{}/{}/{}_{}_{}".format(
        "sec_tr",
        args["dataset"],
        args['tuned_epoch'],
        args["init_lr"],
        args["kshot"],
        args["beta"],
        args["prompt_token_num"],
        args["prompt_pool_num"],
        args.get("prefix", "default")
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    args["base_model_path"] = "saved_model/{}/{}/{}_{}/{}_{}/{}_{}_{}_{}.pth".format(
        "sec_tr",
        args["dataset"],
        args['tuned_epoch'],
        args["init_lr"],
        args["prompt_token_num"],
        args["prompt_pool_num"],
        args["model_prefix"],
        args["tuned_epoch"],
        args["seed"],
        args["batch_size"],
    )


    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )

    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    top1_curve = {"top1": [], "top5": []}
    base_top1_ref = None
    delta_curve_all, forgetting_curve_all = [], []
    delta_curve_base, forgetting_curve_base = [], []
    session_metrics = []
    metrics_json_path = logfilename + "_session_metrics.json"
    metrics_csv_path = logfilename + "_session_metrics.csv"

    max_tasks = args.get("max_tasks", data_manager.nb_tasks)
    for task in range(min(max_tasks, data_manager.nb_tasks)):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )

        model.incremental_train(data_manager)

        top1_accy = model.eval_task()
        model.after_task()

        top1_curve["top1"].append(top1_accy["top1"])

        logging.info("Top1 curve: {}".format(top1_curve["top1"]))

        Hacc, old_acc, new_acc = Harmonic_Accuracy(top1_accy["grouped"], args["init_cls"])

        base_top1 = top1_accy.get("base_top1", None)
        if base_top1_ref is None and base_top1 is not None:
            base_top1_ref = float(base_top1)
        forgetting_base = None
        if task > 0 and base_top1_ref is not None and base_top1 is not None:
            forgetting_base = float(base_top1_ref - float(base_top1))

        nc = top1_accy.get("nc", {})
        nc_base = top1_accy.get("nc_base", {})
        nc_train = top1_accy.get("nc_train", {})
        nc2_decomp = top1_accy.get("nc2_decomp", {})
        nc4_dualhead = top1_accy.get("nc4_dualhead", {})
        nc1 = nc.get("NC1")
        nc2 = nc.get("NC2")
        nc3 = nc.get("NC3")
        nc4 = nc.get("NC4")
        base_nc1 = nc_base.get("NC1")
        base_nc2 = nc_base.get("NC2")
        base_nc3 = nc_base.get("NC3")
        base_nc4 = nc_base.get("NC4")

        if task > 0 and nc2 is not None and forgetting_base is not None:
            delta_curve_all.append(float(nc2))
            forgetting_curve_all.append(float(forgetting_base))
        if task > 0 and base_nc2 is not None and forgetting_base is not None:
            delta_curve_base.append(float(base_nc2))
            forgetting_curve_base.append(float(forgetting_base))

        all_delta_forgetting_corr = pearson_corr(delta_curve_all, forgetting_curve_all)
        base_delta_forgetting_corr = pearson_corr(delta_curve_base, forgetting_curve_base)
        n_new_shot = args["kshot"] if task > 0 and isinstance(args["kshot"], int) else None

        row = {
            "session": int(task),
            "num_classes": int(model._total_classes),
            "top1": float(top1_accy["top1"]),
            "avg_top1_so_far": float(sum(top1_curve["top1"]) / len(top1_curve["top1"])),
            "harmonic_acc": float(Hacc) if Hacc is not None else None,
            "old_acc": float(old_acc),
            "new_acc": float(new_acc) if (new_acc is not None and not isinstance(new_acc, list)) else None,
            "base_top1": float(base_top1) if base_top1 is not None else None,
            "base_forgetting": forgetting_base,
            "NC1": float(nc1) if nc1 is not None else None,
            "NC2": float(nc2) if nc2 is not None else None,
            "NC3": float(nc3) if nc3 is not None else None,
            "NC4": float(nc4) if nc4 is not None else None,
            "base_NC1": float(base_nc1) if base_nc1 is not None else None,
            "base_NC2": float(base_nc2) if base_nc2 is not None else None,
            "base_NC3": float(base_nc3) if base_nc3 is not None else None,
            "base_NC4": float(base_nc4) if base_nc4 is not None else None,
            "train_NC1": float(nc_train.get("NC1")) if nc_train.get("NC1") is not None else None,
            "train_NC2": float(nc_train.get("NC2")) if nc_train.get("NC2") is not None else None,
            "train_NC3": float(nc_train.get("NC3")) if nc_train.get("NC3") is not None else None,
            "train_NC4": float(nc_train.get("NC4")) if nc_train.get("NC4") is not None else None,
            "nc2_bb": float(nc2_decomp["nc2_bb"]) if nc2_decomp.get("nc2_bb") is not None else None,
            "nc2_nn": float(nc2_decomp["nc2_nn"]) if nc2_decomp.get("nc2_nn") is not None else None,
            "nc2_bn": float(nc2_decomp["nc2_bn"]) if nc2_decomp.get("nc2_bn") is not None else None,
            "nc2_new_standalone": float(nc2_decomp["nc2_new_standalone"]) if nc2_decomp.get("nc2_new_standalone") is not None else None,
            "base_delta_forgetting_corr": base_delta_forgetting_corr,
            "all_delta_forgetting_corr": all_delta_forgetting_corr,
            "base_corr_points": len(delta_curve_base),
            "all_corr_points": len(delta_curve_all),
            "n_new_shot": n_new_shot,
            "nc4_fusion_alpha": float(nc4_dualhead["fusion_alpha"]) if nc4_dualhead.get("fusion_alpha") is not None else None,
            "nc4_cls_acc": float(nc4_dualhead["cls_acc"]) if nc4_dualhead.get("cls_acc") is not None else None,
            "nc4_proto_acc": float(nc4_dualhead["proto_acc"]) if nc4_dualhead.get("proto_acc") is not None else None,
            "nc4_fusion_acc": float(nc4_dualhead["fusion_acc"]) if nc4_dualhead.get("fusion_acc") is not None else None,
            "nc4_avg_cls_confidence": float(nc4_dualhead["avg_cls_confidence"]) if nc4_dualhead.get("avg_cls_confidence") is not None else None,
            "nc4_avg_proto_confidence": float(nc4_dualhead["avg_proto_confidence"]) if nc4_dualhead.get("avg_proto_confidence") is not None else None,
            "nc4_avg_fusion_confidence": float(nc4_dualhead["avg_fusion_confidence"]) if nc4_dualhead.get("avg_fusion_confidence") is not None else None,
            "nc4_avg_top1_top2_margin_cls": float(nc4_dualhead["avg_top1_top2_margin_cls"]) if nc4_dualhead.get("avg_top1_top2_margin_cls") is not None else None,
            "nc4_avg_top1_top2_margin_proto": float(nc4_dualhead["avg_top1_top2_margin_proto"]) if nc4_dualhead.get("avg_top1_top2_margin_proto") is not None else None,
            "nc4_avg_top1_top2_margin_fusion": float(nc4_dualhead["avg_top1_top2_margin_fusion"]) if nc4_dualhead.get("avg_top1_top2_margin_fusion") is not None else None,
        }
        session_metrics.append(row)
        _write_session_metrics(metrics_json_path, metrics_csv_path, session_metrics)

        if nc:
            logging.info(
                "[Session %d] BaseTop1=%.2f | Forgetting=%.2f | NC2(all)=%.4f | NC2(base)=%.4f | NC4(all)=%.4f | NC4(base)=%.4f | Corr(base_delta,forget)=%s | Corr(all_delta,forget)=%s",
                task,
                float(base_top1) if base_top1 is not None else float("nan"),
                float(forgetting_base) if forgetting_base is not None else float("nan"),
                float(nc2),
                float(base_nc2) if base_nc2 is not None else float("nan"),
                float(nc4),
                float(base_nc4) if base_nc4 is not None else float("nan"),
                "None" if base_delta_forgetting_corr is None else "{:.4f}".format(base_delta_forgetting_corr),
                "None" if all_delta_forgetting_corr is None else "{:.4f}".format(all_delta_forgetting_corr),
            )

        logging.info("Average Accuracy (Top1): {}   (Harmonic Accuracy): {} (Old Acc): {} (New Acc): {} \n".format(sum(top1_curve["top1"])/len(top1_curve["top1"]),
                                                                            Hacc, old_acc, new_acc))
    relation_summary_base = fit_forgetting_relation(session_metrics, delta_key="base_NC2")
    relation_summary_all = fit_forgetting_relation(session_metrics, delta_key="NC2")
    if relation_summary_base is not None:
        _log_relation_summary("base_NC2", relation_summary_base)
    if relation_summary_all is not None:
        _log_relation_summary("all_NC2", relation_summary_all)
    _write_session_metrics(
        metrics_json_path,
        metrics_csv_path,
        session_metrics,
        relation_summary_base,
        relation_summary_all,
    )

    logging.info("\n")


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1 or str(device).lower() == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def _write_session_metrics(json_path, csv_path, session_metrics, relation_summary=None, relation_summary_all=None):
    payload = {
        "sessions": session_metrics,
        "relation_summary_base": relation_summary,
        "relation_summary_all": relation_summary_all,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    fieldnames = [
        "session",
        "num_classes",
        "top1",
        "avg_top1_so_far",
        "harmonic_acc",
        "old_acc",
        "new_acc",
        "base_top1",
        "base_forgetting",
        "NC1",
        "NC2",
        "NC3",
        "NC4",
        "base_NC1",
        "base_NC2",
        "base_NC3",
        "base_NC4",
        "train_NC1",
        "train_NC2",
        "train_NC3",
        "train_NC4",
        "nc2_bb",
        "nc2_nn",
        "nc2_bn",
        "nc2_new_standalone",
        "base_delta_forgetting_corr",
        "all_delta_forgetting_corr",
        "base_corr_points",
        "all_corr_points",
        "n_new_shot",
        "nc4_fusion_alpha",
        "nc4_cls_acc",
        "nc4_proto_acc",
        "nc4_fusion_acc",
        "nc4_avg_cls_confidence",
        "nc4_avg_proto_confidence",
        "nc4_avg_fusion_confidence",
        "nc4_avg_top1_top2_margin_cls",
        "nc4_avg_top1_top2_margin_proto",
        "nc4_avg_top1_top2_margin_fusion",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in session_metrics:
            writer.writerow(row)


def fit_forgetting_relation(session_metrics, delta_key="base_NC2"):
    xs_delta, xs_shot, ys_forgetting = [], [], []
    for row in session_metrics:
        delta = row.get(delta_key)
        forget = row.get("base_forgetting")
        n_new_shot = row.get("n_new_shot")
        if delta is None or forget is None:
            continue
        if n_new_shot is None or n_new_shot <= 0:
            continue
        xs_delta.append(float(delta))
        xs_shot.append(float(1.0 / np.sqrt(n_new_shot)))
        ys_forgetting.append(float(forget))

    if len(ys_forgetting) < 2:
        return None

    x = np.array(xs_delta, dtype=np.float64)
    shot = np.array(xs_shot, dtype=np.float64)
    y = np.array(ys_forgetting, dtype=np.float64)
    if np.std(shot) < 1e-8:
        x_mat = np.stack([x, np.ones_like(x)], axis=1)
        theta, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
        slope, intercept = float(theta[0]), float(theta[1])
        pred = x_mat @ theta
        logging.warning(
            "fit_forgetting_relation(%s): n_shot term is constant (1/sqrt(n_shot)=%.6f). "
            "Reporting a univariate linear fit with intercept instead of a shot-effect coefficient.",
            delta_key,
            float(shot[0]),
        )
        return {
            "delta_key": delta_key,
            "model_type": "univariate_with_intercept",
            "slope": slope,
            "intercept": intercept,
            "shot_term_constant": True,
            "shot_constant_value": float(shot[0]),
            "pearson_corr": float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else 0.0,
            "max_abs_error": float(np.max(np.abs(y - pred))),
            "mae": float(np.mean(np.abs(y - pred))),
            "num_points": int(len(y)),
        }

    x_mat = np.stack([x, shot], axis=1)
    theta, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    slope, shot_coefficient = float(theta[0]), float(theta[1])
    pred = x_mat @ theta
    return {
        "delta_key": delta_key,
        "model_type": "two_term_no_intercept",
        "slope": slope,
        "shot_coefficient": shot_coefficient,
        "shot_term_constant": False,
        "pearson_corr": float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else 0.0,
        "max_abs_error": float(np.max(np.abs(y - pred))),
        "mae": float(np.mean(np.abs(y - pred))),
        "num_points": int(len(y)),
    }


def _log_relation_summary(name, summary):
    if summary["model_type"] == "univariate_with_intercept":
        logging.info(
            "Linear fit (%s): slope=%.6f intercept=%.6f corr=%.4f max_abs_error=%.4f mae=%.4f points=%d",
            name,
            summary["slope"],
            summary["intercept"],
            summary["pearson_corr"],
            summary["max_abs_error"],
            summary["mae"],
            summary["num_points"],
        )
    else:
        logging.info(
            "Linear fit (%s): slope=%.6f shot_coef=%.6f corr=%.4f max_abs_error=%.4f mae=%.4f points=%d",
            name,
            summary["slope"],
            summary["shot_coefficient"],
            summary["pearson_corr"],
            summary["max_abs_error"],
            summary["mae"],
            summary["num_points"],
        )

def Harmonic_Accuracy(grouped_acc, init_cls):
    old_acc, new_acc = [], []
    for key in grouped_acc.keys():
        if '-' in key:
            if int(key.split('-')[1]) <= init_cls:
                old_acc.append(grouped_acc[key])
            elif int(key.split('-')[1]) > init_cls:
                new_acc.append(grouped_acc[key])
    old_acc = sum(old_acc) / len(old_acc)

    if len(new_acc) > 0:
        new_acc = sum(new_acc) / len(new_acc)
        Hacc = 2 * old_acc * new_acc / (old_acc + new_acc)
    else:
        Hacc = None
    return Hacc, old_acc, new_acc
