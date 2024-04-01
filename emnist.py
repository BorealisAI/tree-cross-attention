# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2022, Tung Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the TNP (https://arxiv.org/abs/2201.12740) implementation
# from https://github.com/tung-nd/TNP-pytorch by Tung Nguyen
####################################################################################


import os
import os.path as osp
import argparse
import yaml
import torch
import time
from attrdict import AttrDict
from tqdm import tqdm

from data.image import img_to_task
from data.emnist import EMNIST
from utils.misc import load_module
from utils.paths import results_path, evalsets_path
from utils.log import get_logger, RunningAverage


def main():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--expid", type=str, default="default")
    parser.add_argument("--resume", action="store_true", default=False)

    # Data
    parser.add_argument("--max_num_points", type=int, default=200)
    parser.add_argument("--class_range", type=int, nargs="*", default=[0, 10])

    # Train
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=100)
    parser.add_argument("--train_num_samples", type=int, default=4)
    parser.add_argument("--train_num_bs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=10)

    # Eval
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--eval_num_bs", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_num_samples", type=int, default=50)
    parser.add_argument("--eval_logfile", type=str, default=None)

    # ReTreever Arguments
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--emb_depth", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="quadratic",
        choices=["quadratic", "constant"],
    )
    parser.add_argument(
        "--decoder_type", type=str, default="tca", choices=["tca", "ca"]
    )
    parser.add_argument(
        "--aggregator_type",
        type=str,
        default="transformer",
        choices=["transformer"],
    )
    parser.add_argument("--bptt", action="store_true")
    parser.add_argument("--branch_factor", type=int, default=2)
    parser.add_argument("--ca_loss_weight", type=float, default=1.0)
    parser.add_argument("--rl_loss_weight", type=float, default=1.0)
    parser.add_argument("--entropy_bonus_weight", type=float, default=0.01)
    parser.add_argument("--num_aggregation_layers", type=int, default=1)

    parser.add_argument(
        "--heuristic",
        type=str,
        default="sort_x2",
        choices=["none", "random_proj", "sort_x1", "sort_x2"],
    )

    args = parser.parse_args()

    if args.expid is not None:
        args.root = osp.join(results_path, "emnist", args.expid)
    else:
        args.root = osp.join(results_path, "emnist")

    model_cls = getattr(load_module(f"models/retreever.py"), "Retreever")
    with open(f"configs/emnist.yaml", "r") as f:
        config = yaml.safe_load(f)

    args.save_path = args.root + "exp_logs.pkl.gz"
    for key, val in vars(args).items():
        if key in config:
            config[key] = val
            print(f"Overriding argument {key}: {config[key]}")

    model = model_cls(**config)
    model.cuda()

    if args.mode == "train":
        train(args, model)
    elif args.mode == "eval":
        eval(args, model)


def train(args, model):
    if osp.exists(args.root + "/ckpt.tar"):
        if args.resume is None:
            raise FileExistsError(args.root)
    else:
        os.makedirs(args.root, exist_ok=True)

    with open(osp.join(args.root, "args.yaml"), "w") as f:
        yaml.dump(args.__dict__, f)

    train_ds = EMNIST(train=True, class_range=args.class_range)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=0
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.num_epochs
    )

    if args.resume:
        ckpt = torch.load(osp.join(args.root, "ckpt.tar"))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_epoch = ckpt.epoch
    else:
        logfilename = osp.join(
            args.root, "train_{}.log".format(time.strftime("%Y%m%d-%H%M"))
        )
        start_epoch = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info(
            "Total number of parameters: {}\n".format(
                sum(p.numel() for p in model.parameters())
            )
        )

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        for x, _ in tqdm(train_loader, ascii=True):
            x = x.cuda()
            batch = img_to_task(x, max_num_points=args.max_num_points)
            optimizer.zero_grad()

            outs = model(batch)

            outs.loss.backward()
            optimizer.step()
            scheduler.step()
            model.reset()

            for key, val in outs.items():
                ravg.update(key, val)

        line = f"Retreever:{args.expid} step {epoch} "
        line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
        line += f"[train_loss] "
        line += ravg.info()
        logger.info(line)

        if epoch % args.eval_freq == 0:
            logger.info(eval(args, model) + "\n")

        ravg.reset()

        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.epoch = epoch + 1
            torch.save(ckpt, osp.join(args.root, "ckpt.tar"))

    args.mode = "eval"
    eval(args, model)


def gen_evalset(args):

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    eval_ds = EMNIST(train=False, class_range=args.class_range)
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0
    )

    batches = []
    for x, _ in tqdm(eval_loader, ascii=True):
        batches.append(img_to_task(x, max_num_points=args.max_num_points))

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path = osp.join(evalsets_path, "emnist")
    if not osp.isdir(path):
        os.makedirs(path)

    c1, c2 = args.class_range
    filename = f"{c1}-{c2}"
    filename += ".tar"

    torch.save(batches, osp.join(path, filename))


def eval(args, model):
    if args.mode == "eval":
        ckpt = torch.load(osp.join(args.root, "ckpt.tar"))
        model.load_state_dict(ckpt.model)
        if args.eval_logfile is None:
            c1, c2 = args.class_range
            eval_logfile = f"eval_{c1}-{c2}"
            eval_logfile += ".log"
        else:
            eval_logfile = args.eval_logfile
        filename = osp.join(args.root, eval_logfile)
        logger = get_logger(filename, mode="w")
    else:
        logger = None

    path = osp.join(evalsets_path, "emnist")
    c1, c2 = args.class_range
    filename = f"{c1}-{c2}"
    filename += ".tar"
    if not osp.isfile(osp.join(path, filename)):
        print("generating evaluation sets...")
        gen_evalset(args)

    eval_batches = torch.load(osp.join(path, filename))

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    ravg = RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_batches, ascii=True):
            for key, val in batch.items():
                batch[key] = val.cuda()

            outs = model(batch)

            for key, val in outs.items():
                ravg.update(key, val)
            model.reset()

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    c1, c2 = args.class_range
    line = f"ReTreever:{args.expid} {c1}-{c2} "
    line += ravg.info()

    if logger is not None:
        logger.info(line)

    return line


if __name__ == "__main__":
    main()
