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

from data.gp import *
from utils.misc import load_module
from utils.paths import results_path, evalsets_path
from utils.log import get_logger, RunningAverage


def main():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument("--mode", default="train")
    parser.add_argument("--expid", type=str, default="default")
    parser.add_argument("--resume", action="store_true")

    # Data
    parser.add_argument("--max_num_points", type=int, default=50)

    # Train
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--save_freq", type=int, default=1000)

    # Eval
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--eval_num_batches", type=int, default=3000)
    parser.add_argument("--eval_batch_size", type=int, default=16)
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

    # OOD settings
    parser.add_argument("--eval_kernel", type=str, default="rbf")

    args = parser.parse_args()

    if args.expid is not None:
        args.root = osp.join(results_path, "gp", args.expid)
    else:
        args.root = osp.join(results_path, "gp")

    model_cls = getattr(load_module(f"models/retreever.py"), "Retreever")
    with open(f"configs/gp.yaml", "r") as f:
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

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print("generating evaluation sets...")
        gen_evalset(args)

    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed(args.train_seed)

    sampler = GPSampler(RBFKernel())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps
    )

    if args.resume:
        ckpt = torch.load(os.path.join(args.root, "ckpt.tar"))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_step = ckpt.step
    else:
        logfilename = os.path.join(
            args.root, f'train_{time.strftime("%Y%m%d-%H%M")}.log'
        )
        start_step = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info(f"Experiment: {args.expid}")
        logger.info(
            f"Total number of parameters: {sum(p.numel() for p in model.parameters())}\n"
        )

    for step in range(start_step, args.num_steps + 1):

        model.reset()
        model.train()
        optimizer.zero_grad()
        batch = sampler.sample(
            batch_size=args.train_batch_size,
            max_num_points=args.max_num_points,
            device="cuda",
        )

        outs = model(batch)

        outs.loss.backward()
        if args.clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        scheduler.step()

        for key, val in outs.items():
            ravg.update(key, val)

        if step % args.print_freq == 0:
            line = f"Retreever:{args.expid} step {step} "
            line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
            line += f"[train_loss] "
            line += ravg.info()
            logger.info(line)

            if step % args.eval_freq == 0:
                line = eval(args, model)
                logger.info(line + "\n")

            ravg.reset()

        if step % args.save_freq == 0 or step == args.num_steps:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.step = step + 1
            torch.save(ckpt, os.path.join(args.root, "ckpt.tar"))
    args.mode = "eval"
    eval(args, model)


def get_eval_path(args):
    path = osp.join(evalsets_path, "gp")
    filename = f"{args.eval_kernel}-seed{args.eval_seed}"
    filename += ".tar"
    return path, filename


def gen_evalset(args):
    if args.eval_kernel == "rbf":
        kernel = RBFKernel()
    elif args.eval_kernel == "matern":
        kernel = Matern52Kernel()
    else:
        raise ValueError(f"Invalid kernel {args.eval_kernel}")
    print(f"Generating Evaluation Sets with {args.eval_kernel} kernel")

    sampler = GPSampler(kernel, seed=args.eval_seed)
    batches = []
    for i in tqdm(range(args.eval_num_batches), ascii=True):
        batches.append(
            sampler.sample(
                batch_size=args.eval_batch_size,
                max_num_points=args.max_num_points,
                device="cuda",
            )
        )

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path, filename = get_eval_path(args)
    if not osp.isdir(path):
        os.makedirs(path)
    torch.save(batches, osp.join(path, filename))


def eval(args, model):
    if args.mode == "eval":
        ckpt = torch.load(os.path.join(args.root, "ckpt.tar"), map_location="cuda")
        model.load_state_dict(ckpt.model)
        if args.eval_logfile is None:
            eval_logfile = f"eval_{args.eval_kernel}"
            eval_logfile += ".log"
        else:
            eval_logfile = args.eval_logfile
        filename = os.path.join(args.root, eval_logfile)
        logger = get_logger(filename, mode="w")
    else:
        logger = None

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print("generating evaluation sets...")
        gen_evalset(args)
    eval_batches = torch.load(osp.join(path, filename))

    if args.mode == "eval":
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed(args.eval_seed)

    ravg = RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_batches, ascii=True):
            model.reset()
            for key, val in batch.items():
                batch[key] = val.cuda()

            outs = model(batch)

            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f"Retreever:{args.expid} {args.eval_kernel} "
    line += ravg.info()

    if logger is not None:
        logger.info(line)

    return line


if __name__ == "__main__":
    main()
