
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import argparse
import json
import os
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from torch import nn
from tqdm import tqdm
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

from datasets import UCF101, HMDB51, Kinetics, PhaseDatasetImages, ToolExistenceImages, SkillAssesementDataset
from models import get_vit_base_patch16_224, get_vit_small_patch16_224, get_vit_tiny_patch16_224, get_aux_token_vit, SwinTransformer3D
from utils import utils
from utils.meters import TestMeter
from utils.parser import load_config
from vision_transformer import DINOHead, MultiDINOHead
from sklearn.metrics import roc_auc_score
import pandas as pd

class Full_Network(nn.Module):
    def __init__(self, backbone, classifier):
        super(Full_Network, self).__init__()  # Call the __init__ method of nn.Module superclass
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def eval_finetune(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(f"{args.output_dir}/config.json", "w"), indent=4)

    # ============ preparing data ... ============
    config = load_config(args)
    config.TEST.NUM_SPATIAL_CROPS = 1
    if args.dataset == "phasedatasetimages":
        dataset_train = PhaseDatasetImages(cfg=config, mode="train")
        dataset_val = PhaseDatasetImages(cfg=config, mode="val")
        config.TEST.NUM_SPATIAL_CROPS = 3
    elif args.dataset == "hmdb51":
        dataset_train = HMDB51(cfg=config, mode="train", num_retries=10)
        dataset_val = HMDB51(cfg=config, mode="val", num_retries=10)
        config.TEST.NUM_SPATIAL_CROPS = 3
    elif args.dataset == "kinetics400":
        dataset_train = Kinetics(cfg=config, mode="train", num_retries=10)
        dataset_val = Kinetics(cfg=config, mode="val", num_retries=10)
        config.TEST.NUM_SPATIAL_CROPS = 3
    elif args.dataset == "complicationsvideos":
        dataset_train = SkillAssesementDataset(cfg=config, mode="train")
        dataset_val = SkillAssesementDataset(cfg=config, mode="val")
        config.TEST.NUM_SPATIAL_CROPS = 3
    else:
        raise NotImplementedError(f"invalid dataset: {args.dataset}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=train_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    if config.DATA.USE_FLOW or config.MODEL.TWO_TOKEN:
        model = get_aux_token_vit(cfg=config, no_head=True)
        model_embed_dim = 2 * model.embed_dim
    else:
        if args.arch == "vit_base":
            model = get_vit_base_patch16_224(cfg=config, no_head=True)
            model_embed_dim = model.embed_dim
            print(model_embed_dim)
            print("**************vit_base****************")
        elif args.arch == "vit_small":
            model = get_vit_small_patch16_224(cfg=config, no_head=True)
            model_embed_dim = model.embed_dim
            print(model_embed_dim)
            print("**************vit_small****************")
        elif args.arch == "vit_tiny":
            model = get_vit_tiny_patch16_224(cfg=config, no_head=True)
            model_embed_dim = model.embed_dim
            print(model_embed_dim)
            print("**************vit_tiny****************")
        elif args.arch == "swin":
            model = SwinTransformer3D(depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32])
            model_embed_dim = 1024
        else:
            raise Exception(f"invalid model: {args.arch}")
    
    if not args.scratch and args.pretrained_weights:
        ckpt = torch.load(args.pretrained_weights, map_location='cpu')
        if args.arch == "vit_small":
            if "teacher" in ckpt:
                ckpt = ckpt["teacher"]
            renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}
            msg = model.load_state_dict(renamed_checkpoint, strict=False)
            print(f"Loaded vit_small model with msg: {msg}")
        elif args.arch == "vit_tiny":
            if "teacher" in ckpt:
                ckpt = ckpt["teacher"]
            renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}
            msg = model.load_state_dict(renamed_checkpoint, strict=False)
            print(f"Loaded vit_tiny model with msg: {msg}")
        elif args.arch == "vit_base":
            if "teacher" in ckpt:
                ckpt = ckpt["teacher"]
            renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}
            msg = model.load_state_dict(renamed_checkpoint, strict=False)
            print(f"Loaded vit_base model with msg: {msg}")
            if args.freeze_backbone == True:
                for param in model.parameters():
                    param.requires_grad = False
    elif args.scratch:
        if args.Transfer_from_CV:
            print("Transfer from CV parameters")
            ckpt = torch.load('checkpoints/kinetics400_vitb_ssl.pth', map_location='cpu')
            if "teacher" in ckpt:
                ckpt = ckpt["teacher"]
            renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}
            msg = model.load_state_dict(renamed_checkpoint, strict=False)
            print(f"Loaded model with msg: {msg}")
        else:
            print("train from sctratch")
    
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    # load weights to evaluate

    print("num_labels:",args.num_labels)

    linear_classifier = LinearClassifier(model_embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens)),
                                         num_labels=args.num_labels)
    
    model_full = Full_Network(model, linear_classifier)
    model_full.cuda()
    model_full = nn.parallel.DistributedDataParallel(model_full, device_ids=[args.gpu])

    if args.test:
        ckpt = torch.load(args.pretrained_model_weights, map_location='cpu')
        if "full_model_dict" in ckpt:
            ckpt = ckpt["full_model_dict"]
        
        msg = model_full.load_state_dict(ckpt, strict=False)
        print("full_model_msg:", msg)

        total_params = sum(p.numel() for p in model_full.parameters())
        print(f"Total number of parameters in the model: {total_params}")

        test_stats, f1, accuracy, precision, recall, jaccard = validate_network(val_loader, model_full, args.n_last_blocks,
                                          args.avgpool_patchtokens)
        print(f"F1 score of the network on the {len(dataset_val)} test images: {f1 * 100:.1f}%")
        print(f"accuracy of the network on the {len(dataset_val)} test images: {accuracy * 100:.1f}%")
        print(f"precision of the network on the {len(dataset_val)} test images: {precision * 100:.1f}%")
        print(f"recall of the network on the {len(dataset_val)} test images: {recall * 100:.1f}%")
        print(f"jaccard of the network on the {len(dataset_val)} test images: {jaccard * 100:.1f}%")
        exit(0)

    scaled_lr = args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.

    # set optimizer
    optimizer = torch.optim.SGD(
        [{'params': model_full.parameters(), 'lr': scaled_lr}],
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0., "best_AUC": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_AUC = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(args, model_full, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats, AUC = validate_network(val_loader, model_full, args.n_last_blocks, args.avgpool_patchtokens, epoch)
            print(f"AUC score at epoch {epoch} of the network on the {len(dataset_val)} test images: {AUC * 100:.1f}%")
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}

            if AUC > best_AUC and utils.is_main_process():
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                save_dict = {
                    "epoch": epoch + 1,
                    "full_model_dict": model_full.state_dict(),
                    "backbone_dict":model.state_dict(),
                    "linear_dict":linear_classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_AUC": best_AUC,
                }
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))

            best_AUC = max(best_AUC, AUC)
            print(f'Max AUC score so far: {best_AUC * 100:.1f}%')

def train(args, model_full, optimizer, loader, epoch, n, avgpool):
    model_full.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target, sample_idx, meta) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model_full(inp)

    
        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model_full, n, avgpool, epcoch):
    model_full.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    all_target = []
    all_output = []
    all_probs = []  # To store the predicted probabilities for the positive class

    for (inp, target, sample_idx, meta) in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model_full(inp)
        
        loss = nn.CrossEntropyLoss()(output, target)
        acc1, = utils.accuracy(output, target, topk=(1,))

        all_target.extend(target.detach().cpu().numpy())
        all_output.extend(np.argmax(output.detach().cpu().numpy(), axis=1))
        
        # Get the predicted probabilities for the positive class (assuming class 1 is positive)
        probs = torch.nn.functional.softmax(output, dim=1)[:, 1]  # Get probabilities for class 1
        all_probs.extend(probs.detach().cpu().numpy())

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # Calculate AUC score
    auc_score = roc_auc_score(all_target, all_probs)
    print(f"AUC Score: {auc_score:.4f}")

    # Save all_target and all_probs into a .csv file
    df = pd.DataFrame({
        'target': all_target,
        'probability': all_probs
    })
    df.to_csv("/output_"+str(epcoch)+".csv", index=False)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, auc_score


@torch.no_grad()
def validate_network_multi_view(val_loader, model, linear_classifier, n, avgpool, cfg):
    linear_classifier.eval()
    test_meter = TestMeter(
        len(val_loader.dataset)
        // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        args.num_labels,
        len(val_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
        )
    test_meter.iter_tic()
    all_target = []
    all_output = []
    for cur_iter, (inp, target, sample_idx, meta) in tqdm(enumerate(val_loader), total=len(val_loader)):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        test_meter.data_toc()

        # forward
        with torch.no_grad():
            output = model(inp)
        output = linear_classifier(output)

        all_target.extend(target.detach().cpu().numpy())
        all_output.extend(np.argmax(output.detach().cpu().numpy(), axis=1))

        output = output.cpu()
        target = target.cpu()
        sample_idx = sample_idx.cpu()

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(
            output.detach(), target.detach(), sample_idx.detach()
        )
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    f1 = f1_score(all_target, all_output, average='micro')

    # Compute accuracy
    accuracy = accuracy_score(all_target, all_output)

    # Compute precision
    precision = precision_score(all_target, all_output, average='micro')

    # Compute recall
    recall = recall_score(all_target, all_output, average='micro')

    # Compute Jaccard similarity score
    jaccard = jaccard_score(all_target, all_output, average='micro')

    test_meter.finalize_metrics(ks=(1, ))
    return test_meter.stats, f1, accuracy, precision, recall, jaccard


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
                        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'swin'],
                        help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--lc_pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    # parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--dataset', default="phasedataset", help='Dataset: ucf101 / hmdb51')
    parser.add_argument('--use_flow', default=False, type=utils.bool_flag, help="use flow teacher")

    parser.add_argument('--datasetpath', default=".", help='Path to dataset')

    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pretrained_model_weights', default='polypdiag.pth', type=str, help='pre-trained weights')

    # config file
    parser.add_argument("--cfg", dest="cfg_file", help="Path to the config file", type=str,
                        default="models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml")
    parser.add_argument("--opts", help="See utils/defaults.py for all options", default=None, nargs=argparse.REMAINDER)

    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    
    parser.add_argument('--freeze_backbone', default=False, type=utils.bool_flag,
                        help="Whether to freeze the backbone (Default: False)")
    parser.add_argument('--Transfer_from_CV', default=False, type=utils.bool_flag,
                        help="True: initialize parameters with kinetics400_vitb_ssl; False: train from scratch")

    args = parser.parse_args()
    eval_finetune(args)

