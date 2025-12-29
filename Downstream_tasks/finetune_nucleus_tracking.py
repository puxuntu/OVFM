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
import matplotlib.pyplot as plt
import cv2

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

from datasets import UCF101, HMDB51, Kinetics, PhaseDatasetImages, SkillAssesementDataset, NucleusDataset
from models import get_vit_base_patch16_224, get_vit_small_patch16_224, get_vit_tiny_patch16_224, get_aux_token_vit, SwinTransformer3D
from utils import utils
from utils.meters import TestMeter
from utils.parser import load_config
from vision_transformer import DINOHead, MultiDINOHead
import pandas as pd


class Full_Network(nn.Module):
    def __init__(self, backbone, head):
        super(Full_Network, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output


class BoundingBoxHead(nn.Module):
    """Head for target localization that predicts both center coordinates and dimensions (normalized)"""
    def __init__(self, dim, hidden_dim=256):
        super(BoundingBoxHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4)  # 4 values: [center_x, center_y, width, height]
        )
        
        # Initialize weights
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Ensure x is flattened to [batch_size, features]
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Predict normalized bounding box parameters [cx, cy, w, h]
        bbox = self.mlp(x)
        
        # Apply sigmoid to ensure all values are in [0, 1] range
        # This is important for normalized coordinates and dimensions
        bbox = torch.sigmoid(bbox)
        
        return bbox


class IoULoss(nn.Module):
    """
    Improved IoU Loss for bounding box prediction
    """
    def __init__(self, eps=1e-6, smooth=1.0):
        super(IoULoss, self).__init__()
        self.eps = eps
        self.smooth = smooth  # Add smoothing to prevent gradient issues
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted bounding boxes [batch_size, 4] in format [cx, cy, w, h]
            target: Ground truth boxes [batch_size, 4] in same format
        Returns:
            IoU loss
        """
        # Extract coordinates and dimensions
        pred_cx, pred_cy, pred_w, pred_h = torch.unbind(pred, dim=1)
        target_cx, target_cy, target_w, target_h = torch.unbind(target, dim=1)
        
        # Convert from center format to min/max format
        pred_x1 = pred_cx - pred_w/2
        pred_y1 = pred_cy - pred_h/2
        pred_x2 = pred_cx + pred_w/2
        pred_y2 = pred_cy + pred_h/2
        
        target_x1 = target_cx - target_w/2
        target_y1 = target_cy - target_h/2
        target_x2 = target_cx + target_w/2
        target_y2 = target_cy + target_h/2
        
        # Calculate intersection area
        x1 = torch.max(pred_x1, target_x1)
        y1 = torch.max(pred_y1, target_y1)
        x2 = torch.min(pred_x2, target_x2)
        y2 = torch.min(pred_y2, target_y2)
        
        # Clip intersection to valid range
        width = torch.clamp(x2 - x1, min=0)
        height = torch.clamp(y2 - y1, min=0)
        intersection = width * height
        
        # Calculate areas with minimum threshold to prevent division by zero
        pred_area = torch.clamp(pred_w * pred_h, min=self.eps)
        target_area = torch.clamp(target_w * target_h, min=self.eps)
        
        # Calculate IoU with smoothing
        union = pred_area + target_area - intersection
        iou = (intersection + self.smooth) / (union + self.smooth + self.eps)
        
        # Return loss - use 1 - IoU but clamp to prevent negative values
        loss = torch.clamp(1 - iou, min=0.0, max=1.0)
        return loss.mean()


class ImprovedCombinedLoss(nn.Module):
    """
    Improved Combined loss with better weight balancing
    """
    def __init__(self, iou_weight=0.3, coord_weight=10.0, dim_weight=10.0, use_focal=False):
        super(ImprovedCombinedLoss, self).__init__()
        self.iou_loss = IoULoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()  # Better than MSE for coordinates
        self.mse_loss = nn.MSELoss()
        self.iou_weight = iou_weight
        self.coord_weight = coord_weight
        self.dim_weight = dim_weight
        self.use_focal = use_focal
        
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """Focal loss for better handling of hard examples"""
        loss = self.mse_loss(pred, target)
        pt = torch.exp(-loss)
        focal_loss = alpha * (1 - pt) ** gamma * loss
        return focal_loss
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted bounding boxes [batch_size, 4] in format [cx, cy, w, h]
            target: Ground truth boxes [batch_size, 4] in same format
        Returns:
            Combined loss and components dict
        """
        # IoU loss (already normalized to [0,1])
        iou_loss = self.iou_loss(pred, target)
        
        # Coordinate loss (center points) - use Smooth L1 for robustness
        if self.use_focal:
            coord_loss = self.focal_loss(pred[:, :2], target[:, :2])
        else:
            coord_loss = self.smooth_l1_loss(pred[:, :2], target[:, :2])
        
        # Dimension loss (width, height) - use MSE but with square root for better scaling
        dim_pred = torch.sqrt(pred[:, 2:] + 1e-8)
        dim_target = torch.sqrt(target[:, 2:] + 1e-8)
        dim_loss = self.mse_loss(dim_pred, dim_target)
        
        # Combine losses with adjusted weights
        total_loss = (
            self.iou_weight * iou_loss +
            self.coord_weight * coord_loss +
            self.dim_weight * dim_loss
        )
        
        return total_loss, {
            'iou_loss': iou_loss.item(),
            'coord_loss': coord_loss.item(),
            'dim_loss': dim_loss.item(),
            'total_loss': total_loss.item()
        }


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in format [cx, cy, w, h]
    
    Args:
        box1: First box [cx, cy, w, h]
        box2: Second box [cx, cy, w, h]
    
    Returns:
        IoU value
    """
    # Convert to [x1, y1, x2, y2] format
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    # Calculate intersection area
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    # Check if boxes intersect
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    
    # Calculate areas
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    # Calculate IoU
    union_area = b1_area + b2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_localization_metrics(pred_boxes, gt_boxes, orig_size):
    """
    Compute metrics for bounding box localization task
    
    Args:
        pred_boxes: Predicted boxes [batch_size, 4] in format [cx, cy, w, h]
        gt_boxes: Ground truth boxes [batch_size, 4] in same format
        orig_size: Original image dimensions [batch_size, 2]
    
    Returns:
        dict: Dictionary of metrics
    """
    # Convert to numpy for easier computation
    pred_boxes = pred_boxes.detach().cpu().numpy()
    gt_boxes = gt_boxes.detach().cpu().numpy()
    orig_size = orig_size.detach().cpu().numpy()
    
    batch_size = pred_boxes.shape[0]
    
    # Calculate IoU for each pair of boxes
    ious = []
    for i in range(batch_size):
        iou = calculate_iou(pred_boxes[i], gt_boxes[i])
        ious.append(iou)
    
    ious = np.array(ious)
    
    # Calculate metrics
    mean_iou = np.mean(ious)
    precision_at_05 = np.mean(ious > 0.5)
    precision_at_07 = np.mean(ious > 0.7)
    precision_at_09 = np.mean(ious > 0.9)
    
    # Calculate center error (Euclidean distance between predicted and ground truth centers)
    center_errors = np.sqrt(
        (pred_boxes[:, 0] - gt_boxes[:, 0])**2 +
        (pred_boxes[:, 1] - gt_boxes[:, 1])**2
    )
    mean_center_error = np.mean(center_errors)
    
    # Calculate relative size error
    size_errors = np.abs(
        (pred_boxes[:, 2] * pred_boxes[:, 3]) - 
        (gt_boxes[:, 2] * gt_boxes[:, 3])
    ) / (gt_boxes[:, 2] * gt_boxes[:, 3] + 1e-8)
    mean_size_error = np.mean(size_errors)
    
    return {
        'mean_iou': float(mean_iou),
        'precision@0.5': float(precision_at_05),
        'precision@0.7': float(precision_at_07),
        'precision@0.9': float(precision_at_09),
        'mean_center_error': float(mean_center_error),
        'mean_size_error': float(mean_size_error)
    }


def visualize_predictions(image, gt_box, pred_box, idx, output_dir, epoch):
    """
    Visualize predictions on images and save them
    
    Args:
        image: Image tensor [C, T, H, W] (with time dimension)
        gt_box: Ground truth box [cx, cy, w, h]
        pred_box: Predicted box [cx, cy, w, h]
        idx: Image index
        output_dir: Output directory
        epoch: Current epoch
    """
    # Create visualization directory if it doesn't exist
    vis_dir = os.path.join(output_dir, 'visualizations', f'epoch_{epoch}')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Extract first frame from time dimension, then permute to HWC format for OpenCV
    img = image[:, 0, :, :].permute(1, 2, 0).cpu().numpy()
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # Convert to uint8 and ensure the array is contiguous in memory
    img = (img * 255).astype(np.uint8)
    img = np.ascontiguousarray(img)  # This ensures OpenCV compatibility
    
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    gt_cx, gt_cy, gt_w, gt_h = gt_box
    pred_cx, pred_cy, pred_w, pred_h = pred_box
    
    # Calculate IoU
    iou = calculate_iou(gt_box, pred_box)
    
    # Convert to pixel coordinates with proper integer casting
    gt_cx, gt_cy = int(gt_cx * w), int(gt_cy * h)
    gt_w, gt_h = max(1, int(gt_w * w)), max(1, int(gt_h * h))  # Ensure minimum size of 1
    pred_cx, pred_cy = int(pred_cx * w), int(pred_cy * h)
    pred_w, pred_h = max(1, int(pred_w * w)), max(1, int(pred_h * h))
    
    # Draw ground truth bounding box (with boundary checking)
    gt_x1, gt_y1 = max(0, gt_cx - gt_w // 2), max(0, gt_cy - gt_h // 2)
    gt_x2, gt_y2 = min(w-1, gt_cx + gt_w // 2), min(h-1, gt_cy + gt_h // 2)
    cv2.rectangle(img, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), 2)  # Green for ground truth
    
    # Draw predicted bounding box
    pred_x1, pred_y1 = max(0, pred_cx - pred_w // 2), max(0, pred_cy - pred_h // 2)
    pred_x2, pred_y2 = min(w-1, pred_cx + pred_w // 2), min(h-1, pred_cy + pred_h // 2)
    cv2.rectangle(img, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 0, 255), 2)  # Red for prediction
    
    # Draw centers
    cv2.circle(img, (gt_cx, gt_cy), 5, (0, 255, 0), -1)  # Green for ground truth
    cv2.circle(img, (pred_cx, pred_cy), 5, (0, 0, 255), -1)  # Red for prediction
    
    # Draw line between centers
    cv2.line(img, (gt_cx, gt_cy), (pred_cx, pred_cy), (255, 0, 0), 2)  # Blue line
    
    # Add text for IoU
    cv2.putText(img, f"IoU: {iou:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save image
    cv2.imwrite(os.path.join(vis_dir, f'sample_{idx}.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def get_improved_scheduler(optimizer, args):
    """Get improved learning rate scheduler with warmup"""
    if args.warmup_epochs > 0:
        # Warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1, 
            total_iters=args.warmup_epochs
        )
        # Main scheduler  
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs - args.warmup_epochs, 
            eta_min=1e-6
        )
        # Sequential scheduler
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[args.warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    
    return scheduler


def train_one_epoch(model, data_loader, optimizer, epoch, args):
    """Improved training loop with better loss handling"""
    model.train()
    
    # Use improved loss with better weight balancing  
    criterion = ImprovedCombinedLoss(
        iou_weight=args.iou_weight,
        coord_weight=args.coord_weight,
        dim_weight=args.dim_weight,
        use_focal=args.use_focal
    )
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('iou_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('coord_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('dim_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Epoch: [{epoch}]'
    
    for samples, targets, idxs, meta in metric_logger.log_every(data_loader, 10, header):
        # Move to GPU
        samples = samples.cuda(non_blocking=True)
        
        # Prepare ground truth bounding box: [cx, cy, w, h]
        gt_bbox = torch.cat([
            targets['normalized_center'],
            targets['normalized_dimensions']
        ], dim=1).cuda(non_blocking=True)
        
        # Add small noise to prevent overfitting to exact coordinates
        if args.add_noise and model.training:
            noise_scale = 0.01
            noise = torch.randn_like(gt_bbox) * noise_scale
            gt_bbox = torch.clamp(gt_bbox + noise, 0.0, 1.0)
        
        # Forward pass
        pred_bbox = model(samples)
        
        # Compute loss
        loss, loss_components = criterion(pred_bbox, gt_bbox)
        
        # Check for NaN values
        if torch.isnan(loss):
            print(f"NaN loss detected at epoch {epoch}")
            print(f"Pred bbox stats: min={pred_bbox.min()}, max={pred_bbox.max()}")
            print(f"GT bbox stats: min={gt_bbox.min()}, max={gt_bbox.max()}")
            continue
            
        # Gradient clipping to prevent exploding gradients
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
        optimizer.step()
        
        # Log metrics
        metric_logger.update(loss=loss_components['total_loss'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(iou_loss=loss_components['iou_loss'])
        metric_logger.update(coord_loss=loss_components['coord_loss'])
        metric_logger.update(dim_loss=loss_components['dim_loss'])
        metric_logger.update(total_loss=loss_components['total_loss'])
    
    # Gather metrics from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate(model, data_loader, epoch, args):
    """Validation loop"""
    model.eval()
    criterion = ImprovedCombinedLoss(
        iou_weight=args.iou_weight,
        coord_weight=args.coord_weight,
        dim_weight=args.dim_weight,
        use_focal=args.use_focal
    )
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation:'
    
    all_predictions = []
    all_targets = []
    all_orig_sizes = []
    all_idxs = []  # Store all image indices
    all_ious = []  # Store all individual IoU values
    
    # Number of samples to visualize
    num_vis_samples = min(10, len(data_loader.dataset))
    vis_indices = np.random.choice(len(data_loader.dataset), num_vis_samples, replace=False)
    
    with torch.no_grad():
        for samples, targets, idxs, meta in metric_logger.log_every(data_loader, 10, header):
            # Move to GPU
            samples = samples.cuda(non_blocking=True)
            
            # Prepare ground truth bounding box: [cx, cy, w, h]
            gt_bbox = torch.cat([
                targets['normalized_center'],
                targets['normalized_dimensions']
            ], dim=1).cuda(non_blocking=True)
            
            # Forward pass
            pred_bbox = model(samples)
            
            # Compute loss
            loss, loss_components = criterion(pred_bbox, gt_bbox)
            
            # Store predictions and targets for metric computation
            all_predictions.append(pred_bbox.cpu())
            all_targets.append(gt_bbox.cpu())
            all_orig_sizes.append(targets['original_size'])
            all_idxs.extend(idxs.tolist())  # Store the indices
            
            # Calculate individual IoU values for each image in the batch
            for i in range(pred_bbox.size(0)):
                iou = calculate_iou(
                    pred_bbox[i].cpu().numpy(),
                    gt_bbox[i].cpu().numpy()
                )
                all_ious.append((idxs[i].item(), iou))  # Store (image_idx, iou) pairs
            
            # Visualize some predictions
            for i, idx in enumerate(idxs):
                if idx.item() in vis_indices:
                    visualize_predictions(
                        samples[i].cpu(),
                        gt_bbox[i].cpu().numpy(),
                        pred_bbox[i].cpu().numpy(),
                        idx.item(),
                        args.output_dir,
                        epoch
                    )
            
            # Update metrics
            metric_logger.update(loss=loss_components['total_loss'])
            metric_logger.update(iou_loss=loss_components['iou_loss'])
            metric_logger.update(coord_loss=loss_components['coord_loss'])
            metric_logger.update(dim_loss=loss_components['dim_loss'])
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_orig_sizes = torch.cat(all_orig_sizes, dim=0)
    
    # Compute localization metrics
    metrics = compute_localization_metrics(all_predictions, all_targets, all_orig_sizes)
    
    # Add metrics to logger
    for k, v in metrics.items():
        metric_logger.meters[k] = utils.SmoothedValue(window_size=1, fmt='{value:.4f}')
        metric_logger.update(**{k: v})
    
    # Print metrics
    print('* Loss {losses.avg:.3f} Mean IoU {mean_iou:.3f} Precision@0.5 {precision05:.3f}'
          .format(losses=metric_logger.loss, 
                  mean_iou=metrics['mean_iou'],
                  precision05=metrics['precision@0.5']))
    
    # Write individual IoU values to a file
    if utils.is_main_process():  # Only write from the main process
        # Create directory for IoU logs if it doesn't exist
        iou_log_dir = os.path.join(args.output_dir, 'iou_logs')
        os.makedirs(iou_log_dir, exist_ok=True)
        
        # Sort by image index for consistent ordering
        all_ious.sort(key=lambda x: x[0])
        
        # Write to file
        with open(os.path.join(iou_log_dir, f'iou_values_epoch_{epoch}.txt'), 'w') as f:
            for idx, iou in all_ious:
                f.write(f"{idx}\t{iou:.6f}\n")
        
        print(f"IoU values for {len(all_ious)} images written to {iou_log_dir}/iou_values_epoch_{epoch}.txt")
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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
    
    if args.dataset == "nucleus_dataset":
        dataset_train = NucleusDataset(cfg=config, mode="train")
        dataset_val = NucleusDataset(cfg=config, mode="val")
        config.TEST.NUM_SPATIAL_CROPS = 1
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
            print(f"Model embed dim: {model_embed_dim}")
            print("**************vit_base****************")
        elif args.arch == "vit_small":
            model = get_vit_small_patch16_224(cfg=config, no_head=True)
            model_embed_dim = model.embed_dim
            print(f"Model embed dim: {model_embed_dim}")
            print("**************vit_small****************")
        elif args.arch == "vit_tiny":
            model = get_vit_tiny_patch16_224(cfg=config, no_head=True)
            model_embed_dim = model.embed_dim
            print(f"Model embed dim: {model_embed_dim}")
            print("**************vit_tiny****************")
        elif args.arch == "swin":
            model = SwinTransformer3D(depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32])
            model_embed_dim = 1024
        else:
            raise Exception(f"invalid model: {args.arch}")
    
    # Load pretrained weights if available
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
            print("Training from scratch")
    
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    
    # Create bounding box head (predicts both center and dimensions)
    feature_dim = model_embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    bbox_head = BoundingBoxHead(feature_dim, hidden_dim=256)
    
    # Combine backbone and bounding box head
    model_full = Full_Network(model, bbox_head)
    model_full.cuda()
    model_full = nn.parallel.DistributedDataParallel(model_full, device_ids=[args.gpu])

    # Scale learning rate based on batch size
    scaled_lr = args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.

    # Set optimizer - Using AdamW for better performance in coordinate regression
    optimizer = torch.optim.AdamW(
        model_full.parameters(),
        lr=scaled_lr,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler with warmup
    scheduler = get_improved_scheduler(optimizer, args)

    # Resume from checkpoint if specified
    to_restore = {"epoch": 0, "best_acc": 0.}
    if args.resume:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, "checkpoint.pth"),
            run_variables=to_restore,
            model=model_full,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    
    start_epoch = to_restore["epoch"]
    best_iou = to_restore["best_acc"]
    
    # Track metrics for each epoch
    train_stats = []
    eval_stats = []
    
    print(f"Starting training from epoch {start_epoch}")
    print(f"Loss weights - IoU: {args.iou_weight}, Coord: {args.coord_weight}, Dim: {args.dim_weight}")
    
    for epoch in range(start_epoch, args.epochs):
        # Set train sampler epoch for correct shuffling
        train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        print(f"Training epoch {epoch}")
        train_metrics = train_one_epoch(model_full, train_loader, optimizer, epoch, args)
        train_stats.append(train_metrics)
        
        # Evaluate
        if epoch % args.val_freq == 0:
            print(f"Validating epoch {epoch}")
            eval_metrics = validate(model_full, val_loader, epoch, args)
            eval_stats.append(eval_metrics)
        
        # Update scheduler
        scheduler.step()
        
        # Save checkpoint
        save_dict = {
            'model': model_full.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'best_acc': best_iou,
        }
        
        # Save latest checkpoint
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        
        # Save best model based on Mean IoU
        if epoch % args.val_freq == 0:
            current_iou = eval_metrics['mean_iou']
            if current_iou > best_iou:
                best_iou = current_iou
                save_dict['best_acc'] = best_iou
                utils.save_on_master(save_dict, os.path.join(args.output_dir, 'best_model.pth'))
                print(f"New best IoU: {best_iou:.4f} at epoch {epoch}")
        
        # Log metrics
        log_stats = {
            **{f'train_{k}': v for k, v in train_metrics.items()},
            'epoch': epoch
        }
        
        if epoch % args.val_freq == 0:
            log_stats.update({f'val_{k}': v for k, v in eval_metrics.items()})
        
        if utils.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
                
    
    print("Training complete")
    print(f"Best IoU achieved: {best_iou:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with bounding box prediction')
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
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--pretrained_model_weights', default='polypdiag.pth', type=str, help='pre-trained weights')

    # Improved Loss weights
    parser.add_argument('--iou_weight', default=0.3, type=float, help='Weight for IoU loss component')
    parser.add_argument('--coord_weight', default=10.0, type=float, help='Weight for coordinate loss component')
    parser.add_argument('--dim_weight', default=10.0, type=float, help='Weight for dimension loss component')
    parser.add_argument('--use_focal', default=False, type=utils.bool_flag, help='Use focal loss for coordinates')

    # Training improvements
    parser.add_argument('--add_noise', default=False, type=utils.bool_flag, help='Add noise during training')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='Number of warmup epochs')
    parser.add_argument('--clip_grad_norm', default=1.0, type=float, help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='Weight decay for AdamW optimizer')

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