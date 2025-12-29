import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from dataset import *
from medpy.metric import binary as medpy_binary
from skimage.measure import label


class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss implementation
    """
    def __init__(self, num_classes, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, outputs, targets, softmax=True):
        """
        Calculate Dice loss for multi-class segmentation
        
        Args:
            outputs: Model predictions [B, C, H, W]
            targets: Ground truth labels [B, H, W] 
            softmax: Whether to apply softmax to outputs
        
        Returns:
            Dice loss value
        """
        if softmax:
            outputs = torch.softmax(outputs, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = torch.zeros_like(outputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1).long(), 1)
        
        # Calculate dice coefficient for each class
        dice_scores = []
        for i in range(self.num_classes):
            pred_i = outputs[:, i, :, :]
            target_i = targets_one_hot[:, i, :, :]
            
            intersection = torch.sum(pred_i * target_i, dim=(1, 2))
            union = torch.sum(pred_i, dim=(1, 2)) + torch.sum(target_i, dim=(1, 2))
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Stack dice scores and calculate mean
        dice_scores = torch.stack(dice_scores, dim=1)  # [B, num_classes]
        dice_loss = 1 - torch.mean(dice_scores)
        
        return dice_loss


def connectivity_loss(pred, smooth=1e-6):
    """
    Calculates connectivity loss to encourage connected regions
    Returns average number of disconnected components per class
    """
    # Get prediction probabilities
    pred_softmax = torch.softmax(pred, dim=1)
    batch_size = pred.size(0)
    num_classes = pred.size(1)
    
    loss = 0.0
    for i in range(batch_size):
        class_loss = 0.0
        for c in range(1, num_classes):  # Skip background class (0)
            # Threshold class probability map
            binary_pred = (pred_softmax[i, c] > 0.5).float().cpu().numpy()
            if binary_pred.sum() > 0:  # Only calculate if class exists in prediction
                labeled_pred, num_components = label(binary_pred, connectivity=2, return_num=True)
                class_loss += max(0, num_components - 1)  # Penalize multiple components
        loss += class_loss / (num_classes - 1)  # Average over classes
    
    return loss / batch_size


def calculate_metric_percase(pred, gt):
    """
    Calculate Dice and HD95 metrics for a single class
    pred and gt should be binary masks for a specific class
    """
    if pred.sum() > 0 and gt.sum() > 0:
        dice = medpy_binary.dc(pred, gt)
        try:
            hd95 = medpy_binary.hd95(pred, gt)
        except Exception:
            # Handle case where HD95 calculation fails (e.g., empty contours)
            hd95 = 0.0
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        # False positive - predicted class when it doesn't exist
        return 0.0, 0.0
    elif pred.sum() == 0 and gt.sum() > 0:
        # False negative - missed class that exists
        return 0.0, 0.0
    else:
        # True negative - correctly predicted absence of class
        return 1.0, 0.0


def eval_dice(pred_y, gt_y, classes=4):
    """
    Evaluate Dice score for classes that exist in ground truth
    Returns dice scores, hd95 scores, and mask for existing classes
    """
    pred_y = torch.argmax(torch.softmax(pred_y, dim=1), dim=1)

    pred_y = pred_y.cpu().detach().numpy()
    gt_y = gt_y.cpu().detach().numpy()

    all_dice = []
    all_hd95 = []
    existing_classes = []  # Track which classes exist in ground truth
    
    for cls in range(classes):
        # Get binary masks for this class
        pred_cls = (pred_y == cls)
        gt_cls = (gt_y == cls)
        
        # Only calculate metrics if class exists in ground truth
        if np.any(gt_cls):
            existing_classes.append(True)
            dice, hd95 = calculate_metric_percase(pred_cls, gt_cls)
            all_dice.append(dice)
            all_hd95.append(hd95)
        else:
            existing_classes.append(False)
            all_dice.append(None)  # Use None for non-existing classes
            all_hd95.append(None)

    return all_dice, all_hd95, existing_classes


def eval(model, val_loader, device, classes, save_path='', epoch=1):
    """
    Evaluate model on validation data
    Only consider classes that exist in ground truth
    """
    all_dice_list = []
    all_hd95_list = []
    all_existing_classes = []
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # Get image and label from batch
            img, label = batch['image'].to(device), batch['label'].to(device)
            
            # The model expects [B, C, H, W] but label might be [B, 1, H, W]
            # Make sure dimensions are correct
            if label.dim() == 4 and label.size(1) == 1:
                label = label.squeeze(1)
                
            output = model(img)
            dice_scores, hd95_scores, existing_classes = eval_dice(output, label, classes=classes)
            all_dice_list.append(dice_scores)
            all_hd95_list.append(hd95_scores)
            all_existing_classes.append(existing_classes)

    # Save per-image Dice scores to text file (write nan for non-existing classes)
    save_metrics_path = "dice_scores_epoch_"+str(epoch)+".txt"
    with open(save_metrics_path, 'w') as f:
        for dice_scores, existing_classes in zip(all_dice_list, all_existing_classes):
            # Include all classes, write nan for classes that don't exist
            all_dice_scores = []
            for i in range(len(dice_scores)):
                if existing_classes[i] and dice_scores[i] is not None:
                    all_dice_scores.append(f"{dice_scores[i]:.4f}")
                else:
                    all_dice_scores.append("nan")
            line = " ".join(all_dice_scores)
            f.write(line + "\n")

    # Compute per-class average Dice and HD95 (only for existing classes)
    mean_dice_per_class = []
    mean_hd95_per_class = []
    
    for cls in range(classes):
        class_dice_scores = []
        class_hd95_scores = []
        
        # Collect dice scores for this class across all images where it exists
        for dice_scores, hd95_scores, existing_classes in zip(all_dice_list, all_hd95_list, all_existing_classes):
            if existing_classes[cls] and dice_scores[cls] is not None:
                class_dice_scores.append(dice_scores[cls])
                class_hd95_scores.append(hd95_scores[cls])
        
        # Calculate mean only if class exists in at least one image
        if class_dice_scores:
            mean_dice_per_class.append(np.mean(class_dice_scores))
            mean_hd95_per_class.append(np.mean(class_hd95_scores))
        else:
            mean_dice_per_class.append(0.0)  # or np.nan if you prefer
            mean_hd95_per_class.append(0.0)  # or np.nan if you prefer

    return np.array(mean_dice_per_class), np.array(mean_hd95_per_class)


def calculate_class_weights(dataset, num_classes=4):
    class_counts = np.zeros(num_classes, dtype=np.float64)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample['label'].numpy()
        for c in range(num_classes):
            class_counts[c] += np.sum(label == c)

    class_counts = np.maximum(class_counts, 1)

    class_weights = 1.0 / class_counts

    class_weights = class_weights / np.sum(class_weights) * num_classes

    return torch.FloatTensor(class_weights)


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # Set up transforms
    train_transform = transforms.Compose([
        Resize((args.img_size + 32, args.img_size + 32)),
        RandomCrop((args.img_size, args.img_size)),
        RandomFlip(),
        RandomRotation(),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    test_transform = transforms.Compose([
        Resize((args.img_size, args.img_size)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Initialize datasets
    try:
        db_train = SegCTDataset(dataroot=args.root_path, mode='train', transforms=train_transform)
        db_test = SegCTDataset(dataroot=args.root_path, mode='test', transforms=test_transform)
    except Exception as e:
        logging.error(f"Failed to initialize datasets: {str(e)}")
        raise

    print("The length of train set is: {}".format(len(db_train)))
    print("The length of test set is: {}".format(len(db_test)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # Set up data loaders
    num_workers = 0 if not hasattr(args, 'num_workers') else args.num_workers
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    # Test mode - load model and evaluate
    if args.test:
        model.eval()
        try:
            model.load_state_dict(torch.load(args.pretrained_model_weights, map_location='cpu'))
        except Exception as e:
            logging.error(f"Failed to load pretrained weights: {str(e)}")
            raise
            
        test_dice, test_hd95 = eval(model, testloader, 'cuda', classes=num_classes)
        dice_str = ' '.join(f"{x * 100:.1f}" for x in test_dice)
        hd95_str = ' '.join(f"{x:.2f}" for x in test_hd95)
        print('Test Dice per class: [%s]' % dice_str)
        print('Test HD95 per class: [%s]' % hd95_str)
        print(f'Mean Dice: {np.mean(test_dice) * 100:.2f}%')
        exit(0)

    # Use DataParallel for multi-GPU training
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    
    # Calculate class weights for loss function
    class_weights = calculate_class_weights(db_train, num_classes=num_classes)
    logging.info(f"Class weights: {class_weights}")
    
    # Initialize loss functions with class weights
    ce_loss = CrossEntropyLoss(weight=class_weights.cuda())
    dice_loss = DiceLoss(num_classes)
    
    # Setup optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=5e-2)

    # Initialize tensorboard writer
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    # Training loop
    best_mean_dice = 0
    for epoch_num in range(max_epoch):
        model.train()

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            
            # Move tensors to GPU
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            # Ensure label has the right shape - if it's [B, 1, H, W], squeeze to [B, H, W]
            if label_batch.dim() == 4 and label_batch.size(1) == 1:
                label_batch = label_batch.squeeze(1)
            
            # Forward pass
            outputs = model(image_batch)
            
            # Calculate losses
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # loss_connect = connectivity_loss(outputs)
            
            # Combine losses with weights
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            
            # Log metrics
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            # Print progress
            if iter_num % 1000 == 1:
                logging.info('iteration %d - loss: %f, loss_ce: %f, loss_dice: %f' % 
                            (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            # Visualize results
            if iter_num % 20 == 0:
                index = 0
                # Normalize image for visualization
                image = image_batch[index]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                
                # Create better visualization for multi-class segmentation
                # Get predictions as class indices
                pred = torch.argmax(torch.softmax(outputs[index], dim=0), dim=0)
                
                # Create a colormap for visualization
                pred_vis = pred.float() / (num_classes - 1)
                # Create RGB channels for better visualization
                pred_vis = pred_vis.unsqueeze(0).repeat(3, 1, 1)
                writer.add_image('train/Prediction', pred_vis, iter_num)
                
                # Visualize ground truth with same colormap
                label_vis = label_batch[index].float() / (num_classes - 1)
                label_vis = label_vis.unsqueeze(0).repeat(3, 1, 1)
                writer.add_image('train/GroundTruth', label_vis, iter_num)

        # Evaluate model after each epoch
        test_dice, test_hd95 = eval(model, testloader, 'cuda', classes=num_classes, 
                                   save_path=snapshot_path, epoch=epoch_num) 
        print(test_dice)
        mean_dice = np.nanmean(test_dice)  # Calculate mean Dice score across all classes
        
        # Save model checkpoint
        save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
        
        # Save best model
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            best_model_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info("New best model saved with mean Dice: {:.4f}".format(best_mean_dice))
        
        # Print epoch summary
        dice_str = ' '.join(f"{x * 100:.1f}" for x in test_dice)
        print('Epoch [%3d/%3d], Loss: %.4f, Dice per class: [%s], Mean Dice: %.2f%%' %
              (epoch_num + 1, max_epoch, loss.item(), dice_str, mean_dice * 100))
        
        # Add epoch metrics to tensorboard
        writer.add_scalar('eval/mean_dice', mean_dice, epoch_num)
        for cls in range(num_classes):
            writer.add_scalar(f'eval/dice_class_{cls}', test_dice[cls], epoch_num)

    writer.close()
    return "Training Finished!"