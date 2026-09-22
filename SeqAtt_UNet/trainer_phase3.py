# =============================================================================
# NOTICE OF MODIFICATION (Apache License 2.0, Section 4(b))
#
# This file is derived from TransUNet:
#     https://github.com/Beckschen/TransUNet -- trainer.py
#     Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical
#     Image Segmentation," arXiv:2102.04306, 2021. Licensed under Apache-2.0.
#
# This file HAS BEEN MODIFIED by the authors of the present study. See NOTICE
# and THIRD_PARTY.md in the repository root, and the full licence text in
# LICENSE.
# =============================================================================
"""
PHASE 3 TRAINER FOR SeqAtt-UNet
===============================
Advanced training techniques, expected to add +1-2% DSC.

INTEGRATES ALL PHASE 3 IMPROVEMENTS:

3.1. Self-configuring, in the spirit of nnU-Net:
    - Auto-detect optimal patch size
    - Linear scaling rule for batch size & LR
    - Automatic network topology adjustment

3.2. Pre-training Strategy:
    - R50+ViT-B/16 ImageNet-21K pretrained weights
    - Warmup + Cosine/Polynomial LR scheduling
    - Layer-wise learning rate decay

3.3. Deep Supervision:
    - Auxiliary losses at the decoder stages
    - Improved gradient flow
    - Weights: [1.0, 0.4, 0.2, 0.1]

COMBINED WITH PHASE 1:
    - Combined loss: 1.0*Dice + 0.5*CE + 1[epoch>=50]*(0.1*Boundary + 0.1*Bmap)
      (see LOSS_CONFIG below and the docstring of losses.py)
    - Advanced Data Augmentation
    - Alpha Tracker for CBAM monitoring
    - Test-Time Augmentation support

Author: SeqAtt-UNet Project - Phase 3 Improvements
"""

import argparse
import logging
import os
import platform
import random
import sys
import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from typing import Dict, List, Tuple, Optional, Any

# Import Phase 1 components
from losses import CombinedLoss, DiceLoss, get_loss_function
from augmentation import AdvancedRandomGenerator, get_augmentation, AUGMENTATION_CONFIGS
from alpha_tracker import AlphaTracker

# Import Phase 3 components
from auto_config import (
    AutoConfig, 
    DatasetAnalyzer, 
    AutoConfigGenerator,
    WarmupCosineScheduler,
    WarmupPolynomialScheduler,
    get_auto_config
)
from deep_supervision import (
    DeepSupervisionLoss,
    DeepSupervisionDiceCELoss,
    get_deep_supervision_loss,
    compute_ds_weights
)


# =============================================================================
# WINDOWS COMPATIBILITY
# =============================================================================

_GLOBAL_SEED = 1234

def worker_init_fn(worker_id):
    """Worker initialization function for the DataLoader."""
    random.seed(_GLOBAL_SEED + worker_id)
    np.random.seed(_GLOBAL_SEED + worker_id)


def get_num_workers():
    """Determine the number of DataLoader workers suited to the current OS."""
    if platform.system() == 'Windows':
        return 0
    return 4


# =============================================================================
# PHASE 3 CONFIGURATION
# =============================================================================

# Loss function config (inherits from Phase 1)
LOSS_CONFIG = {
    'lambda_dice': 1.0,
    'lambda_ce': 0.5,
    'lambda_boundary': 0.1,
    'lambda_hd': 0.1,
    'use_boundary': True,
    'use_hd': True,
    'boundary_start_epoch': 50,
}

# Phase 3 specific config
PHASE3_CONFIG = {
    # 3.1 Self-Configuring
    'use_auto_config': True,           # Enable auto configuration
    'auto_adjust_lr': True,            # Linear scaling for LR
    
    # 3.2 Pre-training Strategy
    'scheduler_type': 'warmup_cosine', # 'warmup_cosine' or 'warmup_polynomial'
    'warmup_epochs': 10,               # Warmup epochs
    'layer_wise_lr_decay': 0.9,        # Layer-wise LR decay (1.0 = disabled)
    
    # 3.3 Deep Supervision
    'use_deep_supervision': True,      # Enable deep supervision
    'ds_weights': (1.0, 0.4, 0.2, 0.1), # DS output weights
    'ds_start_epoch': 0,               # When to start DS (0 = from beginning)
    
    # Training
    'accumulation_steps': 2,           # Gradient accumulation
    'enable_alpha_warmstart': True,    # CBAM alpha warmstart
    'alpha_warmstart_epoch': 5,
    'alpha_warmstart_value': 0.1,
    
    # Best model saving
    'best_model_start_ratio': 0.95,    # Start saving best model at 95% of training
}


# =============================================================================
# LAYER-WISE LEARNING RATE DECAY
# =============================================================================

def get_layer_wise_lr_params(model: nn.Module, 
                              base_lr: float,
                              decay: float = 0.9) -> List[Dict]:
    """
    Build parameter groups with layer-wise learning rate decay.
    
    Layers closer to the output get a higher LR; deeper layers get a lower one.
    
    Formula: lr_layer_i = base_lr * (decay ^ (num_layers - i))
    
    Args:
        model: PyTorch model
        base_lr: Base learning rate
        decay: Decay factor (0.9 = a deeper layer keeps 90% of the LR of the layer above)
        
    Returns:
        List of parameter groups for optimizer
    """
    if decay == 1.0:
        # No decay, single parameter group
        return [{'params': model.parameters(), 'lr': base_lr}]
    
    param_groups = []
    
    # Group parameters by depth
    # TransUNet structure: embeddings -> encoder -> decoder -> segmentation_head
    
    depth_map = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Determine depth based on layer name
        if 'embeddings' in name or 'hybrid_model' in name:
            depth = 0  # Shallowest (encoder backbone)
        elif 'encoder' in name:
            # Extract layer number if available
            if 'layer.' in name:
                try:
                    layer_num = int(name.split('layer.')[1].split('.')[0])
                    depth = 1 + layer_num
                except:
                    depth = 5
            else:
                depth = 5
        elif 'decoder' in name:
            depth = 10
        elif 'segmentation_head' in name or 'ds_heads' in name:
            depth = 15  # Deepest (output layers)
        else:
            depth = 8  # Default
        
        if depth not in depth_map:
            depth_map[depth] = []
        depth_map[depth].append(param)
    
    # Create parameter groups with layer-wise LR
    max_depth = max(depth_map.keys()) if depth_map else 0
    
    for depth, params in sorted(depth_map.items()):
        # Higher depth = closer to output = higher LR
        lr_scale = decay ** (max_depth - depth)
        lr = base_lr * lr_scale
        
        param_groups.append({
            'params': params,
            'lr': lr,
            'depth': depth
        })
    
    return param_groups


# =============================================================================
# DEEP SUPERVISION COMBINED LOSS
# =============================================================================

class Phase3CombinedLoss(nn.Module):
    """
    Combined loss for Phase 3 with Deep Supervision.
    
    Combines:
    - Deep Supervision (multi-scale outputs)
    - Dice Loss
    - Cross Entropy Loss  
    - Boundary Loss (optional)
    - HD Loss (optional)
    """
    
    def __init__(self,
                 n_classes: int,
                 ds_weights: Tuple[float, ...] = (1.0, 0.4, 0.2, 0.1),
                 lambda_dice: float = 1.0,
                 lambda_ce: float = 0.5,
                 lambda_boundary: float = 0.1,
                 lambda_hd: float = 0.1,
                 use_boundary: bool = True,
                 use_hd: bool = True):
        """
        Args:
            n_classes: Number of classes
            ds_weights: Weights for the deep supervision outputs
            lambda_dice, lambda_ce, etc.: Weights for the individual loss components
            use_boundary, use_hd: Flags used to enable/disable those losses
        """
        super(Phase3CombinedLoss, self).__init__()
        
        self.n_classes = n_classes
        self.ds_weights = ds_weights
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.lambda_boundary = lambda_boundary
        self.lambda_hd = lambda_hd
        self.use_boundary = use_boundary
        self.use_hd = use_hd
        
        # Base loss (Dice + CE)
        self.base_loss = CombinedLoss(
            n_classes=n_classes,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            lambda_boundary=lambda_boundary,
            lambda_hd=lambda_hd,
            use_boundary=use_boundary,
            use_hd=use_hd
        )
        
        self._ds_enabled = True
    
    def enable_deep_supervision(self):
        """Enable deep supervision."""
        self._ds_enabled = True
    
    def disable_deep_supervision(self):
        """Disable deep supervision (for inference)."""
        self._ds_enabled = False
    
    def forward(self,
                outputs,
                target: torch.Tensor,
                epoch: int = 0,
                boundary_start_epoch: int = 50) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined loss with deep supervision.
        
        Args:
            outputs: Single tensor, or list of tensors when deep supervision is on
            target: Ground truth
            epoch: Current epoch
            boundary_start_epoch: Epoch at which the boundary/HD loss kicks in
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_dict = {}
        
        # Handle single output
        if not isinstance(outputs, (list, tuple)):
            return self.base_loss(outputs, target, epoch, boundary_start_epoch)
        
        if not self._ds_enabled or len(outputs) == 1:
            # Only use final output
            return self.base_loss(outputs[0], target, epoch, boundary_start_epoch)
        
        # Deep supervision: compute loss for each scale
        total_loss = 0.0
        
        for i, output in enumerate(outputs):
            if i >= len(self.ds_weights):
                break
            
            weight = self.ds_weights[i]
            
            if weight == 0:
                continue
            
            # Resize output if needed
            if output.shape[-2:] != target.shape[-2:]:
                output = F.interpolate(
                    output,
                    size=target.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Compute loss for this scale
            scale_loss, scale_dict = self.base_loss(
                output, target, epoch, boundary_start_epoch
            )
            
            total_loss = total_loss + weight * scale_loss
            
            # Record loss components
            for key, val in scale_dict.items():
                if key == 'total':
                    loss_dict[f'scale{i}_total'] = val
                else:
                    loss_dict[f'scale{i}_{key}'] = val
        
        loss_dict['total'] = total_loss.item()
        loss_dict['ds_enabled'] = True
        loss_dict['num_scales'] = len(outputs)
        
        return total_loss, loss_dict


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_cbam_alphas(model: nn.Module) -> Dict[str, float]:
    """Collect the alpha values from the CBAM modules."""
    alphas = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'alpha_raw'):
            alpha_value = F.softplus(module.alpha_raw).item()
            short_name = name.split('.')[-1] if '.' in name else name
            alphas[short_name] = alpha_value
        elif hasattr(module, 'alpha') and isinstance(module.alpha, nn.Parameter):
            alphas[name.split('.')[-1]] = module.alpha.item()
    
    return alphas


def set_cbam_alphas(model: nn.Module, value: float) -> int:
    """Set the alpha value on every CBAM module."""
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'alpha_raw'):
            if value > 0:
                raw_value = math.log(math.exp(value) - 1) if value < 20 else value
                module.alpha_raw.data.fill_(raw_value)
            else:
                module.alpha_raw.data.fill_(0.0)
            count += 1
        elif hasattr(module, 'alpha') and isinstance(module.alpha, nn.Parameter):
            module.alpha.data.fill_(value)
            count += 1
    return count


def log_cbam_alphas(model: nn.Module, writer: SummaryWriter, iter_num: int, logger):
    """Log CBAM alpha values to TensorBoard."""
    alphas = get_cbam_alphas(model)
    if alphas:
        for name, value in alphas.items():
            writer.add_scalar(f'cbam_alpha/{name}', value, iter_num)
        
        alpha_str = ', '.join([f'{k}: {v:.4f}' for k, v in alphas.items()])
        logger.info(f"[CBAM Alpha] {alpha_str}")
        
        mean_alpha = sum(alphas.values()) / len(alphas)
        writer.add_scalar('cbam_alpha/mean', mean_alpha, iter_num)


# =============================================================================
# MAIN TRAINER - PHASE 3
# =============================================================================

def trainer_synapse_phase3(args, model, snapshot_path):
    """
    Training with the Phase 3 improvements on the Synapse dataset.
    
    Integrates:
    - Auto-config derived from dataset analysis
    - Deep Supervision
    - Advanced LR scheduling
    - Layer-wise LR decay
    - All Phase 1 improvements
    """
    from datasets.dataset_synapse import Synapse_dataset
    
    os.makedirs(snapshot_path, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    # =========================
    # PHASE 3: AUTO-CONFIG
    # =========================
    
    auto_config = None
    if PHASE3_CONFIG['use_auto_config']:
        try:
            analyzer = DatasetAnalyzer(args.root_path, 'Synapse')
            stats = analyzer.analyze()
            
            # Detect GPU memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                gpu_memory = 11.0
            
            generator = AutoConfigGenerator(gpu_memory_gb=gpu_memory, use_amp=False)
            auto_config = generator.generate(stats)
            
            # Override batch size and LR if auto_adjust is enabled
            if PHASE3_CONFIG['auto_adjust_lr']:
                args.batch_size = auto_config.batch_size
                args.base_lr = auto_config.base_lr
                logging.info(f"[AUTO-CONFIG] Adjusted batch_size={args.batch_size}, base_lr={args.base_lr}")
        except Exception as e:
            logging.warning(f"Auto-config failed: {e}. Using default settings.")
    
    # Training parameters
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_epoch = args.max_epochs
    
    # Log configuration
    logging.info("=" * 70)
    logging.info("🚀 SeqAtt-UNet PHASE 3 TRAINING (Synapse)")
    logging.info("=" * 70)
    logging.info("PHASE 3 IMPROVEMENTS:")
    logging.info(f"  [3.1] Auto-Config: {PHASE3_CONFIG['use_auto_config']}")
    logging.info(f"  [3.2] LR Scheduler: {PHASE3_CONFIG['scheduler_type']}")
    logging.info(f"        Warmup epochs: {PHASE3_CONFIG['warmup_epochs']}")
    logging.info(f"        Layer-wise LR decay: {PHASE3_CONFIG['layer_wise_lr_decay']}")
    logging.info(f"  [3.3] Deep Supervision: {PHASE3_CONFIG['use_deep_supervision']}")
    logging.info(f"        DS Weights: {PHASE3_CONFIG['ds_weights']}")
    logging.info("-" * 70)
    logging.info(f"TRAINING CONFIG:")
    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Base LR: {base_lr}")
    logging.info(f"  Epochs: {max_epoch}")
    logging.info(f"  Classes: {num_classes}")
    logging.info("=" * 70)
    
    # =========================
    # DATA LOADING
    # =========================
    
    # Get augmentation - use the config from AUGMENTATION_CONFIGS
    aug_type = args.augmentation if hasattr(args, 'augmentation') else 'strong'
    
    # Pick the config matching the requested augmentation type
    if aug_type in AUGMENTATION_CONFIGS:
        aug_config = AUGMENTATION_CONFIGS[aug_type]
        train_transform = AdvancedRandomGenerator(
            output_size=[args.img_size, args.img_size],
            **aug_config
        )
        logging.info(f"Using '{aug_type}' augmentation config")
    else:
        # Fallback to simple augmentation
        train_transform = get_augmentation(
            aug_type='light',
            output_size=[args.img_size, args.img_size]
        )
        logging.info(f"Using 'light' augmentation (fallback)")
    
    db_train = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=train_transform
    )
    
    logging.info(f"Dataset size: {len(db_train)} samples")
    
    global _GLOBAL_SEED
    _GLOBAL_SEED = args.seed
    
    num_workers = get_num_workers()
    if num_workers == 0:
        logging.info("⚠️ Windows detected: Using num_workers=0")
    
    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        worker_init_fn=worker_init_fn if num_workers > 0 else None
    )
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    model.train()
    
    # =========================
    # PHASE 3: DEEP SUPERVISION LOSS
    # =========================
    
    if PHASE3_CONFIG['use_deep_supervision']:
        criterion = Phase3CombinedLoss(
            n_classes=num_classes,
            ds_weights=PHASE3_CONFIG['ds_weights'],
            lambda_dice=LOSS_CONFIG['lambda_dice'],
            lambda_ce=LOSS_CONFIG['lambda_ce'],
            lambda_boundary=LOSS_CONFIG['lambda_boundary'],
            lambda_hd=LOSS_CONFIG['lambda_hd'],
            use_boundary=LOSS_CONFIG['use_boundary'],
            use_hd=LOSS_CONFIG['use_hd']
        )
        logging.info("[DEEP SUPERVISION] Enabled with weights: " + str(PHASE3_CONFIG['ds_weights']))
    else:
        criterion = CombinedLoss(
            n_classes=num_classes,
            lambda_dice=LOSS_CONFIG['lambda_dice'],
            lambda_ce=LOSS_CONFIG['lambda_ce'],
            lambda_boundary=LOSS_CONFIG['lambda_boundary'],
            lambda_hd=LOSS_CONFIG['lambda_hd'],
            use_boundary=LOSS_CONFIG['use_boundary'],
            use_hd=LOSS_CONFIG['use_hd']
        )
    
    # =========================
    # PHASE 3: LAYER-WISE LR DECAY + OPTIMIZER
    # =========================
    
    if PHASE3_CONFIG['layer_wise_lr_decay'] < 1.0:
        param_groups = get_layer_wise_lr_params(
            model,
            base_lr=base_lr,
            decay=PHASE3_CONFIG['layer_wise_lr_decay']
        )
        logging.info(f"[LAYER-WISE LR] Enabled with decay={PHASE3_CONFIG['layer_wise_lr_decay']}")
        logging.info(f"  Created {len(param_groups)} parameter groups")
    else:
        param_groups = [{'params': model.parameters(), 'lr': base_lr}]
    
    optimizer = optim.SGD(
        param_groups,
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001
    )
    
    # =========================
    # PHASE 3: LR SCHEDULER
    # =========================
    
    if PHASE3_CONFIG['scheduler_type'] == 'warmup_cosine':
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=PHASE3_CONFIG['warmup_epochs'],
            total_epochs=max_epoch,
            base_lr=base_lr,
            min_lr=1e-6
        )
        logging.info(f"[LR SCHEDULER] WarmupCosine with {PHASE3_CONFIG['warmup_epochs']} warmup epochs")
    else:
        scheduler = WarmupPolynomialScheduler(
            optimizer,
            warmup_epochs=PHASE3_CONFIG['warmup_epochs'],
            total_epochs=max_epoch,
            base_lr=base_lr,
            power=0.9
        )
        logging.info(f"[LR SCHEDULER] WarmupPolynomial with power=0.9")
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    
    # Alpha Tracker
    alpha_tracker = None
    if hasattr(args, 'use_cbam') and args.use_cbam:
        alpha_tracker = AlphaTracker(model, snapshot_path)
    
    # Training counters
    iter_num = 0
    max_iterations = max_epoch * len(trainloader)
    
    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")
    
    best_loss = float('inf')
    best_epoch = 0
    best_model_start_epoch = int(max_epoch * PHASE3_CONFIG['best_model_start_ratio'])
    
    # Log initial CBAM alphas
    initial_alphas = get_cbam_alphas(model)
    if initial_alphas:
        logging.info(f"Initial CBAM Alpha values: {initial_alphas}")
    
    # Save config
    config_dict = {
        'phase3_config': PHASE3_CONFIG,
        'loss_config': LOSS_CONFIG,
        'args': vars(args)
    }
    with open(os.path.join(snapshot_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    # =========================
    # TRAINING LOOP
    # =========================
    
    iterator = tqdm(range(max_epoch), ncols=70)
    accumulation_steps = PHASE3_CONFIG['accumulation_steps']
    
    for epoch_num in iterator:
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_loss_dict = {'dice': 0, 'ce': 0, 'hd': 0}
        
        # Update learning rate (epoch-based scheduler)
        current_lr = scheduler.step(epoch_num)
        writer.add_scalar('info/lr', current_lr, epoch_num * len(trainloader))
        
        # CBAM Alpha Warm-start
        if PHASE3_CONFIG['enable_alpha_warmstart'] and epoch_num == PHASE3_CONFIG['alpha_warmstart_epoch']:
            num_alphas = set_cbam_alphas(model, PHASE3_CONFIG['alpha_warmstart_value'])
            logging.info(f"[CBAM WARM-START] Epoch {epoch_num}: Set {num_alphas} alpha(s) = {PHASE3_CONFIG['alpha_warmstart_value']}")
        
        # Enable/disable deep supervision based on epoch (sync model and criterion)
        ds_enabled = epoch_num >= PHASE3_CONFIG['ds_start_epoch']
        
        # Enable/disable on model
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        if hasattr(actual_model, 'enable_deep_supervision'):
            if ds_enabled:
                actual_model.enable_deep_supervision()
            else:
                actual_model.disable_deep_supervision()
        
        # Enable/disable on criterion
        if hasattr(criterion, 'enable_deep_supervision'):
            if ds_enabled:
                criterion.enable_deep_supervision()
            else:
                criterion.disable_deep_supervision()
        
        # Log DS status at start epoch
        if epoch_num == PHASE3_CONFIG['ds_start_epoch'] and PHASE3_CONFIG['use_deep_supervision']:
            logging.info(f"[DEEP SUPERVISION] Enabled at epoch {epoch_num}")
        
        if accumulation_steps > 1:
            optimizer.zero_grad()
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            # Forward pass
            outputs = model(image_batch)
            
            # Handle both single output and deep supervision outputs
            loss, loss_dict = criterion(
                outputs,
                label_batch,
                epoch=epoch_num,
                boundary_start_epoch=LOSS_CONFIG['boundary_start_epoch']
            )
            
            # Backward pass with gradient accumulation
            if accumulation_steps > 1:
                loss_scaled = loss / accumulation_steps
                loss_scaled.backward()
                
                if (i_batch + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    iter_num += 1
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_num += 1
            
            # Accumulate epoch stats
            epoch_loss += loss.item()
            epoch_batches += 1
            
            for key in epoch_loss_dict:
                if key in loss_dict:
                    epoch_loss_dict[key] += loss_dict[key]
                elif f'scale0_{key}' in loss_dict:
                    epoch_loss_dict[key] += loss_dict[f'scale0_{key}']
            
            # Logging
            if accumulation_steps == 1 or (i_batch + 1) % accumulation_steps == 0:
                writer.add_scalar('info/total_loss', loss.item(), iter_num)
                
                for key, val in loss_dict.items():
                    if isinstance(val, (int, float)):
                        writer.add_scalar(f'info/loss_{key}', val, iter_num)
                
                if iter_num % 20 == 0:
                    loss_parts = []
                    for key in ['dice', 'ce', 'boundary', 'hd', 'total']:
                        if key in loss_dict:
                            loss_parts.append(f'{key}: {loss_dict[key]:.4f}')
                        elif f'scale0_{key}' in loss_dict:
                            loss_parts.append(f'{key}: {loss_dict[f"scale0_{key}"]:.4f}')
                    loss_str = ', '.join(loss_parts)
                    
                    ds_info = ""
                    if 'num_scales' in loss_dict:
                        ds_info = f" [DS: {loss_dict['num_scales']} scales]"
                    
                    logging.info(f'iter {iter_num}: {loss_str}{ds_info}')
            
            # Visualization
            if iter_num % 100 == 0 and iter_num > 0:
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                writer.add_image('train/Image', image, iter_num)
                
                # Handle list outputs (deep supervision)
                final_output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                outputs_vis = torch.argmax(torch.softmax(final_output, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_vis[0, ...] * 50, iter_num)
                
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        
        # =========================
        # EPOCH SUMMARY
        # =========================
        
        avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
        writer.add_scalar('epoch/avg_loss', avg_epoch_loss, epoch_num)
        writer.add_scalar('epoch/lr', current_lr, epoch_num)
        
        # Format epoch summary
        avg_loss_dict = {k: v / epoch_batches for k, v in epoch_loss_dict.items() if v > 0}
        loss_parts = []
        for key in ['dice', 'ce', 'boundary', 'hd']:
            if key in avg_loss_dict:
                loss_parts.append(f'{key}: {avg_loss_dict[key]:.4f}')
        loss_summary = ', '.join(loss_parts)
        
        logging.info(f'Epoch {epoch_num} - LR: {current_lr:.6f}, Avg loss: {avg_epoch_loss:.4f} [{loss_summary}]')
        
        # Log CBAM alphas periodically
        if epoch_num % 10 == 0:
            log_cbam_alphas(model, writer, iter_num, logging)
        
        # Alpha Tracker
        if alpha_tracker is not None:
            current_alphas = alpha_tracker.log_epoch(
                epoch=epoch_num,
                loss=avg_epoch_loss,
                writer=writer
            )
            
            if epoch_num % 10 == 0:
                alpha_str = " | ".join([f"{k}: {v:.4f}" for k, v in current_alphas.items()])
                logging.info(f"[Alpha Tracker] Epoch {epoch_num}: {alpha_str}")
        
        # =========================
        # MODEL SAVING
        # =========================
        
        # Best model saving (only in last 5% of training)
        if epoch_num >= best_model_start_epoch:
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_epoch = epoch_num
                
                best_model_path = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"[BEST MODEL] Saved at epoch {epoch_num} with loss {best_loss:.6f}")
        
        # Checkpoint saving
        save_interval = 50
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved checkpoint to {save_mode_path}")
        
        # Final epoch
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved final model to {save_mode_path}")
            
            if alpha_tracker is not None:
                alpha_tracker.save_logs()
                alpha_tracker.plot_evolution(show=False)
            
            logging.info("=" * 70)
            logging.info("🎉 PHASE 3 TRAINING COMPLETED!")
            logging.info(f"Best epoch: {best_epoch} with loss: {best_loss:.6f}")
            logging.info(f"Final CBAM Alphas: {get_cbam_alphas(model)}")
            logging.info("=" * 70)
            
            iterator.close()
            break
    
    writer.close()
    return "Training Finished!"


def trainer_acdc_phase3(args, model, snapshot_path):
    """
    Training with the Phase 3 improvements on the ACDC dataset.
    
    Same as trainer_synapse_phase3 but for the ACDC dataset (cardiac MRI).
    """
    from datasets.dataset_acdc import BaseDataSets, RandomGenerator
    import h5py
    
    os.makedirs(snapshot_path, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    # Training parameters
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_epoch = args.max_epochs
    
    logging.info("=" * 70)
    logging.info("🚀 SeqAtt-UNet PHASE 3 TRAINING (ACDC)")
    logging.info("=" * 70)
    logging.info(f"Deep Supervision: {PHASE3_CONFIG['use_deep_supervision']}")
    logging.info(f"LR Scheduler: {PHASE3_CONFIG['scheduler_type']}")
    logging.info("=" * 70)
    
    # Data loading
    train_transform = transforms.Compose([
        RandomGenerator(output_size=[args.img_size, args.img_size])
    ])
    
    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        transform=train_transform
    )
    
    logging.info(f"ACDC training set size: {len(db_train)}")
    
    global _GLOBAL_SEED
    _GLOBAL_SEED = args.seed
    
    num_workers = get_num_workers()
    
    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        worker_init_fn=worker_init_fn if num_workers > 0 else None
    )
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    model.train()
    
    # Loss function
    if PHASE3_CONFIG['use_deep_supervision']:
        criterion = Phase3CombinedLoss(
            n_classes=num_classes,
            ds_weights=PHASE3_CONFIG['ds_weights'],
            lambda_dice=LOSS_CONFIG['lambda_dice'],
            lambda_ce=LOSS_CONFIG['lambda_ce'],
            lambda_boundary=LOSS_CONFIG['lambda_boundary'],
            lambda_hd=LOSS_CONFIG['lambda_hd'],
            use_boundary=LOSS_CONFIG['use_boundary'],
            use_hd=LOSS_CONFIG['use_hd']
        )
    else:
        criterion = CombinedLoss(
            n_classes=num_classes,
            lambda_dice=LOSS_CONFIG['lambda_dice'],
            lambda_ce=LOSS_CONFIG['lambda_ce'],
            lambda_boundary=LOSS_CONFIG['lambda_boundary'],
            lambda_hd=LOSS_CONFIG['lambda_hd'],
            use_boundary=LOSS_CONFIG['use_boundary'],
            use_hd=LOSS_CONFIG['use_hd']
        )
    
    # Optimizer with layer-wise LR
    if PHASE3_CONFIG['layer_wise_lr_decay'] < 1.0:
        param_groups = get_layer_wise_lr_params(
            model, base_lr=base_lr, decay=PHASE3_CONFIG['layer_wise_lr_decay']
        )
    else:
        param_groups = [{'params': model.parameters(), 'lr': base_lr}]
    
    optimizer = optim.SGD(param_groups, lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    # LR Scheduler
    if PHASE3_CONFIG['scheduler_type'] == 'warmup_cosine':
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=PHASE3_CONFIG['warmup_epochs'],
            total_epochs=max_epoch, base_lr=base_lr
        )
    else:
        scheduler = WarmupPolynomialScheduler(
            optimizer, warmup_epochs=PHASE3_CONFIG['warmup_epochs'],
            total_epochs=max_epoch, base_lr=base_lr
        )
    
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    
    # Alpha Tracker
    alpha_tracker = None
    if hasattr(args, 'use_cbam') and args.use_cbam:
        alpha_tracker = AlphaTracker(model, snapshot_path)
    
    iter_num = 0
    max_iterations = max_epoch * len(trainloader)
    
    best_loss = float('inf')
    best_epoch = 0
    best_model_start_epoch = int(max_epoch * PHASE3_CONFIG['best_model_start_ratio'])
    
    accumulation_steps = PHASE3_CONFIG['accumulation_steps']
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_loss_dict = {'dice': 0, 'ce': 0, 'hd': 0}
        
        current_lr = scheduler.step(epoch_num)
        
        # CBAM Alpha Warm-start
        if PHASE3_CONFIG['enable_alpha_warmstart'] and epoch_num == PHASE3_CONFIG['alpha_warmstart_epoch']:
            num_alphas = set_cbam_alphas(model, PHASE3_CONFIG['alpha_warmstart_value'])
            logging.info(f"[CBAM WARM-START] Epoch {epoch_num}: Set {num_alphas} alpha(s)")
        
        # Enable/disable deep supervision based on epoch (sync model and criterion)
        ds_enabled = epoch_num >= PHASE3_CONFIG['ds_start_epoch']
        
        # Enable/disable on model
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        if hasattr(actual_model, 'enable_deep_supervision'):
            if ds_enabled:
                actual_model.enable_deep_supervision()
            else:
                actual_model.disable_deep_supervision()
        
        # Enable/disable on criterion
        if hasattr(criterion, 'enable_deep_supervision'):
            if ds_enabled:
                criterion.enable_deep_supervision()
            else:
                criterion.disable_deep_supervision()
        
        # Log DS status at start epoch
        if epoch_num == PHASE3_CONFIG['ds_start_epoch'] and PHASE3_CONFIG['use_deep_supervision']:
            logging.info(f"[DEEP SUPERVISION] Enabled at epoch {epoch_num}")
        
        if accumulation_steps > 1:
            optimizer.zero_grad()
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            outputs = model(image_batch)
            
            loss, loss_dict = criterion(
                outputs, label_batch, epoch=epoch_num,
                boundary_start_epoch=LOSS_CONFIG['boundary_start_epoch']
            )
            
            if accumulation_steps > 1:
                loss_scaled = loss / accumulation_steps
                loss_scaled.backward()
                
                if (i_batch + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    iter_num += 1
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_num += 1
            
            epoch_loss += loss.item()
            epoch_batches += 1
            
            for key in epoch_loss_dict:
                if key in loss_dict:
                    epoch_loss_dict[key] += loss_dict[key]
                elif f'scale0_{key}' in loss_dict:
                    epoch_loss_dict[key] += loss_dict[f'scale0_{key}']
            
            if accumulation_steps == 1 or (i_batch + 1) % accumulation_steps == 0:
                writer.add_scalar('info/lr', current_lr, iter_num)
                writer.add_scalar('info/total_loss', loss.item(), iter_num)
                
                if iter_num % 20 == 0:
                    logging.info(f'iter {iter_num}: total: {loss.item():.4f}')
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
        writer.add_scalar('epoch/avg_loss', avg_epoch_loss, epoch_num)
        logging.info(f'Epoch {epoch_num} - LR: {current_lr:.6f}, Avg loss: {avg_epoch_loss:.4f}')
        
        if epoch_num % 10 == 0:
            log_cbam_alphas(model, writer, iter_num, logging)
        
        if alpha_tracker is not None:
            alpha_tracker.log_epoch(epoch=epoch_num, loss=avg_epoch_loss, writer=writer)
        
        # Best model saving
        if epoch_num >= best_model_start_epoch:
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_epoch = epoch_num
                best_model_path = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"[BEST MODEL] Saved at epoch {epoch_num}")
        
        # Checkpoint saving
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % 50 == 0:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), save_mode_path)
        
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), save_mode_path)
            
            if alpha_tracker is not None:
                alpha_tracker.save_logs()
            
            logging.info("=" * 70)
            logging.info("🎉 PHASE 3 TRAINING COMPLETED!")
            logging.info(f"Best epoch: {best_epoch} with loss: {best_loss:.6f}")
            logging.info("=" * 70)
            
            iterator.close()
            break
    
    writer.close()
    return "Training Finished!"


# =============================================================================
# TEST CODE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 3 TRAINER MODULE TEST")
    print("=" * 70)
    
    # Test Phase3CombinedLoss
    print("\n1. Testing Phase3CombinedLoss...")
    
    n_classes = 9
    batch_size = 2
    h, w = 224, 224
    
    loss_fn = Phase3CombinedLoss(
        n_classes=n_classes,
        ds_weights=(1.0, 0.4, 0.2, 0.1),
        use_boundary=False,
        use_hd=False
    )
    
    # Test with deep supervision outputs
    outputs = [
        torch.randn(batch_size, n_classes, h, w),
        torch.randn(batch_size, n_classes, h//2, w//2),
        torch.randn(batch_size, n_classes, h//4, w//4),
        torch.randn(batch_size, n_classes, h//8, w//8),
    ]
    
    target = torch.randint(0, n_classes, (batch_size, h, w))
    
    loss, loss_dict = loss_fn(outputs, target, epoch=0)
    print(f"  DS Loss: {loss.item():.4f}")
    print(f"  Components: {loss_dict}")
    
    # Test with single output
    print("\n2. Testing with single output...")
    single_out = torch.randn(batch_size, n_classes, h, w)
    loss, loss_dict = loss_fn(single_out, target, epoch=0)
    print(f"  Single output loss: {loss.item():.4f}")
    
    # Test layer-wise LR
    print("\n3. Testing layer-wise LR decay...")
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.Linear(20, 10),
    )
    param_groups = get_layer_wise_lr_params(model, base_lr=0.01, decay=0.9)
    print(f"  Created {len(param_groups)} parameter groups")
    for i, pg in enumerate(param_groups):
        print(f"    Group {i}: LR={pg['lr']:.6f}, params={sum(p.numel() for p in pg['params'])}")
    
    print("\n" + "=" * 70)
    print("✓ Phase 3 trainer module test completed!")
    print("=" * 70)
