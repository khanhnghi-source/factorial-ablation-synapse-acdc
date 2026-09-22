"""
PHASE 3 INTEGRATION TEST - SeqAtt-UNet
=====================================
Script that verifies every Phase 3 component behaves correctly.

Run this script before training to make sure that:
1. The VisionTransformerDS model works with Deep Supervision
2. Phase3CombinedLoss handles multi-scale outputs
3. The Auto-Config module works
4. The LR schedulers behave as expected
5. Model and trainer integrate correctly

Usage:
    python test_phase3_integration.py

Author: SeqAtt-UNet Project - Phase 3 Integration Test
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Test configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 2
IMG_SIZE = 224
NUM_CLASSES = 9


def print_header(text):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_result(success, message):
    """Print a single test result."""
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"  {status}: {message}")
    return success


def test_visiontransformer_ds():
    """Test 1: check the VisionTransformerDS model."""
    print_header("TEST 1: VisionTransformerDS Model")
    
    try:
        from networks.vit_seg_modeling_phase3 import VisionTransformerDS, create_model_with_ds, CONFIGS
        
        # Test model creation with Deep Supervision
        model = create_model_with_ds(
            config_name='R50-ViT-B_16',
            img_size=IMG_SIZE,
            num_classes=NUM_CLASSES,
            use_cbam=True,
            mlp_type='bilstm',
            use_deep_supervision=True,
            num_ds_outputs=4
        )
        
        # Check that the model was created successfully
        success = print_result(model is not None, "Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"     Total parameters: {total_params:,}")
        
        # Test the forward pass with Deep Supervision enabled
        model.to(DEVICE).eval()
        x = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(x)
        
        # Check that the outputs form a list of 4 tensors
        is_list = isinstance(outputs, list)
        success = success and print_result(is_list, f"DS output is list: {is_list}")
        
        if is_list:
            success = success and print_result(
                len(outputs) == 4,
                f"DS output has 4 scales: {len(outputs)}"
            )
            
            # Check the shape of each output
            for i, out in enumerate(outputs):
                expected_shape = (BATCH_SIZE, NUM_CLASSES, IMG_SIZE, IMG_SIZE)
                correct_shape = out.shape == expected_shape
                success = success and print_result(
                    correct_shape,
                    f"Output {i} shape: {out.shape} (expected {expected_shape})"
                )
        
        # Test disable Deep Supervision
        model.disable_deep_supervision()
        with torch.no_grad():
            single_output = model(x)
        
        is_tensor = isinstance(single_output, torch.Tensor)
        success = success and print_result(
            is_tensor,
            f"DS disabled returns single tensor: {is_tensor}"
        )
        
        # Re-enable Deep Supervision
        model.enable_deep_supervision()
        with torch.no_grad():
            outputs_again = model(x)
        
        success = success and print_result(
            isinstance(outputs_again, list),
            "DS can be re-enabled successfully"
        )
        
        return success
        
    except Exception as e:
        print_result(False, f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_phase3_combined_loss():
    """Test 2: check Phase3CombinedLoss."""
    print_header("TEST 2: Phase3CombinedLoss")
    
    try:
        from trainer_phase3 import Phase3CombinedLoss
        
        # Create the loss function
        loss_fn = Phase3CombinedLoss(
            n_classes=NUM_CLASSES,
            ds_weights=(1.0, 0.4, 0.2, 0.1),
            lambda_dice=1.0,
            lambda_ce=0.5,
            lambda_boundary=0.1,
            lambda_hd=0.1,
            use_boundary=False,  # Disabled to keep the test simple
            use_hd=False
        )
        
        success = print_result(loss_fn is not None, "Loss function created")
        
        # Test with multi-scale outputs (Deep Supervision)
        outputs = [
            torch.randn(BATCH_SIZE, NUM_CLASSES, IMG_SIZE, IMG_SIZE).to(DEVICE),
            torch.randn(BATCH_SIZE, NUM_CLASSES, IMG_SIZE//2, IMG_SIZE//2).to(DEVICE),
            torch.randn(BATCH_SIZE, NUM_CLASSES, IMG_SIZE//4, IMG_SIZE//4).to(DEVICE),
            torch.randn(BATCH_SIZE, NUM_CLASSES, IMG_SIZE//8, IMG_SIZE//8).to(DEVICE),
        ]
        
        target = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, IMG_SIZE, IMG_SIZE)).to(DEVICE)
        
        loss, loss_dict = loss_fn(outputs, target, epoch=0)
        
        success = success and print_result(
            isinstance(loss, torch.Tensor) and loss.dim() == 0,
            f"DS Loss computed: {loss.item():.4f}"
        )
        
        success = success and print_result(
            'total' in loss_dict,
            f"Loss dict contains 'total': {loss_dict.get('total', 'N/A'):.4f}"
        )
        
        # Test with a single output
        single_output = torch.randn(BATCH_SIZE, NUM_CLASSES, IMG_SIZE, IMG_SIZE).to(DEVICE)
        loss_single, loss_dict_single = loss_fn(single_output, target, epoch=0)
        
        success = success and print_result(
            isinstance(loss_single, torch.Tensor),
            f"Single output loss: {loss_single.item():.4f}"
        )
        
        # Test disable/enable deep supervision
        loss_fn.disable_deep_supervision()
        loss_disabled, _ = loss_fn(outputs, target, epoch=0)
        
        success = success and print_result(
            True,
            f"DS disabled loss: {loss_disabled.item():.4f}"
        )
        
        return success
        
    except Exception as e:
        print_result(False, f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_auto_config():
    """Test 3: check the Auto-Config module."""
    print_header("TEST 3: Auto-Config Module")
    
    try:
        from auto_config import (
            DatasetStatistics,
            AutoConfig,
            AutoConfigGenerator,
            WarmupCosineScheduler,
            WarmupPolynomialScheduler
        )
        
        # Test DatasetStatistics
        stats = DatasetStatistics(
            name='Synapse',
            num_samples=2212,
            num_classes=9,
            image_shape=(224, 224),
            is_3d=False
        )
        
        success = print_result(stats.name == 'Synapse', "DatasetStatistics created")
        
        # Test AutoConfigGenerator
        generator = AutoConfigGenerator(gpu_memory_gb=11.0, use_amp=True)
        config = generator.generate(stats)
        
        success = success and print_result(
            config.patch_size == (224, 224),
            f"Auto patch size: {config.patch_size}"
        )
        
        success = success and print_result(
            config.batch_size > 0,
            f"Auto batch size: {config.batch_size}"
        )
        
        success = success and print_result(
            config.base_lr > 0,
            f"Auto base LR: {config.base_lr}"
        )
        
        # Test WarmupCosineScheduler
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=10, total_epochs=100, base_lr=0.01
        )
        
        # Test warmup phase
        lr_0 = scheduler.step(0)
        lr_9 = scheduler.step(9)
        lr_10 = scheduler.step(10)
        lr_99 = scheduler.step(99)
        
        success = success and print_result(
            lr_0 < lr_9 < lr_10,
            f"Warmup increases LR: {lr_0:.6f} < {lr_9:.6f} < {lr_10:.6f}"
        )
        
        success = success and print_result(
            lr_99 < lr_10,
            f"Cosine decreases LR: {lr_99:.6f} < {lr_10:.6f}"
        )
        
        # Test WarmupPolynomialScheduler
        scheduler_poly = WarmupPolynomialScheduler(
            optimizer, warmup_epochs=10, total_epochs=100, base_lr=0.01
        )
        
        lr_poly_99 = scheduler_poly.step(99)
        success = success and print_result(
            lr_poly_99 > 0,
            f"Polynomial scheduler works: LR at epoch 99 = {lr_poly_99:.6f}"
        )
        
        return success
        
    except Exception as e:
        print_result(False, f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_deep_supervision_module():
    """Test 4: check the Deep Supervision module."""
    print_header("TEST 4: Deep Supervision Module")
    
    try:
        from deep_supervision import (
            DeepSupervisionLoss,
            DeepSupervisionDiceCELoss,
            compute_ds_weights
        )
        
        # Test compute_ds_weights
        weights_exp = compute_ds_weights(4, strategy='exponential')
        weights_linear = compute_ds_weights(4, strategy='linear')
        
        success = print_result(
            len(weights_exp) == 4,
            f"Exponential weights: {weights_exp}"
        )
        
        success = success and print_result(
            len(weights_linear) == 4,
            f"Linear weights: {weights_linear}"
        )
        
        # Test DeepSupervisionDiceCELoss
        ds_loss = DeepSupervisionDiceCELoss(
            n_classes=NUM_CLASSES,
            weights=(1.0, 0.4, 0.2, 0.1)
        )
        
        outputs = [
            torch.randn(BATCH_SIZE, NUM_CLASSES, IMG_SIZE, IMG_SIZE),
            torch.randn(BATCH_SIZE, NUM_CLASSES, IMG_SIZE//2, IMG_SIZE//2),
            torch.randn(BATCH_SIZE, NUM_CLASSES, IMG_SIZE//4, IMG_SIZE//4),
            torch.randn(BATCH_SIZE, NUM_CLASSES, IMG_SIZE//8, IMG_SIZE//8),
        ]
        target = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, IMG_SIZE, IMG_SIZE))
        
        loss, loss_dict = ds_loss(outputs, target)
        
        success = success and print_result(
            isinstance(loss, torch.Tensor),
            f"DS Dice+CE loss: {loss.item():.4f}"
        )
        
        success = success and print_result(
            'scale0' in loss_dict or 'scale0_dice' in loss_dict,
            "Loss dict contains per-scale losses"
        )
        
        return success
        
    except Exception as e:
        print_result(False, f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_layer_wise_lr():
    """Test 5: check the layer-wise learning rate."""
    print_header("TEST 5: Layer-wise Learning Rate")
    
    try:
        from trainer_phase3 import get_layer_wise_lr_params
        from networks.vit_seg_modeling_phase3 import create_model_with_ds
        
        # Create the model
        model = create_model_with_ds(
            config_name='R50-ViT-B_16',
            img_size=IMG_SIZE,
            num_classes=NUM_CLASSES,
            use_cbam=True,
            mlp_type='bilstm',
            use_deep_supervision=True
        )
        
        # Test layer-wise LR
        param_groups = get_layer_wise_lr_params(model, base_lr=0.01, decay=0.9)
        
        success = print_result(
            len(param_groups) > 1,
            f"Created {len(param_groups)} parameter groups"
        )
        
        # Check that different groups get different LRs
        lrs = [pg['lr'] for pg in param_groups]
        unique_lrs = len(set(lrs))
        
        success = success and print_result(
            unique_lrs > 1,
            f"Different LRs for different layers: {unique_lrs} unique LRs"
        )
        
        # Print the details
        print("\n     Parameter groups:")
        for i, pg in enumerate(param_groups[:5]):  # Print only the first 5
            depth = pg.get('depth', 'N/A')
            num_params = sum(p.numel() for p in pg['params'])
            print(f"       Group {i}: depth={depth}, LR={pg['lr']:.6f}, params={num_params:,}")
        
        if len(param_groups) > 5:
            print(f"       ... and {len(param_groups) - 5} more groups")
        
        return success
        
    except Exception as e:
        print_result(False, f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_model_trainer_integration():
    """Test 6: check the integration between model and trainer."""
    print_header("TEST 6: Model-Trainer Integration")
    
    try:
        from networks.vit_seg_modeling_phase3 import create_model_with_ds
        from trainer_phase3 import Phase3CombinedLoss, PHASE3_CONFIG
        
        # Create the model with DS
        model = create_model_with_ds(
            config_name='R50-ViT-B_16',
            img_size=IMG_SIZE,
            num_classes=NUM_CLASSES,
            use_cbam=True,
            mlp_type='bilstm',
            use_deep_supervision=True,
            num_ds_outputs=4
        ).to(DEVICE)
        
        # Create the loss function
        criterion = Phase3CombinedLoss(
            n_classes=NUM_CLASSES,
            ds_weights=PHASE3_CONFIG['ds_weights'],
            use_boundary=False,
            use_hd=False
        )
        
        # Create the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # Simulate a single training step
        model.train()
        
        x = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        target = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, IMG_SIZE, IMG_SIZE)).to(DEVICE)
        
        # Forward
        outputs = model(x)
        
        success = print_result(
            isinstance(outputs, list) and len(outputs) == 4,
            f"Model forward with DS: {len(outputs)} outputs"
        )
        
        # Loss computation
        loss, loss_dict = criterion(outputs, target, epoch=0)
        
        success = success and print_result(
            loss.requires_grad,
            f"Loss requires grad: {loss.requires_grad}"
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients exist
        has_grad = any(p.grad is not None for p in model.parameters())
        success = success and print_result(has_grad, "Gradients computed successfully")
        
        # Optimizer step
        optimizer.step()
        
        success = success and print_result(True, "Optimizer step completed")
        
        # Test disabling DS for inference
        model.disable_deep_supervision()
        criterion.disable_deep_supervision()
        
        model.eval()
        with torch.no_grad():
            output_inference = model(x)
        
        success = success and print_result(
            isinstance(output_inference, torch.Tensor),
            f"Inference (DS disabled): single output shape {output_inference.shape}"
        )
        
        return success
        
    except Exception as e:
        print_result(False, f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test 7: check gradient flow with Deep Supervision."""
    print_header("TEST 7: Deep Supervision Gradient Flow")
    
    try:
        from networks.vit_seg_modeling_phase3 import create_model_with_ds
        from trainer_phase3 import Phase3CombinedLoss
        
        model = create_model_with_ds(
            config_name='R50-ViT-B_16',
            img_size=IMG_SIZE,
            num_classes=NUM_CLASSES,
            use_cbam=True,
            mlp_type='bilstm',
            use_deep_supervision=True
        ).to(DEVICE)
        
        criterion = Phase3CombinedLoss(
            n_classes=NUM_CLASSES,
            ds_weights=(1.0, 0.4, 0.2, 0.1),
            use_boundary=False,
            use_hd=False
        )
        
        model.train()
        
        x = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        target = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, IMG_SIZE, IMG_SIZE)).to(DEVICE)
        
        # Forward and backward
        outputs = model(x)
        loss, _ = criterion(outputs, target, epoch=0)
        loss.backward()
        
        # Check the gradients at the DS heads
        success = True
        grad_info = []
        
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'ds_heads'):
            for i, head in enumerate(model.decoder.ds_heads):
                for name, param in head.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_info.append((f"ds_head_{i}.{name}", grad_norm))
                    else:
                        success = False
                        grad_info.append((f"ds_head_{i}.{name}", "NO GRAD"))
        
        success = success and print_result(
            len(grad_info) > 0,
            f"Gradient computed for {len(grad_info)} DS head parameters"
        )
        
        # Print a few gradients
        print("\n     Sample gradients:")
        for name, grad in grad_info[:3]:
            print(f"       {name}: {grad:.6f}" if isinstance(grad, float) else f"       {name}: {grad}")
        
        # Check the encoder gradients
        encoder_grads = []
        for name, param in model.named_parameters():
            if 'encoder' in name and param.grad is not None:
                encoder_grads.append(param.grad.norm().item())
        
        success = success and print_result(
            len(encoder_grads) > 0,
            f"Encoder receives gradients: {len(encoder_grads)} params with grad"
        )
        
        if encoder_grads:
            print(f"       Encoder grad norm range: [{min(encoder_grads):.6f}, {max(encoder_grads):.6f}]")
        
        return success
        
    except Exception as e:
        print_result(False, f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  SeqAtt-UNet PHASE 3 - INTEGRATION TEST")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Classes: {NUM_CLASSES}")
    
    tests = [
        ("VisionTransformerDS Model", test_visiontransformer_ds),
        ("Phase3CombinedLoss", test_phase3_combined_loss),
        ("Auto-Config Module", test_auto_config),
        ("Deep Supervision Module", test_deep_supervision_module),
        ("Layer-wise LR", test_layer_wise_lr),
        ("Model-Trainer Integration", test_model_trainer_integration),
        ("Gradient Flow", test_gradient_flow),
    ]
    
    results = []
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print("\n" + "-" * 70)
    print(f"  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! Phase 3 integration is ready.")
        print("  You can now run: python train_phase3.py --dataset Synapse --deep_supervision")
    else:
        print(f"\n  ⚠️ {total - passed} test(s) failed. Please review the errors above.")
    
    print("=" * 70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
