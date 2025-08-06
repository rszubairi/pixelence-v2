#!/usr/bin/env python3
"""
Pixelence Model Optimization Example
====================================

This script demonstrates how to optimize your models for faster inference
using TorchScript, ONNX, and quantization techniques.

Usage:
    python optimize_model_example.py
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model_optimization import (
    create_inference_optimized_model,
    optimize_3d_unet_model,
    load_optimized_model,
    benchmark_optimized_models
)

def main():
    print("🚀 PIXELENCE MODEL OPTIMIZATION DEMO")
    print("=" * 60)
    print("This script will optimize your models for faster inference")
    print("Available optimizations:")
    print("  • TorchScript conversion")
    print("  • ONNX export")
    print("  • Dynamic quantization")
    print("  • Memory optimization")
    print("=" * 60)
    
    # Set up paths
    weight_path = None  # Replace with your actual weight path if available
    # Example: weight_path = "/path/to/your/_global1.pth"
    
    save_dir = "optimized_models"
    
    # Create optimized models directory
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Optimize the 3D UNet model
        print("\n🎯 OPTIMIZING 3D UNET MODEL")
        print("-" * 40)
        
        results = optimize_3d_unet_model(
            weight_path=weight_path,
            save_dir=save_dir
        )
        
        print("\n📊 OPTIMIZATION RESULTS:")
        print("-" * 40)
        
        if results:
            original_time = results.get('original_inference_time', 0)
            print(f"Original model inference time: {original_time:.4f}s")
            
            if 'torchscript_inference_time' in results:
                ts_time = results['torchscript_inference_time']
                speedup = results.get('torchscript_speedup', 1.0)
                print(f"TorchScript inference time: {ts_time:.4f}s ({speedup:.2f}x speedup)")
            
            if 'quantized_inference_time' in results:
                quant_time = results['quantized_inference_time']
                speedup = results.get('quantization_speedup', 1.0)
                print(f"Quantized inference time: {quant_time:.4f}s ({speedup:.2f}x speedup)")
            
            if 'parameter_count' in results:
                param_count = results['parameter_count']
                param_size = results['parameter_size_mb']
                print(f"Model size: {param_count:,} parameters ({param_size:.1f}MB)")
        
        # Demonstrate loading optimized models
        print("\n🔄 TESTING OPTIMIZED MODEL LOADING")
        print("-" * 40)
        
        # Try to load TorchScript model
        ts_path = os.path.join(save_dir, "unet3d_deep_supervision_attention_cbam_torchscript.pt")
        if os.path.exists(ts_path):
            print(f"✅ Loading TorchScript model: {ts_path}")
            ts_model = load_optimized_model(ts_path, "torchscript")
            print("✅ TorchScript model loaded successfully!")
            
            # Test inference
            sample_input = torch.randn(1, 3, 64, 64, 32)
            with torch.no_grad():
                output = ts_model(sample_input)
            print(f"✅ TorchScript inference test: Input {sample_input.shape} → Output {output.shape}")
        
        # Try to load ONNX model
        onnx_path = os.path.join(save_dir, "unet3d_deep_supervision_attention_cbam.onnx")
        if os.path.exists(onnx_path):
            print(f"✅ ONNX model exported: {onnx_path}")
            print("💡 To use ONNX model:")
            print("   import onnxruntime as ort")
            print("   session = ort.InferenceSession('model.onnx')")
            print("   output = session.run(None, {'input': input_array})")
        
        print("\n📁 GENERATED FILES:")
        print("-" * 40)
        if os.path.exists(save_dir):
            for file in os.listdir(save_dir):
                file_path = os.path.join(save_dir, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  {file:<40} ({size_mb:.1f}MB)")
        
        print("\n🎯 OPTIMIZATION RECOMMENDATIONS:")
        print("-" * 40)
        if results and 'recommendations' in results:
            for rec in results['recommendations']:
                print(f"  {rec}")
        else:
            print("  📝 Check the optimization report for detailed recommendations")
        
        print("\n🚀 NEXT STEPS:")
        print("-" * 40)
        print("1. Use TorchScript model for production deployment")
        print("2. Use ONNX model for cross-platform inference")
        print("3. Consider quantized model for resource-constrained environments")
        print("4. Benchmark models with your actual data")
        print("5. Test accuracy on validation set before deployment")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        print("\n🛠️  TROUBLESHOOTING:")
        print("1. Ensure PyTorch is installed: pip install torch")
        print("2. Install ONNX packages: pip install onnx onnxruntime")
        print("3. Check if CUDA is available for GPU optimization")
        print("4. Verify model weights path if using pre-trained model")
    
    print("\n" + "=" * 60)
    print("🏁 OPTIMIZATION DEMO COMPLETE")
    print("Check the 'optimized_models' directory for all generated files")

def demonstrate_inference_comparison():
    """
    Demonstrate inference speed comparison between different model formats
    """
    print("\n🏃 INFERENCE SPEED COMPARISON")
    print("-" * 40)
    
    save_dir = "optimized_models"
    sample_input = torch.randn(1, 3, 64, 64, 32)
    
    # Original PyTorch model
    try:
        from src.model import UNet3D_Deep_Supervision_attention_cbam
        original_model = UNet3D_Deep_Supervision_attention_cbam()
        original_model.eval()
        
        # Benchmark original model
        import time
        with torch.no_grad():
            start = time.time()
            for _ in range(10):
                _ = original_model(sample_input)
            original_time = (time.time() - start) / 10
        
        print(f"Original PyTorch model: {original_time:.4f}s per inference")
        
    except Exception as e:
        print(f"Could not benchmark original model: {e}")
    
    # TorchScript model
    ts_path = os.path.join(save_dir, "unet3d_deep_supervision_attention_cbam_torchscript.pt")
    if os.path.exists(ts_path):
        try:
            ts_model = torch.jit.load(ts_path)
            ts_model.eval()
            
            with torch.no_grad():
                start = time.time()
                for _ in range(10):
                    _ = ts_model(sample_input)
                ts_time = (time.time() - start) / 10
            
            print(f"TorchScript model: {ts_time:.4f}s per inference")
            
        except Exception as e:
            print(f"Could not benchmark TorchScript model: {e}")

if __name__ == "__main__":
    main()
    demonstrate_inference_comparison()
