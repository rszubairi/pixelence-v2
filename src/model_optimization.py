import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import os
import time
from pathlib import Path
import json

# Import the models
from .model import UNet3D_Deep_Supervision_attention_cbam

class ModelOptimizer:
    """
    Comprehensive model optimization for inference including:
    - TorchScript conversion
    - ONNX export
    - Quantization
    - JIT optimization
    - Memory optimization
    """
    
    def __init__(self, model, model_name="model"):
        self.model = model
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def optimize_for_inference(self, sample_input=None, save_dir="optimized_models"):
        """
        Complete optimization pipeline for inference
        """
        os.makedirs(save_dir, exist_ok=True)
        optimization_results = {}
        
        print(f"🚀 Starting optimization for {self.model_name}")
        print("=" * 60)
        
        # 1. Prepare model for inference
        self.model.eval()
        
        # 2. Create sample input if not provided
        if sample_input is None:
            sample_input = self._create_sample_input()
            
        # 3. Benchmark original model
        original_time = self._benchmark_model(self.model, sample_input)
        optimization_results['original_inference_time'] = original_time
        
        # 4. TorchScript optimization
        try:
            torchscript_model, ts_time = self._optimize_torchscript(sample_input, save_dir)
            optimization_results['torchscript_inference_time'] = ts_time
            optimization_results['torchscript_speedup'] = original_time / ts_time if ts_time > 0 else 0
            print(f"✅ TorchScript: {ts_time:.4f}s ({optimization_results['torchscript_speedup']:.2f}x speedup)")
        except Exception as e:
            print(f"❌ TorchScript failed: {e}")
            optimization_results['torchscript_error'] = str(e)
        
        # 5. ONNX export
        try:
            onnx_time = self._optimize_onnx(sample_input, save_dir)
            optimization_results['onnx_export_time'] = onnx_time
            print(f"✅ ONNX export completed in {onnx_time:.4f}s")
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            optimization_results['onnx_error'] = str(e)
        
        # 6. Quantization (if applicable)
        try:
            quant_model, quant_time = self._optimize_quantization(sample_input, save_dir)
            optimization_results['quantized_inference_time'] = quant_time
            optimization_results['quantization_speedup'] = original_time / quant_time if quant_time > 0 else 0
            print(f"✅ Quantization: {quant_time:.4f}s ({optimization_results['quantization_speedup']:.2f}x speedup)")
        except Exception as e:
            print(f"❌ Quantization failed: {e}")
            optimization_results['quantization_error'] = str(e)
        
        # 7. Memory optimization
        memory_stats = self._optimize_memory()
        optimization_results.update(memory_stats)
        
        # 8. Save optimization report
        self._save_optimization_report(optimization_results, save_dir)
        
        print("\n" + "=" * 60)
        print("🎯 OPTIMIZATION COMPLETE")
        print(f"📊 Results saved in: {save_dir}")
        
        return optimization_results
    
    def _create_sample_input(self):
        """Create appropriate sample input based on model type"""
        if hasattr(self.model, 'in_channels'):
            # For 3D UNet model
            if hasattr(self.model, 'encoder1'):  # UNet3D model
                return torch.randn(1, 3, 64, 64, 32).to(self.device)
            else:
                return torch.randn(1, self.model.in_channels, 256, 256).to(self.device)
        else:
            # Default fallback
            return torch.randn(1, 3, 256, 256).to(self.device)
    
    def _benchmark_model(self, model, sample_input, num_runs=50):
        """Benchmark model inference time"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(sample_input)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(sample_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        return avg_time
    
    def _optimize_torchscript(self, sample_input, save_dir):
        """Convert model to TorchScript"""
        print("🔄 Optimizing with TorchScript...")
        
        try:
            # Method 1: torch.jit.trace (recommended for most cases)
            traced_model = torch.jit.trace(self.model, sample_input)
            
            # Optimize the traced model
            traced_model = torch.jit.optimize_for_inference(traced_model)
            
            # Save TorchScript model
            ts_path = os.path.join(save_dir, f"{self.model_name}_torchscript.pt")
            traced_model.save(ts_path)
            
            # Benchmark TorchScript model
            ts_time = self._benchmark_model(traced_model, sample_input)
            
            return traced_model, ts_time
            
        except Exception as e:
            print(f"Tracing failed, trying scripting: {e}")
            
            # Method 2: torch.jit.script (fallback)
            try:
                scripted_model = torch.jit.script(self.model)
                scripted_model = torch.jit.optimize_for_inference(scripted_model)
                
                ts_path = os.path.join(save_dir, f"{self.model_name}_torchscript_scripted.pt")
                scripted_model.save(ts_path)
                
                ts_time = self._benchmark_model(scripted_model, sample_input)
                return scripted_model, ts_time
                
            except Exception as e2:
                raise Exception(f"Both tracing and scripting failed: {e2}")
    
    def _optimize_onnx(self, sample_input, save_dir):
        """Export model to ONNX format"""
        print("🔄 Exporting to ONNX...")
        
        start_time = time.time()
        
        onnx_path = os.path.join(save_dir, f"{self.model_name}.onnx")
        
        # ONNX export parameters
        input_names = ['input']
        output_names = ['output']
        
        # Dynamic axes for flexible input sizes (optional)
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            sample_input,
            onnx_path,
            export_params=True,
            opset_version=11,  # Use stable opset version
            do_constant_folding=True,  # Optimize constant operations
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        export_time = time.time() - start_time
        
        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"✅ ONNX model verified successfully")
        except ImportError:
            print("⚠️  ONNX package not available for verification")
        except Exception as e:
            print(f"⚠️  ONNX verification failed: {e}")
        
        return export_time
    
    def _optimize_quantization(self, sample_input, save_dir):
        """Apply dynamic quantization"""
        print("🔄 Applying quantization...")
        
        # Dynamic quantization (works for most models)
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d, nn.Conv3d},  # Layers to quantize
            dtype=torch.qint8
        )
        
        # Save quantized model
        quant_path = os.path.join(save_dir, f"{self.model_name}_quantized.pt")
        torch.save(quantized_model.state_dict(), quant_path)
        
        # Benchmark quantized model
        quant_time = self._benchmark_model(quantized_model, sample_input)
        
        return quantized_model, quant_time
    
    def _optimize_memory(self):
        """Memory optimization recommendations"""
        print("🔄 Analyzing memory usage...")
        
        memory_stats = {}
        
        if torch.cuda.is_available():
            # GPU memory stats
            memory_stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**2  # MB
            memory_stats['gpu_max_memory_allocated'] = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            print(f"💾 GPU Memory - Allocated: {memory_stats['gpu_memory_allocated']:.1f}MB")
            print(f"💾 GPU Memory - Reserved: {memory_stats['gpu_memory_reserved']:.1f}MB")
        
        # Model parameters count
        param_count = sum(p.numel() for p in self.model.parameters())
        param_size_mb = param_count * 4 / 1024**2  # Assuming float32
        
        memory_stats['parameter_count'] = param_count
        memory_stats['parameter_size_mb'] = param_size_mb
        
        print(f"📊 Model Parameters: {param_count:,} ({param_size_mb:.1f}MB)")
        
        return memory_stats
    
    def _save_optimization_report(self, results, save_dir):
        """Save detailed optimization report"""
        report_path = os.path.join(save_dir, f"{self.model_name}_optimization_report.json")
        
        report = {
            'model_name': self.model_name,
            'optimization_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device),
            'pytorch_version': torch.__version__,
            'results': results,
            'recommendations': self._generate_recommendations(results)
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Optimization report saved: {report_path}")
    
    def _generate_recommendations(self, results):
        """Generate optimization recommendations"""
        recommendations = []
        
        if 'torchscript_speedup' in results and results['torchscript_speedup'] > 1.2:
            recommendations.append("✅ TorchScript provides good speedup - recommended for production")
        
        if 'quantization_speedup' in results and results['quantization_speedup'] > 1.1:
            recommendations.append("✅ Quantization provides speedup with minimal accuracy loss")
        
        if 'onnx_export_time' in results:
            recommendations.append("✅ ONNX export successful - can be used with ONNX Runtime for deployment")
        
        if results.get('parameter_size_mb', 0) > 100:
            recommendations.append("⚠️  Large model size - consider model pruning or distillation")
        
        if results.get('gpu_memory_allocated', 0) > 2000:
            recommendations.append("⚠️  High memory usage - consider gradient checkpointing or mixed precision")
        
        return recommendations


def optimize_3d_unet_model(weight_path=None, save_dir="optimized_models"):
    """
    Optimize the 3D UNet model for inference
    """
    print("🎯 OPTIMIZING 3D UNET MODEL FOR INFERENCE")
    print("=" * 60)
    
    # Initialize model
    model = UNet3D_Deep_Supervision_attention_cbam(in_channels=3, out_channels=1, base_filters=32)
    
    # Load weights if provided
    if weight_path and os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        print(f"✅ Loaded weights from: {weight_path}")
    else:
        print("⚠️  No weights loaded - using random initialization")
    
    # Move to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create optimizer and run optimization
    optimizer = ModelOptimizer(model, "unet3d_deep_supervision_attention_cbam")
    
    # Create sample input for 3D model
    sample_input = torch.randn(1, 3, 64, 64, 32).to(device)
    
    results = optimizer.optimize_for_inference(sample_input, save_dir)
    
    return results


def optimize_conditional_gan_model(model_path=None, save_dir="optimized_models"):
    """
    Optimize the Conditional GAN model for inference
    """
    print("🎯 OPTIMIZING CONDITIONAL GAN MODEL FOR INFERENCE")
    print("=" * 60)
    
    # This would require importing the GAN model from model-0907.ipynb
    # For now, we'll provide the structure
    
    print("⚠️  Conditional GAN optimization requires model import from notebook")
    print("📝 To optimize the GAN model:")
    print("1. Extract ConditionalGenerator class to a separate .py file")
    print("2. Load the trained model weights")
    print("3. Run the optimization pipeline")
    
    return None


def create_inference_optimized_model(
    model_type="3d_unet",
    weight_path=None,
    optimization_level="full",
    save_dir="optimized_models"
):
    """
    Main function to create inference-optimized models
    
    Args:
        model_type: "3d_unet" or "conditional_gan"
        weight_path: Path to model weights
        optimization_level: "basic", "advanced", or "full"
        save_dir: Directory to save optimized models
    """
    
    print("🚀 PIXELENCE MODEL OPTIMIZATION SUITE")
    print("=" * 60)
    print(f"Model Type: {model_type}")
    print(f"Optimization Level: {optimization_level}")
    print(f"Save Directory: {save_dir}")
    print("=" * 60)
    
    if model_type == "3d_unet":
        return optimize_3d_unet_model(weight_path, save_dir)
    elif model_type == "conditional_gan":
        return optimize_conditional_gan_model(weight_path, save_dir)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Utility functions for deployment
def load_optimized_model(model_path, model_type="torchscript"):
    """
    Load optimized model for inference
    """
    if model_type == "torchscript":
        return torch.jit.load(model_path)
    elif model_type == "onnx":
        try:
            import onnxruntime as ort
            return ort.InferenceSession(model_path)
        except ImportError:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")
    else:
        return torch.load(model_path)


def benchmark_optimized_models(optimized_dir="optimized_models"):
    """
    Benchmark all optimized models in directory
    """
    print("📊 BENCHMARKING OPTIMIZED MODELS")
    print("=" * 40)
    
    results = {}
    
    # Find all model files
    model_files = list(Path(optimized_dir).glob("*.pt")) + list(Path(optimized_dir).glob("*.onnx"))
    
    for model_file in model_files:
        print(f"Testing: {model_file.name}")
        # Add benchmarking logic here
        
    return results


if __name__ == "__main__":
    # Example usage
    print("🎯 PIXELENCE MODEL OPTIMIZATION")
    print("Available optimizations:")
    print("1. 3D UNet with Deep Supervision + Attention + CBAM")
    print("2. Conditional GAN for FLAIR enhancement")
    print("\nExample usage:")
    print("python model_optimization.py")
    
    # Run optimization for 3D UNet
    results = create_inference_optimized_model(
        model_type="3d_unet",
        optimization_level="full"
    )
    
    print(f"Optimization completed! Check 'optimized_models' directory for results.")
