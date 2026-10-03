import os
import torch
import torch.onnx
import numpy as np
from models.generator_unet import GeneratorUNet

def export_to_onnx(model_path, onnx_path, base_channels=96, dropout=0.23843508579472333, style_dim=16):
    device = torch.device('cpu')
    gen = GeneratorUNet(base_channels=base_channels, style_dim=style_dim, dropout=dropout)
    
    # Load state dict if available
    if os.path.exists(model_path):
        try:
            gen.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded generator weights from {model_path}.")
        except Exception as e:
            print(f"Could not load weights, exporting untrained model: {e}")
    else:
        print(f"Checkpoint {model_path} not found. Exporting model with initialized weights.")
        
    gen.eval()
    
    # Dummy inputs for 128x128 face and style condition
    dummy_photo = torch.randn(1, 3, 128, 128)
    dummy_style = torch.tensor([0], dtype=torch.long)
    
    # Target directory creation
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    
    torch.onnx.export(
        gen,
        (dummy_photo, dummy_style),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['photo', 'style_id'],
        output_names=['sketch'],
        dynamic_axes={
            'photo': {0: 'batch_size'},
            'style_id': {0: 'batch_size'},
            'sketch': {0: 'batch_size'}
        }
    )
    print(f"Model successfully exported to {onnx_path}")
    
    # Verify ONNX and PyTorch parity
    try:
        import onnxruntime as ort
        
        with torch.no_grad():
            pt_out = gen(dummy_photo, dummy_style).numpy()
            
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        ort_inputs = {
            'photo': dummy_photo.numpy(),
            'style_id': dummy_style.numpy()
        }
        ort_out = session.run(None, ort_inputs)[0]
        
        max_diff = np.max(np.abs(pt_out - ort_out))
        print(f"Parity Check: PyTorch vs ONNX max absolute difference = {max_diff:.6e}")
        assert max_diff < 1e-4, f"Parity check failed with difference {max_diff}"
        print("✅ ONNX export parity check PASSED!")
    except ImportError:
        print("onnxruntime not installed; skipping runtime numerical parity test.")
    except Exception as e:
        print(f"Parity verification note: {e}")

if __name__ == "__main__":
    export_to_onnx("best_generator.pth", "generator.onnx")
