import torch
import openvino as ov
from kokoro import KPipeline
import numpy as np
import time
from pathlib import Path

# POC: PyTorch to OpenVINO conversion for Kokoro
# Based on instructions, this script aims to convert the original 
# PyTorch Kokoro model to OpenVINO IR directly to avoid ONNX Sequence issues.

def run_poc():
    print("--- Starting Kokoro PyTorch to OpenVINO POC ---")
    
    # 1. Load the model from the official library
    print("Loading Kokoro model from PyTorch...")
    # 'a' lang_code for English (US)
    pipeline = KPipeline(lang_code='a') 
    
    # Access the internal model
    model = pipeline.model
    model.eval()
    
    print("Model loaded successfully.")
    
    # 2. Define dummy input to trace the model.
    print("Tracing model...")
    # Based on Kokoro source, it expects: tokens, style, speed
    # KPipeline.model usually has a .forward(tokens, style, speed) method
    
    # Example dummy inputs
    tokens = torch.randint(0, 100, (1, 50))
    style = torch.randn(1, 256) # Adjust based on actual style vector shape
    speed = torch.tensor([1.0])
    
    try:
        # 3. Attempt direct OpenVINO conversion via ov.convert_model
        print("Converting model to OpenVINO...")
        
        # We use example_inputs to guide the conversion
        ov_model = ov.convert_model(model, example_input=(tokens, style, speed))
        print("Conversion successful!")
        
        # 4. Save IR
        print("Saving OpenVINO IR...")
        ov.save_model(ov_model, "kokoro_openvino.xml")
        print("Model saved as kokoro_openvino.xml and kokoro_openvino.bin")
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        with open("poc_failure.log", "w") as f:
            traceback.print_exc(file=f)
        print("Traceback saved to poc_failure.log")
        return

if __name__ == "__main__":
    run_poc()
