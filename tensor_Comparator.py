import os
import sys
import torch
from safetensors.torch import load_file
import argparse

def load_file_data(file_path):
    """
    Loads data from a file, supporting both PyTorch (.pth, .pt) and SafeTensors (.safetensors) formats.
    It includes logic to handle raw single-tensor files by wrapping them in a dictionary using the 
    filename as the key, mirroring the behavior of the safetensor_converter.py.
    """
    
    file_path = os.path.abspath(file_path)
    print(f"Loading data from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    
    ext = os.path.splitext(file_path.lower())[1]
    
    try:
        if ext == '.safetensors':
            # Use safetensors load for new format
            data = load_file(file_path)
            print("Successfully loaded as SafeTensors file.")
        elif ext in ['.pth', '.pt', '.bin', '.ckpt']:
            # Use torch load for old formats
            data = torch.load(file_path, map_location="cpu")
            
            # 1. Handle standard checkpoint formats
            if isinstance(data, dict) and any(k in data for k in ['model', 'state_dict', 'weights', 'model_state_dict']):
                if 'model' in data:
                    data = data['model']
                elif 'state_dict' in data:
                    data = data['state_dict']
                elif 'weights' in data:
                    data = data['weights']
                elif 'model_state_dict' in data:
                    data = data['model_state_dict']

            # 2. FIX: Handle raw single-tensor files (like posi_prompt.pth)
            elif isinstance(data, torch.Tensor):
                # If the data is a single tensor, wrap it in a dictionary using the filename as the key.
                file_base_name = os.path.splitext(os.path.basename(file_path))[0]
                # Use filename as key, falling back to "model" if needed (universal approach)
                key_name = file_base_name if file_base_name else "model" 
                
                print(f"  (Comparator Note: Wrapped raw tensor with key '{key_name}' for comparison)")
                data = {key_name: data}
                
            print("Successfully loaded as PyTorch/Checkpoint file.")
        else:
            print(f"Error: Unsupported file extension {ext}. Only .pth, .pt, .bin, .ckpt, and .safetensors are supported.")
            return None
            
        return data

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compare_state_dicts(old_sd, new_sd):
    """Compares two state dictionaries (old model vs. new safetensors) for consistency."""
    
    print("\n--- Starting Tensor Comparison ---")
    
    # 1. Compare Key Sets
    old_keys = set(old_sd.keys())
    new_keys = set(new_sd.keys())
    
    if old_keys != new_keys:
        missing_in_new = old_keys - new_keys
        extra_in_new = new_keys - old_keys
        
        if missing_in_new:
            print(f"FAILURE: {len(missing_in_new)} keys from OLD file are missing in NEW file.")
            print(f"  Missing keys sample: {list(missing_in_new)[:5]}")
        if extra_in_new:
            print(f"FAILURE: {len(extra_in_new)} keys in NEW file are not present in OLD file.")
            print(f"  Extra keys sample: {list(extra_in_new)[:5]}")
        
        print("Comparison stopped: Key sets do not match.")
        return False
    
    print(f"✓ Key Sets Match: Both files have {len(old_keys)} tensors with identical names.")
    
    # 2. Compare Tensors (Shape and Numerical Consistency)
    mismatch_count = 0
    
    for key in old_keys:
        old_tensor = old_sd[key]
        new_tensor = new_sd[key]
        
        # Check if they are tensors (skip metadata)
        if not isinstance(old_tensor, torch.Tensor) or not isinstance(new_tensor, torch.Tensor):
            continue

        # Check Shape
        if old_tensor.shape != new_tensor.shape:
            print(f"FAIL: Tensor '{key}' has mismatched shapes: OLD={old_tensor.shape}, NEW={new_tensor.shape}")
            mismatch_count += 1
            continue

        # Check Numerical Consistency (using allclose for robust float comparison)
        # Using default tolerances (rtol=1e-05, atol=1e-08) which are standard for model weight comparison.
        if not torch.allclose(old_tensor.cpu(), new_tensor.cpu()):
            print(f"FAIL: Tensor '{key}' has mismatched numerical values (outside standard tolerance).")
            mismatch_count += 1
            # Optional: Add mean/max diff debugging here if needed
            # diff = torch.abs(old_tensor - new_tensor).max().item()
            # print(f"  Max Absolute Difference: {diff}")

    if mismatch_count == 0:
        print("\nSUCCESS: All tensors match in shape and are numerically consistent (within floating point tolerance).")
        return True
    else:
        print(f"\nFAILURE: Total of {mismatch_count} tensors failed the shape or numerical consistency check.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Tensor Comparator - Compares the state dictionaries of two model files (PyTorch/SafeTensors) for 100% consistency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  python tensor_comparator.py <original_model.pth> <converted_model.safetensors>
        """
    )
    
    parser.add_argument("original_file", help="Path to the original model file (.pth, .pt, .bin, .ckpt)")
    parser.add_argument("converted_file", help="Path to the converted model file (.safetensors)")
    
    args = parser.parse_args()
    
    old_sd = load_file_data(args.original_file)
    new_sd = load_file_data(args.converted_file)
    
    if old_sd is None or new_sd is None:
        sys.exit(1)
        
    # The converter script now handles cleaning, so we rely on exact key matching here.
    
    is_consistent = compare_state_dicts(old_sd, new_sd)
    
    if is_consistent:
        print("\n--- VERIFICATION SUCCESSFUL ---")
        print("The converted SafeTensors file is 100% consistent with the original file.")
        sys.exit(0)
    else:
        print("\n--- VERIFICATION FAILED ---")
        print("The converted SafeTensors file is NOT consistent with the original file.")
        sys.exit(1)

if __name__ == "__main__":
    # If no arguments provided, show usage
    if len(sys.argv) < 3:
        main()
    else:
        main()
