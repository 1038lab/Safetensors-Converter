#!/usr/bin/env python3
"""
Safetensor Converter
A unified tool to convert various AI model formats to .safetensors format

Features:
- Supports multiple input formats: .pth, .pt, .bin, .ckpt
- Automatic module prefix removal
- Shared tensor detection and cloning
- Error recovery mechanisms
- Automatic validation
- Batch processing
- Comprehensive error handling

Usage:
    python safetensor_converter.py <input_file> [output_file]
    python safetensor_converter.py <input_directory> --batch
    python safetensor_converter.py --help
"""

import os
import sys
import torch
from safetensors.torch import save_file, load_file
import argparse
import glob

# Supported file extensions
SUPPORTED_EXTENSIONS = ['.pth', '.pt', '.bin', '.ckpt']

def is_supported_file(file_path):
    """Check if file has a supported extension"""
    _, ext = os.path.splitext(file_path.lower())
    return ext in SUPPORTED_EXTENSIONS

def get_supported_files(directory):
    """Get all supported files in a directory"""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        pattern = os.path.join(directory, f"*{ext}")
        files.extend(glob.glob(pattern))
    return files

def load_model_file(file_path):
    """Load model file with multiple fallback methods"""
    
    print(f"Loading file: {file_path}")
    
    # Try different loading methods
    loading_methods = [
        # Method 1: Standard loading
        lambda: torch.load(file_path, map_location="cpu"),
        # Method 2: With weights_only=False for PyTorch 2.6+
        lambda: torch.load(file_path, map_location="cpu", weights_only=False),
        # Method 3: With pickle_module for compatibility
        lambda: torch.load(file_path, map_location="cpu", pickle_module=torch.serialization._get_default_pickle_module()),
    ]
    
    for i, method in enumerate(loading_methods, 1):
        try:
            data = method()
            print(f"✓ Loaded using method {i}")
            return data
        except Exception as e:
            print(f"  Method {i} failed: {str(e)[:100]}...")
            if i == len(loading_methods):
                print(f"Error: All loading methods failed for {file_path}")
                raise e
    
    return None

def extract_state_dict(data):
    """Extract state dict from various data structures"""
    
    if isinstance(data, dict):
        # Check for common checkpoint structures
        if 'model' in data:
            print("Found checkpoint format with 'model' key")
            return data['model']
        elif 'state_dict' in data:
            print("Found checkpoint format with 'state_dict' key")
            return data['state_dict']
        elif 'weights' in data:
            print("Found checkpoint format with 'weights' key")
            return data['weights']
        elif 'model_state_dict' in data:
            print("Found checkpoint format with 'model_state_dict' key")
            return data['model_state_dict']
        else:
            # Assume it's already a state dict
            print("Using data as state dict")
            return data
    elif isinstance(data, torch.Tensor):
        print("Warning: Data is a tensor, converting to dict format")
        return {"model": data}
    else:
        print(f"Warning: Unknown data type {type(data)}, converting to dict format")
        return {"model": data}

def clean_state_dict(state_dict):
    """Clean state dict by removing module prefixes and handling special cases"""
    
    if not isinstance(state_dict, dict):
        print("Warning: State dict is not a dictionary")
        return state_dict
    
    # Remove module prefix if present (from DataParallel)
    new_state_dict = {}
    module_prefix_count = 0
    
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
            new_state_dict[new_key] = value
            module_prefix_count += 1
        else:
            new_state_dict[key] = value
    
    if module_prefix_count > 0:
        print(f"Removed 'module.' prefix from {module_prefix_count} keys")
    else:
        print("No 'module.' prefixes found")
    
    return new_state_dict

def prepare_tensors(state_dict):
    """Prepare tensors for saving by making them contiguous and independent"""
    
    print("Preparing tensors...")
    
    # Make all tensors contiguous
    print("Making tensors contiguous...")
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            state_dict[key] = value.contiguous()
    
    # Clone all tensors to ensure independence
    print("Cloning tensors to ensure independence...")
    cloned_count = 0
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            state_dict[key] = value.clone()
            cloned_count += 1
    
    print(f"Cloned {cloned_count} tensors to ensure independence")
    return state_dict

def save_safetensors(state_dict, output_file):
    """Save state dict to safetensors format with error recovery"""
    
    print(f"Saving to safetensors file: {output_file}")
    
    try:
        save_file(state_dict, output_file)
        print("✓ Saved using standard method")
        return True
    except RuntimeError as e:
        if "Some tensors share memory" in str(e):
            print("Error: Shared tensors detected. Trying alternative approach...")
            try:
                # Force all tensors to be completely independent
                independent_state_dict = {}
                for key, value in state_dict.items():
                    if isinstance(value, torch.Tensor):
                        # Create completely new tensor
                        independent_state_dict[key] = torch.tensor(
                            value.cpu().numpy(), 
                            dtype=value.dtype, 
                            device=value.device
                        )
                    else:
                        independent_state_dict[key] = value
                
                save_file(independent_state_dict, output_file)
                print("✓ Saved using alternative approach")
                return True
            except Exception as e2:
                print(f"Alternative approach also failed: {e2}")
                return False
        else:
            print(f"Error during save: {e}")
            return False
    except Exception as e:
        print(f"Unexpected error during save: {e}")
        return False

def verify_saved_file(output_file):
    """Verify the saved safetensors file"""
    
    if not os.path.exists(output_file):
        print("Error: Output file was not created!")
        return False
    
    file_size = os.path.getsize(output_file)
    print(f"✓ File created successfully!")
    print(f"  Output file: {output_file}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    # Verify the saved file structure
    try:
        verify_data = load_file(output_file)
        module_keys = [k for k in verify_data.keys() if k.startswith('module.')]
        if module_keys:
            print(f"  WARNING: Saved file still has {len(module_keys)} keys with 'module.' prefix")
            print(f"  This may cause loading issues!")
            return False
        else:
            print(f"  ✓ Verified: No 'module.' prefixes in saved file")
            return True
    except Exception as e:
        print(f"  WARNING: Could not verify saved file: {e}")
        return False

def convert_file(input_file, output_file=None):
    """Convert a single file to safetensors format"""
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist!")
        return False
    
    # Check if file format is supported
    if not is_supported_file(input_file):
        print(f"Error: File format not supported. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        return False
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.safetensors"
    
    try:
        # Load the model file
        data = load_model_file(input_file)
        
        # Extract state dict
        state_dict = extract_state_dict(data)
        
        # Clean state dict
        state_dict = clean_state_dict(state_dict)
        
        # Show information
        print(f"State dict contains {len(state_dict)} keys")
        total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
        print(f"Total parameters: {total_params:,}")
        
        # Show sample keys
        print("Sample keys:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            if hasattr(state_dict[key], 'shape'):
                print(f"  {key}: {state_dict[key].shape}")
            else:
                print(f"  {key}: {type(state_dict[key])}")
        
        if len(state_dict) > 5:
            print(f"  ... and {len(state_dict) - 5} more keys")
        
        # Prepare tensors
        state_dict = prepare_tensors(state_dict)
        
        # Save to safetensors
        if not save_safetensors(state_dict, output_file):
            return False
        
        # Verify the saved file
        if not verify_saved_file(output_file):
            return False
        
        print("✓ Conversion completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

def batch_convert(input_dir, output_dir=None):
    """Batch convert all supported files in a directory"""
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return
    
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all supported files
    supported_files = get_supported_files(input_dir)
    
    if not supported_files:
        print(f"No supported files found in '{input_dir}'")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        return
    
    print(f"Found {len(supported_files)} supported files to convert:")
    for f in supported_files:
        print(f"  - {f}")
    
    print("\nStarting batch conversion...")
    
    success_count = 0
    for file_path in supported_files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.safetensors')
        
        print(f"\n--- Converting {filename} ---")
        if convert_file(file_path, output_path):
            success_count += 1
    
    print(f"\nBatch conversion complete: {success_count}/{len(supported_files)} files converted successfully")

def main():
    parser = argparse.ArgumentParser(
        description="Safetensor Converter - Convert various model formats to .safetensors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python safetensor_converter.py model.pth
  python safetensor_converter.py model.pt output.safetensors
  python safetensor_converter.py model.bin
  python safetensor_converter.py checkpoint.ckpt
  python safetensor_converter.py models_folder/ --batch
  python safetensor_converter.py models_folder/ -o output_folder/ --batch

Supported formats: .pth, .pt, .bin, .ckpt
        """
    )
    
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("--batch", action="store_true", help="Batch convert all supported files in directory")
    
    args = parser.parse_args()
    
    if args.batch or os.path.isdir(args.input):
        # Batch mode
        batch_convert(args.input, args.output)
    else:
        # Single file mode
        success = convert_file(args.input, args.output)
        if success:
            print("\n✓ Conversion completed successfully!")
        else:
            print("\n✗ Conversion failed!")
            sys.exit(1)

if __name__ == "__main__":
    # If no arguments provided, show usage
    if len(sys.argv) == 1:
        print(__doc__)
        print("\nExamples:")
        print("  python safetensor_converter.py model.pth")
        print("  python safetensor_converter.py model.pt output.safetensors")
        print("  python safetensor_converter.py models_folder/ --batch")
        print(f"\nSupported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(0)
    
    main() 