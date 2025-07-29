# Safetensor Converter

A robust and comprehensive Python tool to convert various AI model formats to `.safetensors` format with advanced error handling and validation.

## 🚀 Features

- ✅ **Multi-format Support** - Convert `.pth`, `.pt`, `.bin`, `.ckpt` files
- ✅ **Automatic Module Prefix Removal** - Handles `module.` prefixes from DataParallel models
- ✅ **Shared Tensor Detection** - Prevents "Some tensors share memory" errors
- ✅ **Multiple Loading Methods** - Fallback strategies for different file formats
- ✅ **Error Recovery** - Alternative approaches when standard conversion fails
- ✅ **Automatic Validation** - Verify converted files are correct
- ✅ **Batch Processing** - Convert multiple files at once
- ✅ **Comprehensive Error Handling** - Clear error messages and solutions

## 📋 Supported Formats

| Input Format | Description | Status |
|--------------|-------------|--------|
| `.pth` | PyTorch model files | ✅ Fully Supported |
| `.pt` | PyTorch tensor files | ✅ Fully Supported |
| `.bin` | Binary model files | ✅ Fully Supported |
| `.ckpt` | Checkpoint files | ✅ Fully Supported |

## 🛠️ Installation

### Prerequisites

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch safetensors
```

### Download

Download the `safetensor_converter.py` script to your working directory.

## 📖 Usage

### Basic Usage

```bash
# Convert a single file (auto-generated output name)
python safetensor_converter.py model.pth

# Convert with custom output name
python safetensor_converter.py model.pt output.safetensors

# Convert different formats
python safetensor_converter.py model.bin
python safetensor_converter.py checkpoint.ckpt
```

### Batch Conversion

```bash
# Convert all supported files in a directory
python safetensor_converter.py models_folder/ --batch

# Convert with custom output directory
python safetensor_converter.py models_folder/ -o output_folder/ --batch
```

### Examples

```bash
# Convert SAM model
python safetensor_converter.py sam_vit_b_01ec64.pth

# Convert GroundingDINO model
python safetensor_converter.py groundingdino_swint_ogc.pth

# Convert PyTorch tensor file
python safetensor_converter.py model.pt

# Convert checkpoint file
python safetensor_converter.py checkpoint.ckpt

# Batch convert all models in current directory
python safetensor_converter.py . --batch
```

## 🔧 How It Works

### 1. File Detection
- Automatically detects supported file formats
- Validates file existence and permissions

### 2. Loading
- Uses multiple loading methods with fallback strategies
- Handles different PyTorch versions and compatibility issues
- Supports various checkpoint structures

### 3. Data Extraction
- Extracts state dict from different data structures
- Handles `model`, `state_dict`, `weights`, `model_state_dict` keys
- Converts tensors to proper format

### 4. Preprocessing
- Removes `module.` prefixes from DataParallel models
- Makes all tensors contiguous
- Clones shared tensors to ensure independence

### 5. Saving
- Saves to `.safetensors` format
- Uses alternative approach if shared tensor errors occur
- Handles memory and storage issues

### 6. Validation
- Verifies file size and existence
- Checks for remaining `module.` prefixes
- Reports conversion success/failure

## 📊 Output Examples

### Successful Conversion
```
Loading file: model.pth
✓ Loaded using method 1
Found checkpoint format with 'model' key
Removed 'module.' prefix from 940 keys
State dict contains 940 keys
Total parameters: 123,456,789
Sample keys:
  transformer.level_embed: torch.Size([256])
  backbone.0.layers.0.blocks.0.attn.qkv.weight: torch.Size([768, 768])
  ... and 935 more keys
Preparing tensors...
Making tensors contiguous...
Cloning tensors to ensure independence...
Cloned 940 tensors to ensure independence
Saving to safetensors file: model.safetensors
✓ Saved using standard method
✓ File created successfully!
  Output file: model.safetensors
  File size: 456,789,012 bytes (435.67 MB)
  ✓ Verified: No 'module.' prefixes in saved file
✓ Conversion completed successfully!
```

### Error Recovery
```
Error: Shared tensors detected. Trying alternative approach...
✓ Saved using alternative approach
✓ File created successfully!
```

## 🚨 Common Issues and Solutions

### "File format not supported" Error
**Cause**: File extension not in supported list
**Solution**: Check file extension is `.pth`, `.pt`, `.bin`, or `.ckpt`

### "All loading methods failed" Error
**Cause**: Corrupted file or incompatible format
**Solution**: Verify file integrity and try with different PyTorch version

### "Some tensors share memory" Error
**Cause**: Model contains tensors that share memory storage
**Solution**: Script automatically detects and handles this

### "module." Prefixes in Output
**Cause**: DataParallel models add `module.` prefixes
**Solution**: Script automatically removes these prefixes

### Large Model Memory Issues
**Cause**: Very large models may exceed available RAM
**Solution**: Script uses CPU loading by default

## 📁 File Structure

### Input Files
- `.pth`, `.pt`, `.bin`, `.ckpt` files
- Can be single files or directories for batch processing

### Output Files
- `.safetensors` files (SafeTensors format)
- Same name as input file with `.safetensors` extension
- Automatically placed in same directory unless specified otherwise

## 🎯 Command Line Options

```bash
python safetensor_converter.py [input] [options]

Arguments:
  input                   Input file or directory

Options:
  -o, --output           Output file or directory
  --batch                Batch convert all supported files in directory
  -h, --help             Show help message
```

## 🔍 Advanced Features

### Multiple Loading Methods
The script tries different loading strategies:
1. Standard `torch.load()`
2. With `weights_only=False` for PyTorch 2.6+
3. With custom pickle module for compatibility

### Checkpoint Structure Detection
Automatically detects and handles:
- `model` key structure
- `state_dict` key structure
- `weights` key structure
- `model_state_dict` key structure

### Error Recovery
When standard saving fails:
- Detects shared tensor issues
- Uses numpy-based tensor reconstruction
- Provides detailed error information

## 🛡️ Safety Features

- **File Validation** - Checks file existence and format
- **Error Recovery** - Multiple fallback strategies
- **Verification** - Automatic validation of output files
- **Safe Loading** - Uses CPU loading to prevent GPU memory issues
- **Backup Friendly** - Doesn't modify original files

## 🔧 Troubleshooting

### Script Won't Run
- Ensure Python 3.6+ is installed
- Install required packages: `pip install -r safetensor_converter_requirements.txt`
- Check file permissions

### Conversion Fails
- Verify input file exists and is not corrupted
- Check available disk space
- Ensure sufficient RAM for large models
- Try with different PyTorch version

### Model Doesn't Work After Conversion
- Check verification output for warnings
- Verify file size is reasonable
- Test loading the converted file manually

## 📈 Performance

- **Memory Efficient** - Uses CPU loading by default
- **Fast Processing** - Optimized tensor operations
- **Batch Support** - Efficient batch processing
- **Error Recovery** - Minimal retry overhead

## 🤝 Contributing

If you encounter issues with specific model types:

1. **Report the issue** with:
   - Model name and type
   - File format (.pth, .pt, .bin, .ckpt)
   - Error message
   - Input file structure (first few keys)

2. **Test with different formats** to verify if it's format-specific

3. **Check the verification output** to identify specific problems

## 📄 License

This script is provided as-is for educational and practical use. Feel free to modify and distribute.

## 🙏 Acknowledgments

- Based on PyTorch and SafeTensors libraries
- Tested with SAM, GroundingDINO, and various other models
- Inspired by the need for reliable multi-format model conversion
- Enhanced error handling based on real-world usage scenarios

---

**Note**: Always backup your original model files before conversion. While this script is designed to be safe, it's good practice to keep originals.

## 📦 Files Included

- `safetensor_converter.py` - Main conversion script
- `safetensor_converter_requirements.txt` - Required dependencies
- `README.md` - This documentation
