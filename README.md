# 🎬 SmolVLM Auto Highlighter

An automated video highlight generator that uses a Vision-Language Model (VLM) to analyze video footage and extract the most interesting moments based on visual descriptions.

## 🚀 Features
- **VLM-Powered Analysis**: Uses SmolVLM to "see" and describe video segments.
- **Customizable Targeting**: Set a target duration (e.g., 60 seconds) and the tool will pick the highest-scoring clips to fit that budget.
- **Flexible Chunking**: Adjustable segment lengths and overlaps to capture everything from long fights to split-second glitches.
- **GPU Accelerated**: Optimized for NVIDIA RTX GPUs using CUDA.

## 🛠️ Installation

### 1. Prerequisites
- **Python 3.12** (Crucial: Python 3.13+ is currently unsupported by PyTorch).
- **NVIDIA GPU** with updated drivers.
- **FFmpeg** installed and added to your System PATH.

### 2. Environment Setup
```powershell
# Create a virtual environment using Python 3.12
py -3.12 -m venv .venv

# Activate the environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python.exe -m pip install --upgrade pip
```

### 3. Install Dependencies
To ensure GPU support, install PyTorch first, then the rest of the requirements:

```powershell
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other required libraries
pip install -r requirements.txt

# Install PyAV (Required for video decoding)
pip install av
```

## 🎮 Usage

Place your videos in the `videos/` folder and run the main module:

```powershell
python -m app.main --input videos --output output --segment 4 --overlap 2 --target 60
```

### Argument Guide:
| Argument | Description | Recommended for "Funny" Clips |
| :--- | :--- | :--- |
| `--input` | Folder containing source videos | `videos` |
| `--output` | Folder where highlights are saved | `output` |
| `--segment` | Duration of each analyzed chunk (seconds) | `4` (keeps clips snappy) |
| `--overlap` | Overlap between segments to avoid cutting events | `2` |
| `--target` | Desired total length of final highlight reel | `60` |

## 💡 Tips for Bodycam Games
To capture "funny" moments (glitches, fails, chaos), avoid generic prompts. Instead, modify the prompt in `config.py` or `model.py` to look for:
- **Physical Glitches**: "Characters clipping through walls or unexpected physics."
- **Visual Chaos**: "Rapid camera movement combined with sudden explosions or absurd combat."
- **Fails**: "Missed shots or clumsy character movements."

## ⚠️ Troubleshooting
- **`WinError 127`**: Install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).
- **`PyAV not installed`**: Run `pip install av`.
- **`CUDA not available`**: Ensure you installed the `cu121` version of torch and not the default CPU version.
- **`Out of Memory (OOM)`**: Close background GPU apps like LM Studio or browser tabs before running.