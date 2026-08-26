# 🎬 SmolVLM Auto Highlighter

An automated video highlight generator that uses a Vision-Language Model (VLM) to analyze video footage and extract the most interesting moments based on visual descriptions.

## 🚀 Features
- **VLM-Powered Analysis**: Uses SmolVLM2-2.2B-Instruct to "see" and describe video segments.
- **Customizable Targeting**: Set a target duration (e.g., 60 seconds) and the tool will pick the highest-scoring clips to fit that budget.
- **Flexible Chunking**: Adjustable segment lengths and overlaps to capture everything from long fights to split-second glitches.
- **GPU Accelerated**: Optimized for NVIDIA RTX GPUs using CUDA.
- **Frame-based Decoding**: Uses PyAV to bypass torchcodec DLL issues on Windows.

## 🛠️ Installation

### 1. Prerequisites
- **Python 3.12** (Crucial: Python 3.13+ is currently unsupported by PyTorch).
- **NVIDIA GPU** with updated drivers.
- **FFmpeg** installed and added to your System PATH.
- **Microsoft Visual C++ Redistributable** (download from Microsoft).

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

**Important**: PyTorch must be installed with CUDA support FIRST, otherwise you'll get the CPU-only version.

```powershell
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyAV for video frame decoding (required for model.py fix)
pip install av

# Install other required libraries
pip install transformers accelerate safetensors pillow numpy tqdm
```

## 🎮 Usage

Place your videos in the `videos/` folder and run the main module:

### Recommended for Funny/Highlight Reels:
```powershell
python -m app.main --input videos --output output --segment 4 --overlap 2 --target 60 --min-score 3.0
```

### Standard Usage:
```powershell
python -m app.main --input videos --output output --segment 15 --overlap 3 --target 180
```

### Argument Guide:
| Argument | Description | Default | Recommended for "Funny" Clips |
|----------|-------------|---------|-------------------------------|
| `--input` | Folder containing source videos | `videos` | `videos` |
| `--output` | Folder where highlights are saved | `output` | `output` |
| `--segment` | Duration of each analyzed chunk (seconds) | `15.0` | `4` (keeps clips snappy) |
| `--overlap` | Overlap between segments | `3.0` | `2` |
| `--target` | Desired total length of final highlight reel | `180.0` | `60` |
| `--min-score` | Minimum score for clip selection | `6.5` | `3.0` |
| `--max-clips` | Maximum number of clips | `25` | `25` |
| `--keep-chunks` | Keep chunk files after processing | `False` | - |
| `--no-render` | Skip rendering final video | `False` | - |

## 📁 Output Structure

After running, your `output/` folder will contain:
- `highlight.mp4` — The final compiled highlight reel
- `edit_decision_list.json` — JSON list of all selected clips
- `highlights.json` — Same as EDL (kept for compatibility)
- `analysis/` — Per-video JSON files with detailed segment scores
- `work/chunks/` — Temporary video chunks (auto-deleted unless `--keep-chunks`)

## 💡 Tips for Bodycam Games

To capture "funny" moments (glitches, fails, chaos), modify the prompt in `app/prompts.py`:

- **Physical Glitches**: "Characters clipping through walls or unexpected physics."
- **Visual Chaos**: "Rapid camera movement combined with sudden explosions or absurd combat."
- **Fails**: "Missed shots or clumsy character movements."

The default prompt in `prompts.py` is already optimized for this — it prioritizes:
- Clear exciting action
- Funny or surprising events
- Strong reactions
- Visually distinctive moments

## ⚠️ Troubleshooting

### Common Issues and Fixes:

**`WinError 127` (DLL not found):**
- Install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- Use CUDA 12.1: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

**`Could not load libtorchcodec`:**
- The `app/model.py` file has been modified to use PyAV instead of torchcodec
- Ensure `pip install av` was successful

**`No highlights passed the selection threshold`:**
- Lower the score: `--min-score 0.0` or `--min-score 3.0`
- Check `output/analysis/` JSON files to see if segments have descriptions
- Modify the prompt in `prompts.py` to be less conservative

**`CUDA not available`:**
- Ensure you installed the `cu121` version of torch, not the default CPU version
- Update NVIDIA drivers to the latest version

**`Out of Memory (OOM)`:**
- Close background GPU apps like LM Studio or browser tabs
- Lower `--max-new-tokens` (try `80`)

## 🔧 Key Configuration Files

- `app/model.py` — Handles model loading and video analysis (uses PyAV for frame extraction)
- `app/pipeline.py` — Orchestrates video chunking, analysis, and rendering
- `app/selector.py` — Scores and filters candidates based on thresholds
- `app/prompts.py` — The instruction prompt sent to the VLM for each segment
- `app/config.py` — Default configuration values

## 📊 How It Works

1. **Chunking**: Videos are split into `--segment`-second clips using FFmpeg.
2. **Frame Extraction**: PyAV extracts frames from each chunk.
3. **VLM Analysis**: SmolVLM2 describes each chunk and returns a JSON score.
4. **Filtering**: `selector.py` filters segments by score, overlap, and target duration.
5. **Rendering**: Selected clips are concatenated into the final `highlight.mp4`.
```