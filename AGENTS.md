# 🤖 Agent Guide: SmolVLM Auto Highlighter

## Project Overview
The `smolvlm_auto_highlighter` is a vision-based pipeline designed to automatically identify "highlight" moments in video footage (specifically optimized for high-chaos gaming/bodycam footage). It leverages a Vision-Language Model (VLM) to describe video segments and a scoring system to select the most relevant clips.

## 🛠 Technical Stack
- **Model**: SmolVLM (Vision-Language Model).
- **Runtime**: Python 3.12 (Strict requirement).
- **Hardware Acceleration**: NVIDIA CUDA (Optimized for RTX 40-series).
- **Video Processing**: FFmpeg (for chunking) and PyAV/Torchvision (for decoding).
- **Core Libraries**: `torch`, `transformers`, `accelerate`, `av`.

## 🏗 Architecture Logic
1. **Preprocessing (`ffmpeg.py`)**: The input video is split into small segments based on the `--segment` and `--overlap` flags.
2. **Inference (`model.py`)**: Each segment is passed to the VLM, which generates a text description of the visual events.
3. **Scoring & Selection (`selector.py`)**: The descriptions are analyzed for keywords or "energy" scores. The top-scoring segments are selected until the `--target` duration is met.
4. **Assembly**: Selected segments are concatenated into a final highlight reel.

## ⚠️ Critical Environment Notes (Agent Warnings)
If you are deploying or debugging this project, be aware of the following solved issues:

- **Python Version**: Do **not** use Python 3.13 or 3.14. PyTorch binaries for these versions are unstable or unavailable. Use **Python 3.12**.
- **CUDA Installation**: Standard `pip install torch` often installs the CPU version. You must use the specific index URL: `--index-url https://download.pytorch.org/whl/cu121`.
- **Dependency Gap**: The project requires `av` (PyAV) for `torchvision` to decode videos. Without it, segments will return an "error" category.
- **DLL Errors**: If `WinError 127` occurs, the system is missing the **Microsoft Visual C++ Redistributable**.
- **VLM Padding**: To avoid `pad_token_id` warnings, ensure `model.config.pad_token_id` is set to `tokenizer.eos_token_id` during initialization.

## 🎯 Optimization for "Funny" Content
For gaming highlights, generic prompts fail. To optimize the agent's selection:
- **Segment Length**: Use short segments (`--segment 4`) to avoid diluting high-action moments.
- **Prompting**: Direct the model to look for "physical glitches," "visual chaos," "unexpected physics," and "absurd combat situations."

## 📋 Quick Start Command
```powershell
python -m app.main --input videos --output output --segment 4 --overlap 2 --target 60
```