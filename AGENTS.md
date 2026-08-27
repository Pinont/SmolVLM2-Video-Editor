# 🤖 Agent Guide: SmolVLM Auto Highlighter

## Project Overview
The `smolvlm_auto_highlighter` is a vision-based pipeline designed to automatically identify "highlight" moments in video footage (specifically optimized for high-chaos gaming/bodycam footage). It leverages a Vision-Language Model (VLM) to describe video segments and a scoring system to select the most relevant clips.

## 🛠 Technical Stack
- **Model**: SmolVLM2-2.2B-Instruct (Vision-Language Model from Hugging Face)
- **Runtime**: Python 3.12 (STRICT requirement — Python 3.13+ breaks PyTorch)
- **Hardware Acceleration**: NVIDIA CUDA (Optimized for RTX 40-series)
- **Video Processing**: FFmpeg (for chunking) and PyAV (for frame extraction)
- **Core Libraries**: `torch`, `transformers`, `accelerate`, `av`

## 🏗 Architecture

### Module Responsibilities:

**`app/main.py`** — Entry point
- Handles CLI argument parsing
- Creates `Config` instance from arguments
- Initializes `SmolVLM` model and `Pipeline`
- Calls `Pipeline.run()`

**`app/config.py`** — Configuration dataclass
- Defines default values for all parameters
- Used as the configuration blueprint for both CLI args and direct usage

**`app/ffmpeg.py`** — Video processing utilities
- `duration()`: Gets video length via ffprobe
- `make_chunk()`: Cuts video into segments using FFmpeg
- `render_clip()`: Re-encodes clips for final output (mixes all audio tracks via `amix`)
- `concat_files()`: Concatenates pre-rendered clips
- `concat_scenes()`: **Single-pass** concat via FFmpeg `filter_complex` (preserves all audio tracks, much faster)

**`app/model.py`** — VLM wrapper ⚠️ **CRITICAL FILE**
- Loads `SmolVLM2-2.2B-Instruct` from Hugging Face
- Uses `PyAV` (not `torchcodec`) for video frame extraction
- The `_load_video_frames()` method bypasses torchcodec DLL requirements
- Handles JSON parsing of model outputs
- `generate_text()` for text-only chat (used by Stage 2)

**`app/pipeline.py`** — Orchestrator
- `discover_videos()`: Finds videos in input folder
- `segment_ranges()`: Calculates chunk boundaries with overlap
- `analyze_one()`: Processes a single video (now runs Stage 1/2 + two-pass Stage 3)
- `run()`: Main execution loop, reconciles two passes and calls `concat_scenes`

**`app/selector.py`** — Filtering logic
- `Candidate` dataclass: Holds segment metadata
- `merge_overlapping()`: Collapses adjacent sequential chunks (20% threshold)
- `select_candidates()`: Filters by score + round-robin per-file + chronological output
- `expand_candidate()`: Adds context (before/after) to selected clips

**`app/stages.py`** — Pre-analysis stages (NEW)
- `describe_video()`: Stage 1 — full-video free-form description
- `derive_highlight_types()`: Stage 2 — list archetypal highlight types from description
- `ensure_meta()`: Caches per-video meta so re-runs don't re-describe
- `chunk_scoring_prompt()`: Stage 3 — per-chunk prompt using cached highlight types

**`app/critic.py`** — Deterministic feedback (NEW)
- `review()`: Runs 8 cheap checks (audio_present, tracks_preserved, sequential_clips, etc.)
- Returns `CriticReport` with `issues`, `axes`, `knob_hints`, `score`
- `apply_knob_hints()`: Adjusts Config within safe bounds

**`app/agent.py`** — Iterative refine loop (NEW)
- Runs Pipeline → Critic → tunes Config → repeats
- Capped by `max_iterations`, `wall_clock_minutes`, or `patience` rounds of no improvement
- Persists each iteration under `output/iterations/<NNN>/` and `output/state.json`

**`app/tui.py`** — Textual dashboard (NEW)
- Multi-pane: Queue | Current chunk | Knobs | Scores-over-iters | Live log
- Bindings: `s` start, `p` pause, `r` re-run, `d` deep review, `q` quit
- Run with: `python -m app.tui`

**`app/prompts.py`** — VLM instructions
- Kept for backwards compatibility (single-pass mode), but Stage 3 uses `app/stages.py::chunk_scoring_prompt` instead
- Defines the original "be generous" rubric — replaced by stricter rubric in `stages.py`

## ⚠️ CRITICAL ENVIRONMENT NOTES

### DO NOT:
- ❌ Use Python 3.13 or 3.14 (breaks PyTorch — use 3.12)
- ❌ Run `pip install torch` without the `--index-url` flag (installs CPU-only)
- ❌ Install `torchcodec` without FFmpeg full-shared DLLs (crashes on Windows)

### DO:
- ✅ Use Python 3.12 with `py -3.12 -m venv .venv`
- ✅ Install PyTorch with CUDA: `--index-url https://download.pytorch.org/whl/cu121`
- ✅ Install `pip install av` (replaces torchcodec)
- ✅ Install Visual C++ Redistributable if getting `WinError 127`

## 🔧 Known Solved Issues

### Issue 1: `WinError 127` (DLL Mismatch)
**Cause**: PyTorch CUDA libraries can't find required Windows DLLs.
**Fix**: Install Visual C++ Redistributable + use CUDA 12.1.

### Issue 2: `Could not load libtorchcodec`
**Cause**: `torchcodec` requires FFmpeg shared DLLs not present on Windows.
**Fix**: Modified `app/model.py` to use PyAV (`av` library) instead of torchcodec for frame extraction.

### Issue 3: `No highlights passed the selection threshold`
**Cause**: Default `min_score=6.5` is too high for bodycam content.
**Fix**: Use `--min-score 3.0` or modify `app/config.py` default.

### Issue 4: `CUDA not available`
**Cause**: CPU-only PyTorch was installed.
**Fix**: Reinstall with `--index-url https://download.pytorch.org/whl/cu121`.

## 🎯 Optimization for Bodycam/Game Content

### Recommended Settings:
```powershell
python -m app.main --input videos --output output --segment 4 --overlap 2 --target 60 --min-score 3.0
```

### Why These Settings:
- **Segment=4**: Captures short, snappy moments without dilution
- **Overlap=2**: Ensures events aren't cut at boundaries
- **Target=60**: Reasonable highlight reel length
- **Min-score=3.0**: Accepts "weak but interesting" moments

### Prompt Engineering:
The default `HIGHLIGHT_PROMPT` in `app/prompts.py` already includes:
- Prioritization for action, achievements, funny moments
- Rejection of setup, menus, dead time
- Conservative scoring (modify if too strict)

## 📋 Quick Start Commands

### First-Time Setup:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python.exe -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install av transformers accelerate safetensors pillow numpy tqdm
```

### Run Highlight Generation:
```powershell
python -m app.main --input videos --output output --segment 4 --overlap 2 --target 60 --min-score 3.0
```

### Verify GPU:
```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## 🧪 Testing Checklist

Before committing changes, verify:
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` returns `True`
- [ ] `python -c "import av; print('av works')"` returns `av works`
- [ ] A test video produces segments with non-zero scores
- [ ] The output folder contains `highlight.mp4`

## 📊 Data Flow

```
Video File (videos/)
    ↓
FFmpeg Chunking (ffmpeg.py:make_chunk)
    ↓
PyAV Frame Extraction (model.py:_load_video_frames)
    ↓
VLM Analysis (model.py:analyze)
    ↓
JSON Parsing (model.py:parse_json)
    ↓
Candidate Filtering (selector.py:select_candidates)
    ↓
Context Expansion (selector.py:expand_candidate)
    ↓
FFmpeg Rendering (ffmpeg.py:render_clip)
    ↓
Final Concatenation (ffmpeg.py:concat_files)
    ↓
Output: output/highlight.mp4
```

## 🚨 If You're Taking Over This Project

1. **Read `app/model.py` first** — it's the most modified file
2. **Verify Python 3.12** is being used (`python --version`)
3. **Verify CUDA works** before debugging anything else
4. **Check `output/analysis/`** for segment JSON files to debug scoring issues
5. **The torchcodec issue is SOLVED** — don't try to install torchcodec or FFmpeg shared DLLs
```

---

These documentation files now reflect the complete state of the project after all our debugging and fixes. The `README.md` is user-facing, while `AGENTS.md` provides technical details for developers/agents working on the codebase.