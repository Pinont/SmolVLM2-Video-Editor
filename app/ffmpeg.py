from pathlib import Path
import json
import subprocess


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, check=True)


def ffprobe_json(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-show_streams",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def duration(path: Path) -> float:
    return float(ffprobe_json(path)["format"]["duration"])


def has_audio(path: Path) -> bool:
    return any(
        s.get("codec_type") == "audio"
        for s in ffprobe_json(path).get("streams", [])
    )


def make_chunk(src: Path, start: float, end: float, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    length = max(0.01, end - start)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{length:.3f}",
        "-map", "0:v:0",
        "-an",
        "-vf", "scale='min(960,iw)':-2:force_original_aspect_ratio=decrease",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(dst),
    ]
    run(cmd)


def render_clip(
    src: Path,
    start: float,
    end: float,
    dst: Path,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    length = max(0.01, end - start)
    audio = has_audio(src)

    video_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={fps},format=yuv420p"
    )

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{length:.3f}",
        "-map", "0:v:0",
    ]

    if audio:
        cmd += ["-map", "0:a:0?"]

    cmd += [
        "-vf", video_filter,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-ac", "2",
        "-shortest",
        str(dst),
    ]
    run(cmd)


def concat_files(files: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    list_file = output.parent / "concat_list.txt"

    with list_file.open("w", encoding="utf-8") as f:
        for p in files:
            value = str(p.resolve()).replace("\\", "/").replace("'", r"'\''")
            f.write(f"file '{value}'\n")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output),
    ]
    run(cmd)
