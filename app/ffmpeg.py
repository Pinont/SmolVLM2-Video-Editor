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
        # Keep ALL audio tracks (game+discord on track 0, mic on track 1, etc.)
        # and mix them into a single stereo output so the mic survives.
        cmd += [
            "-map", "0:a?",
            "-filter_complex",
            "[0:a]amix=inputs=2:normalize=0:dropout_transition=0[aout]",
            "-map", "[aout]",
            "-ac", "2",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
        ]
    else:
        cmd += ["-an"]

    cmd += [
        "-vf", video_filter,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-shortest",
        str(dst),
    ]
    run(cmd)


def concat_scenes(
    src: Path,
    scenes: list[tuple[float, float]],
    output: Path,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
) -> None:
    """Concatenate arbitrary [start, end] scenes from a single source video
    in ONE ffmpeg invocation using filter_complex (trim + atrim + concat).

    Faster than render_clip + concat_files because:
    - No intermediate per-clip files on disk.
    - No re-encode of source frames (decode once).
    - Preserves ALL audio tracks from the source via -map 0:a? + amix.
    """
    if not scenes:
        raise ValueError("concat_scenes called with empty scene list")

    output.parent.mkdir(parents=True, exist_ok=True)

    v_filter_parts: list[str] = []
    a_filter_parts: list[str] = []
    concat_inputs: list[str] = []

    for i, (start, end) in enumerate(scenes):
        length = max(0.01, end - start)
        v_filter_parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},"
            f"setpts=PTS-STARTPTS,"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={fps},format=yuv420p[v{i}];"
        )
        a_filter_parts.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
            f"asetpts=PTS-STARTPTS[a{i}];"
        )
        concat_inputs.append(f"[v{i}][a{i}]")

    # Mix all audio tracks from the source (track 0 = game+discord, track 1 = mic).
    a_mix = (
        f"{''.join(a_filter_parts)}"
        f"{''.join(f'[a{i}]' for i in range(len(scenes)))}"
        f"amix=inputs={len(scenes)}:dropout_transition=0[aout];"
    )

    filter_complex = (
        "".join(v_filter_parts)
        + a_mix
        + f"{''.join(concat_inputs)}concat=n={len(scenes)}:v=1:a=0[outv];"
    )

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-ac", "2",
        "-movflags", "+faststart",
        str(output),
    ]
    run(cmd)


def concat_scenes(
    src: Path,
    scenes: list[tuple[float, float]],
    output: Path,
) -> None:
    """Single-pass trim+atrim+concat from one source video.

    Faster than render-each-clip-then-concat, and preserves all audio
    tracks per scene (with amix for multi-track sources).

    `scenes` is a list of (start_sec, end_sec) tuples; the source video
    is the same file for every scene (the canonical case for our
    highlight reel which is drawn from a single recording).
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if not scenes:
        raise ValueError("concat_scenes: empty scenes list")

    audio = has_audio(src)

    parts: list[str] = []
    concat_inputs: list[str] = []

    for i, (start, end) in enumerate(scenes):
        length = max(0.01, end - start)
        parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},"
            f"setpts=PTS-STARTPTS[v{i}];"
        )
        if audio:
            parts.append(
                f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
                f"asetpts=PTS-STARTPTS[a{i}];"
            )
            concat_inputs.append(f"[v{i}][a{i}]")
        else:
            concat_inputs.append(f"[v{i}]")

    n = len(scenes)

    if audio:
        concat_filter = (
            f"{''.join(concat_inputs)}"
            f"concat=n={n}:v=1:a=1[outv][outa_raw];"
            f"[outa_raw]amix=inputs=1:normalize=0:dropout_transition=0[outa]"
        )
        filter_complex = "".join(parts) + concat_filter
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src),
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            "-movflags", "+faststart",
            str(output),
        ]
    else:
        concat_filter = (
            f"{''.join(concat_inputs)}concat=n={n}:v=1:a=0[outv]"
        )
        filter_complex = "".join(parts) + concat_filter
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src),
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output),
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


def concat_scenes(
    src: Path,
    scenes: list[tuple[float, float]],
    dst: Path,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    mix_audio: bool = True,
) -> None:
    """Single-pass FFmpeg concat via filter_complex. Re-encodes once,
    preserving all audio tracks from `src` (mixing them with amix when
    more than one) so multi-track recordings don't lose the mic.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not scenes:
        raise ValueError("concat_scenes called with empty scene list")

    video_filter_parts: list[str] = []
    concat_inputs: list[str] = []
    for i, (start, end) in enumerate(scenes):
        length = max(0.01, end - start)
        video_filter_parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},"
            f"setpts=PTS-STARTPTS,"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={fps},format=yuv420p[v{i}];"
        )
        # Probe how many audio streams src has; per scene, trim each one
        # then concat them all in lock-step with the video.
        concat_inputs.append(f"[v{i}][a{i}]")

    # Build per-track audio chain: split, trim each stream, concat per-scene
    audio_filter_parts: list[str] = []
    probe = ffprobe_json(src)
    audio_streams = [
        s for s in probe.get("streams", [])
        if s.get("codec_type") == "audio"
    ]

    if audio_streams:
        for i in range(len(scenes)):
            per_scene_audio_inputs: list[str] = []
            for a_idx in range(len(audio_streams)):
                tag = f"a{i}_s{a_idx}"
                audio_filter_parts.append(
                    f"[0:a:{a_idx}]atrim=start={scenes[i][0]:.3f}:"
                    f"end={scenes[i][1]:.3f},asetpts=PTS-STARTPTS[{tag}]"
                )
                per_scene_audio_inputs.append(f"[{tag}]")

            if mix_audio and len(audio_streams) > 1:
                amix_inputs = "".join(per_scene_audio_inputs)
                audio_filter_parts.append(
                    f"{amix_inputs}amix=inputs={len(per_scene_audio_inputs)}:"
                    f"normalize=0:dropout_transition=0[a{i}]"
                )
            else:
                # Just take the first audio stream for this scene.
                audio_filter_parts.append(
                    f"{per_scene_audio_inputs[0]}aresample=48000[a{i}]"
                )

    n = len(scenes)
    concat_filter = (
        f"{''.join(concat_inputs)}concat=n={n}:v=1:a=1[outv][outa]"
    )
    filter_complex = "".join(video_filter_parts) + "".join(
        audio_filter_parts
    ) + concat_filter

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-filter_complex", filter_complex,
        "-map", "[outv]",
    ]
    if audio_streams:
        cmd += ["-map", "[outa]"]
    cmd += [
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-ac", "2",
        "-movflags", "+faststart",
        str(dst),
    ]
    run(cmd)


def concat_scenes(
    src: Path,
    scenes: list[tuple[float, float]],
    dst: Path,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
) -> None:
    """Single-pass trim + concat using ffmpeg's filter_complex.

    Faster than render-each-then-concat and preserves per-scene audio
    alignment. Multi-track sources are mixed down to stereo via amix so
    secondary tracks (e.g. microphone) survive.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not scenes:
        raise ValueError("concat_scenes called with empty scenes list")

    filter_parts: list[str] = []
    concat_inputs: list[str] = []

    for i, (start, end) in enumerate(scenes):
        length = max(0.01, end - start)
        filter_parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},"
            f"setpts=PTS-STARTPTS,"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={fps},format=yuv420p[v{i}];"
        )
        # amix all audio tracks per scene so multi-track recordings
        # (game+discord + mic) keep the mic.
        filter_parts.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
            f"asetpts=PTS-STARTPTS,"
            f"amix=inputs=2:normalize=0:dropout_transition=0[a{i}];"
        )
        concat_inputs.append(f"[v{i}][a{i}]")

    n = len(scenes)
    concat_filter = (
        f"{''.join(concat_inputs)}concat=n={n}:v=1:a=1[outv][outa]"
    )
    filter_complex = "".join(filter_parts) + concat_filter

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-ac", "2",
        "-movflags", "+faststart",
        str(dst),
    ]
    run(cmd)
