from dataclasses import dataclass


@dataclass
class Candidate:
    file: str
    start: float
    end: float
    score: float
    category: str
    description: str
    reason: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def overlap_ratio(a: Candidate, b: Candidate) -> float:
    if a.file != b.file:
        return 0.0

    left = max(a.start, b.start)
    right = min(a.end, b.end)
    overlap = max(0.0, right - left)
    shortest = min(a.duration, b.duration)

    return overlap / shortest if shortest > 0 else 0.0


def merge_overlapping(
    candidates: list[Candidate],
    threshold: float = 0.20,
) -> list[Candidate]:
    """Merge candidates whose [start,end] intervals overlap by more than
    `threshold` (relative to the shorter clip) and live in the same file.

    The kept candidate keeps the higher score; the time range is extended
    to cover both originals so we don't lose footage on either side.
    """
    merged: list[Candidate] = []

    for c in sorted(candidates, key=lambda x: (x.file, x.start)):
        if merged and merged[-1].file == c.file:
            last = merged[-1]
            shortest = min(last.duration, c.duration)
            if shortest > 0 and overlap_ratio(last, c) >= threshold:
                keep = last if last.score >= c.score else c
                merged[-1] = Candidate(
                    file=keep.file,
                    start=min(last.start, c.start),
                    end=max(last.end, c.end),
                    score=keep.score,
                    category=keep.category,
                    description=keep.description,
                    reason=keep.reason,
                )
                continue
        merged.append(c)

    return merged


def select_candidates(
    candidates: list[Candidate],
    target_seconds: float,
    min_score: float,
    max_clips: int,
) -> list[Candidate]:
    """Pick the strongest clips up to `max_clips` / `target_seconds`,
    spread evenly across the timeline so the result isn't just sequential
    chunks glued together.
    """
    pool = [c for c in candidates if c.score >= min_score]
    pool = merge_overlapping(pool)
    pool.sort(key=lambda c: c.score, reverse=True)

    # Group by file so we can spread picks across each source video.
    by_file: dict[str, list[Candidate]] = {}
    for c in pool:
        by_file.setdefault(c.file, []).append(c)

    # Round-robin: always pull the highest-scoring clip per file first,
    # then the next-highest, etc. This naturally spreads selections and
    # prevents a single high-scoring stretch from dominating.
    selected: list[Candidate] = []
    total = 0.0

    while len(selected) < max_clips:
        progressed = False

        for f in list(by_file.keys()):
            if not by_file[f]:
                by_file.pop(f, None)
                continue

            best = by_file[f].pop(0)  # already sorted by score desc
            selected.append(best)
            total += best.duration
            progressed = True

            if len(selected) >= max_clips or total >= target_seconds:
                break

        if not progressed:
            break

    # Chronological playback order: viewers expect highlights to tell a
    # story in the order they happened.
    selected.sort(key=lambda c: (c.file, c.start))
    return selected


def expand_candidate(
    c: Candidate,
    file_duration: float,
    before: float,
    after: float,
) -> Candidate:
    return Candidate(
        file=c.file,
        start=max(0.0, c.start - before),
        end=min(file_duration, c.end + after),
        score=c.score,
        category=c.category,
        description=c.description,
        reason=c.reason,
    )
