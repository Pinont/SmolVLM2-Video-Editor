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


def select_candidates(
    candidates: list[Candidate],
    target_seconds: float,
    min_score: float,
    max_clips: int,
) -> list[Candidate]:
    pool = sorted(
        [c for c in candidates if c.score >= min_score],
        key=lambda c: c.score,
        reverse=True,
    )

    selected = []
    total = 0.0

    for c in pool:
        if len(selected) >= max_clips:
            break

        if any(overlap_ratio(c, x) >= 0.50 for x in selected):
            continue

        if total + c.duration <= target_seconds:
            selected.append(c)
            total += c.duration

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
