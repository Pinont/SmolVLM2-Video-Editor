HIGHLIGHT_PROMPT = """
You are selecting footage for a short, engaging highlight reel.

Watch the entire supplied video segment. Decide whether it contains a
moment that deserves to survive into the final edit.

Be SELECTIVE. Most chunks of a recording are NOT highlights. Mark "keep": true ONLY if the segment contains a moment worth clipping into a short highlight reel.

Prioritize:
- clear exciting action
- impressive results or achievements
- funny or surprising events
- strong reactions
- dramatic moments
- visually distinctive moments
- an obvious payoff
- moments that would make a viewer keep watching

Reject:
- dead time
- loading screens
- menus
- ordinary walking or waiting with no action
- repetitive activity with no payoff
- blurry/unreadable footage
- moments where nothing meaningful happens

Score 0-10:
0-2 = discard (keep=false)
3-4 = boring, skip (keep=false)
5-6 = mild interest but not a highlight (keep=false unless most of the video is dull)
7-8 = solid highlight (keep=true)
9 = excellent highlight (keep=true)
10 = exceptional, must-keep moment (keep=true)

Rule of thumb: set keep=false unless score >= 7. Do NOT mark segments as highlights just because something is happening — only when something MEMORABLE is happening.

Return ONLY valid JSON:
{
  "keep": true,
  "score": 0,
  "category": "action|funny|dramatic|emotional|achievement|unexpected|informative|other",
  "description": "one short sentence",
  "reason": "one short sentence"
}
""".strip()