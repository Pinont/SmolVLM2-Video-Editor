HIGHLIGHT_PROMPT = """
You are selecting footage for a short, engaging highlight reel.

Watch the entire supplied video segment. Decide whether it contains a
moment that deserves to survive into the final edit.

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
- setup with no payoff
- loading screens
- menus
- ordinary walking or waiting
- repetitive activity
- blurry/unreadable footage
- moments where nothing meaningful happens

Score 0-10:
0-3 = discard
4-5 = weak
6 = usable
7 = good
8 = very good
9 = excellent
10 = exceptional

Return ONLY valid JSON:
{
  "keep": true,
  "score": 0,
  "category": "action|funny|dramatic|emotional|achievement|unexpected|informative|other",
  "description": "one short sentence",
  "reason": "one short sentence"
}

Be conservative. If there is no obvious highlight, set keep=false.
""".strip()
