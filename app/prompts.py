HIGHLIGHT_PROMPT = """
You are selecting footage for a short, engaging highlight reel.

Watch the entire supplied video segment. Decide whether it contains a
moment that deserves to survive into the final edit.

Be generous with scoring. Mark segments as "keep": true if they contain ANY notable action or visual interest. Set score to 5 or higher for any segment with movement, action, or interesting visuals.

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
0-2 = discard
3-4 = weak but keepable
5-6 = good highlight
7-8 = very good
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
""".strip()