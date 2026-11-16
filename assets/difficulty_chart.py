"""Render the README difficulty chart as two static SVGs, light and dark.

Run from the repository root: `uv run python assets/difficulty_chart.py assets`.
The numbers are the same as on the difficulty page; edit DATA here and rerun.
"""

import sys
from pathlib import Path

DATA = [  # (topic, difficulty 1-10, hours average)
    (1, 1, "~1"),
    (2, 2, "~3"),
    (3, 2, "~3"),
    (4, 3, "~4"),
    (5, 6, "~8"),
    (6, 5, "~4"),
    (7, 1, "~1"),
    (8, 4, "~4"),
    (9, 4, "~3"),
    (10, 4, "~3"),
    (11, 5, "~4"),
    (12, 7, "~6"),
    (13, 5, "~6"),
    (14, 5, "~5"),
    (15, 9, "~12"),
    (16, 9, "~10"),
    (17, 8, "~8"),
    (18, 10, "~12"),
    (19, 10, "~14"),
    (20, 3, "~4"),
]
PHASES = [
    (1, 7, "tensors"),
    (8, 14, "transformer"),
    (15, 17, "training"),
    (18, 19, "memory"),
    (20, 20, "data"),
]
THEMES = {
    "light": {
        "bg": "#ffffff",
        "ink": "#0f141a",
        "ink2": "#4c5561",
        "ink3": "#828c99",
        "grid": "#e9edf2",
        "axis": "#c4ccd6",
        "rule": "#e0e5eb",
        "lv": ("#a9cdf5", "#4f93e3", "#1f5fb8"),
    },
    "dark": {
        "bg": "#0d1117",
        "ink": "#f1f4f8",
        "ink2": "#b2bbc7",
        "ink3": "#7a8494",
        "grid": "#20252d",
        "axis": "#39414c",
        "rule": "#262c35",
        "lv": ("#a9cdf5", "#4f93e3", "#1f5fb8"),
    },
}
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def fmt(x: float) -> str:
    return f"{x:g}"


def render(t: dict) -> str:
    W, H, L, R, T, PB = 960, 414, 52, 16, 92, 316
    band = (W - L - R) / 20
    bw = 28
    y = lambda v: PB - v / 10 * (PB - T)
    lvl = lambda d: t["lv"][0] if d <= 3 else t["lv"][1] if d <= 6 else t["lv"][2]

    o = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'font-family="{SANS}">'
        ),
        f'<rect width="{W}" height="{H}" rx="6" fill="{t["bg"]}"/>',
        f'<text x="{L}" y="36" font-size="20" font-weight="600" fill="{t["ink"]}">Difficulty by topic</text>',
        (
            f'<text x="{L}" y="60" font-size="13" fill="{t["ink2"]}"> for a Python engineer new to PyTorch, '
            f"first pass with no preparation about {90:g}-{130:g} hours in total</text>"
        ),
    ]
    for k, (name, i) in enumerate([("easy 1-3", 0), ("medium 4-6", 1), ("hard 7-10", 2)]):
        x = W - R - 330 + k * 112
        o.append(f'<rect x="{x}" y="27" width="11" height="11" rx="2" fill="{t["lv"][i]}"/>')
        o.append(f'<text x="{x + 17}" y="37" font-size="12" fill="{t["ink2"]}">{name}</text>')
    for v in (0, 2, 4, 6, 8, 10):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v)}" y2="{y(v)}" stroke="{t["axis"] if v == 0 else t["grid"]}"/>')
        o.append(
            f'<text x="{L - 10}" y="{y(v) + 4}" text-anchor="end" font-family="{MONO}" font-size="11" '
            f'fill="{t["ink3"]}">{v}</text>'
        )
    for n, d, avg in DATA:
        cx = L + band * (n - 1) + band / 2
        o.append(f'<rect x="{cx - bw / 2}" y="{y(d)}" width="{bw}" height="{PB - y(d)}" rx="2" fill="{lvl(d)}"/>')
        o.append(
            f'<text x="{cx}" y="{y(d) - 7}" text-anchor="middle" font-family="{MONO}" font-size="12" '
            f'font-weight="600" fill="{t["ink"]}" stroke="{t["bg"]}" stroke-width="4" paint-order="stroke">{d}</text>'
        )
        o.append(
            f'<text x="{cx}" y="{PB + 18}" text-anchor="middle" font-family="{MONO}" font-size="12" '
            f'fill="{t["ink2"]}">{n:02d}</text>'
        )
        o.append(
            f'<text x="{cx}" y="{PB + 35}" text-anchor="middle" font-family="{MONO}" font-size="9.5" '
            f'fill="{t["ink3"]}">{avg}</text>'
        )
    o.append(
        f'<text x="{L - 10}" y="{PB + 35}" text-anchor="end" font-family="{MONO}" font-size="9.5" '
        f'fill="{t["ink3"]}">hours</text>'
    )
    for a, b, s in PHASES:
        x1, x2, yb = L + band * (a - 1) + 4, L + band * b - 4, PB + 50
        o.append(f'<path d="M{x1} {yb + 5} L{x1} {yb} L{x2} {yb} L{x2} {yb + 5}" fill="none" stroke="{t["rule"]}"/>')
        o.append(
            f'<text x="{(x1 + x2) / 2}" y="{yb + 20}" text-anchor="middle" font-family="{MONO}" '
            f'font-size="11" letter-spacing="0.06em" fill="{t["ink3"]}">{s}</text>'
        )
    o.append("</svg>")
    return "\n".join(o) + "\n"


out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for name, theme in THEMES.items():
    (out / f"difficulty-{name}.svg").write_text(render(theme))
