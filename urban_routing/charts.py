"""Minimal SVG chart writer.

matplotlib is not available in the target environment and the brief asks for
charts, so this module emits standalone SVG -- no third-party dependency, opens
in any browser, and stays diff-readable in the repository.

It draws exactly what the performance evaluation needs: multi-series line charts
with linear or log axes, and grouped bar charts.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from math import log10
from pathlib import Path

_PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]


@dataclass(slots=True)
class Series:
    label: str
    xs: list[float]
    ys: list[float]


def _ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    step = (hi - lo) / count
    return [lo + i * step for i in range(count + 1)]


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.1f}"
    if a >= 0.01:
        return f"{v:.3f}"
    return f"{v:.2e}"


def line_chart(
    path: Path,
    series: list[Series],
    title: str,
    x_label: str,
    y_label: str,
    width: int = 780,
    height: int = 460,
    log_y: bool = False,
    subtitle: str = "",
) -> Path:
    """Write a multi-series line chart to ``path`` as SVG."""
    left, right, top, bottom = 82, 190, 62, 64
    pw, ph = width - left - right, height - top - bottom

    xs = [x for s in series for x in s.xs]
    ys = [y for s in series for y in s.ys if y > 0 or not log_y]
    if not xs or not ys:
        raise ValueError("no data to plot")
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if log_y:
        y0, y1 = log10(max(y0, 1e-12)), log10(max(y1, 1e-12))
    if y1 == y0:
        y1 = y0 + 1.0
    if x1 == x0:
        x1 = x0 + 1.0
    pad = (y1 - y0) * 0.08
    y0, y1 = y0 - pad, y1 + pad

    def px(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * pw

    def py(y: float) -> float:
        v = log10(max(y, 1e-12)) if log_y else y
        return top + ph - (v - y0) / (y1 - y0) * ph

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Inter, Segoe UI, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-size="17" font-weight="600" fill="#111827">'
        f"{html.escape(title)}</text>",
    ]
    if subtitle:
        out.append(
            f'<text x="{left}" y="48" font-size="12" fill="#6b7280">'
            f"{html.escape(subtitle)}</text>"
        )

    # grid + y axis
    for t in _ticks(y0, y1):
        yy = top + ph - (t - y0) / (y1 - y0) * ph
        out.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{left + pw}" y2="{yy:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        label = _fmt(10 ** t if log_y else t)
        out.append(
            f'<text x="{left - 10}" y="{yy + 4:.1f}" font-size="11" fill="#6b7280" '
            f'text-anchor="end">{label}</text>'
        )
    # x axis
    for t in _ticks(x0, x1):
        xx = px(t)
        out.append(
            f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top + ph}" '
            f'stroke="#f3f4f6" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{xx:.1f}" y="{top + ph + 20}" font-size="11" fill="#6b7280" '
            f'text-anchor="middle">{_fmt(t)}</text>'
        )
    out.append(
        f'<line x1="{left}" y1="{top + ph}" x2="{left + pw}" y2="{top + ph}" '
        f'stroke="#9ca3af" stroke-width="1.2"/>'
    )
    out.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + ph}" '
        f'stroke="#9ca3af" stroke-width="1.2"/>'
    )

    # series
    for i, s in enumerate(series):
        colour = _PALETTE[i % len(_PALETTE)]
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(s.xs, s.ys))
        out.append(
            f'<polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="2.2" stroke-linejoin="round"/>'
        )
        for x, y in zip(s.xs, s.ys):
            out.append(
                f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.2" fill="{colour}"/>'
            )
        ly = top + 8 + i * 22
        out.append(
            f'<line x1="{left + pw + 18}" y1="{ly}" x2="{left + pw + 42}" y2="{ly}" '
            f'stroke="{colour}" stroke-width="2.6"/>'
        )
        out.append(
            f'<text x="{left + pw + 48}" y="{ly + 4}" font-size="12" fill="#374151">'
            f"{html.escape(s.label)}</text>"
        )

    out.append(
        f'<text x="{left + pw / 2:.0f}" y="{height - 16}" font-size="12.5" '
        f'fill="#374151" text-anchor="middle">{html.escape(x_label)}</text>'
    )
    out.append(
        f'<text x="20" y="{top + ph / 2:.0f}" font-size="12.5" fill="#374151" '
        f'text-anchor="middle" transform="rotate(-90 20 {top + ph / 2:.0f})">'
        f"{html.escape(y_label)}{' (log scale)' if log_y else ''}</text>"
    )
    out.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def bar_chart(
    path: Path,
    labels: list[str],
    groups: list[tuple[str, list[float]]],
    title: str,
    y_label: str,
    width: int = 780,
    height: int = 430,
    log_y: bool = False,
    subtitle: str = "",
) -> Path:
    """Grouped bar chart.  ``groups`` is [(series label, one value per label)]."""
    left, right, top, bottom = 82, 175, 62, 74
    pw, ph = width - left - right, height - top - bottom
    values = [v for _, vs in groups for v in vs]
    hi = max(values) if values else 1.0
    lo = 0.0
    if log_y:
        positive = [v for v in values if v > 0]
        lo = log10(min(positive)) - 0.4 if positive else 0.0
        hi = log10(hi) if hi > 0 else 1.0

    def bar_top(v: float) -> float:
        vv = log10(max(v, 1e-12)) if log_y else v
        frac = (vv - lo) / (hi - lo) if hi > lo else 0.0
        return top + ph - max(0.0, min(1.0, frac)) * ph

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Inter, Segoe UI, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-size="17" font-weight="600" fill="#111827">'
        f"{html.escape(title)}</text>",
    ]
    if subtitle:
        out.append(
            f'<text x="{left}" y="48" font-size="12" fill="#6b7280">'
            f"{html.escape(subtitle)}</text>"
        )
    for t in _ticks(lo, hi):
        yy = top + ph - (t - lo) / (hi - lo) * ph if hi > lo else top + ph
        out.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{left + pw}" y2="{yy:.1f}" '
            f'stroke="#e5e7eb"/>'
        )
        out.append(
            f'<text x="{left - 10}" y="{yy + 4:.1f}" font-size="11" fill="#6b7280" '
            f'text-anchor="end">{_fmt(10 ** t if log_y else t)}</text>'
        )

    slot = pw / max(1, len(labels))
    bw = slot * 0.74 / max(1, len(groups))
    for gi, (glabel, vs) in enumerate(groups):
        colour = _PALETTE[gi % len(_PALETTE)]
        for li, v in enumerate(vs):
            x = left + li * slot + slot * 0.13 + gi * bw
            y = bar_top(v)
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 2:.1f}" '
                f'height="{top + ph - y:.1f}" fill="{colour}" rx="2"/>'
            )
        ly = top + 8 + gi * 22
        out.append(
            f'<rect x="{left + pw + 18}" y="{ly - 7}" width="18" height="12" '
            f'fill="{colour}" rx="2"/>'
        )
        out.append(
            f'<text x="{left + pw + 42}" y="{ly + 4}" font-size="12" fill="#374151">'
            f"{html.escape(glabel)}</text>"
        )
    for li, lab in enumerate(labels):
        out.append(
            f'<text x="{left + li * slot + slot / 2:.1f}" y="{top + ph + 20}" '
            f'font-size="11" fill="#374151" text-anchor="middle">'
            f"{html.escape(lab)}</text>"
        )
    out.append(
        f'<line x1="{left}" y1="{top + ph}" x2="{left + pw}" y2="{top + ph}" '
        f'stroke="#9ca3af" stroke-width="1.2"/>'
    )
    out.append(
        f'<text x="20" y="{top + ph / 2:.0f}" font-size="12.5" fill="#374151" '
        f'text-anchor="middle" transform="rotate(-90 20 {top + ph / 2:.0f})">'
        f"{html.escape(y_label)}{' (log scale)' if log_y else ''}</text>"
    )
    out.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    return path
