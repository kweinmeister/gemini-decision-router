"""Perception-gated hard items for `s4_visual_and_pdf`."""

from __future__ import annotations

import heapq
import io
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.ticker import MultipleLocator
from matplotlib.tri import Triangulation
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pypdf import PdfReader

matplotlib.use("Agg")

SLICE = "s4_visual_and_pdf"
_RC: dict[str, Any] = {"font.family": "DejaVu Sans", "axes.unicode_minus": False}
_PNG_METADATA: dict[str, Any] = {"Software": None}
_INF = math.inf


def _item(
    item_id: str,
    modality: str,
    subtype: str,
    provenance: str,
    prompt: str,
    attachment: str,
    grader_type: str,
    grader_spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": item_id,
        "slice": SLICE,
        "modality": modality,
        "subtype": subtype,
        "source_provenance": provenance,
        "prompt": prompt,
        "attachments": [attachment],
        "grader_type": grader_type,
        "grader_spec": grader_spec,
    }


def _numeric_spec(value: int) -> dict[str, Any]:
    return {"expected": float(value), "abs_tol": 0.1, "rel_tol": 1e-9}


# ---------------------------------------------------------------------------------------------
# Subtype 0: weighted digraph shortest path
# ---------------------------------------------------------------------------------------------

_G_FIGSIZE = (8.0, 6.0)
_G_DPI = 200
_G_HALF_W, _G_HALF_H = 1.6, 1.2  # same 4:3 aspect as the figure -> 500 px per data unit
_G_PX_PER_UNIT = _G_DPI * _G_FIGSIZE[0] / (2 * _G_HALF_W)
_G_NODE_R = 0.075
_G_ARC_RAD = 0.18
_G_LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"  # no I / O / Q (confusable with 1 / 0)
_G_LABEL_HALF = (0.026, 0.036)  # weight-label box half width / height incl. padding (data units)
_G_LABEL_FONTSIZE = 8.0
_G_MIN_PORT_GAP = math.radians(24.0)
_G_MIN_ANTI_LEN = 0.62
# Label-box clearance from foreign connectors (data units; 0.002 = 1 px): stroke half-width
# (0.0018) / arrowhead half-width incl. stroke (0.018), each plus a visible gap of >= 2 px.
_G_SHAFT_CLEARANCE = 0.016
_G_HEAD_CLEARANCE = 0.024


@dataclass(frozen=True)
class DigraphEdge:
    u: int
    v: int
    weight: int
    rad: float  # arc3 curvature: 0.0 = straight single edge, _G_ARC_RAD = one arc of a 2-cycle


@dataclass(frozen=True)
class DigraphSpec:
    labels: tuple[str, ...]
    pos: tuple[tuple[float, float], ...]
    edges: tuple[DigraphEdge, ...]
    label_xy: tuple[tuple[float, float], ...]  # weight-label centres, parallel to `edges`
    source: int
    target: int
    optimum: int
    path: tuple[int, ...]
    trap_edges: tuple[tuple[int, int], ...]  # single edges whose reversal would beat the optimum


def _arc_ctrl(p0: tuple[float, float], p2: tuple[float, float], rad: float) -> tuple[float, float]:
    """Control point of matplotlib's `arc3` connection style (valid here: equal-aspect axes)."""
    dx, dy = p2[0] - p0[0], p2[1] - p0[1]
    return ((p0[0] + p2[0]) / 2 + rad * dy, (p0[1] + p2[1]) / 2 - rad * dx)


def _bezier(
    p0: tuple[float, float], c: tuple[float, float], p2: tuple[float, float], ts: np.ndarray
) -> np.ndarray:
    t = ts[:, None]
    return (1 - t) ** 2 * np.asarray(p0) + 2 * (1 - t) * t * np.asarray(c) + t**2 * np.asarray(p2)


def _seg_point_dist(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def _digraph_layout(rng: random.Random, n: int) -> list[tuple[float, float]]:
    """Two jittered elliptical rings: 6-7 outer nodes and 3-4 inner nodes."""
    n_inner = 3 if n == 9 else (rng.choice((3, 4)) if n == 10 else 4)
    n_outer = n - n_inner
    pts: list[tuple[float, float]] = []
    phase = rng.uniform(0.0, 2 * math.pi)
    for k in range(n_outer):
        ang = phase + 2 * math.pi * k / n_outer + rng.uniform(-0.14, 0.14)
        pts.append(
            (
                1.36 * rng.uniform(0.95, 1.03) * math.cos(ang),
                0.98 * rng.uniform(0.95, 1.03) * math.sin(ang),
            )
        )
    phase = rng.uniform(0.0, 2 * math.pi)
    for k in range(n_inner):
        ang = phase + 2 * math.pi * k / n_inner + rng.uniform(-0.2, 0.2)
        r = rng.uniform(0.9, 1.1)
        pts.append((0.6 * r * math.cos(ang), 0.44 * r * math.sin(ang)))
    return pts


def _candidate_pairs(pos: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Delaunay edges (planar, no crossings) that are long enough and do not graze other nodes."""
    tri = Triangulation(np.array([p[0] for p in pos]), np.array([p[1] for p in pos]))
    pairs = sorted({(int(min(a, b)), int(max(a, b))) for a, b in tri.edges})
    out = []
    for a, b in pairs:
        if math.dist(pos[a], pos[b]) < 0.42:
            continue
        if all(
            _seg_point_dist(pos[k], pos[a], pos[b]) >= 0.2
            for k in range(len(pos))
            if k not in (a, b)
        ):
            out.append((a, b))
    return out


def _visible_curve(pos: list[tuple[float, float]], a: int, b: int, rad: float) -> np.ndarray:
    """Points (~2 px apart) of the a->b connector that lie outside both node circles."""
    p0, p2 = pos[a], pos[b]
    n = max(40, int(math.dist(p0, p2) / 0.004))
    pts = _bezier(p0, _arc_ctrl(p0, p2, rad), p2, np.linspace(0.0, 1.0, n))
    d0 = np.hypot(pts[:, 0] - p0[0], pts[:, 1] - p0[1])
    d2 = np.hypot(pts[:, 0] - p2[0], pts[:, 1] - p2[1])
    return pts[(d0 >= _G_NODE_R + 0.005) & (d2 >= _G_NODE_R + 0.005)]


def _curve_margins(
    pos: list[tuple[float, float]], curves: list[tuple[int, int, float]], vis: list[np.ndarray]
) -> list[np.ndarray]:
    """Per-point clearance a weight label needs: stroke half-width, or arrowhead half-width."""
    out = []
    for (_a, b, _rad), pts in zip(curves, vis, strict=True):
        near_head = np.hypot(pts[:, 0] - pos[b][0], pts[:, 1] - pos[b][1]) < _G_NODE_R + 0.045
        out.append(np.where(near_head, _G_HEAD_CLEARANCE, _G_SHAFT_CLEARANCE))
    return out


def _geometry_ok(pos: list[tuple[float, float]], curves: list[tuple[int, int, float]]) -> bool:
    """Arrowhead ports well separated, arcs clear of other nodes and of each other."""
    n = len(pos)
    ports: list[list[float]] = [[] for _ in range(n)]
    for a, b, rad in curves:
        c = _arc_ctrl(pos[a], pos[b], rad)
        ports[a].append(math.atan2(c[1] - pos[a][1], c[0] - pos[a][0]) % (2 * math.pi))
        ports[b].append(math.atan2(c[1] - pos[b][1], c[0] - pos[b][0]) % (2 * math.pi))
    for angs in ports:
        if len(angs) >= 2:
            s = sorted(angs)
            gaps = [s[i + 1] - s[i] for i in range(len(s) - 1)] + [s[0] + 2 * math.pi - s[-1]]
            if min(gaps) < _G_MIN_PORT_GAP:
                return False
    vis = [_visible_curve(pos, a, b, rad) for a, b, rad in curves]
    node_xy = np.asarray(pos)
    for (a, b, _rad), pts in zip(curves, vis, strict=True):
        for k in range(n):
            if k not in (a, b) and np.min(np.hypot(*(pts - node_xy[k]).T)) < _G_NODE_R + 0.04:
                return False
    lo = [v.min(axis=0) - 0.03 for v in vis]
    hi = [v.max(axis=0) + 0.03 for v in vis]
    for i in range(len(vis)):
        for j in range(i + 1, len(vis)):
            if np.any(lo[i] > hi[j]) or np.any(lo[j] > hi[i]):
                continue  # bounding boxes far apart
            diff = vis[i][:, None, :] - vis[j][None, :, :]
            if np.min(np.hypot(diff[..., 0], diff[..., 1])) < 0.022:
                return False
    return True


def _label_box_ok(
    xy: tuple[float, float],
    own: int,
    placed: list[tuple[float, float]],
    pos: list[tuple[float, float]],
    vis: list[np.ndarray],
    margins: list[np.ndarray],
) -> bool:
    hw, hh = _G_LABEL_HALF
    x, y = xy
    if abs(x) + hw > _G_HALF_W - 0.03 or abs(y) + hh > _G_HALF_H - 0.03:
        return False
    for nx, ny in pos:  # box-to-circle distance
        dx, dy = max(abs(nx - x) - hw, 0.0), max(abs(ny - y) - hh, 0.0)
        if math.hypot(dx, dy) < _G_NODE_R + 0.012:
            return False
    for px, py in placed:  # box-to-box separation
        if abs(px - x) < 2 * hw + 0.014 and abs(py - y) < 2 * hh + 0.014:
            return False
    for j, (pts, m) in enumerate(zip(vis, margins, strict=True)):
        if j == own:
            continue
        inside = (np.abs(pts[:, 0] - x) < hw + m) & (np.abs(pts[:, 1] - y) < hh + m)
        if inside.any():
            return False
    return True


def _place_labels(
    pos: list[tuple[float, float]], curves: list[tuple[int, int, float]]
) -> list[tuple[float, float]] | None:
    """Greedy weight-label placement on each curve near its midpoint (arc labels nudged outward)."""
    vis = [_visible_curve(pos, a, b, rad) for a, b, rad in curves]
    margins = _curve_margins(pos, curves, vis)
    placed: list[tuple[float, float]] = []
    for idx, (a, b, rad) in enumerate(curves):
        p0, p2 = pos[a], pos[b]
        c = _arc_ctrl(p0, p2, rad)
        length = math.dist(p0, p2)
        t_min = (_G_NODE_R + 0.035 + _G_LABEL_HALF[1] + 0.01) / length
        nx, ny = (p2[1] - p0[1]) / length, -(p2[0] - p0[0]) / length  # bulge side of arc3 rad>0
        for t in (0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65):
            if t < t_min or t > 1 - t_min:
                continue
            bx, by = _bezier(p0, c, p2, np.array([t]))[0]
            xy = (float(bx + 0.008 * nx), float(by + 0.008 * ny)) if rad else (float(bx), float(by))
            if _label_box_ok(xy, idx, placed, pos, vis, margins):
                placed.append(xy)
                break
        else:
            return None
    return placed


def _dijkstra(n: int, adj: list[list[tuple[int, int]]], src: int) -> list[float]:
    dist = [_INF] * n
    dist[src] = 0
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v))
    return dist


def _all_pairs(n: int, edges: list[DigraphEdge]) -> list[list[float]]:
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for e in edges:
        adj[e.u].append((e.v, e.weight))
    return [_dijkstra(n, adj, s) for s in range(n)]


def _unique_shortest_path(
    n: int, edges: list[DigraphEdge], dist_s: list[float], s: int, t: int
) -> tuple[int, list[int]]:
    """Number of distinct shortest s->t paths and one of them (positive weights => tight DAG)."""
    incoming: list[list[DigraphEdge]] = [[] for _ in range(n)]
    for e in edges:
        incoming[e.v].append(e)
    count, pred = [0] * n, [-1] * n
    count[s] = 1
    for _d, v in sorted((d, v) for v, d in enumerate(dist_s) if d < _INF):
        if v == s:
            continue
        for e in incoming[v]:
            if dist_s[e.u] + e.weight == dist_s[v]:
                count[v] += count[e.u]
                pred[v] = e.u
    path = [t]
    while path[-1] != s:
        path.append(pred[path[-1]])
    return count[t], path[::-1]


def _simple_path_costs(n: int, edges: list[DigraphEdge], src: int) -> list[list[int]]:
    """Costs of every simple path from `src` to each node (exhaustive; the graphs are tiny)."""
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for e in edges:
        adj[e.u].append((e.v, e.weight))
    costs: list[list[int]] = [[] for _ in range(n)]
    stack = [(src, 0, 1 << src)]
    while stack:
        u, c, seen = stack.pop()
        for v, w in adj[u]:
            if not (seen >> v) & 1:
                costs[v].append(c + w)
                stack.append((v, c + w, seen | (1 << v)))
    return costs


def _choose_endpoints(
    rng: random.Random, n: int, edges: list[DigraphEdge]
) -> tuple[int, int, int, list[int], list[tuple[int, int]]] | None:
    present = {(e.u, e.v) for e in edges}
    singles = [e for e in edges if (e.v, e.u) not in present]
    dist = _all_pairs(n, edges)
    flipped = [_all_pairs(n, [*edges, DigraphEdge(e.v, e.u, e.weight, 0.0)]) for e in singles]
    swapped: list[list[list[float]]] = []
    for e in edges:
        if (e.v, e.u) in present and e.u < e.v:
            w_back = next(f.weight for f in edges if (f.u, f.v) == (e.v, e.u))
            alt = []
            for f in edges:
                if (f.u, f.v) == (e.u, e.v):
                    alt.append(DigraphEdge(f.u, f.v, w_back, f.rad))
                elif (f.u, f.v) == (e.v, e.u):
                    alt.append(DigraphEdge(f.u, f.v, e.weight, f.rad))
                else:
                    alt.append(f)
            swapped.append(_all_pairs(n, alt))
    path_costs = [_simple_path_costs(n, edges, s) for s in range(n)]
    strong, weak = [], []
    for s in range(n):
        for t in range(n):
            if s == t or (s, t) in present or dist[s][t] == _INF:
                continue
            d = dist[s][t]
            costs = sorted(path_costs[s][t])
            # >= 3 genuine routes, a unique optimum, and a runner-up within 4 (reading matters)
            if len(costs) < 3 or costs[0] != d or not 1 <= costs[1] - d <= 4:
                continue
            count, path = _unique_shortest_path(n, edges, dist[s], s, t)
            if count != 1 or len(path) < 4:
                continue
            traps = [(e.u, e.v) for e, fd in zip(singles, flipped, strict=True) if fd[s][t] < d]
            if not traps or not any(sd[s][t] != d for sd in swapped):
                continue
            cand = (s, t, int(d), path, traps)
            (strong if len(traps) >= 2 else weak).append(cand)
    pool = strong or weak
    return rng.choice(pool) if pool else None


def _sample_digraph(rng: random.Random) -> DigraphSpec:
    for _attempt in range(3000):
        n = rng.randint(9, 11)
        pos = _digraph_layout(rng, n)
        cands = _candidate_pairs(pos)
        n_anti = rng.choice((3, 4, 4, 5, 5, 5))  # 2-cycles place less often: bias the prior up
        lo, hi = 16 - n_anti, min(22 - n_anti, len(cands))
        if hi < lo:
            continue
        for _subset in range(6):
            chosen = sorted(rng.sample(cands, rng.randint(lo, hi)))
            deg = [0] * n
            for a, b in chosen:
                deg[a] += 1
                deg[b] += 1
            long_pairs = [p for p in chosen if math.dist(pos[p[0]], pos[p[1]]) >= _G_MIN_ANTI_LEN]
            if min(deg) < 2 or len(long_pairs) < n_anti:
                continue
            anti = set(rng.sample(long_pairs, n_anti))
            curves: list[tuple[int, int, float]] = []
            for a, b in chosen:
                curves.append((a, b, _G_ARC_RAD if (a, b) in anti else 0.0))
                if (a, b) in anti:
                    curves.append((b, a, _G_ARC_RAD))
            if not _geometry_ok(pos, curves):
                continue
            label_xy = _place_labels(pos, curves)
            if label_xy is None:
                continue
            for _weights in range(10):
                edges: list[DigraphEdge] = []
                k = 0
                while k < len(curves):
                    a, b, rad = curves[k]
                    if rad:
                        w1, w2 = rng.sample(range(1, 10), 2)
                        edges += [DigraphEdge(a, b, w1, rad), DigraphEdge(b, a, w2, rad)]
                        k += 2
                    else:
                        u, v = (a, b) if rng.random() < 0.5 else (b, a)
                        edges.append(DigraphEdge(u, v, rng.randint(1, 9), 0.0))
                        k += 1
                picked = _choose_endpoints(rng, n, edges)
                if picked is None:
                    continue
                s, t, opt, path, traps = picked
                letters = sorted(rng.sample(_G_LETTERS, n))
                rng.shuffle(letters)
                return DigraphSpec(
                    labels=tuple(letters),
                    pos=tuple(pos),
                    edges=tuple(edges),
                    label_xy=tuple(label_xy),
                    source=s,
                    target=t,
                    optimum=opt,
                    path=tuple(path),
                    trap_edges=tuple(traps),
                )
    raise RuntimeError("could not sample a valid weighted digraph")


def _digraph_figure(spec: DigraphSpec) -> Figure:
    """Build (but do not save) the diagram; call inside `matplotlib.rc_context(_RC)`."""
    shrink = _G_NODE_R * _G_PX_PER_UNIT * 72.0 / _G_DPI + 0.9  # points: arrow tip on node rim
    fig = Figure(figsize=_G_FIGSIZE, dpi=_G_DPI)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(-_G_HALF_W, _G_HALF_W)
    ax.set_ylim(-_G_HALF_H, _G_HALF_H)
    ax.set_aspect("equal")
    ax.axis("off")
    for e in spec.edges:
        ax.add_patch(
            FancyArrowPatch(
                spec.pos[e.u],
                spec.pos[e.v],
                arrowstyle="-|>",
                mutation_scale=13,
                connectionstyle=f"arc3,rad={e.rad}",
                shrinkA=shrink,
                shrinkB=shrink,
                linewidth=1.3,
                color="#262626",
                zorder=2,
            )
        )
    for (x, y), name in zip(spec.pos, spec.labels, strict=True):
        ax.add_patch(
            Circle(
                (x, y), _G_NODE_R, facecolor="white", edgecolor="#111111", linewidth=1.6, zorder=3
            )
        )
        ax.text(x, y, name, ha="center", va="center", fontsize=14, fontweight="bold", zorder=4)
    for e, (lx, ly) in zip(spec.edges, spec.label_xy, strict=True):
        ax.text(
            lx,
            ly,
            str(e.weight),
            ha="center",
            va="center",
            fontsize=_G_LABEL_FONTSIZE,
            zorder=5,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none"},
        )
    return fig


def _render_digraph(spec: DigraphSpec, path: Path) -> None:
    with matplotlib.rc_context(cast(Any, _RC)):
        fig = _digraph_figure(spec)
        fig.savefig(path, dpi=_G_DPI, facecolor="white", metadata=_PNG_METADATA)


# ---------------------------------------------------------------------------------------------
# Subtype 1: unlabeled multi-series line chart
# ---------------------------------------------------------------------------------------------

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_SERIES_NAME_SETS = (
    ("North", "South", "West"),
    ("North", "South", "Online"),
    ("Online", "Retail", "Wholesale"),
    ("Direct", "Partner", "Online"),
    ("Domestic", "Export", "Online"),
    ("Enterprise", "SMB", "Consumer"),
    ("East", "West", "Central"),
    ("Plant A", "Plant B", "Plant C"),
)
_SERIES_STYLES = (
    ("#1f77b4", "o"),
    ("#d62728", "s"),
    ("#2ca02c", "^"),
    ("#9467bd", "D"),
    ("#ff7f0e", "v"),
    ("#17becf", "P"),
)
_CHART_TOPICS = (
    ("Customer Satisfaction Index", "Index (0-100)"),
    ("On-Time Fulfilment Rate", "Orders shipped on time (%)"),
    ("Capacity Utilisation", "Utilisation (%)"),
    ("Service Quality Score", "Score (0-100)"),
    ("Order Accuracy", "Accurate orders (%)"),
    ("Repeat-Purchase Rate", "Customers (%)"),
)


@dataclass(frozen=True)
class ChartSpec:
    kind: str  # "max_gap_month" | "months_above"
    title: str
    ylabel: str
    names: tuple[str, ...]  # legend order
    values: tuple[tuple[int, ...], ...]  # parallel to `names`
    styles: tuple[tuple[str, str], ...]  # (colour, marker) parallel to `names`
    x_idx: int  # series X in the question
    y_idx: int  # series Y in the question
    answer: int  # month number (1-12) or count of months


def _random_walk(rng: random.Random) -> list[int]:
    v, vel, out = rng.uniform(22.0, 78.0), rng.uniform(-4.0, 4.0), []
    for _ in range(12):
        out.append(v)
        vel = 0.6 * vel + rng.gauss(0.0, 5.5)
        v += vel
        if v < 10.0:
            v, vel = 20.0 - v, abs(vel)
        elif v > 90.0:
            v, vel = 180.0 - v, -abs(vel)
    return [round(min(90.0, max(10.0, x))) for x in out]


def _sample_series_chart(rng: random.Random, kind: str) -> ChartSpec:
    for _attempt in range(200000):
        x, y, z = _random_walk(rng), _random_walk(rng), _random_walk(rng)
        if kind == "max_gap_month":
            gaps = [abs(a - b) for a, b in zip(x, y, strict=True)]
            order = sorted(range(12), key=lambda m: (-gaps[m], m))
            top, second = gaps[order[0]], gaps[order[1]]
            if top < 12 or not 4 <= top - second <= 6:
                continue
            if sum(g >= top - 8 for g in gaps) < 3:  # competitive months: precision matters
                continue
            m_top = order[0]
            if m_top in (x.index(max(x)), x.index(min(x)), y.index(max(y)), y.index(min(y))):
                continue  # no "biggest gap sits at an extreme" shortcut
            answer = m_top + 1
            min_sep = 2
        else:
            d = [a - b for a, b in zip(x, y, strict=True)]
            if min(abs(v) for v in d) < 3:
                continue
            crossings = sum((d[i] > 0) != (d[i + 1] > 0) for i in range(11))
            if not 3 <= crossings <= 5 or sum(abs(v) <= 6 for v in d) < 3:
                continue
            answer = sum(v > 0 for v in d)
            min_sep = 3
        if any(
            abs(c - a) < min_sep or abs(c - b) < min_sep for a, b, c in zip(x, y, z, strict=True)
        ):
            continue  # the distractor series never hides a queried marker
        names = list(rng.choice(_SERIES_NAME_SETS))
        rng.shuffle(names)
        styles = rng.sample(_SERIES_STYLES, 3)
        topic, ylabel = rng.choice(_CHART_TOPICS)
        year = rng.randint(2021, 2026)
        series = [x, y, z]
        legend_order = [0, 1, 2]
        rng.shuffle(legend_order)
        return ChartSpec(
            kind=kind,
            title=f"Monthly {topic} by Segment, FY{year}",
            ylabel=ylabel,
            names=tuple(names[i] for i in legend_order),
            values=tuple(tuple(series[i]) for i in legend_order),
            styles=tuple(styles[i] for i in legend_order),
            x_idx=legend_order.index(0),
            y_idx=legend_order.index(1),
            answer=answer,
        )
    raise RuntimeError(f"could not sample a valid {kind} chart")


def _render_series_chart(spec: ChartSpec, path: Path) -> None:
    xs = list(range(1, 13))
    with matplotlib.rc_context(cast(Any, _RC)):
        fig = Figure(figsize=(10.0, 6.0), dpi=200)
        ax = fig.add_axes((0.075, 0.17, 0.905, 0.74))
        for i, (name, vals, (color, marker)) in enumerate(
            zip(spec.names, spec.values, spec.styles, strict=True)
        ):
            queried = i in (spec.x_idx, spec.y_idx)
            ax.plot(
                xs,
                vals,
                color=color,
                marker=marker,
                markersize=6.5,
                markeredgecolor="white",
                markeredgewidth=0.7,
                linewidth=1.9,
                label=name,
                zorder=4 if queried else 3,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(_MONTHS, fontsize=11)
        ax.set_xlim(0.5, 12.5)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(which="major", axis="y", color="#b8b8b8", linewidth=0.8)
        ax.grid(which="minor", axis="y", color="#e2e2e2", linewidth=0.6)
        ax.grid(which="major", axis="x", color="#ececec", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(spec.title, fontsize=14)
        ax.set_ylabel(spec.ylabel, fontsize=12)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=12
        )
        fig.savefig(path, dpi=200, facecolor="white", metadata=_PNG_METADATA)


# ---------------------------------------------------------------------------------------------
# Subtype 2: marker-class count inside a dashed region
# ---------------------------------------------------------------------------------------------

_SC_SPAN = 100.0
_SC_EDGE_MARGIN = 2.5  # % of span: every marker this far from the rectangle boundary
_SC_MIN_DIST = 3.0  # % of span: minimum centre-to-centre distance
_SC_BLUE = "#2a62c9"


@dataclass(frozen=True)
class ScatterSpec:
    rect: tuple[float, float, float, float]  # x0, y0, width, height (data units, span 0-100)
    points: tuple[tuple[float, float, str], ...]  # (x, y, cls) cls in filled | hollow | triangle
    answer: int


def _rect_boundary_dist(x: float, y: float, rect: tuple[float, float, float, float]) -> float:
    x0, y0, w, h = rect
    x1, y1 = x0 + w, y0 + h
    if x0 <= x <= x1 and y0 <= y <= y1:
        return min(x - x0, x1 - x, y - y0, y1 - y)
    return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))


def _inside(x: float, y: float, rect: tuple[float, float, float, float]) -> bool:
    return rect[0] < x < rect[0] + rect[2] and rect[1] < y < rect[1] + rect[3]


def _sample_marker_scatter(rng: random.Random) -> ScatterSpec:
    for _attempt in range(500):
        w, h = rng.uniform(32.0, 44.0), rng.uniform(32.0, 44.0)
        rect = (
            round(rng.uniform(12.0, 88.0 - w), 1),
            round(rng.uniform(12.0, 88.0 - h), 1),
            round(w, 1),
            round(h, 1),
        )
        k = rng.randint(7, 14)
        inside_classes = (
            ["hollow"] * k + ["filled"] * rng.randint(4, 9) + ["triangle"] * rng.randint(3, 7)
        )
        n_total = rng.randint(70, 110)
        n_out = n_total - len(inside_classes)
        outside_classes = [
            rng.choice(("hollow", "hollow", "filled", "filled", "triangle")) for _ in range(n_out)
        ]
        pts: list[tuple[float, float, str]] = []
        ok = True
        for want_inside, classes in ((True, inside_classes), (False, outside_classes)):
            for cls in classes:
                for _try in range(400):
                    if want_inside:
                        px = rng.uniform(
                            rect[0] + _SC_EDGE_MARGIN, rect[0] + rect[2] - _SC_EDGE_MARGIN
                        )
                        py = rng.uniform(
                            rect[1] + _SC_EDGE_MARGIN, rect[1] + rect[3] - _SC_EDGE_MARGIN
                        )
                    else:
                        px, py = rng.uniform(3.0, 97.0), rng.uniform(3.0, 97.0)
                        if _inside(px, py, rect):
                            continue
                    px, py = round(px, 2), round(py, 2)
                    if _rect_boundary_dist(px, py, rect) < _SC_EDGE_MARGIN:
                        continue
                    if any(math.hypot(px - qx, py - qy) < _SC_MIN_DIST for qx, qy, _c in pts):
                        continue
                    pts.append((px, py, cls))
                    break
                else:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        rng.shuffle(pts)
        answer = sum(1 for x, y, c in pts if c == "hollow" and _inside(x, y, rect))
        return ScatterSpec(rect=rect, points=tuple(pts), answer=answer)
    raise RuntimeError("could not sample a valid marker scatter")


def _render_marker_scatter(spec: ScatterSpec, path: Path) -> None:
    styles = {
        "filled": {
            "marker": "o",
            "s": 50,
            "facecolors": _SC_BLUE,
            "edgecolors": _SC_BLUE,
            "linewidths": 1.2,
        },
        "hollow": {
            "marker": "o",
            "s": 50,
            "facecolors": "white",
            "edgecolors": _SC_BLUE,
            "linewidths": 1.5,
        },
        "triangle": {
            "marker": "^",
            "s": 58,
            "facecolors": "#f28e2b",
            "edgecolors": "#b35c00",
            "linewidths": 0.6,
        },
    }
    labels = {
        "filled": "filled blue circle",
        "hollow": "hollow blue circle",
        "triangle": "orange triangle",
    }
    with matplotlib.rc_context(cast(Any, _RC)):
        fig = Figure(figsize=(8.0, 8.0), dpi=200)
        ax = fig.add_axes((0.08, 0.06, 0.88, 0.86))
        ax.set_xlim(0, _SC_SPAN)
        ax.set_ylim(0, _SC_SPAN)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.grid(color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
        x0, y0, w, h = spec.rect
        ax.add_patch(
            Rectangle(
                (x0, y0),
                w,
                h,
                fill=False,
                edgecolor="black",
                linewidth=1.6,
                linestyle=(0, (6, 4)),
                zorder=2,
            )
        )
        for cls in ("filled", "hollow", "triangle"):
            xs = [p[0] for p in spec.points if p[2] == cls]
            ys = [p[1] for p in spec.points if p[2] == cls]
            st = styles[cls]
            ax.scatter(
                xs,
                ys,
                zorder=3,
                marker=cast(Any, st["marker"]),
                s=cast(float, st["s"]),
                facecolors=cast(Any, st["facecolors"]),
                edgecolors=cast(Any, st["edgecolors"]),
                linewidths=cast(float, st["linewidths"]),
            )
        handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                marker=str(styles[c]["marker"]),
                markersize=8,
                markerfacecolor=str(styles[c]["facecolors"]),
                markeredgecolor=str(styles[c]["edgecolors"]),
                markeredgewidth=1.4,
                label=labels[c],
            )
            for c in ("filled", "hollow", "triangle")
        ]
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=3,
            frameon=False,
            fontsize=11,
        )
        fig.savefig(path, dpi=200, facecolor="white", metadata=_PNG_METADATA)


# ---------------------------------------------------------------------------------------------
# Image entry point
# ---------------------------------------------------------------------------------------------


def make_hard_image_item(
    rng: random.Random, item_id: str, hard_index: int, img_abs: Path, img_rel: str
) -> dict[str, Any]:
    """Render one perception-gated image asset to `img_abs` and return its dataset item."""
    img_abs = Path(img_abs)
    img_abs.parent.mkdir(parents=True, exist_ok=True)
    variant = hard_index % 3
    if variant == 0:
        g = _sample_digraph(rng)
        _render_digraph(g, img_abs)
        s, t = g.labels[g.source], g.labels[g.target]
        prompt = (
            "The attached diagram shows a directed graph. Each arrow is labeled with its traversal "
            "cost, and arrows can only be followed in the direction they point. What is the minimum "
            f"total cost of a path from node {s} to node {t}? "
            "End your response with `FINAL_ANSWER: <integer>`."
        )
        return _item(
            item_id, "image", "diagram_weighted_digraph_shortest_path",
            "Programmatic Matplotlib weighted directed-graph diagram with antiparallel arcs",
            prompt, img_rel, "numeric_tolerance", _numeric_spec(g.optimum),
        )  # fmt: skip
    if variant == 1:
        kind = "max_gap_month" if (hard_index // 3) % 2 == 0 else "months_above"
        c = _sample_series_chart(rng, kind)
        _render_series_chart(c, img_abs)
        x_name, y_name = c.names[c.x_idx], c.names[c.y_idx]
        series = f"{c.names[0]}, {c.names[1]}, and {c.names[2]}"
        intro = (
            f"The attached line chart plots monthly values (Jan-Dec) for three series: {series}. "
            "No data values are printed; read each plotted point against the y-axis gridlines "
            "(major gridlines every 10 units, lighter minor gridlines every 5 units). "
        )
        if kind == "max_gap_month":
            question = (
                f"In which month is the absolute gap between {x_name} and {y_name} the largest? "
                "Answer with the month number (1 = January, 2 = February, ..., 12 = December). "
                "End your response with `FINAL_ANSWER: <integer>`."
            )
        else:
            question = (
                f"In how many of the 12 months is the value for {x_name} strictly above the value "
                f"for {y_name}? End your response with `FINAL_ANSWER: <integer>`."
            )
        return _item(
            item_id, "image", "chart_unlabeled_series_gap",
            "Programmatic Matplotlib unlabeled three-series monthly line chart",
            intro + question, img_rel, "numeric_tolerance", _numeric_spec(c.answer),
        )  # fmt: skip
    sc = _sample_marker_scatter(rng)
    _render_marker_scatter(sc, img_abs)
    prompt = (
        "The attached scatter plot contains three kinds of markers: filled blue circles, hollow "
        "(unfilled, white-centred) blue circles, and filled orange triangles. A dashed rectangle "
        "outlines a region of the plot. How many hollow blue circles lie inside the dashed "
        "rectangle? Do not count filled circles, triangles, or any marker outside the rectangle. "
        "End your response with `FINAL_ANSWER: <integer>`."
    )
    return _item(
        item_id, "image", "scatter_marker_region_count",
        "Programmatic Matplotlib marker-class scatter plot with dashed region",
        prompt, img_rel, "numeric_tolerance", _numeric_spec(sc.answer),
    )  # fmt: skip


# ---------------------------------------------------------------------------------------------
# PDF: scanned audit packet with errata (no text layer)
# ---------------------------------------------------------------------------------------------

_PAGE_W, _PAGE_H = 612.0, 792.0
# Scan-degradation knobs (difficulty calibration). At these settings 9.5 pt table digits are
# ~13 px tall and clearly legible to a careful human; lower DPI / more blur makes OCR harder.
_SCAN_DPI = 100
_SCAN_BLUR = 0.6  # Gaussian blur radius (px)
_SCAN_NOISE_SD = 3.5  # Gaussian sensor-noise sigma (grey levels)
_SCAN_JPEG_QUALITY = 70  # mild JPEG artefacts, ~60-70 KB per page
_CONFUSABLE_DIGITS = "381756"
_DIGIT_SWAP = {"3": "8", "8": "3", "1": "7", "7": "1", "5": "6", "6": "5"}
_CONFUSION_KEY = str.maketrans("381756", "aabbcc")
_ASSET_PREFIXES = ("TX", "PX", "VX", "HX", "MX", "KX")
_DOC_PREFIXES = ("INV-AUD", "AST-AUD", "EQP-AUD")
_COMPANIES = (
    "Halvorsen Logistics", "Brightwater Utilities", "Kestrel Freight Co.", "Northgate Foods",
    "Meridian Rail Services", "Calder & Voss Manufacturing", "Ostara Cold Chain", "Pinecrest Medical",
)  # fmt: skip
_SITES = (
    "North Yard", "Dock 2", "Dock 5", "Bay C", "Annex", "Plant 4", "Depot West", "Cold Store",
    "Line 3", "Rail Spur",
)  # fmt: skip
_AUDITORS = (
    "Internal Audit",
    "Field Audit",
    "Quality Assurance",
    "Risk Controls",
    "Inventory Review",
    "Compliance Desk",
)
_INSPECTORS = ("QA-01", "QA-02", "QA-03", "OPS-1", "OPS-2", "ENG-1")


@dataclass(frozen=True)
class AssetRow:
    asset_id: str
    site: str
    status: str  # as printed in Table 2
    qty: int  # as printed in Table 2
    unit_cost: int  # as printed in Table 2 (USD)
    result: str  # PASS | FAIL (Table 3)
    inspected: str
    inspector: str


@dataclass(frozen=True)
class Erratum:
    asset_id: str
    column: str  # "Qty" | "Unit Cost" | "Status"
    correct: str
    printed: str


@dataclass(frozen=True)
class AuditSpec:
    doc_id: str
    company: str
    period: str
    auditor: str
    rows: tuple[AssetRow, ...]  # Table 2 order
    table3_order: tuple[int, ...]  # row indices in Table 3 order
    errata: tuple[Erratum, ...]
    filler_pages: tuple[str, ...]  # "site_notes" | "method"
    rotations: tuple[float, ...]  # per-page scan skew (degrees)
    noise_seeds: tuple[int, ...]  # per-page numpy seeds
    expected: dict[str, Any]


def _corrected(
    rows: tuple[AssetRow, ...] | list[AssetRow], errata: list[Erratum] | tuple[Erratum, ...]
) -> list[dict[str, Any]]:
    out = [
        {
            "id": r.asset_id,
            "status": r.status,
            "qty": r.qty,
            "cost": r.unit_cost,
            "result": r.result,
        }
        for r in rows
    ]
    by_id = {d["id"]: d for d in out}
    for e in errata:
        key = {"Qty": "qty", "Unit Cost": "cost", "Status": "status"}[e.column]
        by_id[e.asset_id][key] = e.correct if key == "status" else int(e.correct)
    return out


def _audit_totals(rows: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    active = [r for r in rows if r["status"] == "ACTIVE"]
    active_pass = [r for r in active if r["result"] == "PASS"]
    return (
        sum(r["qty"] for r in active),
        sum(r["qty"] * r["cost"] for r in active_pass),
        len(active),
        len(active_pass),
    )


def _confusable_number(rng: random.Random, lo: int, hi: int) -> int:
    """Two-digit number in [lo, hi] whose digits are mostly drawn from {3,8,1,7,5,6}."""
    while True:
        digits = [
            rng.choice(_CONFUSABLE_DIGITS) if rng.random() < 0.8 else str(rng.randint(0, 9))
            for _ in range(2)
        ]
        v = int("".join(digits))
        if lo <= v <= hi:
            return v


def _confusable_variant(rng: random.Random, value: int) -> int | None:
    """A plausible misprint of `value`: one confusable-digit swap or a transposition."""
    s = str(value)
    opts = {s[:i] + _DIGIT_SWAP[ch] + s[i + 1 :] for i, ch in enumerate(s) if ch in _DIGIT_SWAP}
    if len(s) == 2 and s[0] != s[1]:
        opts.add(s[::-1])
    valid = sorted(o for o in opts if o[0] != "0" and o != s and 10 <= int(o) <= 99)
    return int(rng.choice(valid)) if valid else None


def _sample_audit_packet(rng: random.Random) -> AuditSpec:
    for _attempt in range(20000):
        n_rows = rng.randint(10, 14)
        prefix = rng.choice(_ASSET_PREFIXES)
        ids: list[str] = []
        keys: set[str] = set()
        while len(ids) < n_rows:
            num = str(rng.randint(1, 9)) + "".join(
                rng.choice(_CONFUSABLE_DIGITS) if rng.random() < 0.6 else str(rng.randint(0, 9))
                for _ in range(3)
            )
            if num.translate(_CONFUSION_KEY) not in keys:  # IDs stay distinct even if 3/8 etc. blur
                keys.add(num.translate(_CONFUSION_KEY))
                ids.append(f"{prefix}-{num}")
        n_active = rng.randint(6, 8)
        n_hold = rng.randint(2, 3)
        statuses = (
            ["ACTIVE"] * n_active
            + ["HOLD"] * n_hold
            + ["RETIRED"] * max(1, n_rows - n_active - n_hold)
        )
        statuses = statuses[:n_rows]
        rng.shuffle(statuses)
        results = ["PASS" if rng.random() < 0.6 else "FAIL" for _ in range(n_rows)]
        qty = [_confusable_number(rng, 11, 98) for _ in range(n_rows)]
        cost = [_confusable_number(rng, 12, 98) for _ in range(n_rows)]
        # --- errata: corrections to specific Table 2 cells; the table prints the misprint
        rows_idx = list(range(n_rows))
        ap = [i for i in rows_idx if statuses[i] == "ACTIVE" and results[i] == "PASS"]
        if len(ap) < 3:
            continue
        errata: list[Erratum] = []
        used: set[int] = set()
        i1 = rng.choice(ap)
        bad = _confusable_variant(rng, qty[i1])
        if bad is None:
            continue
        errata.append(Erratum(ids[i1], "Qty", str(qty[i1]), str(bad)))
        printed_qty = list(qty)
        printed_qty[i1] = bad
        printed_cost = list(cost)
        printed_status = list(statuses)
        used.add(i1)
        second = rng.choice(("cost", "status_on", "status_off"))
        if second == "cost":
            i2 = rng.choice([i for i in ap if i not in used])
            bad = _confusable_variant(rng, cost[i2])
            if bad is None:
                continue
            errata.append(Erratum(ids[i2], "Unit Cost", str(cost[i2]), str(bad)))
            printed_cost[i2] = bad
        elif second == "status_on":  # printed HOLD, actually ACTIVE (PASS row -> joins both totals)
            pool = [
                i
                for i in rows_idx
                if statuses[i] == "ACTIVE" and results[i] == "PASS" and i not in used
            ]
            if not pool:
                continue
            i2 = rng.choice(pool)
            errata.append(Erratum(ids[i2], "Status", "ACTIVE", "HOLD"))
            printed_status[i2] = "HOLD"
        else:  # printed ACTIVE, actually RETIRED/HOLD (row must drop out of both totals)
            pool = [i for i in rows_idx if statuses[i] != "ACTIVE" and results[i] == "PASS"]
            if not pool:
                continue
            i2 = rng.choice(pool)
            errata.append(Erratum(ids[i2], "Status", statuses[i2], "ACTIVE"))
            printed_status[i2] = "ACTIVE"
        used.add(i2)
        if rng.random() < 0.5:  # a third, answer-neutral correction on an excluded row
            pool = [i for i in rows_idx if statuses[i] != "ACTIVE" and i not in used]
            if pool:
                i3 = rng.choice(pool)
                bad = _confusable_variant(rng, qty[i3])
                if bad is not None:
                    errata.append(Erratum(ids[i3], "Qty", str(qty[i3]), str(bad)))
                    printed_qty[i3] = bad
        rng.shuffle(errata)
        rows = tuple(
            AssetRow(
                asset_id=ids[i],
                site=rng.choice(_SITES),
                status=printed_status[i],
                qty=printed_qty[i],
                unit_cost=printed_cost[i],
                result=results[i],
                inspected=f"2025-{rng.randint(7, 9):02d}-{rng.randint(1, 28):02d}",
                inspector=rng.choice(_INSPECTORS),
            )
            for i in rows_idx
        )
        units, value, n_act, n_ap = _audit_totals(_corrected(rows, errata))
        _units0, value0, _na0, _nap0 = _audit_totals(_corrected(rows, []))
        if not (5 <= n_act <= 8 and 3 <= n_ap <= 5) or value == value0:
            continue
        relevant_cells = [
            str(r["qty"]) + str(r["cost"])
            for r in _corrected(rows, errata)
            if r["status"] == "ACTIVE" and r["result"] == "PASS"
        ]
        if sum(any(ch in "38" for ch in cell) for cell in relevant_cells) < 2:
            continue
        table3 = list(rows_idx)
        rng.shuffle(table3)
        if sum(a == b for a, b in zip(table3, rows_idx, strict=True)) > 2:
            continue  # Table 3 order must differ from Table 2: the join is by asset ID
        n_filler = rng.choice((0, 1, 1, 2))
        filler = tuple(rng.sample(("site_notes", "method"), n_filler))
        n_pages = 4 + n_filler
        doc_id = f"{rng.choice(_DOC_PREFIXES)}-{rng.randint(1000, 9999)}"
        return AuditSpec(
            doc_id=doc_id,
            company=rng.choice(_COMPANIES),
            period=f"Q{rng.randint(1, 4)} FY{rng.randint(2024, 2026)}",
            auditor=rng.choice(_AUDITORS),
            rows=rows,
            table3_order=tuple(table3),
            errata=tuple(errata),
            filler_pages=filler,
            rotations=tuple(rng.choice((-1, 1)) * rng.uniform(0.4, 1.0) for _ in range(n_pages)),
            noise_seeds=tuple(rng.getrandbits(32) for _ in range(n_pages)),
            expected={"doc_id": doc_id, "active_units": units, "reconciled_value_usd": value},
        )
    raise RuntimeError("could not sample a valid audit packet")


_SCALE = _SCAN_DPI / 72.0
_ANCHORS = {"left": "ls", "right": "rs", "center": "ms"}


def _font(size: float) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=max(6, round(size * _SCALE)))


def _line(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    fill: int = 0,
    width: float = 0.7,
) -> None:
    draw.line(
        [(p0[0] * _SCALE, p0[1] * _SCALE), (p1[0] * _SCALE, p1[1] * _SCALE)],
        fill=fill,
        width=max(1, round(width * _SCALE)),
    )


def _txt(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    s: str,
    size: float = 10.0,
    bold: bool = False,
    align: str = "left",
) -> None:
    draw.text(
        (x * _SCALE, y * _SCALE),
        s,
        font=_font(size),
        fill=0,
        anchor=_ANCHORS[align],
        stroke_width=1 if bold else 0,
    )


def _para(
    draw: ImageDraw.ImageDraw, rect: tuple[float, float, float, float], s: str, size: float = 10.5
) -> None:
    x0, y0, x1, y1 = rect
    font = _font(size)
    max_w = (x1 - x0) * _SCALE
    lines: list[str] = []
    cur = ""
    for word in s.split():
        cand = f"{cur} {word}" if cur else word
        if draw.textlength(cand, font=font) <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    line_step = size * 1.35 * _SCALE
    y_px = y0 * _SCALE + size * _SCALE
    for line in lines:
        if y_px > y1 * _SCALE:
            raise ValueError("paragraph overflow")
        draw.text((x0 * _SCALE, y_px), line, font=font, fill=0, anchor="ls")
        y_px += line_step


def _table(
    draw: ImageDraw.ImageDraw,
    top: float,
    cols: Sequence[tuple[str, float, str]],
    rows: list[list[str]],
    size: float = 10.0,
    row_h: float = 19.0,
) -> float:
    left, right = 62.0, 550.0
    _line(draw, (left, top), (right, top), fill=0, width=0.9)
    y = top + row_h
    for header, x, align in cols:
        _txt(draw, x, y - 6, header, size - 0.5, bold=True, align=align)
    _line(draw, (left, y), (right, y), fill=0, width=0.7)
    for cells in rows:
        y += row_h
        for (_h, x, align), cell in zip(cols, cells, strict=True):
            _txt(draw, x, y - 6, cell, size, align=align)
        _line(draw, (left, y), (right, y), fill=140, width=0.4)
    _line(draw, (left, y), (right, y), fill=0, width=0.9)
    return y


def _compose_packet(spec: AuditSpec) -> list[Image.Image]:
    n_pages = 4 + len(spec.filler_pages)
    kinds = ["cover", "table2", "table3", *spec.filler_pages, "errata"]
    w_px, h_px = round(_PAGE_W * _SCALE), round(_PAGE_H * _SCALE)
    pages: list[Image.Image] = []
    for p, kind in enumerate(kinds, start=1):
        img = Image.new("L", (w_px, h_px), 255)
        draw = ImageDraw.Draw(img)
        _txt(draw, 62, 48, f"{spec.company}  |  Asset Audit Packet", 9)
        _txt(draw, 550, 48, f"{spec.doc_id}   Page {p} of {n_pages}", 9, align="right")
        _line(draw, (62, 55), (550, 55), fill=0, width=0.6)
        _txt(
            draw, 306, 760, "Internal audit use only. Uncontrolled when printed.", 8, align="center"
        )
        if kind == "cover":
            _txt(draw, 62, 125, "Quarterly Asset Audit Packet", 22, bold=True)
            _txt(draw, 62, 152, spec.company, 13)
            _txt(draw, 62, 205, "Document ID:", 13, bold=True)
            _txt(draw, 170, 205, spec.doc_id, 16, bold=True)
            meta = [
                ("Audit period:", spec.period),
                ("Prepared by:", f"{spec.auditor} (Internal Audit)"),
                ("Assets in scope:", f"{len(spec.rows)} tracked assets"),
                ("Packet length:", f"{n_pages} pages"),
            ]
            for k, (label, value) in enumerate(meta):
                _txt(draw, 62, 240 + 20 * k, label, 11, bold=True)
                _txt(draw, 170, 240 + 20 * k, value, 11)
            _txt(draw, 62, 345, "Scope", 12, bold=True)
            _para(
                draw,
                (62, 355, 550, 470),
                "This packet records the physical count of tracked equipment for the audit period "
                "above. Table 2 (page 2) is the asset register as printed, listing each asset's "
                "site, status, counted quantity, and unit cost. Table 3 (page 3) lists the result of "
                "each asset's safety inspection. Corrections found after printing are listed on the "
                "ERRATA page at the end of this packet; they supersede the printed tables.",
            )
            _txt(draw, 62, 505, "Contents", 12, bold=True)
            contents = [
                "Page 1 - Cover and scope",
                "Page 2 - Table 2: Tracked Asset Register",
                "Page 3 - Table 3: Inspection Results",
            ]
            contents += [
                f"Page {4 + i} - Supplementary notes" for i in range(len(spec.filler_pages))
            ]
            contents.append(f"Page {n_pages} - ERRATA and reconciliation rule")
            for k, line in enumerate(contents):
                _txt(draw, 74, 528 + 18 * k, line, 10.5)
        elif kind == "table2":
            _txt(draw, 62, 95, "Table 2 - Tracked Asset Register", 14, bold=True)
            _txt(
                draw,
                62,
                114,
                "Status codes: ACTIVE = in service; HOLD = quarantined; RETIRED = written off.",
                9.5,
            )
            cols = [("Asset ID", 70, "left"), ("Site", 165, "left"), ("Status", 290, "left"),
                    ("Qty", 420, "right"), ("Unit Cost (USD)", 540, "right")]  # fmt: skip
            body = [[r.asset_id, r.site, r.status, str(r.qty), str(r.unit_cost)] for r in spec.rows]
            _table(draw, 132, cols, body, size=9.5)
        elif kind == "table3":
            _txt(draw, 62, 95, "Table 3 - Inspection Results", 14, bold=True)
            _txt(
                draw,
                62,
                114,
                "Listed in inspection order. Result applies to the asset as counted in Table 2.",
                9.5,
            )
            cols = [("Asset ID", 70, "left"), ("Inspected", 200, "left"), ("Inspector", 330, "left"),
                    ("Result", 540, "right")]  # fmt: skip
            body = [
                [
                    spec.rows[i].asset_id,
                    spec.rows[i].inspected,
                    spec.rows[i].inspector,
                    spec.rows[i].result,
                ]
                for i in spec.table3_order
            ]
            _table(draw, 132, cols, body, size=9.5)
        elif kind == "site_notes":
            _txt(draw, 62, 95, "Supplementary Notes - Site Observations", 14, bold=True)
            _para(
                draw,
                (62, 115, 550, 420),
                "Counts were taken during scheduled shift changes to avoid double counting of units "
                "in transit. Dock areas were counted twice by separate teams and reconciled on site. "
                "Units tagged for relocation were counted at their origin site. Seals and tags were "
                "checked against the register; damaged tags were re-issued without changing the "
                "asset ID. Ambient conditions in cold storage areas required gloves, which slowed "
                "the count but did not affect results. No safety incidents were recorded during "
                "the audit window.",
            )
        elif kind == "method":
            _txt(draw, 62, 95, "Supplementary Notes - Count Method and Sign-off", 14, bold=True)
            _para(
                draw,
                (62, 115, 550, 360),
                "Each asset was located by its register entry, scanned, and physically counted. "
                "Quantities are whole units. Unit costs are standard costs in US dollars taken from "
                "the fixed-asset ledger at period open. Inspection results were recorded by the "
                "inspector named in Table 3 and countersigned by the site lead.",
            )
            for k, role in enumerate(("Auditor", "Site lead", "Controller")):
                y = 430 + 60 * k
                _line(draw, (62, y), (300, y), fill=0, width=0.6)
                _txt(draw, 62, y + 14, f"{role} signature / date", 9)
        else:  # errata
            _txt(draw, 62, 100, "ERRATA", 20, bold=True)
            _para(
                draw,
                (62, 115, 550, 175),
                "The following corrections to Table 2 (page 2) were identified after printing. "
                "Apply every correction to Table 2 before computing any total.",
            )
            y = 200.0
            for k, e in enumerate(spec.errata, start=1):
                _txt(draw, 74, y, f"{k}.", 11.5, bold=True)
                _txt(
                    draw,
                    96,
                    y,
                    f"{e.asset_id}: {e.column} should read {e.correct} (printed as {e.printed}).",
                    11.5,
                )
                y += 24
            y += 24
            _txt(draw, 62, y, "Reconciliation rule (apply after all errata)", 12.5, bold=True)
            rule = [
                "(a) Active units = sum of Qty over assets whose Status is ACTIVE.",
                "(b) Reconciled value (USD) = sum of Qty x Unit Cost over ACTIVE assets",
                "     whose inspection Result in Table 3 is PASS.",
                "(c) HOLD and RETIRED assets are excluded from both totals.",
            ]
            for k, line in enumerate(rule):
                _txt(draw, 74, y + 24 + 19 * k, line, 11)
        pages.append(img)
    return pages


def _scan_page(clean: Image.Image, angle: float, seed: int) -> Image.Image:
    """Rasterise a composed page into a skewed, blurred, noisy grayscale 'scan'."""
    img = clean.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(radius=_SCAN_BLUR))
    nrng = np.random.default_rng(seed)
    a = np.asarray(img, dtype=np.float32) / 255.0
    h, w = a.shape
    gx, gy = nrng.uniform(-6.0, 6.0, size=2)
    bg = (
        236.0
        + gx * (np.linspace(-0.5, 0.5, w)[None, :])
        + gy * (np.linspace(-0.5, 0.5, h)[:, None])
    )
    ink = 28.0
    out = ink + (bg - ink) * a + nrng.normal(0.0, _SCAN_NOISE_SD, size=(h, w))
    # Sparse dust specks, kept well away from any ink so they never touch a digit.
    far = np.asarray(img.filter(ImageFilter.MinFilter(15)), dtype=np.float32) > 250.0
    n_dust = int(nrng.integers(40, 110))
    ys, xs = nrng.integers(0, h, n_dust), nrng.integers(0, w, n_dust)
    keep = far[ys, xs]
    out[ys[keep], xs[keep]] -= nrng.uniform(70.0, 140.0, int(keep.sum()))
    scan = Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    scan.save(buf, format="JPEG", quality=_SCAN_JPEG_QUALITY, optimize=True)
    buf.seek(0)
    with Image.open(buf) as jpg_img:
        return jpg_img.convert("L")


def _render_audit_packet(spec: AuditSpec, path: Path) -> None:
    clean_pages = _compose_packet(spec)
    scanned = [
        _scan_page(page, angle, seed)
        for page, angle, seed in zip(clean_pages, spec.rotations, spec.noise_seeds, strict=True)
    ]
    scanned[0].save(
        path,
        format="PDF",
        save_all=True,
        append_images=scanned[1:],
        resolution=_SCAN_DPI,
    )
    reader = PdfReader(path)
    for pg_num, page in enumerate(reader.pages):
        if (page.extract_text() or "").strip() != "":
            raise AssertionError(f"scanned page {pg_num + 1} unexpectedly has a text layer")


def make_hard_pdf_item(
    rng: random.Random, item_id: str, hard_index: int, pdf_abs: Path, pdf_rel: str
) -> dict[str, Any]:
    """Render one scanned (text-layer-free) errata audit packet to `pdf_abs` and return its item."""
    del hard_index  # every hard PDF uses the same subtype; kept for API symmetry
    pdf_abs = Path(pdf_abs)
    pdf_abs.parent.mkdir(parents=True, exist_ok=True)
    spec = _sample_audit_packet(rng)
    _render_audit_packet(spec, pdf_abs)
    prompt = (
        "The attached PDF is a scanned asset audit packet. Use the cover page, Table 2 (asset "
        "register, page 2), Table 3 (inspection results, page 3), and the ERRATA page at the end. "
        "Apply every erratum to Table 2 before computing anything, then follow the reconciliation "
        "rule printed on the ERRATA page. Return ONLY a JSON object with keys `doc_id` (string, the "
        "document ID on the cover page), `active_units` (int, total Qty of assets whose Status is "
        "ACTIVE after the errata), and `reconciled_value_usd` (int, the reconciled value defined by "
        "the rule)."
    )
    return _item(
        item_id, "pdf", "scanned_pdf_errata_reconciliation",
        "Programmatic scanned-style audit packet: Pillow layout rasterized, no text layer",
        prompt, pdf_rel, "exact_json", {"expected": dict(spec.expected)},
    )  # fmt: skip
