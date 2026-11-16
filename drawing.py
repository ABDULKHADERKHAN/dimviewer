"""Reference drawing from a 3D model (opens in any browser, prints to PDF).

Sheet 1: front, top and right views (third angle) plus an isometric view, overall sizes,
and the features that belong to those views.
Far-side sheet (only when needed): bottom, back or left view for features that open there,
so no single view carries everything.
Last sheet: two section views with the most common wall and floor thickness.

Dimensioning rules, to keep every view readable:
  - each feature gets one size callout and one centre position, nothing repeated
  - equal features in a view share one callout with a count (2X ...)
  - a countersink or counterbore is part of its hole's callout
  - a through hole is dimensioned once, in the main view; a blind feature in the view
    that looks at the side it opens on
  - a view with more than six features shows tags only, and the table carries the numbers

Openings are read from the rims they leave in flat faces, so straight, tapered and
chamfered holes are all found. A model file holds no tolerances and no design intent,
so no tolerance values or feature control frames are invented.

    from model import Model, Units
    from drawing import make_drawing
    make_drawing(Model("part.stl"), Units("mm", "mm"), "part.html")
"""
# Part of DimViewer. Copyright (c) 2026 Abdul Khader Khan.
# Free for noncommercial use. License and contact details are in README.md.

import datetime
import math
import os

import numpy as np

from model import DIA, _circle2d, axis_name

SHEET_W, SHEET_H = 431.8, 279.4                 # ANSI B (11 x 17 in), in mm
TXT = 3.0                                       # dimension text height, mm on the sheet
SCALES = [(50, "50:1"), (20, "20:1"), (10, "10:1"), (5, "5:1"), (4, "4:1"), (2, "2:1"), (1, "1:1"),
          (0.5, "1:2"), (0.25, "1:4"), (0.2, "1:5"), (0.1, "1:10"), (0.05, "1:20"), (0.02, "1:50"),
          (0.01, "1:100"), (0.005, "1:200"), (0.002, "1:500"), (0.001, "1:1000")]
_Z = np.array([0.0, 0.0, 1.0])
_ISO = np.array([1.0, -1.0, 1.0]) / math.sqrt(3.0)
VIEWS = {                                        # name: (direction toward the viewer, right, up)
    "FRONT": (np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), _Z),
    "TOP": (_Z, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    "RIGHT": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), _Z),
}
_iso_up = _Z - (_Z @ _ISO) * _ISO
_iso_up /= np.linalg.norm(_iso_up)
VIEWS["ISO"] = (_ISO, np.cross(-_ISO, _iso_up), _iso_up)
AXIS_VIEW = {"Z": "TOP", "Y": "FRONT", "X": "RIGHT"}    # view that looks along each axis
# views of the far sides, used only when a feature opens there
VIEWS["BOTTOM"] = (-_Z, np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]))
VIEWS["BACK"] = (np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0]), _Z)
VIEWS["LEFT"] = (np.array([-1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), _Z)
PRIMARY_OF = {"TOP": "TOP", "BOTTOM": "TOP", "FRONT": "FRONT", "BACK": "FRONT", "RIGHT": "RIGHT", "LEFT": "RIGHT"}
SIDE_VIEW = {2: {1: "TOP", -1: "BOTTOM"}, 1: {-1: "FRONT", 1: "BACK"}, 0: {1: "RIGHT", -1: "LEFT"}}   # axis, side it opens on
VIEW_ORDER = ["TOP", "FRONT", "RIGHT", "BOTTOM", "BACK", "LEFT"]


# ----------------------------------------------------------------------------
# Hidden-line removal
# ----------------------------------------------------------------------------
def _near3(z, fn):
    """3x3 neighbourhood max or min of a 2D array."""
    p = np.pad(z, 1, mode="edge")
    h, w = z.shape
    return fn(np.stack([p[i:i + h, j:j + w] for i in range(3) for j in range(3)]), axis=0)


def _depth_buffer(m, d, r, u, x0, y0, px, W, H):
    """Nearest surface depth per pixel, built by scattering sample points over every
    triangle that faces the viewer (no per-triangle loop, so it stays fast)."""
    nd = m.N @ d
    tris = np.flatnonzero(nd > 1e-9)
    z = np.full(W * H, -np.inf)
    if len(tris) == 0:
        return z.reshape(H, W)
    for pix, depth in _scatter(m, tris, nd[tris], d, r, u, x0, y0, px, W, H):
        np.maximum.at(z, pix, depth)
    # Scattered samples miss a few pixels of the nearest surface, which would let whatever
    # is behind show through. Taking the nearest depth of each 3x3 block closes those gaps.
    return _near3(z.reshape(H, W), np.max)


def view_lines(m, d, r, u):
    """Visible and hidden edge segments of the model seen along d.
    Returns (visible, hidden), each an (n,4) array of x1, y1, x2, y2 in model units."""
    V = m.V
    two = m.et1 >= 0
    nd = m.N @ d
    s0, s1 = nd[m.et0], nd[np.maximum(m.et1, 0)]
    la, lb = m.labels[m.et0], m.labels[np.maximum(m.et1, 0)]
    crease = ~two | (m.ecos < math.cos(math.radians(40))) \
        | ((la != lb) & (m.ecos < math.cos(math.radians(12))))
    eps = 1e-9
    outline = two & (((s0 > eps) & (s1 <= eps)) | ((s1 > eps) & (s0 <= eps)))
    e = np.flatnonzero(crease | outline)
    A, B = V[m.ev[e, 0]], V[m.ev[e, 1]]
    a2, b2 = np.c_[A @ r, A @ u], np.c_[B @ r, B @ u]
    xy = np.c_[V @ r, V @ u]
    x0, y0 = xy.min(0)
    ext = xy.max(0) - xy.min(0)
    px = max(ext.max(), 1e-12) / 900.0
    W, H = int(ext[0] / px) + 13, int(ext[1] / px) + 13    # 6 empty pixels all round, so the
    x0, y0 = x0 - 6 * px, y0 - 6 * px                        # part's outline has background behind it
    # ... and stepping back two pixels keeps edges that lie right on an outline visible
    zfar = _near3(_near3(_depth_buffer(m, d, r, u, x0, y0, px, W, H), np.min), np.min)

    length = np.linalg.norm(b2 - a2, axis=1) / px
    keep = length > 0.5                         # edges seen end-on project to a point
    A, B, a2, b2, length = A[keep], B[keep], a2[keep], b2[keep], length[keep]
    if len(A) == 0:
        return np.zeros((0, 4)), np.zeros((0, 4))
    k = np.clip(np.ceil(length / 4.0), 1, 500).astype(np.int64)
    idx = np.repeat(np.arange(len(A)), k)
    i = np.arange(len(idx)) - np.repeat(np.cumsum(k) - k, k)
    t = (i + 0.5) / k[idx]
    M = A[idx] + t[:, None] * (B - A)[idx]
    ix = np.clip(((M @ r - x0) / px).astype(np.int64), 0, W - 1)
    iy = np.clip(((M @ u - y0) / px).astype(np.int64), 0, H - 1)
    vis = (M @ d) >= zfar[iy, ix] - (2.5 * px + 1e-5 * m.diag)

    brk = np.ones(len(idx), bool)               # merge neighbouring pieces of equal visibility
    brk[1:] = (idx[1:] != idx[:-1]) | (vis[1:] != vis[:-1])
    rs = np.flatnonzero(brk)
    re = np.append(rs[1:], len(idx))
    ej = idx[rs]
    t0, t1 = i[rs] / k[ej], (i[re - 1] + 1) / k[ej]
    dv = (b2 - a2)[ej]
    seg = np.c_[a2[ej] + t0[:, None] * dv, a2[ej] + t1[:, None] * dv]
    q = np.round(seg / (0.05 * px)).astype(np.int64)
    swap = (q[:, 0] > q[:, 2]) | ((q[:, 0] == q[:, 2]) & (q[:, 1] > q[:, 3]))
    q[swap] = q[swap][:, [2, 3, 0, 1]]
    v = vis[rs]
    out = []
    for flag in (True, False):                  # drop exact duplicates (stacked edges)
        _, first = np.unique(q[v == flag], axis=0, return_index=True)
        out.append(seg[v == flag][first])
    return out[0], out[1]


# ----------------------------------------------------------------------------
# Sheet: drawing primitives collected in mm with y up, written out as SVG
# ----------------------------------------------------------------------------
class Sheet:
    """One drawing sheet. Shapes are added in mm with y pointing up, then written out as SVG."""
    STYLE = {"vis": "stroke-width:0.5", "hid": "stroke-width:0.22;stroke-dasharray:2 1",
             "thin": "stroke-width:0.18", "ctr": "stroke-width:0.18;stroke-dasharray:7 1.5 1 1.5",
             "frame": "stroke-width:0.7", "cut": "stroke-width:0.35;stroke-dasharray:9 1.5 1.5 1.5 1.5 1.5"}

    def __init__(self):
        """Start an empty sheet."""
        self.paths = {k: [] for k in self.STYLE}
        self.items = []
        self.under = []                                  # drawn first, beneath the lines

    def line(self, x1, y1, x2, y2, cls="thin"):
        """One straight line. cls picks the line style (vis, hid, thin, ctr, frame)."""
        self.paths[cls].append("M%.3f %.3fL%.3f %.3f" % (x1, SHEET_H - y1, x2, SHEET_H - y2))

    def lines(self, seg, cls):
        """Many lines from an (n,4) array of x1, y1, x2, y2."""
        for x1, y1, x2, y2 in seg:
            self.line(x1, y1, x2, y2, cls)

    def rect(self, x, y, w, h, cls="thin"):
        """Rectangle outline with its lower-left corner at (x, y)."""
        for a, b, c, d in ((x, y, x + w, y), (x + w, y, x + w, y + h), (x + w, y + h, x, y + h), (x, y + h, x, y)):
            self.line(a, b, c, d, cls)

    def circle(self, x, y, r, cls="thin"):
        """Circle outline."""
        self.items.append('<circle cx="%.3f" cy="%.3f" r="%.3f" fill="none" stroke="#000" style="%s"/>'
                          % (x, SHEET_H - y, r, self.STYLE[cls]))

    def dot(self, x, y, r=0.55):
        """Small filled dot: marks the point a position value is measured to."""
        self.items.append('<circle cx="%.3f" cy="%.3f" r="%.3f" fill="#000"/>' % (x, SHEET_H - y, r))

    def poly(self, pts):
        """Filled black polygon (used for arrowheads and datum triangles)."""
        self.items.append('<polygon points="%s" fill="#000"/>'
                          % " ".join("%.3f,%.3f" % (x, SHEET_H - y) for x, y in pts))

    def text(self, x, y, s, size=TXT, anchor="middle", bold=False, box=False, rot=0):
        """Text with its vertical centre at y. box=True puts a white patch behind it.
        rot=90 turns the text to read upward."""
        s = str(s)
        if box and not rot:
            w = 0.62 * size * len(s) + 1.2
            x0 = x - w / 2 if anchor == "middle" else (x - 0.6 if anchor == "start" else x - w + 0.6)
            self.items.append('<rect x="%.3f" y="%.3f" width="%.3f" height="%.3f" fill="#fff"/>'
                              % (x0, SHEET_H - y - size * 0.75, w, size * 1.5))
        esc = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        turn = ' transform="rotate(%d %.3f %.3f)"' % (-rot, x, SHEET_H - y) if rot else ""
        self.items.append('<text x="%.3f" y="%.3f" font-size="%.2f" text-anchor="%s"%s%s>%s</text>'
                          % (x, SHEET_H - y + size * 0.36, size, anchor,
                             ' font-weight="bold"' if bold else "", turn, esc))

    def arrow(self, x, y, dx, dy, size=3.0):
        """Filled arrowhead with its tip at (x, y), pointing along (dx, dy)."""
        n = math.hypot(dx, dy) or 1.0
        dx, dy = dx / n, dy / n
        bx, by = x - dx * size, y - dy * size
        self.poly([(x, y), (bx - dy * size / 6, by + dx * size / 6), (bx + dy * size / 6, by - dx * size / 6)])

    def dim(self, p1, p2, at, label, vertical=False):
        """Linear dimension between two points. `at` is where the dimension line sits:
        its y for a horizontal dimension, its x for a vertical one."""
        (x1, y1), (x2, y2) = p1, p2
        w = 0.62 * TXT * len(label) + 2.0
        if not vertical:
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            for x, y in ((x1, y1), (x2, y2)):                      # extension lines
                s = 1.0 if at > y else -1.0
                self.line(x, y + s * 1.2, x, at + s * 2.0)
            if x2 - x1 >= w + 8.0:
                self.line(x1, at, x2, at)
                self.arrow(x1, at, -1, 0)
                self.arrow(x2, at, 1, 0)
                self.text((x1 + x2) / 2, at, label, box=True)
            else:                                                  # no room: arrows outside
                self.line(x1 - 7, at, x2 + 9, at)
                self.arrow(x1, at, 1, 0)
                self.arrow(x2, at, -1, 0)
                self.text(x2 + 10, at, label, anchor="start", box=True)
        else:
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            for x, y in ((x1, y1), (x2, y2)):
                s = 1.0 if at > x else -1.0
                self.line(x + s * 1.2, y, at + s * 2.0, y)
            if y2 - y1 >= TXT * 1.6 + 8.0:
                self.line(at, y1, at, y2)
                self.arrow(at, y1, 0, -1)
                self.arrow(at, y2, 0, 1)
                self.text(at, (y1 + y2) / 2, label, box=True)
            else:
                self.line(at, y1 - 7, at, y2 + 7)
                self.arrow(at, y1, 0, 1)
                self.arrow(at, y2, 0, -1)
                self.text(at, y2 + 7 + TXT, label, box=True)

    def datum(self, x, y, nx, ny, letter):
        """Datum feature symbol: filled triangle on the surface line, stem, boxed letter."""
        t = 2.6
        px, py = -ny, nx
        self.poly([(x - px * t / 2, y - py * t / 2), (x + px * t / 2, y + py * t / 2), (x + nx * t, y + ny * t)])
        self.line(x + nx * t, y + ny * t, x + nx * 5.5, y + ny * 5.5)
        cx, cy = x + nx * 8.5, y + ny * 8.5
        self.rect(cx - 3, cy - 3, 6, 6)
        self.text(cx, cy, letter, size=3.5, bold=True)

    def hatch(self, loops):
        """Cut material: the outlines filled with section lining (45 degree hatch)."""
        d = "".join("M" + "L".join("%.3f %.3f" % (x, SHEET_H - y) for x, y in lp) + "Z" for lp in loops)
        self.under.append('<path d="%s" fill="url(#hatch)" fill-rule="evenodd" stroke="#000" '
                          'stroke-linejoin="round" style="stroke-width:0.5"/>' % d)

    def svg(self):
        """The whole sheet as SVG text."""
        out = ['<svg xmlns="http://www.w3.org/2000/svg" width="17in" height="11in" viewBox="0 0 %.1f %.1f" '
               'font-family="Arial, Helvetica, sans-serif">' % (SHEET_W, SHEET_H),
               '<defs><pattern id="hatch" width="2.2" height="2.2" patternUnits="userSpaceOnUse" '
               'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="2.2" stroke="#000" '
               'stroke-width="0.16"/></pattern></defs>',
               '<rect width="100%" height="100%" fill="#fff"/>'] + self.under
        for cls in ("hid", "ctr", "thin", "cut", "vis", "frame"):
            if self.paths[cls]:
                out.append('<path d="%s" fill="none" stroke="#000" stroke-linecap="round" style="%s"/>'
                           % ("".join(self.paths[cls]), self.STYLE[cls]))
        return "\n".join(out + self.items + ["</svg>"])


def _num(v_mm, unit):
    """Number formatting per ASME Y14.5: mm has no trailing zeros, inch has no leading zero."""
    if unit == "mm":
        s = ("%.3f" % v_mm).rstrip("0").rstrip(".")
        return "0" if s in ("", "-0") else s
    s = "%.4f" % (v_mm / 25.4)
    return s[1:] if s.startswith("0.") else s


# ----------------------------------------------------------------------------
# Reading features off the model for the drawing
# ----------------------------------------------------------------------------
def _walk_loops(edges, other):
    """Join loose edges (pairs of node numbers) into closed loops.
    Returns [(node numbers in order, set of the `other` values met along the loop)]."""
    pairs = edges.tolist()
    adj = {}
    for i, (a, b) in enumerate(pairs):
        adj.setdefault(a, []).append((b, i))
        adj.setdefault(b, []).append((a, i))
    used = [False] * len(pairs)
    loops = []
    for i0, (a, b) in enumerate(pairs):
        if used[i0]:
            continue
        used[i0] = True
        nodes, seen, cur = [a, b], {int(other[i0])}, b
        while cur != a:
            nxt = next(((v, i) for v, i in adj[cur] if not used[i]), None)
            if nxt is None:
                break
            used[nxt[1]] = True
            seen.add(int(other[nxt[1]]))
            cur = nxt[0]
            nodes.append(cur)
        if cur == a and len(nodes) > 3:
            loops.append((np.array(nodes[:-1]), seen))
    return loops


def _region_edges(m):
    """For every surface: the mesh edges on its boundary and the surface across each edge."""
    feat = m.feat
    la = m.labels[m.et0[feat]]
    lb = np.where(m.et1[feat] >= 0, m.labels[np.maximum(m.et1[feat], 0)], -1)
    lab, other = np.concatenate([la, lb]), np.concatenate([lb, la])
    idx = np.concatenate([np.arange(len(feat))] * 2)
    order = np.argsort(lab, kind="stable")
    out = {}
    if len(order) == 0:
        return out
    ls = lab[order]
    for grp in np.split(order, np.flatnonzero(ls[1:] != ls[:-1]) + 1):
        if lab[grp[0]] >= 0:
            out[int(lab[grp[0]])] = (m.ev[feat][idx[grp]], other[grp])
    return out


def classify_loop(P, tol):
    """Name the shape of a closed 2D outline: round, slot (two straight sides with full
    round ends), rect, or free. Returns a dict with its centre and sizes."""
    x, y = P[:, 0], P[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y2 - x2 * y
    area = cr.sum() / 2.0
    if abs(area) < tol * tol:
        return None
    c = np.array([((x + x2) * cr).sum(), ((y + y2) * cr).sum()]) / (6.0 * area)
    out = dict(kind="free", c=c, area=abs(area), w=float(np.ptp(x)), l=float(np.ptp(y)),
               c_box=np.array([(x.max() + x.min()) / 2, (y.max() + y.min()) / 2]))
    if len(P) >= 6:
        cx, cy, r = _circle2d(P)
        if r is not None and np.abs(np.hypot(x - cx, y - cy) - r).max() <= 0.02 * r + tol:
            out.update(kind="round", c=np.array([cx, cy]), d=2 * r)
            return out
    seg = np.c_[x2 - x, y2 - y]
    L = np.linalg.norm(seg, axis=1)
    mid = np.c_[(x + x2) / 2, (y + y2) / 2] - c
    frames = [np.eye(2)]                                 # try the view's own axes, then the outline's
    w, v = np.linalg.eigh((mid * L[:, None]).T @ mid)
    if w[1] > 1.02 * w[0]:
        frames.append(np.c_[v[:, 1], v[:, 0]])
    for fr in frames:
        a, b = (P - c) @ fr[:, 0], (P - c) @ fr[:, 1]
        ca, cb = (a.max() + a.min()) / 2, (b.max() + b.min()) / 2
        a, b = a - ca, b - cb
        la, lb = np.ptp(a), np.ptp(b)
        if la < lb:
            a, b, la, lb = b, a, lb, la
        centre = c + fr[:, 0] * ca + fr[:, 1] * cb
        h = (la - lb) / 2.0
        if h > 0.05 * lb:
            dist = np.hypot(np.maximum(np.abs(a) - h, 0.0), b)
            if np.abs(dist - lb / 2).max() <= 0.03 * lb + tol:
                out.update(kind="slot", c=centre, w=float(lb), l=float(la))
                return out
        if np.minimum(np.abs(np.abs(a) - la / 2), np.abs(np.abs(b) - lb / 2)).max() <= 0.01 * la + tol \
                and len(P) >= 4:
            out.update(kind="rect", c=centre, w=float(lb), l=float(la))
            return out
    if out["area"] >= 0.8 * out["w"] * out["l"]:           # a rectangle with rounded corners
        w_, l_ = sorted((out["w"], out["l"]))
        out.update(kind="rect", c=out["c_box"], w=w_, l=l_)
    return out


class _Rays:
    """Finds where a line parallel to X, Y or Z passes through the surface of the part."""

    def __init__(self, m):
        self.m, self.cache = m, {}

    def hits(self, a, p):
        """Coordinates along axis a where the line through point p meets the surface."""
        i, j = [q for q in range(3) if q != a]
        if a not in self.cache:
            T = self.m.V[self.m.F]
            self.cache[a] = (T[:, :, i], T[:, :, j], T[:, :, a])
        xi, yj, za = self.cache[a]
        px, py = p[i], p[j]
        c = np.flatnonzero((xi.min(1) <= px) & (xi.max(1) >= px) & (yj.min(1) <= py) & (yj.max(1) >= py))
        if len(c) == 0:
            return np.zeros(0)
        ax, ay, bx, by, cx, cy = xi[c, 0], yj[c, 0], xi[c, 1], yj[c, 1], xi[c, 2], yj[c, 2]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        ok = np.abs(det) > 1e-300
        det = np.where(ok, det, 1.0)
        l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
        l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
        l3 = 1.0 - l1 - l2
        inside = ok & (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        return (l1 * za[c, 0] + l2 * za[c, 1] + l3 * za[c, 2])[inside]


def find_features(m):
    """Openings and raised features, read from the outlines of the flat faces that lie square
    to X, Y or Z. Works for straight, tapered and chamfered holes alike, because it looks at
    the rim of each opening, not at the wall behind it.
    Returns a list of dicts: axis, kind, sense ('hole' or 'boss'), centre, sizes, thru, depth."""
    if m.lazy:
        return []
    tol = 2e-6 * m.diag
    edges = _region_edges(m)
    rays = _Rays(m)
    mean_cache = {}

    def mean_coord(j, a):
        """Average position of surface j along axis a."""
        if (j, a) not in mean_cache:
            mean_cache[(j, a)] = float(m.V[np.unique(m.F[m.regs[j]])][:, a].mean())
        return mean_cache[(j, a)]

    stacks = {}
    for rid, (ev, other) in edges.items():
        f = m.sfit(rid)
        if f["kind"] != "plane" or not axis_name(f["normal"]):
            continue
        a = "XYZ".index(axis_name(f["normal"]))
        sgn = 1.0 if f["normal"][a] > 0 else -1.0
        level = float(f["point"][a])
        d, r, u = VIEWS[AXIS_VIEW["XYZ"[a]]]
        loops = _walk_loops(ev, other)
        if len(loops) < 2:
            continue
        shapes = [classify_loop(np.c_[m.V[n] @ r, m.V[n] @ u], tol) for n, _ in loops]
        areas = [s["area"] if s else 0.0 for s in shapes]
        outer = int(np.argmax(areas))
        for li, ((nodes, nbrs), sh) in enumerate(zip(loops, shapes)):
            if li == outer or sh is None:
                continue
            nb = [j for j in nbrs if j >= 0 and j != rid]
            if not nb:
                continue
            off = np.mean([(mean_coord(j, a) - level) * sgn for j in nb])
            sense = "hole" if off < 0 else "boss"
            c2 = sh["c"] if sh["kind"] != "free" else sh["c_box"]
            c3 = r * c2[0] + u * c2[1] + np.eye(3)[a] * level
            key = (a, sense, sh["kind"], int(round(c2[0] / (0.004 * m.diag))), int(round(c2[1] / (0.004 * m.diag))))
            stacks.setdefault(key, []).append(dict(sh, c3=c3, level=level, sgn=sgn,
                                                   pts=np.c_[m.V[nodes] @ r, m.V[nodes] @ u]))
    def inward(a, p, level, sgn):
        """Distance from a rim at `level` down into the part to the first surface under point p
        (None if the line meets nothing: the opening goes right through)."""
        h = (rays.hits(a, p) - level) * -sgn
        h = h[h > 1e-4 * m.diag]
        return float(h.min()) if len(h) else None

    feats = []
    for (a, sense, kind, _, _), loops in stacks.items():
        size = (lambda q: q["d"]) if kind == "round" else (lambda q: math.sqrt(q["w"] * q["l"]))
        loops.sort(key=size)
        clusters = [[loops[0]]]                          # rims of one opening differ little in size
        for lp in loops[1:]:                             # (taper, chamfer); a much larger rim around
            if size(lp) <= 1.3 * size(clusters[-1][0]):  # the same centre is a separate feature,
                clusters[-1].append(lp)                  # such as a counterbore or a cavity
            else:
                clusters.append([lp])
        d_, r, u = VIEWS[AXIS_VIEW["XYZ"[a]]]
        for ci, cl in enumerate(clusters):
            best = cl[0]                                 # the tightest rim is the size that matters
            ft = dict(axis=a, sense=sense, kind=kind, c3=best["c3"], pts=best["pts"], area=best["area"],
                      d=best.get("d"), w=best.get("w"), l=best.get("l"), thru=False, depth=None, taper=False,
                      side=best["sgn"])
            if sense == "hole" and ci == 0:              # look straight down the middle from each rim
                found = [inward(a, lp["c3"], lp["level"], lp["sgn"]) for lp in cl]
                if any(v is None for v in found):
                    ft["thru"] = True
                else:
                    ft["depth"] = max(found)
            elif sense == "hole":                        # a smaller opening sits inside: probe the ledge around it
                inner = size(clusters[ci - 1][-1]) / size(best)
                c2 = best["c"] if kind != "free" else best["c_box"]
                ext = best["pts"][[np.argmax(best["pts"][:, 0]), np.argmin(best["pts"][:, 0]),
                                   np.argmax(best["pts"][:, 1]), np.argmin(best["pts"][:, 1])]]
                found = []
                for frac in (0.3, 0.7):
                    scale = inner + frac * (1.0 - inner)
                    probes = c2 + scale * (ext - c2)
                    vals = [inward(a, r * p[0] + u * p[1] + np.eye(3)[a] * best["level"], best["level"], best["sgn"])
                            for p in probes]
                    vals = [v for v in vals if v is not None]
                    found.append(float(np.median(vals)) if len(vals) >= 2 else None)
                if found[0] is None or found[1] is None:
                    ft["thru"] = True
                elif abs(found[0] - found[1]) > 0.1 * max(found):
                    ft["taper"] = True                   # depth changes with radius: a countersink or draft
                else:
                    ft["depth"] = (found[0] + found[1]) / 2.0
            feats.append(ft)
    out = []                                             # drop repeats of the same feature
    for f in sorted(feats, key=lambda q: q["area"]):
        ij = [q for q in range(3) if q != f["axis"]]
        if not any(g["axis"] == f["axis"] and g["sense"] == f["sense"] and g["kind"] == f["kind"]
                   and np.linalg.norm(g["c3"][ij] - f["c3"][ij]) < 0.004 * m.diag
                   and f["area"] <= 1.7 * g["area"] for g in out):
            out.append(f)
    return out


def _scatter(m, tris, weight, d, r, u, x0, y0, px, W, H):
    """Scatter sample points over triangles; yields (pixel number, depth) in chunks."""
    n = np.ceil(m.A[tris] * weight / (px * px) * 3.0).astype(np.int64) + 3
    rng = np.random.default_rng(0)
    V, F = m.V, m.F
    start, csum = 0, np.cumsum(n)
    while start < len(tris):                            # chunks of about 1.5 million samples
        stop = int(np.searchsorted(csum, (csum[start - 1] if start else 0) + 1_500_000)) + 1
        t = np.repeat(tris[start:stop], n[start:stop])
        a, b = rng.random(len(t)), rng.random(len(t))
        flip = a + b > 1.0
        a[flip], b[flip] = 1.0 - a[flip], 1.0 - b[flip]
        p0 = V[F[t, 0]]
        P = p0 + a[:, None] * (V[F[t, 1]] - p0) + b[:, None] * (V[F[t, 2]] - p0)
        ix = np.clip(((P @ r - x0) / px).astype(np.int64), 0, W - 1)
        iy = np.clip(((P @ u - y0) / px).astype(np.int64), 0, H - 1)
        yield iy * W + ix, P @ d
        start = stop


def typical_thickness(m, view="TOP"):
    """Most common material thickness measured along a view direction (the wall or sheet
    thickness of a shell-like part). Returns [(thickness, share of the plan area)], largest share first."""
    d, r, u = VIEWS[view]
    xy = np.c_[m.V @ r, m.V @ u]
    ext = np.ptp(xy, axis=0)
    px = max(ext.max(), 1e-12) / 500.0
    W, H = int(ext[0] / px) + 13, int(ext[1] / px) + 13
    x0, y0 = xy[:, 0].min() - 6 * px, xy[:, 1].min() - 6 * px
    nd = m.N @ d
    front = _depth_buffer(m, d, r, u, x0, y0, px, W, H).ravel()
    back = np.full(W * H, -np.inf)
    tris = np.flatnonzero(nd < -1e-9)
    if len(tris) == 0:
        return []
    for pix, depth in _scatter(m, tris, -nd[tris], d, r, u, x0, y0, px, W, H):
        ok = depth < front[pix] - 1e-4 * m.diag          # first surface behind the front one
        np.maximum.at(back, pix[ok], depth[ok])
    both = np.isfinite(front) & np.isfinite(back)
    t = front[both] - back[both]
    if len(t) < 50:
        return []
    span = float(np.ptp(m.V @ d)) or 1.0
    q = np.round(t / (span / 250.0)).astype(np.int64)
    vals, counts = np.unique(q, return_counts=True)
    out = []
    for i in np.argsort(-counts)[:6]:
        near = np.abs(q - vals[i]) <= 1
        share = float(near.mean())
        v = float(np.median(t[near]))
        if share >= 0.12 and all(abs(v - o[0]) > 0.03 * span for o in out):
            out.append((v, share))
    return out[:2]


def section_loops(m, a, value):
    """Outline of the part where the plane (axis a = value) cuts it: a list of closed
    loops, each an (n,3) array of points."""
    V, ev = m.V, m.ev
    dist = V[:, a] - value
    d0, d1 = dist[ev[:, 0]], dist[ev[:, 1]]
    cross = (d0 > 0) != (d1 > 0)
    t = np.where(cross, d0 / np.where(cross, d0 - d1, 1.0), 0.0)
    pts = V[ev[:, 0]] + t[:, None] * (V[ev[:, 1]] - V[ev[:, 0]])
    F, nV = m.F, len(V)
    E = np.empty((len(F), 3, 2), np.int64)
    E[:, 0], E[:, 1], E[:, 2] = F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]
    E.sort(axis=2)
    te = np.searchsorted(ev[:, 0].astype(np.int64) * nV + ev[:, 1], E[:, :, 0] * nV + E[:, :, 1])
    c = cross[te]
    sel = c.sum(1) == 2                                  # triangles the plane passes through
    pairs = te[sel][c[sel]].reshape(-1, 2)
    if len(pairs) == 0:
        return []
    return [pts[nodes] for nodes, _ in _walk_loops(pairs, np.zeros(len(pairs), np.int64))]


def _common_chord(loops, vertical, extent):
    """Scan straight lines across a section and return the most common material thickness
    along them, with one place to show it: (position of the scan line, start, end)."""
    S0 = np.concatenate([lp for lp in loops])
    S1 = np.concatenate([np.roll(lp, -1, axis=0) for lp in loops])
    ax = 0 if vertical else 1
    lo, hi = S0[:, ax].min(), S0[:, ax].max()
    chords = []
    for c in np.linspace(lo, hi, 163)[1:-1] + 1e-7 * extent:
        a0, a1 = S0[:, ax] - c, S1[:, ax] - c
        cr = (a0 > 0) != (a1 > 0)
        t = a0[cr] / (a0[cr] - a1[cr])
        y = np.sort(S0[cr, 1 - ax] + t * (S1[cr, 1 - ax] - S0[cr, 1 - ax]))
        chords += [(c, y[i], y[i + 1]) for i in range(0, len(y) - 1, 2)]
    if len(chords) < 4:
        return None
    ch = np.array(chords)
    length = ch[:, 2] - ch[:, 1]
    q = np.round(length / (extent / 600.0)).astype(np.int64)
    vals, counts = np.unique(q, return_counts=True)
    best = vals[np.argmax(counts)]
    if counts.max() < 4:
        return None
    pick = ch[q == best]
    pick = pick[np.argsort(pick[:, 0])]
    return tuple(pick[len(pick) // 2])


# ----------------------------------------------------------------------------
# The drawing
# ----------------------------------------------------------------------------
def _title_block(sh, m, U, k, scale, sheet_no, sheets, notes_w=128.0):
    """Border and title block in the lower right corner of a sheet."""
    X, Y, Z = m.hi - m.lo
    sh.rect(5, 5, SHEET_W - 10, SHEET_H - 10, "frame")
    tb_w, tb_h = notes_w - 6.0, 38.0
    tx, ty = SHEET_W - 5.0 - tb_w, 5.0
    sh.rect(tx, ty, tb_w, tb_h, "frame")
    sh.line(tx, ty + 14, tx + tb_w, ty + 14)
    sh.line(tx, ty + 26, tx + tb_w, ty + 26)
    for xx in (30, 62, 92):
        sh.line(tx + xx, ty, tx + xx, ty + 14)
    title = os.path.splitext(os.path.basename(m.path))[0]
    sh.text(tx + 3, ty + 32, title[:44], size=4.2, anchor="start", bold=True)
    sh.text(tx + 3, ty + 22.5, "SOURCE: %s" % os.path.basename(m.path)[:50], size=2.4, anchor="start")
    sh.text(tx + 3, ty + 17.5, "OVERALL: %s x %s x %s %s" % (_num(X * k, U.display), _num(Y * k, U.display),
                                                         _num(Z * k, U.display), U.display.upper()),
            size=2.4, anchor="start")
    sh.text(tx + 3, ty + 10, "SCALE", size=2.2, anchor="start")
    sh.text(tx + 3, ty + 4.5, scale, size=3.6, anchor="start", bold=True)
    sh.text(tx + 33, ty + 10, "DATE", size=2.2, anchor="start")
    sh.text(tx + 33, ty + 4.5, datetime.date.today().isoformat(), size=2.8, anchor="start")
    sh.text(tx + 65, ty + 10, "SHEET", size=2.2, anchor="start")
    sh.text(tx + 65, ty + 4.5, "%d OF %d" % (sheet_no, sheets), size=3.2, anchor="start", bold=True)
    sh.text(tx + 95, ty + 10, "THIRD ANGLE", size=2.2, anchor="start")
    px, py = tx + 103, ty + 4.5                          # projection symbol: cone, then its end view
    sh.line(px - 6, py - 1.4, px, py - 2.8)
    sh.line(px - 6, py + 1.4, px, py + 2.8)
    sh.line(px - 6, py - 1.4, px - 6, py + 1.4)
    sh.line(px, py - 2.8, px, py + 2.8)
    sh.circle(px + 7, py, 2.8)
    sh.circle(px + 7, py, 1.4)


class _View:
    """One orthographic view placed on a sheet: turns model points into sheet positions."""

    def __init__(self, m, name, ox, oy, sc):
        self.name, self.ox, self.oy, self.sc = name, ox, oy, sc
        self.d, self.r, self.u = VIEWS[name]
        c8 = np.array([[a, b, c] for a in (m.lo[0], m.hi[0]) for b in (m.lo[1], m.hi[1]) for c in (m.lo[2], m.hi[2])])
        self.h0, self.v0 = float((c8 @ self.r).min()), float((c8 @ self.u).min())
        self.w, self.h = float(np.ptp(c8 @ self.r)) * sc, float(np.ptp(c8 @ self.u)) * sc
        pr, pu = VIEWS[PRIMARY_OF[name]][1:]             # feature outlines are stored in the main view's axes
        self.flip = np.array([float(self.r @ pr), float(self.u @ pu)])

    def pt(self, p3):
        """Sheet position of a 3D model point."""
        return self.ox + (float(p3 @ self.r) - self.h0) * self.sc, self.oy + (float(p3 @ self.u) - self.v0) * self.sc

    def outline(self, pts):
        """Sheet positions of a feature outline stored in the main view's axes."""
        return (pts * self.flip - np.array([self.h0, self.v0])) * self.sc + np.array([self.ox, self.oy])

    def draw(self, sheet, vis, hid, hidden):
        """Draw the view's visible (and optionally hidden) edges."""
        off, base = np.array([self.h0, self.v0] * 2), np.array([self.ox, self.oy] * 2)
        if hidden:
            sheet.lines((hid - off) * self.sc + base, "hid")
        sheet.lines((vis - off) * self.sc + base, "vis")


def make_drawing(m, U, path, max_hidden=6000, max_rows=30, dual=False, crowd=6):
    """Write a reference drawing of model m.
    Sheet 1: main views with feature dimensions. Next sheet (only if needed): views of the
    far sides, for features that open there. Last sheet: two sections with thicknesses.
    A view with more than `crowd` features shows tags only and leaves sizes and positions to
    the table. dual=True adds the other unit in brackets to overall sizes.
    A path ending in .svg gives one SVG per sheet; anything else gives one HTML file."""
    k = 25.4 if U.model == "in" else 1.0                # model units -> mm
    other = "in" if U.display == "mm" else "mm"
    unit_names = {"mm": "MILLIMETERS", "in": "INCHES"}

    def one(v):
        """Dimension text in the display units."""
        return _num(v * k, U.display)

    def lab(v):
        """Overall-size text: display units, plus the other unit in brackets when dual is on."""
        return "%s [%s]" % (one(v), _num(v * k, other)) if dual else one(v)

    sh = Sheet()
    lo, hi = m.lo, m.hi
    X, Y, Z = hi - lo
    tol = 1e-5 * m.diag

    # ---- 1. read the features ------------------------------------------------------------
    plan_area = {0: Y * Z, 1: X * Z, 2: X * Y}
    feats = [f for f in find_features(m)                 # small odd-shaped outlines are lettering or texture
             if not (f["kind"] == "free" and (f["sense"] != "hole" or f["area"] < 0.04 * plan_area[f["axis"]]))]

    def view_of(f):
        """The view a feature is dimensioned in: a through hole in the main view, anything
        else in the view that looks at the side it opens on."""
        if f["sense"] == "hole" and f["thru"]:
            return AXIS_VIEW["XYZ"[f["axis"]]]
        return SIDE_VIEW[f["axis"]][1 if f["side"] > 0 else -1]

    for f in feats:
        f["extra"], f["merged"] = "", False
    for g in sorted(feats, key=lambda q: q["area"]):     # a countersink or counterbore joins its hole's callout
        if g["sense"] != "hole" or g["kind"] != "round" or g["thru"] or not (g["taper"] or g["depth"]):
            continue
        ij = [q for q in range(3) if q != g["axis"]]
        inner = [f for f in feats if f is not g and not f["merged"] and f["axis"] == g["axis"]
                 and f["sense"] == "hole" and f["kind"] == "round" and f["d"] < g["d"] <= 3.0 * f["d"]
                 and np.linalg.norm(f["c3"][ij] - g["c3"][ij]) < 0.004 * m.diag]
        if inner:
            f = max(inner, key=lambda q: q["d"])
            f["extra"] += ", %s %s%s%s" % ("CSK" if g["taper"] else "CBORE", DIA + one(g["d"]),
                                           "" if g["taper"] else " ↧ " + one(g["depth"]),
                                           "" if view_of(g) == view_of(f) else " FAR SIDE")
            g["merged"] = True
    feats = [f for f in feats if not f["merged"]]

    def size_text(f):
        """Size of a feature as drawing text."""
        if f["kind"] == "round":
            return DIA + one(f["d"])
        txt = "%s x %s" % (one(f["w"]), one(f["l"]))
        return txt + (" SLOT" if f["kind"] == "slot" else " OPENING" if f["kind"] == "free" else "")

    def depth_text(f):
        """THRU, or the depth of a blind opening."""
        if f["sense"] != "hole":
            return ""
        if f["taper"]:
            return " CSK"
        return " THRU" if f["thru"] else (" ↧ %s" % one(f["depth"]) if f["depth"] else "")

    grouped = {}                                         # equal features in one view share one callout
    for f in feats:
        grouped.setdefault((view_of(f), f["sense"], size_text(f), depth_text(f), f["extra"]), []).append(f)
    order = sorted(grouped, key=lambda q: (VIEW_ORDER.index(q[0]), q[1], -max(g["area"] for g in grouped[q])))
    by_view, rows = {}, []
    for gi, key in enumerate(order):
        view, sense, size, depth, extra = key
        letter = chr(ord("A") + gi % 26) * (1 + gi // 26)
        members = sorted(grouped[key], key=lambda q: tuple(np.round(q["c3"], 6)))
        n = len(members)
        by_view.setdefault(view, []).append(dict(letter=letter, feats=members,
                                                 label="%s%s%s%s" % ("%dX " % n if n > 1 else "", size, depth, extra)))
        for j, f in enumerate(members):
            f["tag"] = "%s%d" % (letter, j + 1)
            pos = (f["c3"] - lo) * k
            cell = ["-" if a == f["axis"] else _num(pos[a], U.display) for a in range(3)]
            rows.append([f["tag"], sense.upper(), size.replace(" SLOT", "").replace(" OPENING", ""), view]
                        + cell + [depth.replace("↧", "").strip() or "-"])
            if extra:
                rows.append(["", "", extra.lstrip(", ").replace("↧", "DEEP"), "", "", "", "", ""])
    table_mode = {v: sum(len(g["feats"]) for g in gs) > crowd for v, gs in by_view.items()}

    radii, radius_count, levels = {}, {}, {}
    if not m.lazy:
        for i in range(len(m.regs)):
            f = m.sfit(i)
            if f["kind"] == "cyl" and not f["full"]:
                rk = (round(f["r"] * k, 3), f["external"])
                radius_count[rk] = radius_count.get(rk, 0) + 1
                if axis_name(f["axis"]):
                    radii.setdefault(rk + (axis_name(f["axis"]),), []).append((i, f))
            elif f["kind"] == "plane" and axis_name(f["normal"]) == "Z" and f["area"] >= 0.03 * X * Y:
                v = float(f["point"][2] - lo[2])
                if tol < v < Z - tol:
                    key = int(round(v / tol))
                    levels[key] = (v, max(f["area"], levels.get(key, (0, 0))[1]))
    rad_by_view = {}                                     # the commonest radii get one leader each
    for rk in sorted(radii, key=lambda q: -len(radii[q]))[:4]:
        i, f = radii[rk][0]
        view = AXIS_VIEW[rk[2]]
        d, r, u = VIEWS[view]
        rel = m.V[np.unique(m.F[m.regs[i]])] - f["point"]
        mid = np.array([float((rel @ r).mean()), float((rel @ u).mean())])
        if np.linalg.norm(mid) < 0.2 * f["r"] or table_mode.get(view):
            continue                                     # arc too close to a full circle, or view already full
        n = len(radii[rk])
        rad_by_view.setdefault(view, []).append((f["point"], f["r"], "%sR%s" % ("%dX " % n if n > 1 else "", one(f["r"])),
                                                 mid / np.linalg.norm(mid)))

    def callout_texts(view):
        """Text of every leader callout a view will carry (none in table mode)."""
        if table_mode.get(view):
            return []
        return ["%s  %s" % (g["letter"], g["label"]) for g in by_view.get(view, [])] + [t[2] for t in rad_by_view.get(view, [])]

    stack_w = {v: (9.0 + max(0.62 * 2.7 * len(t) for t in callout_texts(v)) if callout_texts(v) else 0.0)
               for v in VIEW_ORDER}
    thick = typical_thickness(m, "TOP")
    skipped = 0

    # ---- 2. tools shared by every view ---------------------------------------------------
    def ordinate(sheet, box, values, side, axis, dots=True):
        """Coordinates along one edge of a view, written as the axis letter and the distance
        from the origin corner (X 17.794). Each line runs from the value to a dot on the
        point being located. values: [(model distance, sheet position, line start, line style)]."""
        nonlocal skipped
        ox, oy, vw, vh = box
        merged = {}
        for v, p, start, cls in values:
            key = int(round(v / (2 * tol)))
            cur = merged.get(key)
            if cur is not None:                          # same value twice: one line, from the farther feature
                if dots:
                    sheet.dot(*((p, start) if side == "top" else (start, p)))
                start = min(cur[2], start) if side == "top" else max(cur[2], start)
            merged[key] = (v, p, start, cls)
        last, prev = [None, None], None                  # two rows of labels so close values still fit
        for v, p, start, cls in sorted(merged.values(), key=lambda q: q[1]):
            if dots:
                sheet.dot(*((p, start) if side == "top" else (start, p)))
            if prev is not None and p - prev < 0.6:
                continue                                 # on top of the last line: one value serves both
            col = next((c for c in (0, 1) if last[c] is None or p - last[c] >= 3.3), None)
            if col is None:
                skipped += 1
                continue
            last[col], prev = p, p
            off = 7.0 + 13.0 * col
            txt = "%s %s" % (axis, _num(v * k, U.display))
            if side == "top":
                sheet.line(p, start, p, oy + vh + off - 1.0, cls)
                sheet.text(p, oy + vh + off, txt, size=2.5, anchor="start", rot=90)
            elif side == "left":
                sheet.line(start, p, ox - off + 1.0, p, cls)
                sheet.text(ox - off, p, txt, size=2.5, anchor="end", box=True)
            else:
                sheet.line(start, p, ox + vw + off - 1.0, p, cls)
                sheet.text(ox + vw + off, p, txt, size=2.5, anchor="start")

    def annotate(sheet, vw, pad):
        """Dimension the features that belong to one view. With few features: one leader
        callout per group plus the centre position of each feature. With many: tags only."""
        ox_, oy_ = vw.pt(lo)                             # origin mark: positions are from this corner
        sheet.circle(ox_, oy_, 1.1)
        sheet.line(ox_ - 2.2, oy_, ox_ + 2.2, oy_)
        sheet.line(ox_, oy_ - 2.2, ox_, oy_ + 2.2)
        sheet.text(ox_ - 3.0, oy_ - 3.6, "ORIGIN", size=2.2, anchor="end")
        table = table_mode.get(vw.name, False)
        across, up, callouts = [], [], []
        for g in by_view.get(vw.name, []):
            for f in g["feats"]:
                cx, cy = vw.pt(f["c3"])
                pts = vw.outline(f["pts"])
                half = (f["d"] / 2 if f["kind"] == "round" else max(f["w"], f["l"]) / 2 if f["kind"] != "free" else 0.0)
                half = half * vw.sc + 1.8 if f["kind"] != "free" else 3.0
                sheet.line(cx - half, cy, cx + half, cy, "ctr")
                sheet.line(cx, cy - half, cx, cy + half, "ctr")
                if table:
                    top_left = pts[np.argmax(pts[:, 1] - pts[:, 0])]
                    sheet.text(top_left[0] - 0.6, top_left[1] + 1.8, f["tag"], size=2.4, anchor="end", box=True)
                else:
                    across.append((abs(float((f["c3"] - lo) @ vw.r)), cx, cy, "ctr"))
                    up.append((abs(float((f["c3"] - lo) @ vw.u)), cy, cx, "ctr"))
            if not table:
                f = max(g["feats"], key=lambda q: q["area"])
                callouts.append((vw.pt(f["c3"]), vw.outline(f["pts"]), None, "%s  %s" % (g["letter"], g["label"]), None))
        if table:
            return
        for p3, rad, text, aim in rad_by_view.get(vw.name, []):
            callouts.append((vw.pt(p3), None, rad * vw.sc, text, aim * vw.flip))
        box = (vw.ox, vw.oy, vw.w, vw.h)
        if across:                                       # the zero lines, so every value has a visible start
            across.append((0.0, ox_, oy_, "thin"))
            up.append((0.0, oy_, ox_, "thin"))
        ordinate(sheet, box, across, "top", "XYZ"[int(np.argmax(np.abs(vw.r)))])    # line runs up from the centre
        ordinate(sheet, box, up, "left", "XYZ"[int(np.argmax(np.abs(vw.u)))])       # line runs left from the centre
        xs = vw.ox + vw.w + pad                          # callouts: one column right of the view
        y_prev = None
        for (cx, cy), pts, rr, text, aim in sorted(callouts, key=lambda q: -q[0][1]):
            y = cy if y_prev is None else min(cy, y_prev - 4.6)
            y_prev = y
            if pts is not None:                          # leader touches the outline nearest the text
                tip = pts[np.argmin(np.hypot(pts[:, 0] - xs, pts[:, 1] - y))]
            else:                                        # radius: arrow touches the middle of the arc
                tip = (cx + aim[0] * rr, cy + aim[1] * rr)
            sheet.line(tip[0], tip[1], xs, y)
            sheet.arrow(tip[0], tip[1], tip[0] - xs, tip[1] - y, size=2.6)
            sheet.line(xs, y, xs + 3.0, y)
            sheet.text(xs + 4.0, y, text, size=2.7, anchor="start")

    def title(sheet, vw, drop, extra=""):
        """Name under a view, with a pointer to the table when the view shows tags only."""
        sheet.text(vw.ox + vw.w / 2, vw.oy - drop, vw.name + " VIEW" + extra, size=2.6)
        if table_mode.get(vw.name):
            sheet.text(vw.ox + vw.w / 2, vw.oy - drop - 4.0, "SIZES AND POSITIONS OF TAGGED FEATURES: SEE TABLE", size=2.2)

    # ---- 3. sheet 1: main views ------------------------------------------------------------
    notes_w, m_left, m_bot, m_top = 128.0, 64.0, 36.0, 34.0
    gap_y = 46.0
    gap_x = max(42.0, stack_w["FRONT"] + 8.0)
    right_room = stack_w["RIGHT"] + 26.0
    area_w = SHEET_W - notes_w - 16.0 - m_left - right_room
    area_h = SHEET_H - 16.0 - m_bot - m_top
    s, s_label = SCALES[-1]
    for val, name in SCALES:                            # largest standard scale that fits
        if (X + Y) * val + gap_x <= area_w and (Z + Y) * val + gap_y <= area_h:
            s, s_label = val, name
            break
    fx, fy = 8.0 + m_left, 8.0 + m_bot
    V = {"FRONT": _View(m, "FRONT", fx, fy, s), "TOP": _View(m, "TOP", fx, fy + Z * s + gap_y, s),
         "RIGHT": _View(m, "RIGHT", fx + X * s + gap_x, fy, s)}
    lines = {n: view_lines(m, *VIEWS[n]) for n in V}
    hidden_drawn = sum(len(h) for _, h in lines.values()) <= max_hidden    # busy part: hidden lines only add clutter
    for name, vw in V.items():
        vw.draw(sh, lines[name][0], lines[name][1], hidden_drawn)
        title(sh, vw, 27.0 if name == "FRONT" else 17.0)

    d, r, u = VIEWS["ISO"]                               # isometric view fills the free corner
    vis, _ = view_lines(m, d, r, u)
    c8 = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
    ix0, iy0 = (c8 @ r).min(), (c8 @ u).min()
    iw, ih = np.ptp(c8 @ r), np.ptp(c8 @ u)
    bx = max(V["RIGHT"].ox + 4.0, fx + X * s + stack_w["TOP"] + 16.0)
    by = V["TOP"].oy + 6.0
    bw, bh = SHEET_W - notes_w - 12.0 - bx, SHEET_H - 12.0 - by
    if bw > 15 and bh > 15 and len(vis):
        si = min(bw / iw, bh / ih)
        base = np.array([bx + (bw - iw * si) / 2, by + (bh - ih * si) / 2] * 2)
        sh.lines((vis - np.array([ix0, iy0, ix0, iy0])) * si + base, "vis")
        sh.text(bx + bw / 2, by - 3.0, "ISOMETRIC VIEW (REFERENCE, NOT TO SCALE)", size=2.6)

    f0, t0 = (fx, fy), (fx, V["TOP"].oy)                 # overall length, height and width
    sh.dim((f0[0], f0[1]), (f0[0] + X * s, f0[1]), f0[1] - 18.0, lab(X))
    sh.dim((f0[0], f0[1]), (f0[0], f0[1] + Z * s), f0[0] - 46.0, lab(Z), vertical=True)
    sh.dim((t0[0], t0[1]), (t0[0], t0[1] + Y * s), t0[0] - 46.0, lab(Y), vertical=True)
    for name, vw in V.items():
        annotate(sh, vw, 26.0 if name == "RIGHT" else 8.0)
    vr = V["RIGHT"]                                      # heights of the main flat faces, beside the right view
    main_levels = sorted(levels.values(), key=lambda q: -q[1])[:4]
    ordinate(sh, (vr.ox, vr.oy, vr.w, vr.h), [(v, vr.oy + v * s, vr.ox + vr.w, "thin") for v, _ in main_levels], "right", "Z", dots=False)

    # proposed datum features: the largest outer faces, mutually perpendicular
    datums = []
    if not m.lazy:
        planes = []
        for i in range(len(m.regs)):
            f = m.sfit(i)
            if f["kind"] != "plane" or not axis_name(f["normal"]):
                continue
            a = "XYZ".index(axis_name(f["normal"]))
            sign = 1.0 if f["normal"][a] > 0 else -1.0
            edge = hi[a] if sign > 0 else lo[a]
            if abs(f["point"][a] - edge) <= 1e-6 * m.diag:      # lies on the outside of the part
                planes.append((f["area"], a, sign, f))
        planes.sort(key=lambda q: -q[0])
        used = set()
        for area, a, sign, f in planes:
            if a in used:
                continue
            used.add(a)
            datums.append((a, sign, f))
            if len(datums) == 3:
                break
        for letter, (a, sign, f) in zip("ABC", datums):
            if a == 2:      # horizontal line in the front view
                x, y = V["FRONT"].pt(f["point"])
                sh.datum(min(max(x, f0[0] + 6), f0[0] + X * s - 6) + 9.0, y, 0, sign, letter)
            elif a == 1:    # on the face's extension line left of the top view, clear of the values
                sh.datum(t0[0] - 38.0, t0[1] + (Y * s if sign > 0 else 0.0), 0, sign, letter)
            else:           # on the face's extension line below the front view
                sh.datum(f0[0] + (X * s if sign > 0 else 0.0), f0[1] - 9.0, sign, 0, letter)

    # where the two sections are cut, marked on the top view
    plan = sorted((f for f in feats if f["axis"] == 2 and f["sense"] == "hole"), key=lambda q: -q["area"])
    mid = (lo + hi) / 2.0
    cut_y = float(plan[0]["c3"][1]) if plan else float(mid[1])
    far = [f for f in plan[1:] if abs(f["c3"][0] - plan[0]["c3"][0]) > 0.05 * X] if plan else []
    cut_x = float((far[0] if far else plan[0])["c3"][0]) if plan else float(mid[0])
    nudge = 1.2345e-6 * m.diag                           # keeps the plane off mesh vertices
    sec = [("D", 1, cut_y + nudge), ("E", 0, cut_x + nudge)]
    tx0, ty0, tw, th = t0[0], t0[1], X * s, Y * s
    yy = ty0 + (cut_y - lo[1]) * s                       # section D-D: seen from the front, arrows point back
    sh.line(tx0 - 5, yy, tx0 + tw + 5, yy, "cut")
    for xx in (tx0 - 5, tx0 + tw + 5):
        sh.line(xx, yy, xx, yy + 5.0, "vis")
        sh.arrow(xx, yy + 5.0, 0, 1, size=2.6)
        sh.text(xx, yy + 8.0, "D", size=3.4, bold=True)
    xx = tx0 + (cut_x - lo[0]) * s                       # section E-E: seen from the right, arrows point left
    sh.line(xx, ty0 - 5, xx, ty0 + th + 5, "cut")
    for yy2 in (ty0 - 5, ty0 + th + 5):
        sh.line(xx, yy2, xx - 5.0, yy2, "vis")
        sh.arrow(xx - 5.0, yy2, -1, 0, size=2.6)
        sh.text(xx - 8.0, yy2, "E", size=3.4, bold=True)

    # ---- 4. far-side sheet: only the views that have features of their own -----------------
    far_names = [n for n in ("BOTTOM", "BACK", "LEFT") if by_view.get(n)]
    s_far = None
    if far_names:
        s_far = Sheet()
        sizes = {n: _View(m, n, 0, 0, 1.0) for n in far_names}
        fixed = sum(64.0 + stack_w[n] + 30.0 for n in far_names)
        sf, sf_label = SCALES[-1]
        for val, name in SCALES:
            if sum(sizes[n].w for n in far_names) * val + fixed <= SHEET_W - 30.0 \
                    and max(sizes[n].h for n in far_names) * val <= SHEET_H - 130.0:
                sf, sf_label = val, name
                break
        seen = {"BOTTOM": " (SEEN FROM BELOW)", "BACK": " (SEEN FROM BEHIND)", "LEFT": " (SEEN FROM THE LEFT)"}
        x_cur = 20.0
        for n in far_names:
            vw = _View(m, n, x_cur + 64.0, 0.0, sf)
            vw.oy = 70.0 + (SHEET_H - 130.0 - vw.h) / 2.0
            vis, hid = view_lines(m, vw.d, vw.r, vw.u)
            vw.draw(s_far, vis, hid, hidden_drawn)
            title(s_far, vw, 14.0, seen[n])
            annotate(s_far, vw, 8.0)
            x_cur = vw.ox + vw.w + stack_w[n] + 30.0
        s_far.text(14.0, SHEET_H - 14.0, "FAR-SIDE VIEWS", size=3.4, anchor="start", bold=True)
        s_far.text(14.0, SHEET_H - 20.0, "FEATURES THAT OPEN ON THE FAR SIDES OF THE PART. SAME ORIGIN CORNER AND FEATURE TABLE AS SHEET 1.",
                   size=2.4, anchor="start")

    # ---- 5. right-hand column of sheet 1: table, notes -------------------------------------
    n_sheets = 3 if s_far else 2
    nx = SHEET_W - notes_w - 3.0
    y = SHEET_H - 12.0
    if rows:
        sh.text(nx, y, "FEATURES  (centre position from origin mark, %s)" % U.display.upper(),
                size=2.8, anchor="start", bold=True)
        y -= 5.0
        cols = [0, 9, 20, 46, 60, 76, 92, 108]
        head = ["TAG", "TYPE", "SIZE", "VIEW", "X", "Y", "Z", "DEPTH"]
        shown = rows[:max_rows]
        sh.rect(nx - 1, y - 4.2 * len(shown) - 2.4, notes_w - 6, 4.2 * (len(shown) + 1) + 0.6)
        for r_i, row in enumerate([head] + shown):
            for c, cell in zip(cols, row):
                sh.text(nx + c, y, cell, size=2.4, anchor="start", bold=(r_i == 0))
            y -= 4.2
        if len(rows) > max_rows:
            sh.text(nx, y, "+ %d more rows not shown." % (len(rows) - max_rows), size=2.4, anchor="start")
            y -= 4.2
        y -= 5.0
    notes = ["DIMENSIONING AND TOLERANCING PER ASME Y14.5-2018.",
             "DIMENSIONS ARE IN %s%s." % (unit_names[U.display], " [%s]" % unit_names[other] if dual else ""),
             "DIMENSIONS ARE NOMINAL, READ FROM THE 3D MODEL.",
             "THE MODEL CARRIES NO TOLERANCES: APPLY THE RELEASED DRAWING."]
    if rows:
        notes.append("POSITIONS: X, Y, Z VALUES BESIDE A VIEW ARE DISTANCES FROM THE ORIGIN CORNER")
        notes.append("   TO THE DOT AT THE OTHER END OF THE LINE, WHICH IS THE CENTRE OF A FEATURE.")
        notes.append("SIZES: ONE LEADER PER FEATURE GROUP. ITS LETTER MATCHES THE TABLE.")
    if any(table_mode.values()):
        notes.append("A VIEW WITH MORE THAN %d FEATURES SHOWS TAGS ONLY: READ THE TABLE." % crowd)
    if s_far:
        notes.append("FEATURES THAT OPEN ON THE FAR SIDES ARE ON SHEET 2.")
    if any(f["thru"] is False and f["depth"] for f in feats):
        notes.append("↧ = DEPTH. SIZES ARE TAKEN AT THE NARROWEST RIM OF EACH OPENING.")
    elif feats:
        notes.append("SIZES ARE TAKEN AT THE NARROWEST RIM OF EACH OPENING.")
    if thick:
        notes.append("MOST COMMON THICKNESS ALONG Z: %s."
                     % ", ".join("%s (%d%% OF PLAN AREA)" % (one(v), round(100 * p)) for v, p in thick))
    if datums:
        notes.append("PROPOSED DATUM FEATURES (LARGEST OUTER FACES): %s. CONFIRM THEM."
                     % ", ".join("ABC"[:len(datums)]))
    if radius_count:
        txt = ", ".join("R%s (%dX)" % (_num(rk[0], U.display), n)
                        for rk, n in sorted(radius_count.items(), key=lambda q: -q[1])[:6])
        notes.append("RADII: %s%s." % (txt, " AND OTHERS" if len(radius_count) > 6 else ""))
    if skipped:
        notes.append("%d CLOSELY SPACED POSITION VALUES OMITTED: READ THEM FROM THE TABLE." % skipped)
    if not hidden_drawn:
        notes.append("HIDDEN LINES OMITTED FOR CLARITY.")
    if m.lazy:
        notes.append("MESH TOO IRREGULAR FOR FEATURE RECOGNITION: OUTLINES ONLY.")
    notes.append("SECTIONS D-D AND E-E ARE ON SHEET %d." % n_sheets)
    notes.append("REFERENCE DRAWING. NOT FOR MANUFACTURE OR INSPECTION.")
    sh.text(nx, y, "NOTES", size=2.8, anchor="start", bold=True)
    y -= 4.6
    n_i = 0
    for note in notes:
        if note.startswith(" "):                         # second line of the note above
            sh.text(nx + 4.0, y, note.strip(), size=2.2, anchor="start")
        else:
            n_i += 1
            sh.text(nx, y, "%d. %s" % (n_i, note), size=2.2, anchor="start")
        y -= 4.0

    # ---- 6. last sheet: the two sections -------------------------------------------------------
    s2 = Sheet()
    avail_w, avail_h = SHEET_W - 60.0, SHEET_H - 110.0
    sc2, sc2_label = SCALES[-1]
    for val, name in SCALES:
        if (X + Y) * val + 70.0 <= avail_w and Z * val <= avail_h:
            sc2, sc2_label = val, name
            break
    x_cursor = 45.0
    base_y = 75.0 + (avail_h - Z * sc2) / 2.0
    n_dims = 0
    for letter, a, value in sec:
        hax = 0 if a == 1 else 1                         # across the page: X for D-D, Y for E-E
        width = (X if a == 1 else Y) * sc2
        loops3 = section_loops(m, a, value)
        loops = [np.c_[(lp[:, hax] - lo[hax]) * sc2 + x_cursor, (lp[:, 2] - lo[2]) * sc2 + base_y] for lp in loops3]
        if loops:
            s2.hatch(loops)
            for vertical in (True, False):               # one thickness each way, where it is most common
                ch = _common_chord(loops, vertical, max(width, Z * sc2))
                if ch is None:
                    continue
                c, a0, a1 = ch
                if a1 - a0 >= 0.985 * (Z * sc2 if vertical else width):
                    continue                             # same as the overall size already shown
                txt = one((a1 - a0) / sc2)
                if vertical:
                    s2.dim((c, a0), (c, a1), c, txt, vertical=True)
                else:
                    s2.dim((a0, c), (a1, c), c, txt)
                n_dims += 1
        s2.dim((x_cursor, base_y), (x_cursor + width, base_y), base_y - 16.0, lab(X if a == 1 else Y))
        s2.dim((x_cursor, base_y), (x_cursor, base_y + Z * sc2), x_cursor - 16.0, lab(Z), vertical=True)
        where = "%s = %s" % ("Y" if a == 1 else "X", one(value - lo[a]))
        s2.text(x_cursor + width / 2, base_y - 28.0, "SECTION %s-%s" % (letter, letter), size=3.6, bold=True)
        s2.text(x_cursor + width / 2, base_y - 33.5, "CUT AT %s FROM THE ORIGIN.  SCALE %s" % (where, sc2_label), size=2.4)
        x_cursor += width + 70.0
    ny = SHEET_H - 14.0
    s2.text(14.0, ny, "SECTIONS", size=3.4, anchor="start", bold=True)
    lines2 = ["THE HATCHED AREA IS MATERIAL CUT BY THE SECTION PLANE. EDGES BEHIND THE PLANE ARE NOT SHOWN.",
              "EACH SECTION SHOWS ITS MOST COMMON THICKNESS, ONCE UPRIGHT AND ONCE ACROSS."]
    if thick:
        lines2.append("MOST COMMON THICKNESS ALONG Z OVER THE WHOLE PART: %s."
                      % ", ".join("%s %s (%d%% OF PLAN AREA)" % (one(v), U.display, round(100 * p)) for v, p in thick))
    for i, ln in enumerate(lines2):
        s2.text(14.0, ny - 6.0 - 4.2 * i, ln, size=2.4, anchor="start")

    # ---- 7. title blocks, then write the file(s) -------------------------------------------------
    sheets = [(sh, s_label)] + ([(s_far, sf_label)] if s_far else []) + [(s2, sc2_label)]
    for i, (sheet, scale) in enumerate(sheets):
        _title_block(sheet, m, U, k, scale, i + 1, len(sheets))
    svgs = [sheet.svg() for sheet, _ in sheets]
    if path.lower().endswith(".svg"):
        files = [path] + ["%s_sheet%d.svg" % (path[:-4], i + 2) for i in range(len(svgs) - 1)]
        for p, content in zip(files, svgs):
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(content)
    else:
        html = ("<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>%s drawing</title>\n<style>\n"
                "@page { size: 17in 11in; margin: 0 }\nbody { margin: 0; background: #777 }\n"
                "svg { display: block; margin: 14px auto; width: min(98vw, 17in); height: auto;"
                " box-shadow: 0 2px 10px rgba(0,0,0,.5); page-break-after: always }\n"
                "@media print { body { background: #fff } svg { margin: 0; width: 17in; height: 11in; box-shadow: none } }\n"
                "</style></head><body>\n%s\n</body></html>\n" % (os.path.basename(m.path), "\n".join(svgs)))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        files = [path]
    return dict(path=path, files=files, svgs=svgs, sheets=len(svgs), scale=s_label, features=len(feats),
                datums=len(datums), hidden=hidden_drawn, thickness=thick, section_dims=n_dims,
                table_views=[v for v, t in table_mode.items() if t])
