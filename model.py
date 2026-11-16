"""Geometry: recognises surfaces and edges in a triangle mesh and measures them.

No graphics in this file, so it can be used and tested without a window:

    from model import Model, Units
    m = Model("part.stl")
    m.report(Units("mm", "mm"))
"""
# Part of DimViewer. Copyright (c) 2026 Abdul Khader Khan.
# Free for noncommercial use. License and contact details are in README.md.

import math
import os
import sys
import time

import numpy as np

from readers import load_mesh

DIA, DEG, SQ, CUBE = "\u00d8", "\u00b0", "\u00b2", "\u00b3"
INCH = 25.4
TWO_PI = 2.0 * math.pi


# ----------------------------------------------------------------------------
# Units
# ----------------------------------------------------------------------------
class Units:
    """Model units are what the STL numbers mean. Display units are what is shown."""
    DEC = {("mm", 1): 3, ("in", 1): 4, ("mm", 2): 2, ("in", 2): 4, ("mm", 3): 1, ("in", 3): 4}

    def __init__(self, model="mm", display="mm"):
        """model: units of the file's numbers. display: units shown first. Each is 'mm' or 'in'."""
        self.model, self.display = model, display

    def _fmt(self, v, unit, power):
        """Format a value given in model units as text in `unit` (power 1 length, 2 area, 3 volume)."""
        if unit != self.model:
            v = v * ((1.0 / INCH) if unit == "in" else INCH) ** power
        return "%.*f %s%s" % (self.DEC[(unit, power)], v, unit, {1: "", 2: SQ, 3: CUBE}[power])

    def _dual(self, v, power):
        """Value in the chosen display units only."""
        return self._fmt(v, self.display, power)

    def one(self, v):
        """Length in display units only."""
        return self._fmt(v, self.display, 1)

    def L(self, v):
        """Length in the display units, for example 10.000 mm."""
        return self._dual(v, 1)

    def A(self, v):
        """Area in the display units."""
        return self._dual(v, 2)

    def Vol(self, v):
        """Volume in the display units."""
        return self._dual(v, 3)

    def P(self, p):
        """A point (x, y, z) in display units."""
        f = 1.0 if self.display == self.model else ((1.0 / INCH) if self.display == "in" else INCH)
        d = self.DEC[(self.display, 1)]
        return "(%.*f, %.*f, %.*f) %s" % (d, p[0] * f, d, p[1] * f, d, p[2] * f, self.display)


def vec(d):
    """Direction vector as text."""
    return "(%.4f, %.4f, %.4f)" % tuple(round(float(x), 4) + 0.0 for x in d)


def axis_name(d, signed=False):
    """'X', 'Y' or 'Z' if direction d lies along that axis, else ''. signed adds + or -."""
    for i, nm in enumerate("XYZ"):
        if abs(d[i]) > 0.99999:
            return (("+" if d[i] > 0 else "-") + nm) if signed else nm
    return ""


def mem_mb():
    """Memory used by this program in MB, or None if it cannot be read."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        pass
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1e6 if sys.platform == "darwin" else r / 1e3
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Small numeric helpers
# ----------------------------------------------------------------------------
def _cc(n, a, b):
    """Connected-component label for each of n nodes, given undirected links a-b."""
    if len(a) == 0:
        return np.arange(n)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        g = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(n, n))
        return connected_components(g, directed=False)[1]
    except ImportError:
        pass
    lab = np.arange(n)
    while True:
        la, lb = lab[a], lab[b]
        m = np.minimum(la, lb)
        new = lab.copy()
        for idx in (a, b, la, lb):
            np.minimum.at(new, idx, m)
        while True:
            nn = new[new]
            if np.array_equal(nn, new):
                break
            new = nn
        if np.array_equal(new, lab):
            return lab
        lab = new


def _groups(lab):
    """Split indices 0..n-1 into lists that share the same label."""
    order = np.argsort(lab, kind="stable")
    ls = lab[order]
    return np.split(order, np.flatnonzero(ls[1:] != ls[:-1]) + 1)


def _closest_on_tris(p, T):
    """Closest point to p on any of the triangles T (n,3,3). Returns (distance, point)."""
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4

    def div(n, d):
        return n / np.where(np.abs(d) > 1e-300, d, 1.0)

    den = div(1.0, va + vb + vc)
    q = a + ab * (vb * den)[:, None] + ac * (vc * den)[:, None]      # inside the triangle
    cases = [((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b + (c - b) * div(d4 - d3, (d4 - d3) + (d5 - d6))[:, None]),
             ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * div(d2, d2 - d6)[:, None]),
             ((d6 >= 0) & (d5 <= d6), c),
             ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * div(d1, d1 - d3)[:, None]),
             ((d3 >= 0) & (d4 <= d3), b),
             ((d1 <= 0) & (d2 <= 0), a)]
    for cond, pts in cases:                              # later cases take priority
        q[cond] = pts[cond]
    dist = np.linalg.norm(q - p, axis=1)
    i = int(np.argmin(dist))
    return float(dist[i]), q[i]


def _closest_on_segs(p, s0, s1):
    """Closest point to p on any of the line segments s0-s1. Returns (distance, point)."""
    d = s1 - s0
    t = np.clip(((p - s0) * d).sum(1) / np.maximum((d * d).sum(1), 1e-300), 0.0, 1.0)
    q = s0 + t[:, None] * d
    dist = np.linalg.norm(q - p, axis=1)
    i = int(np.argmin(dist))
    return float(dist[i]), q[i]


def _lstsq(A, b):
    """Least-squares solution of A x = b."""
    return np.linalg.lstsq(A, b, rcond=None)[0]


def _circle2d(q):
    """Least-squares circle through 2D points. Returns cx, cy, r (r=None if it fails)."""
    cx, cy, D = _lstsq(np.c_[2 * q, np.ones(len(q))], (q * q).sum(1))
    r2 = D + cx * cx + cy * cy
    return cx, cy, (math.sqrt(r2) if r2 > 0 else None)


# ----------------------------------------------------------------------------
# Geometry model: surfaces and edges recognised from the triangles
# ----------------------------------------------------------------------------
GDT = {
    "plane": ["Form: flatness",
              "Orientation to a datum: parallelism, perpendicularity, angularity",
              "Profile of a surface; position when it is one side of a width"],
    "cyl": ["Size: diameter (Rule #1 envelope at MMC)",
            "Form: circularity, cylindricity, straightness",
            "Orientation of axis: perpendicularity, parallelism, angularity",
            "Location: position; circular / total runout to a datum axis"],
    "cone": ["Form: circularity, straightness of surface elements",
             "Angularity; profile of a surface; circular / total runout"],
    "sphere": ["Size: spherical diameter; form: circularity",
               "Location: position; profile of a surface"],
    "free": ["Profile of a surface (the control that fits a freeform face)"],
}
KIND_NAME = {"plane": "PLANE", "cyl": "CYLINDER", "cone": "CONE", "sphere": "SPHERE",
             "free": "FREEFORM / BLEND"}


class Model:
    """A loaded part: its triangles, the surfaces and edges found in them, and their measurements."""
    def __init__(self, path, angle=30.0, log=print):
        """Load a file and recognise its features. angle: faces meeting at more than this many degrees are separate surfaces."""
        t0 = time.time()
        self.path = path
        self.V, self.F, self.units = load_mesh(path)
        self._prep()
        self._segment(angle)
        self._build_chains()
        self.load_seconds = time.time() - t0
        log("Loaded %s: %s triangles, %d surfaces, %d edges in %.1f s"
            % (os.path.basename(path), format(len(self.F), ","), len(self.regs),
               len(self.chains), self.load_seconds))

    # -- preparation ---------------------------------------------------------
    def _prep(self):
        """Clean the mesh (drop empty triangles, fix inside-out facets) and build the table of which triangles share each edge."""
        V = self.V
        self.lo, self.hi = V.min(0), V.max(0)
        self.diag = float(np.linalg.norm(self.hi - self.lo))
        nV = len(V)
        for attempt in (0, 1):
            F = self.F
            p0 = V[F[:, 0]]
            cr = np.cross(V[F[:, 1]] - p0, V[F[:, 2]] - p0)
            dbl = np.linalg.norm(cr, axis=1)
            keep = dbl > 1e-12 * self.diag ** 2
            if not keep.all():                          # drop zero-area triangles
                F, cr, dbl, p0 = F[keep], cr[keep], dbl[keep], p0[keep]
            vol = float((p0 * cr).sum() / 6.0)
            if vol < 0:                                 # make normals point outward
                F, cr, vol = F[:, [0, 2, 1]], -cr, -vol
            self.F = F = np.ascontiguousarray(F)
            self.A = dbl / 2.0
            self.N = cr / dbl[:, None]
            self.volume, self.area = vol, float(self.A.sum())

            nT = len(F)
            E = np.empty((nT, 3, 2), np.int32)
            E[:, 0], E[:, 1], E[:, 2] = F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]
            E = E.reshape(-1, 2)
            fwd = E[:, 0] < E[:, 1]
            E.sort(axis=1)
            key = E[:, 0].astype(np.int64) * nV + E[:, 1]
            order = np.argsort(key, kind="stable")
            ks = key[order]
            first = np.ones(len(ks), bool)
            first[1:] = ks[1:] != ks[:-1]
            start = np.flatnonzero(first)
            two = np.diff(np.append(start, len(ks))) >= 2
            i0 = order[start]
            i1 = order[np.minimum(start + 1, len(ks) - 1)]
            self.ev = E[i0]                             # unique mesh edges (vertex pairs)
            self.et0 = i0 // 3                          # triangle on one side
            self.et1 = np.where(two, i1 // 3, -1)       # triangle on the other (-1 = open)
            self.ecos = np.full(len(start), -2.0)
            self.ecos[two] = (self.N[self.et0[two]] * self.N[self.et1[two]]).sum(1)
            nbr = np.full(nT * 3, -1, np.int64)
            ncos = np.full(nT * 3, -2.0)
            nbr[i0[two]], nbr[i1[two]] = self.et1[two], self.et0[two]
            ncos[i0[two]] = ncos[i1[two]] = self.ecos[two]
            self.nbr, self.ncos = nbr.reshape(nT, 3), ncos.reshape(nT, 3)
            bad = two & (fwd[i0] == fwd[i1])            # neighbours wound opposite ways
            if attempt == 0 and bad.any():
                par = np.zeros(nT * 3, np.int8)
                par[i0[bad]] = par[i1[bad]] = 1
                self._reorient(par.reshape(nT, 3))
                continue
            break

    def _reorient(self, par):
        """Flip triangles so every neighbour pair agrees on which side is outside."""
        nl, pl = self.nbr.tolist(), par.tolist()
        fl = [-1] * len(nl)
        for seed in range(len(nl)):
            if fl[seed] >= 0:
                continue
            fl[seed] = 0
            stack = [seed]
            while stack:
                t = stack.pop()
                for j in (0, 1, 2):
                    u = nl[t][j]
                    if u >= 0 and fl[u] < 0:
                        fl[u] = fl[t] ^ pl[t][j]
                        stack.append(u)
        flip = np.array(fl, bool)
        self.F[flip] = self.F[flip][:, [0, 2, 1]]

    # -- surface fitting -----------------------------------------------------
    def _fit(self, tris):
        """Fit plane / cylinder / cone / sphere to a group of triangles."""
        V, F = self.V, self.F
        P = V[np.unique(F[tris])]
        n, a = self.N[tris], self.A[tris]
        at = float(a.sum())
        c = P.mean(0)
        Q = P - c
        size = float(np.linalg.norm(P.max(0) - P.min(0)))
        floor = 2e-6 * self.diag
        out = dict(kind="free", area=at, size=size, point=c, box=P.max(0) - P.min(0))
        nf = len(np.unique(self.facet[tris]))           # number of flat facets
        tc = V[F[tris]].mean(1)                         # triangle centroids

        nb = (n * a[:, None]).sum(0)
        ln = float(np.linalg.norm(nb))
        if ln > 0.9995 * at:                            # ---- plane
            nn = np.linalg.eigh(Q.T @ Q)[1][:, 0]
            if abs(nn @ nb) < 0.99 * ln:
                nn = nb / ln
            if nn @ nb < 0:
                nn = -nn
            d = Q @ nn
            if np.ptp(d) <= max(2e-4 * size, floor):
                out.update(kind="plane", normal=nn, flatness=float(np.ptp(d)))
                return out

        w, v = np.linalg.eigh((n * a[:, None]).T @ n / at)
        if nf >= 3 and w[0] < 1e-5 and w[1] > 1e-6:     # ---- cylinder
            ax, b1, b2 = v[:, 0], v[:, 1], v[:, 2]
            q = np.c_[Q @ b1, Q @ b2]
            cx, cy, r = _circle2d(q)
            if r is not None and r < 1e3 * size:
                rad = np.hypot(q[:, 0] - cx, q[:, 1] - cy)
                cen = c + cx * b1 + cy * b2
                if np.abs(rad - r).max() <= max(2e-3 * r, floor) and self._even(tris, cen, b1, b2, r, nf):
                    t = Q @ ax
                    rel = tc - cen
                    rel -= np.outer(rel @ ax, ax)
                    span, full = self._span(tris, cen, b1, b2)
                    out.update(kind="cyl", axis=ax, r=r, length=float(np.ptp(t)),
                               point=cen + ax * (t.max() + t.min()) / 2.0,
                               external=bool(((rel * n).sum(1) * a).sum() > 0),
                               span=span, full=full, cylindricity=float(np.ptp(rad)))
                    return out

        nm = nb / at
        Xn = n - nm
        w, v = np.linalg.eigh((Xn * a[:, None]).T @ Xn / at)
        if nf >= 4 and w[1] > 1e-6 and w[0] < 1e-6 and w[0] < 1e-2 * w[1]:   # ---- cone
            ax, b1, b2 = v[:, 0], v[:, 1], v[:, 2]
            s = n @ ax
            sm = float((s * a).sum() / at)
            if np.abs(s - sm).max() < 0.02 and 0.003 < abs(sm) < 0.9995:
                q = np.c_[Q @ b1, Q @ b2]
                t = Q @ ax
                lev = np.round(t / max(1e-6 * self.diag, 1e-12)).astype(np.int64)
                ul = np.unique(lev)
                K = B = 0.0
                if len(ul) == 2:                        # two rings only: fit both circles at once
                    hi_ring = lev == ul[1]
                    cx, cy, Da, Db = _lstsq(np.c_[2 * q, ~hi_ring, hi_ring], (q * q).sum(1))
                    ra = math.sqrt(max(Da + cx * cx + cy * cy, 0.0))
                    rb = math.sqrt(max(Db + cx * cx + cy * cy, 0.0))
                    ta, tb = float(t[~hi_ring].mean()), float(t[hi_ring].mean())
                    if abs(rb - ra) > 0:
                        K = ((rb - ra) / (tb - ta)) ** 2
                        B = K * (ta - ra * (tb - ta) / (rb - ra))
                elif len(ul) > 2:
                    cx, cy, K, B, _ = _lstsq(np.c_[2 * q, t * t, -2 * t, np.ones(len(t))],
                                             (q * q).sum(1))
                if K > 0:
                    k, t0 = math.sqrt(K), B / K
                    half = math.atan(k)
                    rr = k * np.abs(t - t0)
                    res = (np.hypot(q[:, 0] - cx, q[:, 1] - cy) - rr) * math.cos(half)
                    if (np.abs(res).max() <= max(2e-3 * rr.max(), floor)
                            and abs(math.degrees(half) - math.degrees(math.asin(abs(sm)))) < 1.0):
                        cen = c + cx * b1 + cy * b2
                        straddle = t.min() < t0 < t.max()
                        rel = tc - cen
                        rel -= np.outer(rel @ ax, ax)
                        out.update(kind="cone", axis=ax, half=half, apex=cen + ax * t0,
                                   d_small=0.0 if straddle else float(2 * rr.min()),
                                   d_large=float(2 * rr.max()), length=float(np.ptp(t)),
                                   point=cen + ax * (t.max() + t.min()) / 2.0, k=k,
                                   external=bool(((rel * n).sum(1) * a).sum() > 0))
                        return out

        if nf >= 8 and len(P) >= 10 and w[0] > 1e-6:    # ---- sphere (normals must spread in 3D)
            sol = _lstsq(np.c_[2 * Q, np.ones(len(Q))], (Q * Q).sum(1))
            r2 = sol[3] + sol[:3] @ sol[:3]
            if r2 > 0:
                r = math.sqrt(r2)
                res = np.linalg.norm(Q - sol[:3], axis=1) - r
                if r < 1e3 * size and np.abs(res).max() <= max(2e-3 * r, floor):
                    cen = c + sol[:3]
                    out.update(kind="sphere", r=r, point=cen, sphericity=float(np.ptp(res)),
                               external=bool((((tc - cen) * n).sum(1) * a).sum() > 0))
        return out

    def _even(self, tris, cen, b1, b2, r, nf):
        """With only a few facets, a real arc has roughly even steps. A long flat between
        two short ones also fits a circle, but it is not a cylinder."""
        if nf >= 8:
            return True
        tv = self.V[self.F[tris]] - cen
        q = np.stack([tv @ b1, tv @ b2], axis=2)
        chord = np.max(np.linalg.norm(q - np.roll(q, 1, axis=1), axis=2), axis=1)
        step = 2 * np.arcsin(np.clip(chord / (2 * r), 0, 1))
        return step.max() <= 2.5 * step.min()

    def _span(self, tris, cen, b1, b2):
        """Angle a cylindrical face covers around its axis (radians), and whether it is full."""
        tv = self.V[self.F[tris]] - cen
        x, y = tv @ b1, tv @ b2
        ang = np.arctan2(y, x)
        mc = np.arctan2(y.sum(1), x.sum(1))
        dev = (ang - mc[:, None] + math.pi) % TWO_PI - math.pi
        nbin = 720
        dth = TWO_PI / nbin
        s = np.floor((mc + dev.min(1) + 2 * TWO_PI) / dth).astype(np.int64)
        e = np.floor((mc + dev.max(1) + 2 * TWO_PI) / dth).astype(np.int64)
        diff = np.zeros(4 * nbin + 2)
        np.add.at(diff, s, 1)
        np.add.at(diff, e + 1, -1)
        cov = (np.cumsum(diff)[:4 * nbin] > 0).reshape(4, nbin).any(0)
        if cov.all():
            return TWO_PI, True
        th = np.unique(np.round(ang.ravel() % TWO_PI, 6))
        gap = np.diff(np.append(th, th[0] + TWO_PI))
        mid = ((th + gap / 2.0) % TWO_PI / dth).astype(np.int64) % nbin
        return float(TWO_PI - gap[~cov[mid]].sum()), False

    def _on_prim(self, P, f):
        """True if all points P lie on the fitted cylinder, cone or sphere f."""
        floor = 2e-6 * self.diag
        rel = P - f["point"] if f["kind"] != "cone" else P - f["apex"]
        if f["kind"] == "sphere":
            return np.abs(np.linalg.norm(rel, axis=1) - f["r"]).max() <= max(2e-3 * f["r"], floor)
        t = rel @ f["axis"]
        rad = np.linalg.norm(rel - np.outer(t, f["axis"]), axis=1)
        if f["kind"] == "cyl":
            return np.abs(rad - f["r"]).max() <= max(2e-3 * f["r"], floor)
        rr = f["k"] * np.abs(t)
        return np.abs(rad - rr).max() <= max(2e-3 * rr.max(), floor)

    # -- segmentation into surfaces -------------------------------------------
    def _segment(self, angle):
        """Group triangles into surfaces: first by smoothness, then by splitting patches that are not one plane / cylinder / cone / sphere."""
        nT = len(self.F)
        cos_s = math.cos(math.radians(angle))
        inner = self.et1 >= 0
        sm = inner & (self.ecos > cos_s)
        fl = inner & (self.ecos > math.cos(math.radians(0.12)))
        self.facet = _cc(nT, self.et0[fl], self.et1[fl])
        comps = _groups(_cc(nT, self.et0[sm], self.et1[sm]))
        self.lazy = len(comps) > 50000                  # scan-like mesh: fit only on demand
        regs, fits, from_split = [], [], []
        if self.lazy:
            regs, fits = comps, [None] * len(comps)
        else:
            inmask, tmp = np.zeros(nT, bool), np.zeros(nT, np.int64)
            stack = [(g, 0, cos_s, None) for g in comps]
            while stack:
                tris, depth, cs, fit = stack.pop()
                if fit is None:
                    fit = self._fit(tris)
                if fit["kind"] == "free" and depth < 10 and len(tris) > 1:
                    res = self._split(tris, cs, inmask, tmp)
                    if res:
                        for tr, ft in res[0]:
                            from_split.append(len(regs))
                            regs.append(tr)
                            fits.append(ft)
                        stack.extend((g, depth + 1, cs, None) for g in res[1])
                        continue
                    res = self._split_angle(tris, cs, inmask, tmp)
                    if res:
                        stack.extend((g, depth + 1, res[1], ft) for g, ft in res[0])
                        continue
                regs.append(tris)
                fits.append(fit)
        labels = np.empty(nT, np.int64)
        for i, tr in enumerate(regs):
            labels[tr] = i
        # a "plane" that was split off but lies on a neighbouring cylinder/cone/sphere
        # is really one facet of that surface: put it back
        for rid in from_split:
            tris = regs[rid]
            nb = self.nbr[tris]
            ok = (nb >= 0) & (self.ncos[tris] > cos_s)
            P = None
            for r2 in np.unique(labels[nb[ok]]):
                if r2 == rid or regs[r2] is None or fits[r2]["kind"] not in ("cyl", "cone", "sphere"):
                    continue
                if P is None:
                    P = self.V[np.unique(self.F[tris])]
                if self._on_prim(P, fits[r2]):
                    regs[r2] = np.concatenate([regs[r2], tris])
                    labels[tris] = r2
                    fits[r2] = self._fit(regs[r2])
                    regs[rid] = None
                    break
        keep = [i for i, r in enumerate(regs) if r is not None]
        self.regs = [regs[i] for i in keep]
        self.fits = [fits[i] for i in keep]
        self.labels = labels
        for i, tr in enumerate(self.regs):
            labels[tr] = i

    def _split(self, tris, cos_s, inmask, tmp):
        """Pull true planes out of a smooth patch that is not one single primitive
        (for example a flat that runs tangent into a fillet)."""
        uf, inv = np.unique(self.facet[tris], return_inverse=True)
        inv = inv.reshape(-1)
        k = len(uf)
        if k < 2:
            return None
        fa = np.bincount(inv, self.A[tris], k)
        inmask[tris] = True
        tmp[tris] = inv
        nb = self.nbr[tris]
        ok = (nb >= 0) & (self.ncos[tris] > cos_s)
        ok &= inmask[np.maximum(nb, 0)]
        fi = np.repeat(inv, 3).reshape(-1, 3)[ok]
        fj = tmp[nb[ok]]
        inmask[tris] = False
        pk = np.unique(fi[fi != fj] * k + fj[fi != fj])
        pi, pj = pk // k, pk % k
        cnt = np.bincount(pi, minlength=k)
        mx = np.zeros(k)
        np.maximum.at(mx, pi, fa[pj])
        multi = np.bincount(inv, minlength=k) >= 2      # a facet made of 2+ triangles
        cand = ((cnt <= 1) | (fa > 1.25 * mx)) & ((k <= 12) | multi | (fa >= 4 * np.median(fa)))
        if not cand.any():
            return None
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(k + 1))
        planes, taken = [], np.zeros(k, bool)
        for i in np.flatnonzero(cand):
            tr = tris[order[bounds[i]:bounds[i + 1]]]
            ft = self._fit(tr)
            if ft["kind"] == "plane":
                planes.append((tr, ft))
                taken[i] = True
        if not planes:
            return None
        rest = tris[~taken[inv]]
        if len(rest) == 0:
            return planes, []
        inmask[rest] = True
        tmp[rest] = np.arange(len(rest))
        nb = self.nbr[rest]
        ok = (nb >= 0) & (self.ncos[rest] > cos_s)
        ok &= inmask[np.maximum(nb, 0)]
        a = np.repeat(np.arange(len(rest)), 3).reshape(-1, 3)[ok]
        b = tmp[nb[ok]]
        inmask[rest] = False
        return planes, [rest[g] for g in _groups(_cc(len(rest), a, b))]

    def _split_angle(self, tris, cos_s, inmask, tmp):
        """Cut a smooth patch at its sharpest creases (for example a cylinder running into
        a shallow taper). Kept only if a piece then fits a plane / cylinder / cone / sphere."""
        inmask[tris] = True
        tmp[tris] = np.arange(len(tris))
        nb, nc = self.nbr[tris], self.ncos[tris]
        ok = (nb >= 0) & (nc > cos_s)
        ok &= inmask[np.maximum(nb, 0)]
        inmask[tris] = False
        if not ok.any():
            return None
        rows = np.repeat(np.arange(len(tris)), 3).reshape(-1, 3)
        ang = math.acos(min(1.0, float(nc[ok].min())))
        for _ in range(6):
            ang *= 0.95
            if ang < math.radians(2.0):
                return None
            cs = math.cos(ang)
            ok2 = ok & (nc > cs)
            groups = _groups(_cc(len(tris), rows[ok2], tmp[nb[ok2]]))
            if len(groups) > 1:
                parts = [(tris[g], self._fit(tris[g])) for g in groups]
                area = self.A[tris].sum()
                if any(ft["kind"] in ("cyl", "cone", "sphere")
                       or (ft["kind"] == "plane" and ft["area"] >= 0.05 * area) for _, ft in parts):
                    return parts, cs
                return None
            left = nc[ok2]
            if len(left) == 0:
                return None
            ang = math.acos(min(1.0, float(left.min())))
        return None

    def sfit(self, rid):
        """Fitted shape of surface number rid (computed on first use for rough meshes)."""
        if self.fits[rid] is None:
            self.fits[rid] = self._fit(self.regs[rid])
        return self.fits[rid]

    # -- edges ---------------------------------------------------------------
    def _build_chains(self):
        """Find the edges: runs of mesh edges that separate the same two surfaces."""
        nR, nV = len(self.regs), len(self.V)
        la = self.labels[self.et0]
        lb = np.where(self.et1 >= 0, self.labels[np.maximum(self.et1, 0)], -1)
        feat = np.flatnonzero(la != lb)
        lo, hi = np.minimum(la, lb)[feat], np.maximum(la, lb)[feat]
        self.chains, self.chain_pair = [], []
        self.rchains = [[] for _ in range(nR)]
        self.feat = feat
        if len(feat) == 0:
            return
        pid = np.unique((lo + 1) * (nR + 2) + hi + 1, return_inverse=True)[1].reshape(-1)
        ev = self.ev[feat].astype(np.int64)
        ni = np.unique(np.concatenate([pid * nV + ev[:, 0], pid * nV + ev[:, 1]]),
                       return_inverse=True)[1].reshape(-1)
        m = len(feat)
        lab = _cc(int(ni.max()) + 1, ni[:m], ni[m:])
        for g in _groups(lab[ni[:m]]):
            cid = len(self.chains)
            self.chains.append(feat[g])
            pair = (int(lo[g[0]]), int(hi[g[0]]))
            self.chain_pair.append(pair)
            for r in pair:
                if r >= 0:
                    self.rchains[r].append(cid)
        self._efits = {}

    def segs(self, cid):
        """Start and end points of every small segment in edge number cid."""
        ev = self.ev[self.chains[cid]]
        return self.V[ev[:, 0]], self.V[ev[:, 1]]

    def efit(self, cid):
        """Fitted shape of edge number cid: straight line, circle, arc or general curve."""
        if cid in self._efits:
            return self._efits[cid]
        ev = self.ev[self.chains[cid]]
        vid, inv = np.unique(ev, return_inverse=True)
        P = self.V[vid]
        deg = np.bincount(inv.reshape(-1), minlength=len(vid))
        closed = len(vid) == len(ev) and bool((deg == 2).all())
        sl = np.linalg.norm(self.V[ev[:, 1]] - self.V[ev[:, 0]], axis=1)
        L = float(sl.sum())
        c = P.mean(0)
        Q = P - c
        v = np.linalg.eigh(Q.T @ Q)[1]
        d = v[:, 2]
        tol = max(1e-4 * L, 2e-6 * self.diag)
        t = Q @ d
        out = dict(kind="polyline", length=L, point=c, closed=closed)
        if np.linalg.norm(Q - np.outer(t, d), axis=1).max() <= tol:
            out.update(kind="line", length=float(np.ptp(t)), dir=d,
                       point=c + d * (t.max() + t.min()) / 2.0)
        elif len(P) >= 4 and np.abs(Q @ v[:, 0]).max() <= tol:
            q = np.c_[Q @ v[:, 1], Q @ v[:, 2]]
            cx, cy, r = _circle2d(q)
            if r is not None and np.abs(np.hypot(q[:, 0] - cx, q[:, 1] - cy) - r).max() \
                    <= max(2e-3 * r, 2e-6 * self.diag):
                ang = TWO_PI if closed else float((2 * np.arcsin(np.clip(sl / (2 * r), 0, 1))).sum())
                out.update(kind="circle" if closed else "arc", r=r, angle=ang, dir=v[:, 0],
                           length=r * ang, point=c + cx * v[:, 1] + cy * v[:, 2])
        if out["kind"] == "polyline" and not closed and (deg == 1).sum() == 2:
            ends = P[deg == 1]
            out["chord"] = float(np.linalg.norm(ends[1] - ends[0]))
        self._efits[cid] = out
        return out

    # -- picking -------------------------------------------------------------
    def circle_chains(self):
        """Numbers of all edges that are circles or arcs."""
        if not hasattr(self, "_circ"):
            self._circ = [c for c in range(len(self.chains)) if self.efit(c)["kind"] in ("circle", "arc")]
        return self._circ

    def pick(self, cell, p, tol, mode="auto"):
        """Decide whether a click at point p on triangle `cell` hit an edge or a surface.
        mode: auto (edge if the click is near one), edges, circles (nearest such edge), faces."""
        rid = int(self.labels[cell])
        if mode == "faces":
            return ("surf", rid)
        cands = self.rchains[rid]
        if mode == "circles":
            cands = [c for c in cands if self.efit(c)["kind"] in ("circle", "arc")]
        best, best_cid = float("inf"), None
        for cid in cands:
            s0, s1 = self.segs(cid)
            d = s1 - s0
            u = np.clip(((p - s0) * d).sum(1) / np.maximum((d * d).sum(1), 1e-300), 0, 1)
            dist = float(np.linalg.norm(s0 + u[:, None] * d - p, axis=1).min())
            if dist < best:
                best, best_cid = dist, cid
        if best_cid is not None and (best <= tol or mode in ("edges", "circles")):
            return ("edge", best_cid)
        return ("surf", rid)

    # -- descriptions --------------------------------------------------------
    def name(self, sel):
        """Short name of a selection, such as S12 for a surface or E40 for an edge."""
        return ("S%d" if sel[0] == "surf" else "E%d") % sel[1]

    def describe(self, sel, U):
        """Returns (title, detail lines, short label for the 3D view)."""
        if sel[0] == "edge":
            return self._describe_edge(sel[1], U)
        rid = sel[1]
        f = self.sfit(rid)
        k = f["kind"]
        lines = []
        what = ""
        short = KIND_NAME[k].title()
        if k == "plane":
            an = axis_name(f["normal"], True)
            lines += ["Area: " + U.A(f["area"]),
                      "Normal: %s%s" % (vec(f["normal"]), "  (faces %s)" % an if an else ""),
                      "Flatness measured on mesh: " + U.L(f["flatness"])]
            short = "Plane"
        elif k == "cyl":
            an = axis_name(f["axis"])
            deg = math.degrees(f["span"])
            if f["full"]:
                what = " (boss / shaft, external)" if f["external"] else " (hole, internal)"
            else:
                what = " (convex radius, external)" if f["external"] else " (concave radius, internal)"
            if f["full"] or deg > 180.5:
                lines.append("Diameter: %s%s" % (DIA, U.L(2 * f["r"]))
                             + ("" if f["full"] else "   arc %.1f%s" % (deg, DEG)))
                short = DIA + U.one(2 * f["r"])
            else:
                lines.append("Radius: R%s   arc %.1f%s" % (U.L(f["r"]), deg, DEG))
                short = "R" + U.one(f["r"])
            lines += ["Length along axis: " + U.L(f["length"]),
                      "Axis: %s%s" % (vec(f["axis"]), "  (parallel to %s)" % an if an else ""),
                      "Cylindricity measured on mesh: " + U.L(f["cylindricity"])]
        elif k == "cone":
            what = " (external)" if f["external"] else " (internal)"
            lines += ["Included angle: %.2f%s  (half angle %.2f%s)"
                      % (2 * math.degrees(f["half"]), DEG, math.degrees(f["half"]), DEG),
                      "Small end: %s%s" % (DIA, U.L(f["d_small"])),
                      "Large end: %s%s" % (DIA, U.L(f["d_large"])),
                      "Length along axis: " + U.L(f["length"])]
            short = "Cone %.1f%s" % (2 * math.degrees(f["half"]), DEG)
        elif k == "sphere":
            what = " (external)" if f["external"] else " (internal)"
            lines += ["Spherical diameter: S%s%s" % (DIA, U.L(2 * f["r"])),
                      "Center: " + U.P(f["point"])]
            short = "S" + DIA + U.one(2 * f["r"])
        else:
            b = f["box"]
            lines += ["Area: " + U.A(f["area"]),
                      "Extent X, Y, Z: %s, %s, %s" % (U.one(b[0]), U.one(b[1]), U.one(b[2]))]
            short = "Freeform"
        lines.append("Y14.5 controls that apply:")
        lines += ["  " + s for s in GDT[k]]
        return "S%d  %s%s" % (rid, KIND_NAME[k], what), lines, short

    def _describe_edge(self, cid, U):
        """Title, detail lines and short label for edge number cid."""
        f = self.efit(cid)
        a, b = self.chain_pair[cid]
        between = "Between %s and %s" % ("open boundary" if a < 0 else "S%d" % a, "S%d" % b)
        k = f["kind"]
        if k == "line":
            an = axis_name(f["dir"])
            return ("E%d  STRAIGHT EDGE" % cid,
                    ["Length: " + U.L(f["length"]),
                     "Direction: %s%s" % (vec(f["dir"]), "  (parallel to %s)" % an if an else ""),
                     between], U.one(f["length"]))
        if k in ("circle", "arc"):
            deg = math.degrees(f["angle"])
            an = axis_name(f["dir"])
            if k == "circle" or deg > 180.5:
                first = "Diameter: %s%s" % (DIA, U.L(2 * f["r"]))
                short = DIA + U.one(2 * f["r"])
            else:
                first = "Radius: R" + U.L(f["r"])
                short = "R" + U.one(f["r"])
            lines = [first]
            if k == "arc":
                lines.append("Arc angle: %.1f%s   arc length: %s" % (deg, DEG, U.L(f["length"])))
            else:
                lines.append("Circumference: " + U.L(f["length"]))
            lines += ["Center: " + U.P(f["point"]),
                      "Axis: %s%s" % (vec(f["dir"]), "  (parallel to %s)" % an if an else ""),
                      between]
            return ("E%d  %s" % (cid, "CIRCULAR EDGE" if k == "circle" else "ARC EDGE"), lines, short)
        lines = ["Length along curve: " + U.L(f["length"])]
        if "chord" in f:
            lines.append("End to end: " + U.L(f["chord"]))
        lines.append(between)
        return "E%d  CURVED EDGE%s" % (cid, " (closed)" if f["closed"] else ""), lines, U.one(f["length"])

    # -- relation between two features ----------------------------------------
    def _geom(self, sel):
        """Reduce a feature to what distance / angle maths needs: kind (plane, axis, point), a point, a direction, a length."""
        if sel[0] == "surf":
            f = self.sfit(sel[1])
            if f["kind"] == "plane":
                return "plane", f["point"], f["normal"], 0.0
            if f["kind"] in ("cyl", "cone"):
                return "axis", f["point"], f["axis"], f["length"]
            return "point", f["point"], None, 0.0
        f = self.efit(sel[1])
        if f["kind"] == "line":
            return "axis", f["point"], f["dir"], f["length"]
        if f["kind"] in ("circle", "arc"):
            return "axis", f["point"], f["dir"], 0.0
        return "point", f["point"], None, 0.0

    def _verts(self, sel):
        """All vertices of a selected surface (None for an edge)."""
        return self.V[np.unique(self.F[self.regs[sel[1]]])] if sel[0] == "surf" else None

    def closest(self, a, b):
        """Shortest distance between two selected features, measured on the features
        themselves (the actual faces and edges, not their infinite extensions).
        Returns (distance, point on a, point on b)."""
        def parts(sel):
            if sel[0] == "surf":
                T = self.V[self.F[self.regs[sel[1]]]]
                return (lambda p: _closest_on_tris(p, T)), np.concatenate([T.reshape(-1, 3), T.mean(1)])
            s0, s1 = self.segs(sel[1])
            return (lambda p: _closest_on_segs(p, s0, s1)), np.concatenate([s0, s1, (s0 + s1) / 2.0])

        on_a, pa = parts(a)
        on_b, pb = parts(b)
        pa, pb = pa[::max(1, len(pa) // 1500)], pb[::max(1, len(pb) // 1500)]
        best_d, best_j = np.empty(len(pa)), np.empty(len(pa), np.int64)
        for i in range(0, len(pa), 256):                 # coarse search over sample points
            d2 = ((pa[i:i + 256, None, :] - pb[None, :, :]) ** 2).sum(2)
            best_j[i:i + 256] = d2.argmin(1)
            best_d[i:i + 256] = d2.min(1)
        out = None
        for i in np.argsort(best_d)[:3]:                 # then refine exactly from the best few starts
            p = pa[i]
            for _ in range(12):
                _, q = on_b(p)
                _, p2 = on_a(q)
                moved = float(np.linalg.norm(p2 - p))
                p = p2
                if moved <= 1e-9 * self.diag:
                    break
            dist, q = on_b(p)
            if out is None or dist < out[0]:
                out = (dist, p, q)
        return out

    def caliper(self, a, b):
        """Length a caliper would read across two features. The jaws are held square to
        the X, Y or Z axis that best lines up with the two features.
        Returns (axis number, outside length, inside length or None, lowest point, highest point):
        outside spans the outer extremes of both features, inside is the clear gap between them."""
        def pts(sel):
            if sel[0] == "surf":
                return self.V[np.unique(self.F[self.regs[sel[1]]])]
            return np.concatenate(self.segs(sel[1]))
        Pa, Pb = pts(a), pts(b)
        ax = int(np.argmax(np.abs(Pb.mean(0) - Pa.mean(0))))
        if Pa[:, ax].mean() > Pb[:, ax].mean():
            Pa, Pb = Pb, Pa
        both = np.concatenate([Pa, Pb])
        p_lo, p_hi = both[np.argmin(both[:, ax])], both[np.argmax(both[:, ax])]
        inside = float(Pb[:, ax].min() - Pa[:, ax].max())
        return ax, float(p_hi[ax] - p_lo[ax]), (inside if inside > 1e-9 * self.diag else None), p_lo, p_hi

    def relate(self, a, b, U):
        """Distance / angle of feature b relative to feature a (a acts as the reference)."""
        ta, pa, da, _ = self._geom(a)
        tb, pb, db, Lb = self._geom(b)
        par, near1 = math.sin(math.radians(0.5)), math.cos(math.radians(0.5))
        w = pb - pa
        out = []
        if ta == "plane" and tb == "plane":
            c = min(1.0, abs(float(da @ db)))
            ang = math.degrees(math.acos(c))
            Pb = self._verts(b)
            if c > near1:
                out.append("Parallel planes, distance: " + U.L(abs(float(w @ da))))
                out.append("Parallelism to reference, measured: " + U.L(float(np.ptp(Pb @ da))))
            else:
                out.append("Angle between planes (normals): %.2f%s" % (ang, DEG))
                if c < par:
                    m = db - (db @ da) * da
                    m /= np.linalg.norm(m)
                    out.append("Perpendicularity to reference, measured: " + U.L(float(np.ptp(Pb @ m))))
        elif {ta, tb} == {"plane", "axis"}:
            pn, pp = (da, pa) if ta == "plane" else (db, pb)
            ad, ap, AL = (db, pb, Lb) if ta == "plane" else (da, pa, 0.0)
            s = min(1.0, abs(float(ad @ pn)))
            if s < par:
                out.append("Axis parallel to plane, distance: " + U.L(abs(float((ap - pp) @ pn))))
                if AL:
                    out.append("Parallelism of axis, measured: " + U.L(AL * s))
            elif s > near1:
                out.append("Axis perpendicular to plane")
                out.append("Feature center to plane: " + U.L(abs(float((ap - pp) @ pn))))
                if AL:
                    out.append("Perpendicularity of axis, measured: %s%s"
                               % (DIA, U.L(AL * math.sqrt(max(0.0, 1 - s * s)))))
            else:
                out.append("Angle between axis and plane: %.2f%s" % (math.degrees(math.asin(s)), DEG))
        elif ta == "axis" and tb == "axis":
            c = min(1.0, abs(float(da @ db)))
            if c > near1:
                off = w - (w @ da) * da
                dist = float(np.linalg.norm(off))
                if dist <= 1e-5 * self.diag:
                    out.append("Coaxial (axis offset %s)" % U.L(dist))
                else:
                    out.append("Parallel axes, center distance: " + U.L(dist))
                    out.append("Offset X, Y, Z: %s, %s, %s"
                               % (U.one(abs(off[0])), U.one(abs(off[1])), U.one(abs(off[2]))))
            else:
                cr = np.cross(da, db)
                out.append("Angle between axes: %.2f%s" % (math.degrees(math.acos(c)), DEG))
                out.append("Closest distance between axes: "
                           + U.L(abs(float(w @ cr)) / float(np.linalg.norm(cr))))
        elif "plane" in (ta, tb):
            pn = da if ta == "plane" else db
            out.append("Distance, point to plane: " + U.L(abs(float(w @ pn))))
        elif "axis" in (ta, tb):
            ad = da if ta == "axis" else db
            out.append("Distance, point to axis: " + U.L(float(np.linalg.norm(w - (w @ ad) * ad))))
        else:
            out.append("Distance between centers: " + U.L(float(np.linalg.norm(w))))
            out.append("Offset X, Y, Z: %s, %s, %s"
                       % (U.one(abs(w[0])), U.one(abs(w[1])), U.one(abs(w[2]))))
        return out

    # -- text outputs --------------------------------------------------------
    def summary(self, U):
        """Overall size, volume and surface area as text lines, in the display units."""
        s = self.hi - self.lo
        return ["Overall X, Y, Z: %s, %s, %s" % (U.one(s[0]), U.one(s[1]), U.one(s[2])),
                "Volume: %s   Surface area: %s" % (U.Vol(self.volume), U.A(self.area))]

    def report(self, U, out=print):
        """Print every surface and edge with its measurements."""
        out("%s   (model units: %s)" % (os.path.basename(self.path), U.model))
        for ln in self.summary(U):
            out(ln)
        counts = {}
        for i in range(len(self.regs)):
            counts[self.sfit(i)["kind"]] = counts.get(self.sfit(i)["kind"], 0) + 1
        out("Surfaces: " + ", ".join("%d %s" % (n, KIND_NAME[k].lower()) for k, n in sorted(counts.items())))
        out("")
        for i in range(len(self.regs)):
            title, lines, _ = self.describe(("surf", i), U)
            cut = lines.index("Y14.5 controls that apply:")
            out("%-44s %s" % (title, " | ".join(lines[:cut])))
        out("")
        for c in range(len(self.chains)):
            title, lines, _ = self.describe(("edge", c), U)
            out("%-24s %s" % (title, " | ".join(lines)))

    def write_csv(self, path):
        """Write every surface and edge to a CSV file, in model units."""
        import csv
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "class", "type", "internal_external", "diameter", "length",
                        "area", "angle_deg", "direction_axis", "dir_x", "dir_y", "dir_z"])
            for i in range(len(self.regs)):
                f = self.sfit(i)
                d = f.get("normal", f.get("axis"))
                ie = "" if "external" not in f else ("external" if f["external"] else "internal")
                dia = 2 * f["r"] if "r" in f else f.get("d_large", "")
                ang = math.degrees(f["span"]) if "span" in f else (
                    2 * math.degrees(f["half"]) if "half" in f else "")
                w.writerow(["S%d" % i, "surface", f["kind"], ie, dia, f.get("length", ""), f["area"],
                            ang, axis_name(d) if d is not None else ""]
                           + (list(d) if d is not None else ["", "", ""]))
            for c in range(len(self.chains)):
                f = self.efit(c)
                d = f.get("dir")
                w.writerow(["E%d" % c, "edge", f["kind"], "", 2 * f["r"] if "r" in f else "",
                            f["length"], "", math.degrees(f["angle"]) if "angle" in f else "",
                            axis_name(d) if d is not None else ""]
                           + (list(d) if d is not None else ["", "", ""]))
