"""Part info for design for additive manufacturing (DfAM).

part_info() counts the features that matter when a part is printed and sintered:
holes, bosses, slots, patterns, fillets, rounds, chamfers, sharp edges and corners,
walls and ribs, thin walls, narrow gaps, overhangs for each build direction, and
enclosed voids. It returns plain text lines, so the same result can go to the
window, the terminal or a file.

No graphics in this file. Everything is counted on the model as loaded; the limits
(thin wall, small hole, overhang angle) are settings, not process rules.
"""
# Part of DimViewer. Copyright (c) 2026 Abdul Khader Khan.
# Free for noncommercial use. License and contact details are in README.md.

import math
import os

import numpy as np

from model import DEG, DIA, INCH, KIND_NAME, _cc
from measure import _rounds, chamfer, face_thickness, patterns, pattern_text, slots, thickness_map

DIRECTIONS = [("+Z", (0, 0, 1)), ("-Z", (0, 0, -1)), ("+X", (1, 0, 0)), ("-X", (-1, 0, 0)),
              ("+Y", (0, 1, 0)), ("-Y", (0, -1, 0))]


# ----------------------------------------------------------------------------
# Building blocks (each returns numbers, not text, so later tools can reuse them)
# ----------------------------------------------------------------------------
def _islands(m, mask):
    """Number of separate patches formed by the triangles where mask is True."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0
    num = np.full(len(mask), -1, np.int64)
    num[idx] = np.arange(len(idx))
    a = np.repeat(idx, 3)
    b = m.nbr[idx].ravel()
    ok = (b >= 0) & mask[np.maximum(b, 0)]
    return len(np.unique(_cc(len(idx), num[a[ok]], num[b[ok]])))


def overhang(m, up, angle=45.0):
    """Faces that look downward for build direction `up` and lie flatter than `angle`
    degrees from horizontal. The face resting on the build plate is left out.
    Returns (area, number of separate patches, share of the whole surface)."""
    up = np.asarray(up, float)
    h = m.V[m.F].mean(1) @ up
    mask = (m.N @ up < -math.cos(math.radians(angle))) & (h > float((m.V @ up).min()) + 1e-4 * m.diag)
    area = float(m.A[mask].sum())
    return area, _islands(m, mask), area / m.area


def shells(m):
    """Separate closed skins in the mesh. Returns (solid bodies, enclosed voids):
    a skin with its normals pointing inward encloses an empty space inside the part."""
    two = m.et1 >= 0
    lab = _cc(len(m.F), m.et0[two], m.et1[two])
    vol = np.bincount(lab, weights=(m.V[m.F[:, 0]] * m.N).sum(1) * m.A / 3.0)
    vol = vol[np.bincount(lab) > 0]
    return int((vol > 0).sum()), int((vol < 0).sum())


def edge_classes(m, smooth_deg=15.0):
    """Sort the edges between surfaces into sharp outside edges, sharp inside edges and
    smooth (tangent) joins. Also counts corners: points where three or more sharp edges meet.
    Returns dict(convex, concave, smooth, corners, shortest)."""
    cen = m.V[m.F].mean(1)
    out = dict(convex=0, concave=0, smooth=0, corners=0, shortest=None)
    ends = []
    for cid, ch in enumerate(m.chains):
        t0, t1 = m.et0[ch], m.et1[ch]
        ok = t1 >= 0
        if not ok.any():
            continue
        t0, t1 = t0[ok], t1[ok]
        if float(m.ecos[ch][ok].mean()) > math.cos(math.radians(smooth_deg)):
            out["smooth"] += 1
            continue
        bulge = float((((cen[t1] - cen[t0]) * m.N[t0]).sum(1)).mean())
        out["convex" if bulge < 0 else "concave"] += 1
        L = m.efit(cid)["length"]
        if out["shortest"] is None or L < out["shortest"]:
            out["shortest"] = L
        vid, cnt = np.unique(m.ev[ch], return_counts=True)
        ends.append(vid[cnt == 1])
    if ends:
        out["corners"] = int((np.bincount(np.concatenate(ends)) >= 3).sum())
    return out


def walls(m, ratio=3.0):
    """Pairs of flat faces that are parallel, back to back with material between, and at
    least `ratio` times wider than the material is thick. A pair where both faces rise
    from an inside corner (joined to a base on both sides) is counted as a rib.
    Returns a list of dicts: a, b (surface numbers), t (thickness), rib (True / False)."""
    planes = [i for i in range(len(m.regs)) if m.sfit(i)["kind"] == "plane"]
    if len(planes) > 1500:
        return []
    cen = m.V[m.F].mean(1)
    inside = set()                                       # faces that touch a sharp inside edge
    for cid, ch in enumerate(m.chains):
        t0, t1 = m.et0[ch], m.et1[ch]
        ok = t1 >= 0
        if ok.any() and float(m.ecos[ch][ok].mean()) < 0.966 \
                and float((((cen[t1[ok]] - cen[t0[ok]]) * m.N[t0[ok]]).sum(1)).mean()) > 0:
            inside.update(m.chain_pair[cid])
    out, used = [], set()
    for k, a in enumerate(planes):
        fa = m.sfit(a)
        for b in planes[k + 1:]:
            if a in used or b in used:
                continue
            fb = m.sfit(b)
            if float(fa["normal"] @ fb["normal"]) > -0.99996:
                continue
            t = -float((fb["point"] - fa["point"]) @ fa["normal"])
            if t <= 1e-6 * m.diag:
                continue
            wide = min(sorted(fa["box"])[1], sorted(fb["box"])[1])       # middle extent of the smaller face
            if wide < ratio * t:
                continue
            ta, tb = face_thickness(m, a), face_thickness(m, b)
            if not ta or not tb or abs(ta[0] - t) > 0.02 * t or abs(tb[0] - t) > 0.02 * t:    # thinnest behind each face
                continue
            used.update((a, b))
            out.append(dict(a=a, b=b, t=t, rib=(a in inside and b in inside)))
    return out


def blind_and_through(m):
    """Openings found on the flat faces square to X, Y or Z: (through, blind, deepest blind
    depth over width). Uses the feature finder of the drawing module."""
    try:
        from drawing import find_features
        feats = [f for f in find_features(m) if f["sense"] == "hole"]
    except Exception:
        return None
    thru = sum(1 for f in feats if f["thru"])
    blind = [f for f in feats if not f["thru"] and f["depth"]]
    deep = max((f["depth"] / (f["d"] or f["w"] or f["depth"]) for f in blind), default=None)
    return thru, len(blind), deep


# ----------------------------------------------------------------------------
# The text report
# ----------------------------------------------------------------------------
def part_info(m, U, thin_mm=1.0, hole_mm=1.0, overhang_deg=45.0):
    """All counts as text lines. thin_mm: walls and gaps under this are flagged.
    hole_mm: holes under this diameter are flagged. overhang_deg: see overhang()."""
    to_model = 1.0 if U.model == "mm" else 1.0 / INCH
    thin, small = thin_mm * to_model, hole_mm * to_model
    size = m.hi - m.lo
    L = ["PART INFO: %s" % os.path.basename(m.path),
         "  Limits used for the flags:",
         "  thin wall or gap under %s" % U.L(thin),
         "  small hole under %s%s" % (DIA, U.L(small)),
         "  overhang flatter than %.0f%s from level" % (overhang_deg, DEG), ""]
    flags = []

    L.append("SIZE AND MASS")
    L.append("  Overall X, Y, Z: %s, %s, %s" % (U.one(size[0]), U.one(size[1]), U.L(size[2])))
    box = float(np.prod(np.maximum(size, 1e-300)))
    L.append("  Volume: %s   Surface area: %s" % (U.Vol(m.volume), U.A(m.area)))
    T = m.V[m.F]
    tet = (T[:, 0] * m.N).sum(1) * m.A / 3.0             # signed volume of the wedge under each triangle
    cg = (tet[:, None] * T.sum(1) / 4.0).sum(0) / max(float(tet.sum()), 1e-300)
    L.append("  Centre of volume (centre of gravity): " + U.P(cg))
    L.append("  Fills %.0f%% of its bounding box   Longest / shortest side: %.1f"
             % (100.0 * m.volume / box, float(size.max() / max(size.min(), 1e-300))))
    L.append("  Triangles: %s   Surfaces: %d   Edges: %d" % (format(len(m.F), ","), len(m.regs), len(m.chains)))
    solid, void = shells(m)
    L.append("  Solid bodies: %d   Enclosed voids: %d" % (solid, void))
    if void:
        flags.append("%d enclosed void(s): powder cannot get out" % void)
    if solid > 1:
        flags.append("%d separate bodies in one file" % solid)
    if m.lazy:
        L += ["", "This mesh is too rough for feature recognition (scan-like), so features are not counted."]
        return L + _overhang_lines(m, U, overhang_deg)

    fits = [m.sfit(i) for i in range(len(m.regs))]
    kinds = {}
    for f in fits:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    L += ["", "SURFACES"]
    L.append("  " + ", ".join("%d %s" % (n, KIND_NAME[k].lower()) for k, n in sorted(kinds.items())))
    free = sum(f["area"] for f in fits if f["kind"] == "free")
    if free:
        L.append("  Freeform share of the surface: %.0f%%" % (100.0 * free / m.area))
    flat = max((f for f in fits if f["kind"] == "plane"), key=lambda f: f["area"], default=None)
    if flat is not None:
        L.append("  Largest flat face: %s, faces %s" % (U.A(flat["area"]), _dir(flat["normal"])))

    rounds = _rounds(m)
    holes = [r for r in rounds if not r["external"]]
    bosses = [r for r in rounds if r["external"]]
    L += ["", "HOLES, BOSSES, SLOTS"]
    L.append("  Round holes: %d%s" % (len(holes), _sizes(U, [2 * r["r"] for r in holes], DIA)))
    tiny = [r for r in holes if 2 * r["r"] < small]
    if tiny:
        flags.append("%d hole(s) under %s%s" % (len(tiny), DIA, U.L(small)))
    if holes:
        deep = max(m.sfit(r["ids"][0])["length"] / (2 * r["r"]) for r in holes)
        L.append("  Deepest hole, length over diameter: %.1f" % deep)
        if deep > 8:
            flags.append("hole %.1f x deeper than wide: hard to clear of powder" % deep)
    bt = blind_and_through(m)
    if bt:
        L.append("  Openings on the main faces: %d through, %d blind" % (bt[0], bt[1]))
        if bt[2] is not None:
            L.append("    deepest blind opening: %.1f x its width" % bt[2])
    L.append("  Round bosses and pins: %d%s" % (len(bosses), _sizes(U, [2 * r["r"] for r in bosses], DIA)))
    sl = slots(m)
    L.append("  Slots: %d%s" % (len(sl), _sizes(U, [s["width"] for s in sl], "width ")))
    pats = patterns(m)
    L.append("  Patterns: %d" % len(pats))
    L += ["    " + pattern_text(g, U) for g in pats[:8]]

    fil = [f for f in fits if f["kind"] == "cyl" and not f["full"] and not f["external"] and f["span"] < 2.36]
    rnd = [f for f in fits if f["kind"] == "cyl" and not f["full"] and f["external"] and f["span"] < 2.36]
    cham = sum(1 for i, f in enumerate(fits) if f["kind"] == "plane" and chamfer(m, i))
    cones = kinds.get("cone", 0)
    ed = edge_classes(m)
    L += ["", "EDGES AND CORNERS"]
    L.append("  Fillets (inside radius): %d%s" % (len(fil), _sizes(U, [f["r"] for f in fil], "R")))
    L.append("  Rounds (outside radius): %d%s" % (len(rnd), _sizes(U, [f["r"] for f in rnd], "R")))
    L.append("  Chamfers between flat faces: %d" % cham)
    L.append("  Conical faces (countersinks, tapers): %d" % cones)
    L.append("  Sharp outside edges: %d" % ed["convex"])
    L.append("  Sharp inside edges: %d" % ed["concave"])
    L.append("  Smooth (tangent) joins: %d" % ed["smooth"])
    L.append("  Corners (3 or more sharp edges meet): %d" % ed["corners"])
    if ed["shortest"] is not None:
        L.append("  Shortest sharp edge: " + U.L(ed["shortest"]))
    if ed["concave"]:
        flags.append("%d sharp inside edge(s) with no fillet: crack risk" % ed["concave"])

    t, measured = thickness_map(m)
    g, _ = thickness_map(m, gap=True)
    L += ["", "WALLS AND GAPS"]
    w = walls(m)
    ribs = [x for x in w if x["rib"]]
    L.append("  Wall sections (parallel flat faces): %d%s"
             % (len(w) - len(ribs), _sizes(U, [x["t"] for x in w if not x["rib"]], "")))
    L.append("  Ribs (thin walls standing on a base): %d%s"
             % (len(ribs), _sizes(U, [x["t"] for x in ribs], "")))
    good = np.isfinite(t)
    if good.any():
        mask = good & (t < thin)
        typical = _wmedian(t[good], m.A[good])
        low = _wmedian(t[good], m.A[good], 0.01)         # ignore the knife edges where two faces meet
        L.append("  Wall thickness: thinnest %s, typical %s" % (U.one(low), U.L(typical)))
        if float(t[good].min()) < 0.5 * low:
            L.append("    (down to %s right at sharp edges)" % U.L(float(t[good].min())))
        L.append("  Longest path through material: " + U.L(float(t[good].max())))
        L.append("  Thin walls (under %s): %d area(s), %.1f%% of surface"
                 % (U.L(thin), _islands(m, mask), 100.0 * float(m.A[mask].sum()) / m.area))
        if mask.any():
            flags.append("walls under %s on %.1f%% of the surface" % (U.L(thin), 100.0 * float(m.A[mask].sum()) / m.area))
        ratio = typical / max(low, 1e-300)
        if ratio > 4:
            flags.append("typical wall is %.0f x the thinnest: uneven shrinkage" % ratio)
    gg = np.isfinite(g)
    if gg.any():
        mask = gg & (g < thin)
        L.append("  Narrowest gap between faces: %s" % U.L(float(g[gg].min())))
        L.append("  Narrow gaps (under %s): %d area(s)" % (U.L(thin), _islands(m, mask)))
        if mask.any():
            flags.append("gaps under %s (narrowest %s): may fuse or hold powder" % (U.L(thin), U.one(float(g[gg].min()))))
    else:
        L.append("  Gaps between faces: none")
    if measured < len(t):
        L.append("  (measured at %s of %s triangles)" % (format(measured, ","), format(len(t), ",")))

    L += _overhang_lines(m, U, overhang_deg)
    L += ["", "POINTS TO CHECK"] + (["  - " + f for f in flags] if flags else ["  none found with these limits"])
    return L


def _overhang_lines(m, U, angle):
    """The overhang table: one line per build direction, least overhang marked."""
    rows = [(name,) + overhang(m, d, angle) for name, d in DIRECTIONS]
    best = min(r[1] for r in rows)
    out = ["", "OVERHANGS BY BUILD DIRECTION",
           "  (faces looking down, flatter than %.0f%s)" % (angle, DEG),
           "  build up along: area, patches, share of surface"]
    for name, area, n, share in rows:
        out.append("  %s: %12s  %4d  %5.1f%%%s"
                   % (name, U.A(area), n, 100.0 * share, "  <- least" if area <= best + 1e-9 * m.area else ""))
    return out


def _dir(n):
    """Direction of a normal as +X, -Z, ... or as numbers when it is not along an axis."""
    i = int(np.argmax(np.abs(n)))
    return ("+" if n[i] > 0 else "-") + "XYZ"[i] if abs(n[i]) > 0.9999 else "(%.2f, %.2f, %.2f)" % tuple(n)


def _sizes(U, vals, prefix):
    """'  (sizes ...)' text for a list of sizes: the distinct values, smallest first, at most five."""
    if not vals:
        return ""
    v = sorted({U.one(x) for x in vals}, key=lambda s: float(s.split()[0]))
    more = "" if len(v) <= 5 else ", ... up to " + prefix + v[-1]
    return "   (" + ", ".join(prefix + x for x in v[:5]) + more + ")"


def _wmedian(x, w, q=0.5):
    """Value of x below which the share q of the weight w lies (q=0.5: area-weighted median)."""
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][min(len(x) - 1, np.searchsorted(c, c[-1] * q))])
