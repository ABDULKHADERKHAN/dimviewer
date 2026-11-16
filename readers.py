"""Readers for the supported model files: STL, 3MF, AMF and STEP.

Every reader returns triangle corner coordinates; load_mesh() welds them into
shared vertices. 3MF, AMF and STEP state their units and are converted to mm.
STL has no units, so the caller decides (mm is assumed by default).
"""
# Part of DimViewer. Copyright (c) 2026 Abdul Khader Khan.
# Free for noncommercial use. License and contact details are in README.md.

import math
import os
import re
import struct

import numpy as np

NATIVE_MSG = ("SolidWorks (.sldprt) and Creo (.prt) files are proprietary and can only be "
              "opened by those programs. In SolidWorks or Creo use File > Save As and export "
              "STEP (.step), 3MF or STL, then open that file here.")
UNIT_MM = {"micron": 1e-3, "micrometer": 1e-3, "millimeter": 1.0, "centimeter": 10.0,
           "inch": 25.4, "foot": 304.8, "feet": 304.8, "meter": 1000.0}
FILE_TYPES = [("3D models", "*.stl *.3mf *.amf *.step *.stp *.STL *.3MF *.AMF *.STEP *.STP"),
              ("All files", "*.*")]


def _local(tag):
    """XML tag name without its namespace prefix."""
    return tag.rsplit("}", 1)[-1]


def read_stl(path):
    """Binary or ASCII STL -> (n,3,3) triangle corner coordinates."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(84)
        n = struct.unpack("<I", head[80:84])[0] if len(head) == 84 else 0
        binary = len(head) == 84 and (size == 84 + 50 * n or head[:5].lower() != b"solid")
        if binary:
            n = min(n, (size - 84) // 50)
            dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
            return np.fromfile(f, dtype=dt, count=n)["v"]
        f.seek(0)
        nums = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", f.read())
        flat = b" ".join(b" ".join(t) for t in nums).decode("ascii", "ignore")
        return np.array(flat.split(), dtype=np.float32).reshape(-1, 3, 3)


def read_3mf(path):
    """3MF (zipped XML). Applies component and build transforms. Returns mm."""
    import zipfile
    import xml.etree.ElementTree as ET
    from array import array
    z = zipfile.ZipFile(path)
    parts = {}

    def xf(s):
        """Parse a 3MF transform string into (3x3 matrix, translation), or None."""
        if not s:
            return None
        m = np.array(s.split(), float).reshape(4, 3)
        return m[:3], m[3]

    def compose(inner, outer):                          # apply inner first, then outer
        """Combine two transforms into one."""
        if inner is None or outer is None:
            return inner if outer is None else outer
        return inner[0] @ outer[0], inner[1] @ outer[0] + outer[1]

    def ref(el, here):
        """Where a component or build item points: (model part name, object id, transform)."""
        p = next((v for k, v in el.attrib.items() if _local(k) == "path"), None)
        return (p or here).lstrip("/"), el.get("objectid"), xf(el.get("transform"))

    def part(name):
        """Parse one model part inside the 3MF archive (cached). Returns objects, build items, unit scale."""
        if name in parts:
            return parts[name]
        objs, build, unit = {}, [], 1.0
        vx = tr = comps = oid = box = None
        count = 0
        with z.open(name) as fh:
            for ev, el in ET.iterparse(fh, events=("start", "end")):
                tag = _local(el.tag)
                if ev == "start":
                    if tag == "model":
                        unit = UNIT_MM.get((el.get("unit") or "millimeter").lower(), 1.0)
                    elif tag == "object":
                        oid, vx, tr, comps = el.get("id"), array("d"), array("q"), []
                    elif tag in ("vertices", "triangles"):
                        box = el
                    continue
                if tag == "vertex" and vx is not None:
                    vx.extend((float(el.get("x")), float(el.get("y")), float(el.get("z"))))
                elif tag == "triangle" and tr is not None:
                    tr.extend((int(el.get("v1")), int(el.get("v2")), int(el.get("v3"))))
                elif tag == "component" and comps is not None:
                    comps.append(ref(el, name))
                elif tag == "object":
                    objs[oid] = (np.array(vx).reshape(-1, 3),
                                 np.array(tr, dtype=np.int64).reshape(-1, 3), comps)
                    vx = tr = comps = None
                elif tag == "item":
                    build.append(ref(el, name))
                if tag in ("vertex", "triangle"):
                    count += 1
                    if count % 20000 == 0 and box is not None:
                        box.clear()                     # keep the XML tree small
        parts[name] = (objs, build, unit)
        return parts[name]

    out = []

    def collect(name, oid, M, depth=0):
        """Gather the triangles of one object and everything nested inside it."""
        objs, _, unit = part(name)
        if oid not in objs or depth > 16:
            return
        V, T, comps = objs[oid]
        if len(T):
            P = V if M is None else V @ M[0] + M[1]
            out.append(P[T] * unit)
        for n2, o2, M2 in comps:
            collect(n2, o2, compose(M2, M), depth + 1)

    root = next((n for n in z.namelist() if n.lower().endswith("3dmodel.model")), None) \
        or next((n for n in z.namelist() if n.lower().endswith(".model")), None)
    if root is None:
        raise ValueError("No model found inside the 3MF file")
    objs, build, _ = part(root)
    for name, oid, M in (build or [(root, o, None) for o in objs]):
        collect(name, oid, M)
    if not out:
        raise ValueError("The 3MF file has no mesh")
    return np.concatenate(out)


def read_amf(path):
    """AMF (XML, optionally zipped). Returns mm."""
    import zipfile
    import xml.etree.ElementTree as ET
    from array import array
    fh = open(path, "rb")
    if fh.read(2) == b"PK":
        fh.close()
        z = zipfile.ZipFile(path)
        fh = z.open(z.namelist()[0])
    else:
        fh.seek(0)
    vx, tr, cur = array("d"), array("q"), {}
    base, unit, count, box = 0, 1.0, 0, None
    with fh:
        for ev, el in ET.iterparse(fh, events=("start", "end")):
            tag = _local(el.tag)
            if ev == "start":
                if tag == "amf":
                    unit = UNIT_MM.get((el.get("unit") or "millimeter").lower(), 1.0)
                elif tag == "object":
                    base = len(vx) // 3
                elif tag in ("vertices", "volume"):
                    box = el
                continue
            if tag in ("x", "y", "z"):
                cur[tag] = float(el.text)
            elif tag in ("v1", "v2", "v3"):
                cur[tag] = int(el.text)
            elif tag == "vertex":
                vx.extend((cur["x"], cur["y"], cur["z"]))
            elif tag == "triangle":
                tr.extend((base + cur["v1"], base + cur["v2"], base + cur["v3"]))
            if tag in ("vertex", "triangle"):
                count += 1
                if count % 20000 == 0 and box is not None:
                    box.clear()
    V = np.array(vx).reshape(-1, 3) * unit
    T = np.array(tr, dtype=np.int64).reshape(-1, 3)
    if len(T) == 0:
        raise ValueError("The AMF file has no mesh")
    return V[T]


def _step_ocp(path):
    """STEP through Open CASCADE (pip install cadquery-ocp). Each CAD face is meshed
    by the CAD kernel, so curved faces come out as accurate triangles."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    to_face = getattr(TopoDS, "Face_s", None) or TopoDS.Face
    triangulation = getattr(BRep_Tool, "Triangulation_s", None) or BRep_Tool.Triangulation
    reader = STEPControl_Reader()
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise ValueError("The STEP file could not be read")
    reader.TransferRoots()
    shape = reader.OneShape()
    # 0.001 = chord error as a fraction of each edge's size; 0.15 rad = about 9 degrees per facet
    BRepMesh_IncrementalMesh(shape, 0.001, True, 0.15, True)
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = to_face(exp.Current())
        loc = TopLoc_Location()
        tri = triangulation(face, loc)
        if tri is not None and tri.NbTriangles() > 0:
            trsf = loc.Transformation()
            pts = np.empty((tri.NbNodes(), 3))
            for i in range(tri.NbNodes()):
                p = tri.Node(i + 1).Transformed(trsf)
                pts[i] = (p.X(), p.Y(), p.Z())
            idx = np.empty((tri.NbTriangles(), 3), np.int64)
            for i in range(tri.NbTriangles()):
                t = tri.Triangle(i + 1)
                idx[i] = (t.Value(1), t.Value(2), t.Value(3))
            idx -= 1
            if face.Orientation() == TopAbs_REVERSED:   # keep facet normals pointing outward
                idx = idx[:, [0, 2, 1]]
            out.append(pts[idx])
        exp.Next()
    if not out:
        raise ValueError("The STEP file has no surfaces")
    return np.concatenate(out)


def _step_gmsh(path):
    """STEP through gmsh. Kept only as a fallback for machines that already have gmsh."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(path)
        bb = gmsh.model.getBoundingBox(-1, -1)
        diag = math.dist(bb[:3], bb[3:]) or 1.0
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 40)
        gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 15.0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", diag / 2000.0)
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        conn = gmsh.model.mesh.getElementsByType(2)[1]
    finally:
        gmsh.finalize()
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    tags = np.asarray(tags, np.int64)
    lut = np.zeros(int(tags.max()) + 1, np.int64)
    lut[tags] = np.arange(len(tags))
    return xyz[lut[np.asarray(conn, np.int64)].reshape(-1, 3)]


def read_step(path):
    """STEP holds exact CAD surfaces, so a CAD kernel is needed to turn them into triangles.
    Uses Open CASCADE (cadquery-ocp) when installed, otherwise gmsh if that is present."""
    problem = None
    try:
        return _step_ocp(path)
    except ImportError:
        pass
    except ValueError:
        raise
    except Exception as exc:                            # installed, but this version behaves differently
        problem = "Open CASCADE reader failed (%s: %s)" % (type(exc).__name__, exc)
    try:
        return _step_gmsh(path)
    except ImportError:
        pass
    raise RuntimeError(problem or "STEP files need a CAD kernel. Install it with:  pip install cadquery-ocp")


def load_mesh(path):
    """Read any supported file. Returns welded vertices, faces, and units if the file states them."""
    low = path.lower()
    ext = os.path.splitext(low)[1]
    if ext in (".sldprt", ".sldasm", ".prt", ".asm") or re.search(r"\.(prt|asm)\.\d+$", low):
        raise ValueError(NATIVE_MSG)
    if ext == ".3mf":
        tri, unit = read_3mf(path), "mm"
    elif ext == ".amf":
        tri, unit = read_amf(path), "mm"
    elif ext in (".step", ".stp"):
        tri, unit = read_step(path), "mm"
    else:
        tri, unit = read_stl(path), None                # STL carries no units
    if len(tri) == 0:
        raise ValueError("No triangles found in %s" % path)
    P = tri.reshape(-1, 3) + tri.dtype.type(0)          # +0 turns -0.0 into 0.0
    order = np.lexsort((P[:, 2], P[:, 1], P[:, 0]))
    Ps = P[order]
    new = np.ones(len(Ps), bool)
    new[1:] = np.any(Ps[1:] != Ps[:-1], axis=1)
    inv = np.empty(len(Ps), np.int32)
    inv[order] = np.cumsum(new) - 1
    return Ps[new].astype(np.float64), inv.reshape(-1, 3), unit
