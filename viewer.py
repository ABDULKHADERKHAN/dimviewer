"""The window: 3D view, on-screen buttons, click-to-measure. Uses VTK."""
# Part of DimViewer. Copyright (c) 2026 Abdul Khader Khan.
# Free for noncommercial use. License and contact details are in README.md.

import math
import os

import numpy as np

from model import Model, Units, mem_mb
from readers import FILE_TYPES

HELP = """Left-drag rotate, wheel zoom
Middle-drag or Shift+drag pan
Click: measure edge or surface
Shift+click: second feature
Measure button (T): click two features
  for the shortest distance between them
Caliper button (K): same two clicks, but
  shows the outside length across both
Space: measure under the cursor
Keys: 1-7 views, R fit, U units
I model info, M model units
G save a drawing (2 sheets)
D datum A (reference feature)
O open, Esc clear, Q quit"""

# on-screen buttons, top row first, right-aligned: (label, action); action None = plain text
BUTTONS = [
    [("View:", None), ("Front", "1"), ("Back", "2"), ("Top", "3"), ("Bottom", "4"),
     ("Right", "5"), ("Left", "6"), ("Iso", "7"), ("Fit", "r"), ("Open", "o"), ("Drawing", "g"), ("Info", "i"), ("Keys", "h")],
    [("Select:", None), ("Auto", "sel:auto"), ("Edges", "sel:edges"), ("Faces", "sel:faces"),
     ("Circles", "sel:circles"), ("mm/in", "u"), ("Wire", "w"), ("Colour", "c"),
     ("Axes", "b"), ("Edge lines", "e"), ("Measure", "t"), ("Caliper", "k")],
]

VIEWS = {"1": ((0, -1, 0), (0, 0, 1)), "2": ((0, 1, 0), (0, 0, 1)),
         "3": ((0, 0, 1), (0, 1, 0)), "4": ((0, 0, -1), (0, 1, 0)),
         "5": ((1, 0, 0), (0, 0, 1)), "6": ((-1, 0, 0), (0, 0, 1)),
         "7": ((1, -1, 1), (0, 0, 1))}
BASE, SEL, SEL2, DATUM = (205, 208, 212), (255, 150, 40), (80, 190, 255), (120, 220, 140)
TYPECOL = {"plane": (176, 196, 222), "cyl": (240, 196, 130), "cone": (200, 170, 225),
           "sphere": (160, 215, 175), "free": (205, 205, 205)}


def _nice(x):
    """Smallest of 1, 2, 5 x 10^n that is not below x."""
    e = 10.0 ** math.floor(math.log10(x))
    return next(f * e for f in (1, 2, 5, 10) if f * e >= x * 0.999999)


def ask_save(default):
    """File dialog for saving. Returns a path, "" if cancelled, None if no dialog is available."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        p = filedialog.asksaveasfilename(title="Save drawing", defaultextension=".html",
                                         initialdir=os.path.dirname(default) or ".",
                                         initialfile=os.path.basename(default),
                                         filetypes=[("Drawing, both sheets in one file", "*.html"),
                                                    ("SVG, one file per sheet", "*.svg")])
        root.destroy()
        return p or ""
    except Exception:
        return None


def ask_open():
    """File dialog for opening a model. Returns a path, or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        p = filedialog.askopenfilename(title="Open model", filetypes=FILE_TYPES)
        root.destroy()
        return p or None
    except Exception:
        return None


class Viewer:
    """The application window. Holds the open model, what is selected, and the on-screen controls."""
    def __init__(self, args):
        """Build the window, its text areas and buttons, and connect mouse and keyboard events."""
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        self.vtk, self.n2v, self.n2id = vtk, numpy_to_vtk, numpy_to_vtkIdTypeArray
        self.args = args
        self.U = Units(args.units, args.display or args.units)
        self.model = None
        self.sel = self.sel2 = self.datum = None
        self.p1 = self.p2 = None
        self.measure = False                             # Measure mode: clicks alternate feature 1, feature 2
        self.caliper = False                             # draw the outside (caliper) length, not the shortest gap
        self._gap = (None, None)                         # cached shortest distance for the current pair
        self.by_type = False
        self.show_help = False
        self.show_info = False                           # model data block, toggled by Info
        self.mode = "auto"                               # selection filter
        self.wire, self.show_edges, self.show_axes = False, True, True
        self.press = None
        self.swallow = False
        self._size = None
        self.last_mem = 0.0

        vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.09, 0.10, 0.12)
        self.ren.SetBackground2(0.24, 0.26, 0.30)
        self.ren.GradientBackgroundOn()
        self.rw = vtk.vtkRenderWindow()
        self.rw.AddRenderer(self.ren)
        self.rw.SetSize(1280, 820)
        self.rw.SetWindowName("DimViewer")
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.rw)
        style = vtk.vtkInteractorStyleTrackballCamera()
        style.AddObserver("CharEvent", lambda o, e: None)   # switch off VTK's own hot keys
        self.iren.SetInteractorStyle(style)
        self.style = style
        self.ren.GetActiveCamera().ParallelProjectionOn()

        self.mesh_actor = self.edge_actor = self.circ_actor = None
        self.scale_actors, self._scale_key = [], None
        self.hi_actors = [self._line_actor(SEL, 5.0), self._line_actor(SEL2, 5.0),
                          self._line_actor(DATUM, 4.0)]
        # top left: file name. bottom left: model data, with the picked feature above it.
        # top right: buttons (and the key list under them when Keys is on).
        self.t_title = self._text(0.01, 0.985, top=True, size=17)
        self.t_info = self._text(0.01, 0.01)
        self.t_sel = self._text(0.01, 0.01, color=(1.0, 0.86, 0.55))
        for t, y in ((self.t_info, 10), (self.t_sel, 136)):
            t.GetPositionCoordinate().SetCoordinateSystemToDisplay()
            t.GetPositionCoordinate().SetValue(12, y)
        self.t_help = self._text(0.99, 0.90, top=True, right=True, size=13, color=(0.72, 0.76, 0.80))
        self.t_help.GetPositionCoordinate().SetCoordinateSystemToDisplay()
        self._build_toolbar()
        self.labels = [self._label(SEL), self._label(SEL2)]
        self.gap_actor = self._line_actor((255, 230, 60), 3.0)       # shortest-distance line
        self.gap_label = self._label((255, 230, 60))
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.0005)
        self.picker.PickFromListOn()

        self.axes = vtk.vtkAxesActor()
        self.marker = vtk.vtkOrientationMarkerWidget()
        self.marker.SetOrientationMarker(self.axes)
        self.marker.SetInteractor(self.iren)
        self.marker.SetViewport(0.86, 0.0, 1.0, 0.18)
        self.marker.SetEnabled(1)
        self.marker.InteractiveOff()

        # Mouse clicks are watched on the camera style, not on the window: once the button
        # is down the style takes over the mouse, and the window never reports the release.
        style.AddObserver("LeftButtonPressEvent", self.on_press)
        style.AddObserver("LeftButtonReleaseEvent", self.on_release)
        style.AddObserver("EndInteractionEvent", self.on_end)
        self.ren.AddObserver("StartEvent", lambda o, e: self._layout())   # follow window resizes
        self.iren.AddObserver("KeyPressEvent", self.on_key)
        self.iren.AddObserver("TimerEvent", self.on_timer)

    # -- small VTK builders ---------------------------------------------------
    def _add(self, prop):
        """Add anything drawable. AddViewProp exists in every VTK version; AddActor2D does not."""
        add = getattr(self.ren, "AddViewProp", None) or self.ren.AddActor
        add(prop)

    def _remove(self, prop):
        """Remove something from the scene (works across VTK versions)."""
        rem = getattr(self.ren, "RemoveViewProp", None) or self.ren.RemoveActor
        rem(prop)

    def _text(self, x, y, top=False, right=False, size=15, color=(0.93, 0.94, 0.95)):
        """Make a text area fixed to the screen at (x, y), given as fractions of the window."""
        t = self.vtk.vtkTextActor()
        tp = t.GetTextProperty()
        tp.SetFontFamilyToCourier()
        tp.SetFontSize(size)
        tp.SetColor(*color)
        if top:
            tp.SetVerticalJustificationToTop()
        if right:
            tp.SetJustificationToRight()
        t.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        t.GetPositionCoordinate().SetValue(x, y)
        self._add(t)
        return t

    def _label(self, rgb):
        """Make a floating label that follows a 3D point on the model."""
        t = self.vtk.vtkTextActor()
        tp = t.GetTextProperty()
        tp.SetFontSize(18)
        tp.BoldOn()
        tp.SetColor(1, 1, 1)
        tp.SetBackgroundColor(rgb[0] / 400.0, rgb[1] / 400.0, rgb[2] / 400.0)
        tp.SetBackgroundOpacity(0.85)
        t.GetPositionCoordinate().SetCoordinateSystemToWorld()
        t.SetVisibility(0)
        self._add(t)
        return t

    def _build_toolbar(self):
        """Rows of clickable buttons in the top right corner, sized in screen pixels."""
        vtk = self.vtk
        self.btn_poly = vtk.vtkPolyData()
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(self.btn_poly)
        back = vtk.vtkActor2D()
        back.SetMapper(mapper)
        back.SetLayerNumber(0)
        self._add(back)
        self.btn_back = back
        self.btn_spec = []          # (distance of left edge from window right, top offset, width, action, text)
        for r, row in enumerate(BUTTONS):
            widths = [9 * len(label) + 14 for label, _ in row]
            left = sum(widths) + 6 * (len(row) - 1) + 10
            for (label, act), w in zip(row, widths):
                t = vtk.vtkTextActor()
                t.SetInput(label)
                tp = t.GetTextProperty()
                tp.SetFontFamilyToCourier()
                tp.SetFontSize(14)
                tp.SetColor(*((0.95, 0.96, 0.97) if act else (0.70, 0.74, 0.78)))
                tp.SetJustificationToCentered()
                tp.SetVerticalJustificationToCentered()
                t.SetLayerNumber(1)
                self._add(t)
                self.btn_spec.append((left, 8 + r * 32, w, act, t))
                left -= w + 6
        n = sum(1 for b in self.btn_spec if b[3])
        idt = np.int64 if vtk.vtkIdTypeArray().GetDataTypeSize() == 8 else np.int32
        conn = np.empty((n, 5), idt)
        conn[:, 0] = 4
        conn[:, 1:] = np.arange(4 * n).reshape(n, 4)
        ca = vtk.vtkCellArray()
        ca.SetCells(n, self.n2id(conn.ravel(), deep=True))
        self.btn_poly.SetPolys(ca)
        self.btn = []
        self._layout()
        self._paint_buttons()

    def _layout(self):
        """Place the buttons against the top right corner for the current window size."""
        W, H = self.rw.GetSize()[:2]
        if (W, H) == self._size or W < 2 or H < 2:
            return
        self._size = (W, H)
        self.btn, quads = [], []
        for left, top, w, act, t in self.btn_spec:
            x0, y1 = W - left, H - top
            y0 = y1 - 26
            t.SetDisplayPosition(int(x0 + w // 2), int(y0 + 13))
            if act:
                self.btn.append((x0, y0, x0 + w, y1, act))
                quads += [(x0, y0, 0), (x0 + w, y0, 0), (x0 + w, y1, 0), (x0, y1, 0)]
        vp = self.vtk.vtkPoints()
        vp.SetData(self.n2v(np.array(quads, np.float32), deep=True))
        self.btn_poly.SetPoints(vp)
        self.t_help.GetPositionCoordinate().SetValue(W - 12, H - 80)

    def _button_on(self, act):
        """Whether a button should be shown lit."""
        if act.startswith("sel:"):
            return self.mode == act[4:]
        return {"u": self.U.display == "in", "w": self.wire, "c": self.by_type, "b": self.show_axes,
                "e": self.show_edges, "h": self.show_help, "i": self.show_info, "t": self.measure, "k": self.caliper}.get(act, False)

    def _paint_buttons(self):
        """Colour each button by whether it is on."""
        col = np.array([(64, 132, 190) if self._button_on(b[4]) else (62, 66, 76) for b in self.btn],
                       np.uint8)
        arr = self.n2v(col, deep=True, array_type=self.vtk.VTK_UNSIGNED_CHAR)
        arr.SetName("colors")
        self.btn_poly.GetCellData().SetScalars(arr)

    def _button_at(self, pos):
        """Action of the button under screen position pos, or None."""
        for x0, y0, x1, y1, act in self.btn:
            if x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1:
                return act
        return None

    def _poly(self, pts, cells):
        """Turn numpy points and cells (triangles or line segments) into VTK geometry."""
        vtk = self.vtk
        pd = vtk.vtkPolyData()
        vp = vtk.vtkPoints()
        vp.SetData(self.n2v(np.ascontiguousarray(pts, dtype=np.float32), deep=True))
        pd.SetPoints(vp)
        n, k = cells.shape
        idt = np.int64 if vtk.vtkIdTypeArray().GetDataTypeSize() == 8 else np.int32
        conn = np.empty((n, k + 1), idt)
        conn[:, 0] = k
        conn[:, 1:] = cells
        ca = vtk.vtkCellArray()
        ca.SetCells(n, self.n2id(conn.ravel(), deep=True))
        if k == 3:
            pd.SetPolys(ca)
        else:
            pd.SetLines(ca)
        return pd

    def _seg_poly(self, s0, s1):
        """VTK geometry for a set of line segments."""
        P = np.empty((2 * len(s0), 3))
        P[0::2], P[1::2] = s0, s1
        return self._poly(P, np.arange(2 * len(s0)).reshape(-1, 2))

    def _line_actor(self, rgb, width):
        """Make an empty, hidden line object in the given colour and width."""
        a = self.vtk.vtkActor()
        a.SetMapper(self.vtk.vtkPolyDataMapper())
        pr = a.GetProperty()
        pr.SetColor(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
        pr.SetLineWidth(width)
        pr.LightingOff()
        a.PickableOff()
        a.SetVisibility(0)
        self._add(a)
        return a

    # -- loading -------------------------------------------------------------
    def load(self, path):
        """Open a model file and show it. Returns False (and shows the reason) if it cannot be read."""
        vtk = self.vtk
        est = os.path.getsize(path) / 50.0 * 600 / 1e6      # rough peak MB while analysing
        if est > self.args.mem_warn:
            print("Note: this file may need about %.0f MB while loading (limit set to %d MB)."
                  % (est, self.args.mem_warn))
        try:
            model = Model(path, angle=self.args.angle)
        except Exception as exc:
            print("Could not load %s: %s" % (path, exc))
            import textwrap
            self.t_sel.SetInput("Could not load %s:\n%s"
                                % (os.path.basename(path), "\n".join(textwrap.wrap(str(exc), 70))))
            self.rw.Render()
            return False
        self.model = model
        if model.units:
            self.U.model = model.units
        for a in (self.mesh_actor, self.edge_actor, self.circ_actor):
            if a is not None:
                self._remove(a)
        self.poly = self._poly(model.V, model.F)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.poly)
        self.mesh_actor = vtk.vtkActor()
        self.mesh_actor.SetMapper(mapper)
        pr = self.mesh_actor.GetProperty()
        pr.SetInterpolationToFlat()
        pr.SetAmbient(0.25)
        pr.SetDiffuse(0.75)
        pr.SetSpecular(0.08)
        self._add(self.mesh_actor)
        self.picker.InitializePickList()
        self.picker.AddPickList(self.mesh_actor)

        self.edge_actor = self._line_actor((35, 38, 45), 1.5)
        if len(model.feat):
            ev = model.ev[model.feat]
            self.edge_actor.GetMapper().SetInputData(self._seg_poly(model.V[ev[:, 0]], model.V[ev[:, 1]]))
        self.circ_actor = None                          # built the first time Circles mode is used
        self.wire = False

        self._scale_key = None                          # size scale is rebuilt in update()

        self.sel = self.sel2 = self.datum = None
        self.rw.SetWindowName("DimViewer - " + os.path.basename(path))
        self.set_view("7")
        self.update()
        return True

    def _build_scale(self):
        """Box around the part with size ticks on X, Y and Z, measured from one corner.
        All three axes share one tick step, so a thin part gets one or two labels on its
        short side instead of a pile of overlapping numbers."""
        vtk, m, U = self.vtk, self.model, self.U
        for a in self.scale_actors:
            self._remove(a)
        self.scale_actors = []
        self._scale_key = (U.model, U.display)
        lo, size = m.lo, m.hi - m.lo
        f = 1.0 if U.display == U.model else ((1.0 / 25.4) if U.display == "in" else 25.4)
        step = _nice(max(float(size.max()) * f, 1e-12) / 5.0) / f
        tick = 0.012 * m.diag
        s0, s1 = [], []
        corners = [np.array([i, j, k], float) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        for a in corners:                                # the 12 box edges
            for ax in range(3):
                if a[ax] == 0:
                    b = a.copy()
                    b[ax] = 1
                    s0.append(lo + a * size)
                    s1.append(lo + b * size)
        outward = (np.array([0, -1.0, 0]), np.array([-1.0, 0, 0]), np.array([-0.7071, -0.7071, 0]))
        dec = 3 if U.display == "mm" else 4
        for ax in range(3):
            vals = [v for v in np.arange(0.0, size[ax], step) if size[ax] - v > 0.45 * step] or [0.0]
            vals.append(float(size[ax]))
            for j, v in enumerate(vals):
                p = lo.copy()
                p[ax] += v
                s0.append(p)
                s1.append(p + outward[ax] * tick)
                if j == len(vals) - 1:
                    txt = "%s %.*f %s" % ("XYZ"[ax], dec, v * f, U.display)
                elif v == 0.0 and (ax > 0 or size[ax] < 0.6 * step):
                    continue                             # one "0" at the corner is enough
                else:
                    txt = "%g" % round(v * f, 6)
                t = vtk.vtkTextActor()
                t.SetInput(txt)
                tp = t.GetTextProperty()
                tp.SetFontFamilyToCourier()
                tp.SetFontSize(13)
                tp.SetColor(0.80, 0.84, 0.88)
                tp.SetJustificationToCentered()
                tp.SetVerticalJustificationToCentered()
                t.GetPositionCoordinate().SetCoordinateSystemToWorld()
                q = p + outward[ax] * tick * 3.5
                t.GetPositionCoordinate().SetValue(float(q[0]), float(q[1]), float(q[2]))
                self._add(t)
                self.scale_actors.append(t)
        box = self._line_actor((150, 158, 168), 1.0)
        box.GetMapper().SetInputData(self._seg_poly(np.array(s0), np.array(s1)))
        self.scale_actors.append(box)

    def save_drawing(self):
        """Write the two-sheet drawing of the open model to a file and open it in the browser."""
        if self.model is None:
            return
        default = os.path.splitext(self.model.path)[0] + "_drawing.html"
        path = ask_save(default)
        if path == "":
            return                                       # cancelled
        path = path or default
        try:
            from drawing import make_drawing
            info = make_drawing(self.model, self.U, path)
            msg = "Drawing saved (%d sheets): %s\nScale %s, %d features dimensioned" % (
                info["sheets"], path, info["scale"], info["features"])
            try:
                if hasattr(os, "startfile"):
                    os.startfile(path)                   # Windows: open in the default program
                else:
                    import webbrowser
                    webbrowser.open("file://" + os.path.abspath(path))
            except Exception:
                pass
        except Exception as exc:
            msg = "Could not make the drawing: %s" % exc
        print(msg)
        self.t_sel.SetInput(msg)
        self.rw.Render()

    # -- display state -------------------------------------------------------
    def set_view(self, key):
        """Point the camera along a standard direction (keys 1 to 7) and fit the part."""
        if self.model is None:
            return
        d, up = VIEWS[key]
        d = np.array(d, float) / np.linalg.norm(d)
        c = (self.model.lo + self.model.hi) / 2.0
        cam = self.ren.GetActiveCamera()
        cam.SetFocalPoint(*c)
        cam.SetPosition(*(c + d * max(self.model.diag, 1e-9) * 3.0))
        cam.SetViewUp(*up)
        self.ren.ResetCamera()
        self.ren.ResetCameraClippingRange()
        self.rw.Render()

    def _recolor(self):
        """Colour the triangles: base grey or by surface type, then the selected surfaces."""
        m = self.model
        col = np.empty((len(m.F), 3), np.uint8)
        col[:] = BASE
        if (self.by_type or self.mode == "faces") and not m.lazy:
            lut = np.array([TYPECOL[m.sfit(i)["kind"]] for i in range(len(m.regs))], np.uint8)
            col[:] = lut[m.labels]
        for s, rgb in ((self.datum, DATUM), (self.sel2, SEL2), (self.sel, SEL)):
            if s and s[0] == "surf":
                col[m.regs[s[1]]] = rgb
        arr = self.n2v(col, deep=True, array_type=self.vtk.VTK_UNSIGNED_CHAR)
        arr.SetName("colors")
        self.poly.GetCellData().SetScalars(arr)

    def update(self):
        """Redraw everything that depends on the current selection, mode, units and toggles."""
        m, U = self.model, self.U
        if m is None:
            self.t_title.SetInput("No file loaded. Click Open (or press O) to open a model.")
            self.t_info.SetInput("")
            self.t_help.SetInput(HELP if self.show_help else "")
            self._paint_buttons()
            self.rw.Render()
            return
        self._recolor()
        self._paint_buttons()
        all_edges = self.mode == "edges"                 # Edges mode: every edge lit up
        pr = self.edge_actor.GetProperty()
        pr.SetColor(*((0.25, 0.90, 1.0) if all_edges else (0.14, 0.15, 0.18)))
        pr.SetLineWidth(3.0 if all_edges else 1.5)
        self.edge_actor.SetVisibility(1 if (len(m.feat) and (self.show_edges or all_edges)) else 0)
        if self.mode == "circles" and self.circ_actor is None:
            self.circ_actor = self._line_actor((255, 225, 60), 4.0)
            cc = m.circle_chains()
            if cc:
                s0, s1 = zip(*(m.segs(c) for c in cc))
                self.circ_actor.GetMapper().SetInputData(self._seg_poly(np.concatenate(s0), np.concatenate(s1)))
        if self.circ_actor is not None:
            self.circ_actor.SetVisibility(1 if (self.mode == "circles" and m.circle_chains()) else 0)
        if self._scale_key != (U.model, U.display):
            self._build_scale()
        for a in self.scale_actors:
            a.SetVisibility(1 if self.show_axes else 0)
        for actor, s in zip(self.hi_actors, (self.sel, self.sel2, self.datum)):
            if s and s[0] == "edge":
                actor.GetMapper().SetInputData(self._seg_poly(*m.segs(s[1])))
                actor.SetVisibility(1)
            else:
                actor.SetVisibility(0)
        txt = []
        for tag, s, p, lab in (("1", self.sel, self.p1, self.labels[0]),
                               ("2", self.sel2, self.p2, self.labels[1])):
            if not s:
                lab.SetVisibility(0)
                continue
            title, lines, short = m.describe(s, U)
            txt.append("[%s] %s" % (tag, title))
            txt += ["    " + ln for ln in lines]
            lab.SetInput(" %s " % short)
            lab.GetPositionCoordinate().SetValue(float(p[0]), float(p[1]), float(p[2]))
            lab.SetVisibility(1)
        if self.sel and self.sel2 and self.sel != self.sel2:
            if self._gap[0] != (self.sel, self.sel2):
                self._gap = ((self.sel, self.sel2), m.closest(self.sel, self.sel2))
            dist, pa, pb = self._gap[1]
            txt.append("[1] to [2]:")
            txt.append("    Shortest distance: " + U.L(dist))
            if dist > 1e-9 * m.diag:
                dv = np.abs(pb - pa)
                txt.append("    Along X, Y, Z: %s, %s, %s" % (U.one(dv[0]), U.one(dv[1]), U.one(dv[2])))
            ax, outside, inside, p_lo, p_hi = m.caliper(self.sel, self.sel2)
            txt.append("    Caliper, outside along %s: %s" % ("XYZ"[ax], U.L(outside)))
            if inside is not None:
                txt.append("    Caliper, inside along %s: %s" % ("XYZ"[ax], U.L(inside)))
            txt += ["    " + ln for ln in m.relate(self.sel, self.sel2, U)]
            if self.caliper:                             # measuring line along the axis, then the jaw to the far point
                end = p_lo.copy()
                end[ax] = p_hi[ax]
                self.gap_actor.GetMapper().SetInputData(self._seg_poly(np.array([p_lo, end]), np.array([end, p_hi])))
                mid, dist = (p_lo + end) / 2.0, outside
            else:
                self.gap_actor.GetMapper().SetInputData(self._seg_poly(pa[None, :], pb[None, :]))
                mid = (pa + pb) / 2.0
            self.gap_actor.SetVisibility(1)
            self.gap_label.SetInput(" %s " % U.one(dist))
            self.gap_label.GetPositionCoordinate().SetValue(float(mid[0]), float(mid[1]), float(mid[2]))
            self.gap_label.SetVisibility(1)
        else:
            self.gap_actor.SetVisibility(0)
            self.gap_label.SetVisibility(0)
            if self.measure:
                txt.append("Measure: click the %s feature" % ("second" if self.sel else "first"))
        if self.datum:
            txt.append("Datum A = %s" % m.name(self.datum))
            if self.sel and self.sel != self.datum:
                txt += ["    " + ln for ln in m.relate(self.datum, self.sel, U)]
        self.t_sel.SetInput("\n".join(txt))
        self.t_help.SetInput(HELP if self.show_help else "")
        self._info()
        self.rw.Render()

    def _info(self):
        """File name top left (details on a second line). The model data block at the bottom
        only shows when Info is on; a memory warning always shows."""
        m, U = self.model, self.U
        if m is None:
            return
        self.t_title.SetInput("%s\n%s triangles, %d surfaces, %d edges"
                              % (os.path.basename(m.path), format(len(m.F), ","), len(m.regs),
                                 len(m.chains)))
        lines = []
        if self.show_info:
            lines.append("Model units: %s (%s)   Display: %s"
                         % (U.model, "from file" if m.units else "assumed, STL has none, M to change",
                            U.display))
            lines += m.summary(U)
        mb = mem_mb()
        if mb is None:
            if self.show_info:
                lines.append("Memory: not available (pip install psutil)")
        else:
            over = mb > self.args.mem_warn
            if over:
                lines.append("Memory: %.0f MB   ** ABOVE %d MB LIMIT **" % (mb, self.args.mem_warn))
                if self.last_mem <= self.args.mem_warn:
                    print("WARNING: memory use %.0f MB is above the %d MB limit." % (mb, self.args.mem_warn))
            elif self.show_info:
                lines.append("Memory: %.0f MB" % mb)
            self.last_mem = mb
        self.t_info.SetInput("\n".join(lines))
        self.t_sel.GetPositionCoordinate().SetValue(12, 12 + 19 * len(lines) + (10 if lines else 0))

    # -- events --------------------------------------------------------------
    def on_press(self, obj, event):
        """Left button down: a button acts at once, anything else may start a camera move."""
        self._layout()
        pos = self.iren.GetEventPosition()
        act = self._button_at(pos)
        if act:                                          # a button: act now, keep the camera still
            self.press, self.swallow = None, True
            self.do(act)
            return
        self.press, self.swallow = pos, False
        self.style.OnLeftButtonDown()                    # let the camera start rotating / panning

    def on_release(self, obj, event):
        """Left button up."""
        if self.swallow:
            self.swallow = False
            return
        self.style.OnLeftButtonUp()
        self._clicked()

    def on_end(self, obj, event):
        """The camera style finished a move (also fires when the left button comes up)."""
        self._clicked()                                  # second route to the same handler

    def _clicked(self):
        """Runs once when the left button comes up. A press and release at the same spot
        is a click (measure); anything else was a drag (camera move)."""
        press, self.press = self.press, None
        if press is None or self.model is None:
            return
        pos = self.iren.GetEventPosition()
        if abs(pos[0] - press[0]) > 3 or abs(pos[1] - press[1]) > 3:
            return
        shift = bool(self.iren.GetShiftKey())
        if not self.picker.Pick(pos[0], pos[1], 0, self.ren) or self.picker.GetCellId() < 0:
            if not shift:
                self.sel = self.sel2 = None
                self.update()
            return
        p = np.array(self.picker.GetPickPosition(), float)
        cam = self.ren.GetActiveCamera()
        tol = 8.0 * 2.0 * cam.GetParallelScale() / max(1, self.rw.GetSize()[1])
        hit = self.model.pick(int(self.picker.GetCellId()), p, tol, self.mode)
        if self.sel and (shift or (self.measure and not self.sel2)):
            self.sel2, self.p2 = hit, p
        else:
            self.sel, self.p1, self.sel2 = hit, p, None
        self.update()

    def on_key(self, obj, event):
        """Keyboard shortcut."""
        k = self.iren.GetKeySym() or ""
        k = k[3:] if k.startswith("KP_") else k
        self.do(k.lower())

    def do(self, k):
        """Carry out one action: a key press or an on-screen button."""
        if k.startswith("sel:"):
            self.mode = k[4:]
        elif k in VIEWS:
            self.set_view(k)
        elif k == "r":
            self.ren.ResetCamera()
            self.ren.ResetCameraClippingRange()
        elif k == "u":
            self.U.display = "in" if self.U.display == "mm" else "mm"
        elif k == "m" and not (self.model and self.model.units):
            self.U.model = "in" if self.U.model == "mm" else "mm"
        elif k == "d":
            self.datum = None if (self.sel is None or self.sel == self.datum) else self.sel
        elif k == "e":
            self.show_edges = not self.show_edges
        elif k == "w" and self.mesh_actor is not None:
            self.wire = not self.wire
            pr = self.mesh_actor.GetProperty()
            if self.wire:
                pr.SetRepresentationToWireframe()
            else:
                pr.SetRepresentationToSurface()
        elif k == "c":
            self.by_type = not self.by_type
        elif k == "b":
            self.show_axes = not self.show_axes
        elif k == "h":
            self.show_help = not self.show_help
        elif k == "i":
            self.show_info = not self.show_info
        elif k == "t":
            self.measure = not self.measure
            self.sel = self.sel2 = None
        elif k == "k":
            self.caliper = not self.caliper
            if self.caliper and not self.measure:        # caliper needs two picks, so switch Measure on too
                self.measure = True
                self.sel = self.sel2 = None
        elif k == "g":
            self.save_drawing()
            return
        elif k == "escape":
            self.sel = self.sel2 = None
        elif k == "space":
            self.press = self.iren.GetEventPosition()
            self._clicked()
            return
        elif k == "o":
            p = ask_open()
            if p:
                self.load(p)
            return
        elif k == "q":
            self.rw.Finalize()
            self.iren.TerminateApp()
            return
        else:
            return
        self.update()

    def on_timer(self, obj, event):
        """Every few seconds: refresh the memory readout."""
        self._info()
        self.rw.Render()

    def start(self, path):
        """Show the window, open the first file if one was given, and run until closed."""
        self.iren.Initialize()
        if path:
            self.load(path)
        self.update()
        self.iren.CreateRepeatingTimer(3000)             # refresh the memory readout
        self.iren.Start()
