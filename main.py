#!/usr/bin/env python3
"""
DimViewer: a lightweight 3D model dimensional viewer
====================================================
Open a model, rotate / zoom / snap to standard views, and click to measure:

  click an edge      -> length (straight), diameter or radius (circle / arc)
  click a surface    -> type (plane, cylinder, cone, sphere), size, measured form
                        error, and the ASME Y14.5 controls that apply to it
  Shift+click        -> second feature: distance / angle between the two
  Drawing button     -> two-sheet drawing: dimensioned views, then section views

Files:    STL (binary or ASCII), 3MF, AMF. STEP if cadquery-ocp is installed.
          SolidWorks .sldprt and Creo .prt cannot be read directly: export STEP.

Run:      python main.py part.stl
          python main.py                          (opens a file dialog)
          python main.py part.3mf --report        (text only, no window)
          python main.py part.stl --csv features.csv
          python main.py part.stl --drawing part.html

Needs:    pip install -r requirements.txt   (numpy, vtk, psutil; optional cadquery-ocp for STEP)
Help:     python main.py --help

How the code is organised
  main.py     this file: reads the command line and starts the window or a text output
  readers.py  turns STL / 3MF / AMF / STEP files into triangles
  model.py    finds surfaces and edges in the triangles and measures them (no graphics)
  viewer.py   the window: 3D view, buttons, click-to-measure (uses VTK)
  drawing.py  the two-sheet drawing (views, feature dimensions, sections)
"""
import argparse
import sys

from model import Model, Units, mem_mb
from viewer import Viewer, ask_open

# Copyright (c) 2026 Abdul Khader Khan. Free for noncommercial use; see README.md.
__version__ = "0.1.0"
__author__ = "Abdul Khader Khan"
__email__ = "akhader6076@yahoo.in"          # questions and bug reports


def main():
    """Read the command line, then either print / export and exit, or open the window."""
    ap = argparse.ArgumentParser(description="DimViewer: lightweight 3D model dimensional viewer (STL, 3MF, AMF, STEP)")
    ap.add_argument("stl", nargs="?", metavar="file",
                    help="STL, 3MF, AMF or STEP file (a file dialog opens if omitted)")
    ap.add_argument("--units", choices=["mm", "in"], default="mm", help="units the STL is in (default mm)")
    ap.add_argument("--display", choices=["mm", "in"], help="units to show first (default: same as --units)")
    ap.add_argument("--angle", type=float, default=30.0,
                    help="edge angle in degrees: faces meeting at more than this are separate surfaces")
    ap.add_argument("--mem-warn", type=float, default=1024, help="warn above this memory use, MB")
    ap.add_argument("--report", action="store_true", help="print all features as text, no window")
    ap.add_argument("--csv", help="write all features to a CSV file, no window")
    ap.add_argument("--dual", action="store_true",
                    help="drawing: also show overall sizes in the other unit, in brackets")
    ap.add_argument("--drawing", metavar="FILE",
                    help="write the two-sheet drawing, no window (.html = one file, .svg = one file per sheet)")
    ap.add_argument("--version", action="version",
                    version="DimViewer %s, by %s <%s>" % (__version__, __author__, __email__))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    if args.report or args.csv or args.drawing:
        if not args.stl:
            ap.error("--report, --csv and --drawing need a model file")
        try:
            model = Model(args.stl, angle=args.angle)
        except (ValueError, RuntimeError, OSError) as exc:
            sys.exit(str(exc))
        if args.report:
            model.report(Units(model.units or args.units, args.display or args.units))
        if args.csv:
            model.write_csv(args.csv)
            print("Wrote " + args.csv)
        if args.drawing:
            from drawing import make_drawing
            info = make_drawing(model, Units(model.units or args.units, args.display or args.units),
                                args.drawing, dual=args.dual)
            print("Wrote %s (%d sheets, scale %s)" % (args.drawing, info["sheets"], info["scale"]))
        mb = mem_mb()
        if mb is not None:
            print("Peak memory about %.0f MB" % mb)
        return

    try:
        viewer = Viewer(args)
    except ImportError:
        sys.exit("The viewer window needs VTK. Install it with:  pip install vtk")
    path = args.stl or ask_open()
    if not path:
        print("No file chosen. Press O in the window to open a model.")
    viewer.start(path)


if __name__ == "__main__":
    main()
