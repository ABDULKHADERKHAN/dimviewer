# DimViewer

A lightweight 3D model viewer for checking dimensions. Open a part, click an edge
or a face, and read its size. It can also write a simple three-view drawing.

## Install

Python 3.10 to 3.12 is recommended.

    pip install -r requirements.txt

To open STEP files, also run `pip install cadquery-ocp`.

## Run

    python main.py                 (opens a file dialog)
    python main.py part.stl

| File type | Support |
|---|---|
| STL (binary or ASCII) | Yes. STL has no units, so mm is assumed (press M to switch). |
| 3MF, AMF | Yes. Units are read from the file. |
| STEP (.step, .stp) | Yes, with `cadquery-ocp` installed. |
| SolidWorks .sldprt, Creo .prt | No. Export STEP from the CAD program first. |

## Using the window

- **Left-drag** rotates, **wheel** zooms, **middle-drag** or **Shift+drag** pans.
- **Click an edge** for its length, or the diameter or radius of a circle or arc.
- **Click a face** for its type (plane, cylinder, cone, sphere), size, and the
  ASME Y14.5 controls that apply to that kind of feature.
- **Measure** (or key T): click one feature, then another. A line is drawn across
  the shortest distance between the two, with its length and X, Y, Z parts.
  Shift+click a second feature does the same without switching mode.
- **Caliper** (or key K): same two clicks, but the line shows the outside length
  across both features, as a caliper held square to X, Y or Z would read it.
  The inside length (clear gap) is listed too.
- **Space** measures whatever is under the cursor.
- **Select** buttons limit what a click picks: Auto, Edges, Faces or Circles.
- **View** buttons snap to Front, Back, Top, Bottom, Right, Left and Iso.
- **Drawing** saves a two-sheet drawing and opens it in the browser.
- **Info** shows units, overall size, volume, surface area and memory use.
- **Keys** lists the keyboard shortcuts.

## Without the window

    python main.py part.stl --report                 print every feature
    python main.py part.stl --csv features.csv       feature table for a spreadsheet
    python main.py part.stl --drawing part.html      two-sheet drawing

## The drawing

The drawing is one HTML file. It opens in any browser; print it to get a PDF.
Everything is in the units chosen in the viewer (mm or inch).

- **Sheet 1:** front, top and right views plus an isometric view, the overall
  sizes, and the features that belong to those views.
- **Far-side sheet (only when needed):** bottom, back or left view for features
  that open on those sides.
- **Last sheet:** sections D-D and E-E with the most common wall and floor
  thickness.

Rules that keep the views readable:

- each feature gets one size callout and one centre position, nothing repeated
- equal features in a view share one callout with a count, such as `2X`
- a countersink or counterbore is part of its hole's callout
- a through hole is dimensioned once, in the main view; a blind feature is
  dimensioned in the view that looks at the side it opens on
- a view with more than six features shows tags only (A1, B1, ...) and the
  feature table carries the sizes and positions

Limits: a model file holds no tolerances, so the drawing shows nominal sizes and
no feature control frames. Features on curved or sloping faces are not
dimensioned. Sections show the cut surface only. It is a reference drawing, not
a replacement for a released drawing.

## Files

| File | Contents |
|---|---|
| `main.py` | Start-up and command-line options |
| `readers.py` | Reading STL, 3MF, AMF and STEP |
| `model.py` | Feature recognition, measuring, units, report and CSV export |
| `viewer.py` | The window, buttons and clicking |
| `drawing.py` | The two-sheet drawing: views, feature dimensions, sections |

## Limits

- Features are recognised from triangles. Very coarse arcs (fewer than about five
  facets) are treated as flats, and blended or organic faces show as freeform.
- Scanned meshes are too rough for feature recognition; they can still be viewed.

## Help

    python main.py --help          all command-line options
    python main.py --version

In the window, the **Keys** button lists the keyboard shortcuts.

## Author and contact

Abdul Khader Khan, akhader6076@yahoo.in

Questions and bug reports are welcome by email or as a GitHub issue. When reporting
a problem, include the full error text and the file type you opened.

## License

Copyright (c) 2026 Abdul Khader Khan.

DimViewer is free for noncommercial use: individuals, students, teaching and
research. It is released under the PolyForm Noncommercial License 1.0.0; see the
`LICENSE` and `NOTICE` files, or https://polyformproject.org/licenses/noncommercial/1.0.0

Required Notice: Copyright (c) 2026 Abdul Khader Khan

Use by or for a company or any other commercial purpose needs a separate
commercial license. Contact the author.

## Third-party software

DimViewer does not include any third-party code. It uses these packages, which
the user installs separately:

| Package | Used for | License |
|---|---|---|
| NumPy | numerics | BSD |
| VTK | 3D window | BSD |
| psutil | memory readout | BSD |
| cadquery-ocp (optional) | reading STEP | Apache 2.0 wrapper around Open CASCADE (LGPL 2.1 with exception) |
