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
  The text also lists what applies to that pair (see "What Measure reports").
- **Caliper** (or key K): same two clicks, but the line shows the outside length
  across both features, as a caliper held square to X, Y or Z would read it.
  The inside length (clear gap) is listed too.
- **Space** measures whatever is under the cursor.
- **Datum A** (or key D): makes the picked feature the reference. Later picks
  are related to it, and a pair of features gets its symmetry about it.
- **Thickness** (or key X): colours the part by wall thickness, red thin to
  blue thick, with the values in the text.
- **Part info** (or key A): counts the features that matter for additive
  manufacturing and lists them in the log bar (see "Part info").
- **Log bar:** the dark strip at the bottom holds the text for whatever is
  picked or analysed, set out in columns to suit the window width. The part
  moves up so the bar never covers it. **Copy log** (key L) puts all of it on the clipboard;
  **Save log** (key S) writes it to `<part name>_info.txt` and opens it.
- **Wireframe** (or key F): edges only, with the faces almost clear, as in a
  CAD wireframe view. **Mesh** (or key W) shows the triangles.
- **Select** buttons limit what a click picks: Auto, Edges, Faces, Circles or
  Points (the nearest corner point of the mesh).
- **View** buttons snap to Front, Back, Top, Bottom, Right, Left and Iso.
- **Drawing** saves a two-sheet drawing and opens it in the browser.
- **Info** shows units, overall size, volume, surface area and memory use.
- **Keys** lists the keyboard shortcuts.

## What Measure reports

Every pair gets a RESULT block with three values: the **angle** between the two,
the **shortest distance** (on the real faces and edges) and the **normal
distance** (square to the first feature), plus the caliper lengths. A curved
face is treated as facing its average direction and a curved edge as running
along its general line, so these three are always given; such values are marked
"approx.". Under "More on this pair":

| Pair | Reported |
|---|---|
| Two flat faces | angle; normal plane-to-plane distance, marked as thickness, gap or step; parallelism or perpendicularity; mid-plane |
| Two holes, shafts or straight edges | angle between the axes; centre-to-centre distance with X, Y, Z offsets; coaxiality and concentricity when they share an axis; closest distance when they are skew |
| Axis and flat face | angle; normal distance; perpendicularity or parallelism of the axis |
| Point and anything | point-to-point, point-to-plane or point-to-axis distance |
| Any pair plus datum A | symmetry of the pair about the datum |

A single face also shows, where it applies: draft angle, chamfer size, fillet
or round radius, the hole pattern it belongs to (bolt circle, row or grid, with
pitch), slot width and length, and the wall thickness behind it.

All of these are measured on the model. They are nominal values, useful for
checking a design and for planning an inspection, not a measurement of a real part.

## Part info

One click gives a design-for-additive-manufacturing overview of the part:

- size, volume, how much of its bounding box it fills, solid bodies, enclosed voids
- surfaces by type, largest flat face
- round holes and bosses with their sizes, through and blind openings, slots, patterns
- fillets, rounds, chamfers, sharp outside and inside edges, corners
- wall sections and ribs, thinnest and typical wall, thin-wall areas, narrowest gap
- overhang area for each of the six build directions, with the least marked
- a short "points to check" list

The limits are settings, not process rules: `--thin` (mm, default 1.0),
`--min-hole` (mm, default 1.0) and `--overhang` (degrees from horizontal,
default 45). Set them to suit your process. Ribs and walls are found by rule
of thumb (two parallel flat faces with material between), so curved walls are
covered by the thickness numbers but not counted as walls.

## Without the window

    python main.py part.stl --report                 print every feature
    python main.py part.stl --info                   part info for additive manufacturing
    python main.py part.stl --info --thin 0.8        same, with your own thin-wall limit
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
| `dfam.py` | Part info for additive manufacturing: feature counts, thin walls, gaps, overhangs |
| `measure.py` | Angles, normal and centre distances, GD&T-style relations, patterns, slots, wall thickness |
| `viewer.py` | The window, buttons and clicking |
| `drawing.py` | The two-sheet drawing: views, feature dimensions, sections |

## Limits

- Features are recognised from triangles. Very coarse arcs (fewer than about five
  facets) are treated as flats, and blended or organic faces show as freeform.
- Scanned meshes are too rough for feature recognition; they can still be viewed.
- Wall thickness is measured straight in from the surface, square to it. Above
  8,000 triangles it is measured at 8,000 places and spread to the rest.
- Form and orientation values (flatness, parallelism, coaxiality, ...) describe
  the model, so on clean CAD they read zero. They do not replace a CMM report.

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
