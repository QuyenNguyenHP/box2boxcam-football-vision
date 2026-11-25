"""Parametric prototype CAD, millimetres. See requirements-cad.txt.

Nominal hardware interfaces MUST be measured before production printing.
STLs use print coordinates (bottom at Z=0); assembly GLB uses assembled coordinates.
"""

import json
import math
from pathlib import Path

import numpy as np
import trimesh as tm

OUT = Path(__file__).parent / "exports"
OUT.mkdir(exist_ok=True)
P = {
    "box_x": 150,
    "box_y": 140,
    "box_h": 72,
    "wall": 3,
    "floor": 9,
    "lid_t": 6,
    "bearing_height": 10,
    "bearing_pitch": 58,
    "bearing_hole": 4.5,
    "rotor_d": 110,
    "rotor_t": 6,
    "servo_body_x": 42,
    "servo_body_y": 22,
    "servo_body_center_x": 10,
    "servo_ear_pitch_x": 49.5,
    "servo_ear_pitch_y": 10,
    "tray_top": 64,
    "horn_pitch": 24,
    "horn_top": 84,
    "clamp_offset_y": 22,
    "stop_angle": 80,
    "stop_radius": 62,
}


def box(size, center):
    m = tm.creation.box(extents=size)
    m.apply_translation(center)
    return m


def cyl(d, h, center):
    m = tm.creation.cylinder(radius=d / 2, height=h, sections=64)
    m.apply_translation(center)
    return m


def union(*parts):
    return tm.boolean.union(parts, engine="manifold")


def cut(body, *tools):
    return tm.boolean.difference([body, *tools], engine="manifold")


def slot(x1, x2, y, d, h, z):
    return tm.boolean.union(
        [
            cyl(d, h, (x1, y, z)),
            cyl(d, h, (x2, y, z)),
            box((max(x2 - x1, 0.01), d, h), ((x1 + x2) / 2, y, z)),
        ],
        engine="manifold",
    )


corners = [(x, y) for x in [-66, 66] for y in [-61, 61]]
supports = [(x, y) for x in [-45, 45] for y in [-23, 23]]
bearings = [
    (x, y)
    for x in [-P["bearing_pitch"] / 2, P["bearing_pitch"] / 2]
    for y in [-P["bearing_pitch"] / 2, P["bearing_pitch"] / 2]
]

# Open top box, integral lid posts; bottom hex pocket takes a metal 1/4-20 nut.
base = cut(box((150, 140, 72), (0, 0, 36)), box((144, 134, 70), (0, 0, 44)))
base = union(base, *[cyl(15, 63, (x, y, 40.5)) for x, y in corners])
holes = [cyl(3.4, 30, (x, y, 61)) for x, y in corners]
# Hex pockets under lid, sized for M3 nuts, accessible from above before lid assembly.
for x, y in corners:
    m = tm.creation.cylinder(radius=6.6 / math.sqrt(3), height=3, sections=6)
    m.apply_translation((x, y, 70.5))
    holes.append(m)
nut = tm.creation.cylinder(radius=11.8 / math.sqrt(3), height=6.3, sections=6)
nut.apply_translation((0, 0, 5.95))
holes += [nut, cyl(6.9, 20, (0, 0, 5))]
# Rear cable opening: grommet/strain relief required. Side toggle hole.
holes += [box((14, 12, 10), (48, 69, 19))]
sw = cyl(6.5, 12, (0, 0, 0))
sw.apply_transform(tm.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
sw.apply_translation((-74, -38, 24))
holes.append(sw)
# Cable-tie floor slots for two electronics boards; adhesive insulating pads also required.
for x in [-58, -32, 36, 60]:
    for y in [-18, 18]:
        holes.append(box((3, 10, 20), (x, y, 5)))
base = cut(base, *holes)

# Removable lid: servo axis opening, bearing mounting, tray rods, case fasteners.
lid = box((150, 140, 6), (0, 0, 75))
holes = [cyl(46, 12, (0, 0, 75))]
holes += [cyl(3.4, 12, (x, y, 75)) for x, y in corners + supports]
holes += [cyl(P["bearing_hole"], 12, (x, y, 75)) for x, y in bearings]
lid = cut(lid, *holes)

# Fixed stop posts OUTSIDE rotor sweep; rotor has a radial stop tab.
# Top of these posts is 93, intersecting tab at z=88..94 only near +/-80 deg.
stop_points = []
for a in [-P["stop_angle"], P["stop_angle"]]:
    x = P["stop_radius"] * math.sin(math.radians(a))
    y = -P["stop_radius"] * math.cos(math.radians(a))
    stop_points.append((x, y))
    post = cut(cyl(12, 15, (x, y, 85.5)), cyl(4.5, 24, (x, y, 85.5)))
    lid = union(lid, post)

# Suspended servo tray. Ear pattern is nominal: check your actual servo.
tray = box((112, 64, 5), (0, 0, P["tray_top"] - 2.5))
holes = [
    box((P["servo_body_x"], P["servo_body_y"], 12), (P["servo_body_center_x"], 0, 62))
]
holes += [cyl(3.4, 12, (x, y, 62)) for x, y in supports]
for x in [10 - P["servo_ear_pitch_x"] / 2, 10 + P["servo_ear_pitch_x"] / 2]:
    for y in [-P["servo_ear_pitch_y"] / 2, P["servo_ear_pitch_y"] / 2]:
        holes.append(slot(x - 1, x + 1, y, 3.4, 12, 62))
tray = cut(tray, *holes)

# Rotor: bearing supported, offset tripod slot avoids servo horn centre screw.
rotor_z = 78 + P["bearing_height"]
rotor = cyl(P["rotor_d"], 6, (0, 0, rotor_z + 3))
holes = [
    cyl(12, 20, (0, 0, rotor_z + 1)),
    slot(-16, 16, P["clamp_offset_y"], 6.9, 16, rotor_z + 3),
]
holes += [cyl(P["bearing_hole"], 16, (x, y, rotor_z + 3)) for x, y in bearings]
holes += [
    cyl(3.4, 20, (x, 0, rotor_z + 1))
    for x in [-P["horn_pitch"] / 2, P["horn_pitch"] / 2]
]
# Underside drive spacer ring; horn bears on this ring, not on rotating bearing races.
boss = cut(
    cyl(36, rotor_z - P["horn_top"] + 0.2, (0, 0, (rotor_z + P["horn_top"]) / 2)),
    cyl(12, 20, (0, 0, rotor_z)),
)
rotor = cut(union(rotor, boss), *holes)
# Broad tab touches buffered stop posts before nominal +/-80 deg. Not a running stop.
rotor = union(rotor, box((10, 14, 6), (0, -59, rotor_z + 3)))

# Fit coupon: print first. Nominal screw and servo/bearing interfaces.
coupon = box((76, 46, 3), (0, 0, 1.5))
coupon = cut(
    coupon,
    box((42, 22, 10), (5, 0, 1.5)),
    *[
        cyl(d, 10, (x, 17, 1.5))
        for x, d in [(-27, 3.4), (-13, 4.5), (2, 6.9), (19, 12)]
    ],
)

parts = {
    "base": base,
    "lid": lid,
    "servo_tray": tray,
    "rotor": rotor,
    "fit_coupon": coupon,
}
report = {}
for name, m in parts.items():
    m.remove_unreferenced_vertices()
    assert m.is_watertight and m.volume > 0, name
    assert len(m.split()) == 1, f"{name}: disconnected bodies"
    printed = m.copy()
    printed.apply_translation((0, 0, -m.bounds[0, 2]))
    printed.export(OUT / f"{name}.stl")
    report[name] = {
        "watertight": bool(m.is_watertight),
        "volume_cm3": round(m.volume / 1000, 2),
        "size_mm": np.round(m.extents, 2).tolist(),
    }

# Check the moving printed part against fixed printed parts at 5-degree samples.
# This checks CAD solids only, not purchased fasteners, tolerances or load deformation.
pan_checks = []
for angle in range(-60, 61, 5):
    moving = rotor.copy()
    moving.apply_transform(
        tm.transformations.rotation_matrix(math.radians(angle), [0, 0, 1])
    )
    overlap = 0.0
    for fixed in [base, lid, tray]:
        intersection = tm.boolean.intersection([moving, fixed], engine="manifold")
        if not intersection.is_empty:
            overlap += max(0.0, intersection.volume)
    assert overlap < 0.01, f"printed-part collision at {angle} degrees"
    pan_checks.append({"angle_deg": angle, "overlap_mm3": round(overlap, 6)})

# Assembly visualization. Grey parts are reference envelopes, NOT printable parts.
scene = tm.Scene()
colors = {
    "base": [42, 59, 78, 255],
    "lid": [67, 114, 142, 255],
    "servo_tray": [232, 164, 71, 255],
    "rotor": [47, 156, 146, 255],
}
for name in ["base", "lid", "servo_tray", "rotor"]:
    m = parts[name].copy()
    m.visual.vertex_colors = colors[name]
    scene.add_geometry(m, node_name=name)
ring = cut(box((76.2, 76.2, 10), (0, 0, 83)), cyl(46, 14, (0, 0, 83)))
ring.visual.vertex_colors = [150, 160, 170, 255]
scene.add_geometry(ring, node_name="bearing_reference_nominal")
horn = cut(cyl(34, 2, (0, 0, 83)), cyl(6, 5, (0, 0, 83)))
horn.visual.vertex_colors = [200, 180, 110, 255]
scene.add_geometry(horn, node_name="servo_horn_reference")
servo = box((40.8, 20.1, 39), (10, 0, 44.5))
servo.visual.vertex_colors = [190, 90, 60, 255]
scene.add_geometry(servo, node_name="servo_reference_nominal")
# Bought clamp is symbolic; phone is landscape with lens side facing -Y.
stem = box((32, 24, 28), (0, 22, 108))
stem.visual.vertex_colors = [90, 95, 100, 255]
scene.add_geometry(stem, node_name="clamp_reference")
phone = box((159.9, 8.25, 76.7), (0, 22, 158.35))
phone.visual.vertex_colors = [90, 105, 125, 255]
scene.add_geometry(phone, node_name="iphone15promax_reference")
export_scene = scene.copy()
export_scene.apply_scale(0.001)
export_scene.export(OUT / "assembly.glb")
(OUT / "geometry_check.json").write_text(
    json.dumps({"parameters": P, "parts": report, "pan_checks": pan_checks}, indent=2)
    + "\n"
)

# Exploded, inspectable preview without an OpenGL dependency.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure(figsize=(13, 9), facecolor="#f5f7fa")
ax = fig.add_subplot(111, projection="3d")


def draw(m, color, offset=0, alpha=1):
    triangles = m.triangles.copy()
    triangles[:, :, 2] += offset
    ax.add_collection3d(
        Poly3DCollection(triangles, facecolors=color, edgecolors="none", alpha=alpha)
    )


draw(base, "#44566a")
draw(tray, "#e4a344", 35)
draw(lid, "#497e9c", 60)
draw(ring, "#acb4bf", 85)
draw(rotor, "#299e94", 110)
draw(stem, "#69727e", 125)
draw(phone, "#748598", 125)
fig.subplots_adjust(right=0.72)
for i, (text, col) in enumerate(
    [
        ("PHONE + METAL CLAMP", "#748598"),
        ("ROTOR + HORN SPACER", "#299e94"),
        ("TURNTABLE BEARING", "#acb4bf"),
        ("LID + BACKUP STOPS", "#497e9c"),
        ("ADJUSTABLE SERVO TRAY", "#e4a344"),
        ("BASE / ELECTRONICS", "#44566a"),
    ]
):
    fig.text(0.76, 0.78 - i * 0.085, text, fontsize=10, color=col, weight="bold")
fig.text(
    0.76,
    0.19,
    "Purchased parts are\nreference envelopes.\n\nSTLs: mm\nGLB: metres\n\nPan target: +/-60 deg",
    fontsize=10,
    color="#243447",
    linespacing=1.6,
)
ax.set(
    xlim=(-100, 100),
    ylim=(-100, 100),
    zlim=(0, 335),
    xlabel="X mm",
    ylabel="Y mm",
    zlabel="Z mm",
)
ax.view_init(elev=24, azim=-62)
ax.set_box_aspect((1, 0.8, 1.25))
ax.set_title(
    "PAN V1 - iPhone 15 Pro Max\nExploded concept | nominal bearing/horn fit must be measured",
    fontsize=15,
    pad=20,
)
fig.savefig(OUT / "exploded_preview.png", dpi=170, bbox_inches="tight")
plt.close(fig)
# Dimensioned plan and stack: geometric reference, not a vendor bearing drawing.
from matplotlib.patches import Circle, Rectangle

fig, (a, b) = plt.subplots(1, 2, figsize=(13, 7), facecolor="white")
a.add_patch(Rectangle((-75, -70), 150, 140, fill=False, lw=2, color="#44566a"))
a.add_patch(Rectangle((-38.1, -38.1), 76.2, 76.2, fill=False, lw=1.5, color="#9ba4ae"))
a.add_patch(Circle((0, 0), 55, fill=False, lw=2, color="#299e94"))
a.add_patch(Circle((0, 0), 23, fill=False, ls="--", color="#497e9c"))
for x, y in bearings:
    a.add_patch(Circle((x, y), 2.25, color="#44566a"))
a.plot([-16, 16], [22, 22], lw=5, color="#e4a344")
a.plot(0, 0, "+", color="black", ms=12)
a.annotate("", (-75, -83), (75, -83), arrowprops={"arrowstyle": "<->"})
a.text(0, -90, "150 mm", ha="center")
a.annotate("", (-88, -70), (-88, 70), arrowprops={"arrowstyle": "<->"})
a.text(-94, 0, "140 mm", rotation=90, va="center")
a.text(0, 64, "ROTOR DIA 110", ha="center", color="#299e94")
a.text(0, -58, "Bearing pitch: 58 x 58 NOMINAL", ha="center", fontsize=9)
a.set(
    xlim=(-105, 100),
    ylim=(-100, 95),
    aspect="equal",
    title="TOP VIEW / nominal interfaces",
)
a.axis("off")
for x, y, w, h, col in [
    (-75, 0, 150, 72, "#44566a"),
    (-75, 72, 150, 6, "#497e9c"),
    (-38.1, 78, 76.2, 10, "#acb4bf"),
    (-55, 88, 110, 6, "#299e94"),
]:
    b.add_patch(Rectangle((x, y), w, h, fill=False, lw=2, color=col))
for z, label in [
    (0, "0 / tripod nut"),
    (64, "64 / adjustable servo tray"),
    (72, "72 / lid bottom"),
    (78, "78 / bearing bottom"),
    (88, "88 / rotor bottom"),
    (94, "94 / clamp mounting face"),
]:
    b.plot([75, 92], [z, z], color="#a5a5a5", lw=0.7)
    # Spread labels, join to true elevation.
    tz = {0: 0, 64: 44, 72: 63, 78: 82, 88: 101, 94: 120}[z]
    b.annotate(
        label,
        xy=(83, z),
        xytext=(98, tz),
        fontsize=9,
        va="center",
        arrowprops={"arrowstyle": "-", "color": "#a5a5a5"},
    )
b.add_patch(Rectangle((-56, 59), 112, 5, fill=False, lw=2, color="#e4a344"))
b.set(
    xlim=(-90, 245),
    ylim=(-10, 140),
    aspect="equal",
    title="STACK / mm; purchased clamp height excluded",
)
b.axis("off")
fig.suptitle(
    "PAN V1 - PROTOTYPE DIMENSIONS / MEASURE HARDWARE BEFORE PRINTING", fontsize=13
)
fig.tight_layout()
fig.savefig(OUT / "dimensions.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print(json.dumps(report, indent=2))
