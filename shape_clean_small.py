"""
GEOMETRIC SHAPES DATASET GENERATOR v2.1
Refactored with a clean CONFIG section.
"""

# ------------------------------------------------------------------
# CONFIG – change only the values below
# ------------------------------------------------------------------
CONFIG = {
    # Canvas
    "canvas_size": (64, 64),          # (width, height)

    # Dataset size
    "samples_per_shape": 2000,

    # Output
    "output_root": "shape_data",

    # Random-parameter ranges
    "margin": 10,                      # keep shapes away from edges
    "size_range": (10, 50),            # (min, max) radius / half-size
    "line_width_range": (2, 6),        # stroke width
    "color_range": (0, 200),           # RGB channels (avoid very light colors)
    "rotation_range": (0, 360),        # degrees

    # Points parameter ranges
    "polygon_points": (3, 9),          # sides
    "star_points": (3, 8),             # spikes
    "wavy_points": (1, 4),             # waves
    "fallback_points": (3, 8),         # for shapes that ignore 'points'
}

# ------------------------------------------------------------------
# Everything below is implementation – you normally don’t edit it
# ------------------------------------------------------------------
import os
import json
import math
import random
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------------
def create_canvas(width: int, height: int, background_color=(0, 0, 0)):
    img = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(img)
    return img, draw


def rotate_point(x, y, cx, cy, angle):
    angle_rad = math.radians(angle)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    tx, ty = x - cx, y - cy
    rx = tx * cos_a - ty * sin_a
    ry = tx * sin_a + ty * cos_a
    return rx + cx, ry + cy


# ------------------------------------------------------------------
# Shape drawers
# ------------------------------------------------------------------
def draw_circle(canvas_size, params):
    img, draw = create_canvas(*canvas_size)
    x, y = params["position"]
    r = params["size"]
    color = params["color"]
    lw = params["line_width"]
    bbox = [x - r, y - r, x + r, y + r]
    draw.ellipse(bbox, outline=color, width=lw)
    return img


def draw_polygon(canvas_size, params):
    img, draw = create_canvas(*canvas_size)
    x, y = params["position"]
    s = params["size"]
    rot = params["rotation"]
    color = params["color"]
    lw = params["line_width"]
    n = max(3, int(params["points"]))

    pts = []
    for i in range(n):
        angle = 2 * math.pi * i / n + math.radians(rot)
        px = x + s * math.cos(angle)
        py = y + s * math.sin(angle)
        pts.append((px, py))

    for i in range(n):
        draw.line([pts[i], pts[(i + 1) % n]], fill=color, width=lw)
    return img


def draw_star(canvas_size, params):
    img, draw = create_canvas(*canvas_size)
    x, y = params["position"]
    outer = params["size"]
    inner = outer * 0.4
    rot = params["rotation"]
    color = params["color"]
    lw = params["line_width"]
    spikes = max(3, int(params["points"]))

    pts = []
    for i in range(spikes * 2):
        angle = (i * 180 / spikes + rot) * math.pi / 180
        r = outer if i % 2 == 0 else inner
        pts.append((x + r * math.cos(angle), y + r * math.sin(angle)))

    for i in range(len(pts)):
        draw.line([pts[i], pts[(i + 1) % len(pts)]], fill=color, width=lw)
    return img


def draw_wavy_bar(canvas_size, params):
    img, draw = create_canvas(*canvas_size)
    x, y = params["position"]
    size = params["size"]
    rot = params["rotation"]
    color = params["color"]
    lw = params["line_width"]
    waves = max(1, int(params["points"]))

    wave_length = size * 2
    amplitude = size * 0.3
    segments = 100
    path = []

    for i in range(segments + 1):
        px = x - wave_length / 2 + i * (wave_length / segments)
        py = y + amplitude * math.sin(2 * math.pi * waves * i / segments)
        path.append(rotate_point(px, py, x, y, rot))

    for i in range(len(path) - 1):
        draw.line([path[i], path[i + 1]], fill=color, width=lw)
    return img


def draw_cross(canvas_size, params):
    img, draw = create_canvas(*canvas_size)
    x, y = params["position"]
    arm = params["size"]
    rot = params["rotation"]
    color = params["color"]
    lw = params["line_width"]

    aw = arm * 0.3 / 2
    pts = [
        (x - aw, y - arm),
        (x + aw, y - arm),
        (x + aw, y - aw),
        (x + arm, y - aw),
        (x + arm, y + aw),
        (x + aw, y + aw),
        (x + aw, y + arm),
        (x - aw, y + arm),
        (x - aw, y + aw),
        (x - arm, y + aw),
        (x - arm, y - aw),
        (x - aw, y - aw),
    ]
    pts = [rotate_point(px, py, x, y, rot) for px, py in pts]

    for i in range(len(pts)):
        draw.line([pts[i], pts[(i + 1) % len(pts)]], fill=color, width=lw)
    return img


SHAPE_DRAWERS = [draw_circle, draw_polygon, draw_star, draw_wavy_bar, draw_cross]
SHAPE_NAMES = ["circle", "polygon", "star", "wavy_bar", "cross"]

# ------------------------------------------------------------------
# Dataset generation
# ------------------------------------------------------------------
def random_params(shape_id: int, cfg: dict) -> dict:
    w, h = cfg["canvas_size"]
    margin = cfg["margin"]

    x = random.randint(margin, w - margin)
    y = random.randint(margin, h - margin)
    rot = random.uniform(*cfg["rotation_range"])
    size = random.randint(*cfg["size_range"])
    r = random.randint(*cfg["color_range"])
    g = random.randint(*cfg["color_range"])
    b = random.randint(*cfg["color_range"])
    lw = random.randint(*cfg["line_width_range"])

    if shape_id == 1:  # polygon
        points = random.randint(*cfg["polygon_points"])
    elif shape_id == 2:  # star
        points = random.randint(*cfg["star_points"])
    elif shape_id == 3:  # wavy_bar
        points = random.randint(*cfg["wavy_points"])
    else:
        points = random.randint(*cfg["fallback_points"])

    return {
        "position": (x, y),
        "rotation": rot,
        "size": size,
        "color": (r, g, b),
        "line_width": lw,
        "points": points,
    }


def generate_dataset(cfg: dict):
    os.makedirs(cfg["output_root"], exist_ok=True)
    total = 0

    for shape_id, name in enumerate(SHAPE_NAMES):
        print(f"\n{'='*50}")
        print(f"Generating {name.upper()} samples...")
        print(f"{'='*50}")

        shape_dir = os.path.join(cfg["output_root"], name)
        img_dir = os.path.join(shape_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        params_list = []
        for idx in tqdm(range(cfg["samples_per_shape"]), desc=name):
            params = random_params(shape_id, cfg)
            img = SHAPE_DRAWERS[shape_id](cfg["canvas_size"], params)
            img.save(os.path.join(img_dir, f"{name}_{idx:04d}.png"))

            # Normalized parameter vector
            vec = torch.tensor(
                [
                    params["position"][0] / cfg["canvas_size"][0],
                    params["position"][1] / cfg["canvas_size"][1],
                    params["rotation"] / 360.0,
                    params["size"] / 100.0,
                    params["color"][0] / 255.0,
                    params["color"][1] / 255.0,
                    params["color"][2] / 255.0,
                    params["points"] / 12.0,
                ],
                dtype=torch.float32,
            )
            params_list.append(vec)
            total += 1

        if params_list:
            torch.save(torch.stack(params_list), os.path.join(shape_dir, "parameters.pt"))
            with open(os.path.join(shape_dir, "dataset_info.json"), "w") as f:
                json.dump(
                    {
                        "shape": name,
                        "shape_id": shape_id,
                        "count": len(params_list),
                        "parameter_format": {
                            "dimensions": 8,
                            "order": ["x", "y", "rotation", "size", "r", "g", "b", "points"],
                        },
                    },
                    f,
                    indent=2,
                )
            print(f"✅ {len(params_list)} {name} samples saved")
        else:
            print(f"❌ No {name} samples generated")

    # Overall metadata
    with open(os.path.join(cfg["output_root"], "dataset_info.json"), "w") as f:
        json.dump(
            {
                "dataset_name": "Geometric Shapes Dataset",
                "total_samples": total,
                "shapes": SHAPE_NAMES,
                "samples_per_shape": cfg["samples_per_shape"],
                "canvas_size": list(cfg["canvas_size"]),
                "config": cfg,
            },
            f,
            indent=2,
        )
    print(f"\n{'='*60}")
    print("DATASET GENERATION COMPLETE!")
    print(f"Total samples: {total}")
    print(f"Output: {cfg['output_root']}")
    print(f"{'='*60}")


# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------
def visualize(cfg: dict, samples_per_shape: int = 5):
    fig, axes = plt.subplots(
        len(SHAPE_NAMES),
        samples_per_shape,
        figsize=(samples_per_shape * 2, len(SHAPE_NAMES) * 2),
    )
    if len(SHAPE_NAMES) == 1:
        axes = axes.reshape(1, -1)

    for row, name in enumerate(SHAPE_NAMES):
        img_dir = os.path.join(cfg["output_root"], name, "images")
        if not os.path.exists(img_dir):
            continue
        files = sorted(os.listdir(img_dir))[:samples_per_shape]
        for col, fname in enumerate(files):
            img = Image.open(os.path.join(img_dir, fname))
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(name.title(), rotation=0, labelpad=50)

    plt.suptitle("Generated Shape Samples", fontsize=16)
    plt.tight_layout()
    out_path = os.path.join(cfg["output_root"], "sample_visualization.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Visualization saved to", out_path)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("GEOMETRIC SHAPES DATASET GENERATOR v2.1")
    generate_dataset(CONFIG)
    visualize(CONFIG)