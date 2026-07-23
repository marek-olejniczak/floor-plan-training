#!/usr/bin/env python3
"""
wall_postprocessor.py — Post-processing YOLO bboxów na wektory ścian.

Pipeline:
  filter_noise → classify_orientation → to_centerline
  → find_intersections → snap_walls → merge_collinear → unify_thickness
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


# ──────────────────────────────────────────────
# Struktury danych
# ──────────────────────────────────────────────

@dataclass
class Wall:
    orientation: str       # 'H' | 'V'
    x1: float              # lewy kraniec (H) / lewa krawędź (V)
    x2: float              # prawy kraniec (H) / prawa krawędź (V)
    y1: float              # górny kraniec (V) / górna krawędź (H)
    y2: float              # dolny kraniec (V) / dolna krawędź (H)
    center: float          # y_center (H) lub x_center (V)
    thickness: float
    confidence: float
    merged: bool = False   # znacznik dla collinear merge


# ──────────────────────────────────────────────
# Krok 1: Filtracja szumu
# ──────────────────────────────────────────────

def filter_noise(detections: List[dict], min_area: float = 50.0) -> List[dict]:
    filtered = []
    for d in detections:
        if d.get("class", "") != "wall":
            continue
        verts = d.get("vertices", [])
        if len(verts) < 4:
            continue
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)

        w = x2 - x1
        h = y2 - y1
        area = w * h
        if area < min_area:
            continue

        aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 999
        if 0.8 <= aspect <= 1.2:
            continue

        filtered.append({"bbox": [x1, y1, x2, y2], "confidence": d["confidence"]})
    return filtered


# ──────────────────────────────────────────────
# Krok 2: Klasyfikacja orientacji
# ──────────────────────────────────────────────

def classify_orientation(x1: float, y1: float, x2: float, y2: float) -> str:
    w = x2 - x1
    h = y2 - y1
    return "H" if w >= h else "V"


# ──────────────────────────────────────────────
# Krok 3: Konwersja na centerline
# ──────────────────────────────────────────────

def to_centerline(bbox: List[float], confidence: float) -> Wall:
    x1, y1, x2, y2 = bbox
    orient = classify_orientation(x1, y1, x2, y2)

    if orient == "H":
        center = (y1 + y2) / 2
        thickness = y2 - y1
        return Wall(
            orientation="H",
            x1=x1, x2=x2,
            y1=center - thickness / 2, y2=center + thickness / 2,
            center=center,
            thickness=thickness,
            confidence=confidence,
        )
    else:
        center = (x1 + x2) / 2
        thickness = x2 - x1
        return Wall(
            orientation="V",
            x1=center - thickness / 2, x2=center + thickness / 2,
            y1=y1, y2=y2,
            center=center,
            thickness=thickness,
            confidence=confidence,
        )


# ──────────────────────────────────────────────
# Krok 4: Macierz przecięć
# ──────────────────────────────────────────────

@dataclass
class Intersection:
    h_wall: Wall
    v_wall: Wall
    ix: float   # x przecięcia (zawsze V.x_center)
    iy: float   # y przecięcia (zawsze H.y_center)
    overlap_ok: bool = True


def bbox_overlap_with_margin(h: Wall, v: Wall, margin_factor: float = 1.5) -> bool:
    hx1, hx2 = min(h.x1, h.x2), max(h.x1, h.x2)
    hy1, hy2 = min(h.y1, h.y2), max(h.y1, h.y2)
    vx1, vx2 = min(v.x1, v.x2), max(v.x1, v.x2)
    vy1, vy2 = min(v.y1, v.y2), max(v.y1, v.y2)

    margin = max(h.thickness, v.thickness) * margin_factor
    return not (hx2 + margin < vx1 - margin or
                hx1 - margin > vx2 + margin or
                hy2 + margin < vy1 - margin or
                hy1 - margin > vy2 + margin)


def find_intersections(h_walls: List[Wall], v_walls: List[Wall]) -> List[Intersection]:
    intersections = []
    for h in h_walls:
        for v in v_walls:
            if bbox_overlap_with_margin(h, v):
                intersections.append(Intersection(
                    h_wall=h, v_wall=v,
                    ix=v.center, iy=h.center,
                ))
    return intersections


# ──────────────────────────────────────────────
# Krok 5: Dociąganie (Snap & Trim)
# ──────────────────────────────────────────────

def get_tolerance(h: Wall, v: Wall) -> float:
    return max(h.thickness, v.thickness) * 1.5


def is_end_near(value: float, end: float, tol: float) -> bool:
    return abs(value - end) <= tol


def is_inside(value: float, a: float, b: float, margin: float = 2.0) -> bool:
    lo, hi = min(a, b), max(a, b)
    return lo + margin <= value <= hi - margin


def snap_walls(intersections: List[Intersection]) -> None:
    for inter in intersections:
        h = inter.h_wall
        v = inter.v_wall
        tol = get_tolerance(h, v)
        ix, iy = inter.ix, inter.iy

        h_left_near = is_end_near(ix, h.x1, tol)
        h_right_near = is_end_near(ix, h.x2, tol)
        v_top_near = is_end_near(iy, v.y1, tol)
        v_bottom_near = is_end_near(iy, v.y2, tol)

        h_end_near = h_left_near or h_right_near
        v_end_near = v_top_near or v_bottom_near

        if not h_end_near and not v_end_near:
            continue

        if h_end_near and v_end_near:
            # Typ L: narożnik
            if h_left_near:
                h.x1 = ix
            if h_right_near:
                h.x2 = ix
            if v_top_near:
                v.y1 = iy
            if v_bottom_near:
                v.y2 = iy

        elif v_end_near and not h_end_near and is_inside(ix, h.x1, h.x2):
            # Typ T: koniec V w środku H
            if v_top_near:
                v.y1 = iy
            if v_bottom_near:
                v.y2 = iy

        elif h_end_near and not v_end_near and is_inside(iy, v.y1, v.y2):
            # Typ T: koniec H w środku V
            if h_left_near:
                h.x1 = ix
            if h_right_near:
                h.x2 = ix


# ──────────────────────────────────────────────
# Krok 6: Scalanie współliniowych segmentów
# ──────────────────────────────────────────────

def merge_collinear(walls: List[Wall], center_tolerance: float = 2.0) -> List[Wall]:
    result = []
    for w in walls:
        if w.merged:
            continue
        merged = w
        for other in walls:
            if other is w or other.merged:
                continue
            if other.orientation != merged.orientation:
                continue
            if abs(other.center - merged.center) > center_tolerance:
                continue

            if merged.orientation == "H":
                m1, m2 = sorted([merged.x1, merged.x2])
                o1, o2 = sorted([other.x1, other.x2])
            else:
                m1, m2 = sorted([merged.y1, merged.y2])
                o1, o2 = sorted([other.y1, other.y2])

            gap = max(m1, o1) - min(m2, o2)
            if gap <= center_tolerance:
                merged.x1 = min(merged.x1, other.x1)
                merged.x2 = max(merged.x2, other.x2)
                merged.y1 = min(merged.y1, other.y1)
                merged.y2 = max(merged.y2, other.y2)
                merged.confidence = max(merged.confidence, other.confidence)
                other.merged = True

        result.append(merged)
    return result


# ──────────────────────────────────────────────
# Krok 7: Unifikacja grubości w narożnikach
# ──────────────────────────────────────────────

def unify_thickness(h_walls: List[Wall], v_walls: List[Wall], max_diff: float = 3.0) -> None:
    for h in h_walls:
        for v in v_walls:
            diff = abs(h.thickness - v.thickness)
            if diff > max_diff or diff < 0.5:
                continue
            avg = (h.thickness + v.thickness) / 2
            h.thickness = avg
            v.thickness = avg


# ──────────────────────────────────────────────
# Główny pipeline
# ──────────────────────────────────────────────

def process_walls(detections: List[dict]) -> Dict[str, list]:
    filtered = filter_noise(detections)
    h_walls: List[Wall] = []
    v_walls: List[Wall] = []

    for d in filtered:
        w = to_centerline(d["bbox"], d["confidence"])
        if w.orientation == "H":
            h_walls.append(w)
        else:
            v_walls.append(w)

    intersections = find_intersections(h_walls, v_walls)
    snap_walls(intersections)

    h_walls = merge_collinear(h_walls)
    v_walls = merge_collinear(v_walls)

    unify_thickness(h_walls, v_walls)

    return {
        "horizontal": [
            {
                "x1": round(h.x1, 1),
                "x2": round(h.x2, 1),
                "y_center": round(h.center, 1),
                "thickness": round(h.thickness, 1),
                "confidence": round(h.confidence, 4),
            }
            for h in h_walls
        ],
        "vertical": [
            {
                "y1": round(v.y1, 1),
                "y2": round(v.y2, 1),
                "x_center": round(v.center, 1),
                "thickness": round(v.thickness, 1),
                "confidence": round(v.confidence, 4),
            }
            for v in v_walls
        ],
    }
