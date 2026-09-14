"""Pure NumPy geometry helpers for the experimental OpenGL renderer."""

from __future__ import annotations

import math
import numpy as np


def _normalise(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    length = float(np.linalg.norm(value))
    if length <= 1e-12:
        return np.zeros(3, dtype=float)
    return value / length


def sphere_mesh(
    center: np.ndarray,
    radius: float,
    *,
    latitude_segments: int = 16,
    longitude_segments: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Return triangle-list positions and normals for a UV sphere."""

    centre = np.asarray(center, dtype=float).reshape(3)
    radius = max(0.0, float(radius))
    latitude_segments = max(4, int(latitude_segments))
    longitude_segments = max(6, int(longitude_segments))
    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []

    for lat in range(latitude_segments):
        phi0 = -0.5 * math.pi + math.pi * lat / latitude_segments
        phi1 = -0.5 * math.pi + math.pi * (lat + 1) / latitude_segments
        cp0, sp0 = math.cos(phi0), math.sin(phi0)
        cp1, sp1 = math.cos(phi1), math.sin(phi1)
        for lon in range(longitude_segments):
            theta0 = 2.0 * math.pi * lon / longitude_segments
            theta1 = 2.0 * math.pi * (lon + 1) / longitude_segments
            c0, s0 = math.cos(theta0), math.sin(theta0)
            c1, s1 = math.cos(theta1), math.sin(theta1)
            n00 = np.array([cp0 * c0, cp0 * s0, sp0], dtype=float)
            n01 = np.array([cp0 * c1, cp0 * s1, sp0], dtype=float)
            n10 = np.array([cp1 * c0, cp1 * s0, sp1], dtype=float)
            n11 = np.array([cp1 * c1, cp1 * s1, sp1], dtype=float)
            if lat > 0:
                for normal in (n00, n10, n01):
                    positions.append(centre + radius * normal)
                    normals.append(normal)
            if lat < latitude_segments - 1:
                for normal in (n01, n10, n11):
                    positions.append(centre + radius * normal)
                    normals.append(normal)
    return np.asarray(positions, dtype=np.float32), np.asarray(normals, dtype=np.float32)


def cylinder_mesh(
    first: np.ndarray,
    second: np.ndarray,
    radius: float,
    *,
    segments: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an open cylinder joining two 3-D points."""

    start = np.asarray(first, dtype=float).reshape(3)
    finish = np.asarray(second, dtype=float).reshape(3)
    axis = finish - start
    length = float(np.linalg.norm(axis))
    if length <= 1e-12 or radius <= 0.0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty.copy()
    axis /= length
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(axis, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    u = _normalise(np.cross(axis, reference))
    v = _normalise(np.cross(axis, u))
    radius = float(radius)
    segments = max(6, int(segments))
    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []

    for index in range(segments):
        angle0 = 2.0 * math.pi * index / segments
        angle1 = 2.0 * math.pi * (index + 1) / segments
        radial0 = math.cos(angle0) * u + math.sin(angle0) * v
        radial1 = math.cos(angle1) * u + math.sin(angle1) * v
        a0 = start + radius * radial0
        a1 = start + radius * radial1
        b0 = finish + radius * radial0
        b1 = finish + radius * radial1
        for point, normal in (
            (a0, radial0), (b0, radial0), (a1, radial1),
            (a1, radial1), (b0, radial0), (b1, radial1),
        ):
            positions.append(point)
            normals.append(normal)
    return np.asarray(positions, dtype=np.float32), np.asarray(normals, dtype=np.float32)


def polygon_triangles(
    vertices: np.ndarray,
    polyhedron_center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate one planar polygon as a fan and return outward normals."""

    polygon = np.asarray(vertices, dtype=float).reshape(-1, 3)
    if polygon.shape[0] < 3:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty.copy()
    normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
    normal = _normalise(normal)
    if polyhedron_center is not None and np.linalg.norm(normal) > 0.0:
        centre = np.asarray(polyhedron_center, dtype=float).reshape(3)
        face_centre = polygon.mean(axis=0)
        if float(np.dot(normal, face_centre - centre)) < 0.0:
            polygon = polygon[::-1].copy()
            normal = -normal
    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for index in range(1, polygon.shape[0] - 1):
        for point in (polygon[0], polygon[index], polygon[index + 1]):
            positions.append(point)
            normals.append(normal)
    return np.asarray(positions, dtype=np.float32), np.asarray(normals, dtype=np.float32)


def line_vertices(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray([first, second], dtype=np.float32).reshape(2, 3)
    normals = np.zeros_like(positions)
    return positions, normals


__all__ = ["cylinder_mesh", "line_vertices", "polygon_triangles", "sphere_mesh"]
