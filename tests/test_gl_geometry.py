import unittest
import numpy as np

from unit_cell_gui.gl_geometry import cylinder_mesh, polygon_triangles, sphere_mesh


class OpenGLGeometryTests(unittest.TestCase):
    def test_sphere_vertices_are_on_requested_radius(self):
        centre = np.array([1.0, -2.0, 3.0])
        positions, normals = sphere_mesh(centre, 0.75, latitude_segments=8, longitude_segments=12)
        self.assertGreater(len(positions), 0)
        np.testing.assert_allclose(np.linalg.norm(positions - centre, axis=1), 0.75, atol=2e-6)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=2e-6)

    def test_cylinder_vertices_keep_constant_radial_distance(self):
        first = np.array([0.0, 0.0, 0.0])
        second = np.array([0.0, 0.0, 2.0])
        positions, normals = cylinder_mesh(first, second, 0.2, segments=10)
        self.assertGreater(len(positions), 0)
        np.testing.assert_allclose(np.linalg.norm(positions[:, :2], axis=1), 0.2, atol=2e-6)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=2e-6)

    def test_polygon_is_triangulated_without_changing_original_corners(self):
        polygon = np.array([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ])
        positions, normals = polygon_triangles(polygon, np.array([0.0, 0.0, -1.0]))
        self.assertEqual(positions.shape, (6, 3))
        for corner in polygon:
            self.assertTrue(any(np.allclose(corner, point) for point in positions))
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
