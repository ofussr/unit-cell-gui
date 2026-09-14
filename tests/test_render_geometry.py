import unittest

import numpy as np

from unit_cell_gui.render_geometry import (
    atom_screen_distances,
    bond_screen_distances,
    clip_polygon_vertices_3d,
    clip_projected_bond,
    draw_order_key,
    face_screen_distances,
    inset_polygon,
    screen_distance,
    screen_distances,
)


class RenderGeometryTests(unittest.TestCase):
    def test_screen_distance_uses_camera_z_only(self):
        self.assertEqual(screen_distance(np.array([8.0, -4.0, -3.0])), 3.0)
        self.assertEqual(screen_distance(np.array([0.0, 0.0, 2.0])), -2.0)


    def test_specific_distance_calculators_use_requested_points(self):
        self.assertEqual(atom_screen_distances(np.array([9.0, 8.0, -2.0])), (2.0,))
        self.assertEqual(
            bond_screen_distances(
                np.array([0.0, 0.0, -3.0]),
                np.array([0.0, 0.0, 4.0]),
            ),
            (3.0, -4.0),
        )
        self.assertEqual(
            face_screen_distances(
                np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 2.0], [0.0, 0.0, -5.0]])
            ),
            (1.0, -2.0, 5.0),
        )

    def test_characteristic_point_distances_and_order_key(self):
        face = np.array([
            [0.0, 0.0, -4.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 3.0],
        ])
        distances = screen_distances(face)
        self.assertEqual(distances, (4.0, -1.0, -3.0))
        self.assertEqual(draw_order_key(distances), (4.0, 0.0, -3.0))

    def test_far_primitive_sorts_before_near_primitive(self):
        far = draw_order_key(screen_distances(np.array([[0, 0, -5], [0, 0, -4]], float)))
        near = draw_order_key(screen_distances(np.array([[0, 0, 1], [0, 0, 2]], float)))
        ordered = sorted([("near", near), ("far", far)], key=lambda item: item[1], reverse=True)
        self.assertEqual([name for name, _key in ordered], ["far", "near"])

    def test_projected_bond_is_clipped_exactly_by_atom_radii(self):
        result = clip_projected_bond(
            np.array([0.0, 0.0]),
            np.array([100.0, 0.0]),
            20.0,
            10.0,
        )
        self.assertIsNotNone(result)
        first, second, t_first, t_second = result
        np.testing.assert_allclose(first, [20.0, 0.0])
        np.testing.assert_allclose(second, [90.0, 0.0])
        self.assertAlmostEqual(t_first, 0.2)
        self.assertAlmostEqual(t_second, 0.9)

    def test_projected_bond_disappears_when_atom_discs_cover_it(self):
        self.assertIsNone(
            clip_projected_bond(
                np.array([0.0, 0.0]),
                np.array([20.0, 0.0]),
                12.0,
                9.0,
            )
        )

    def test_diagonal_clip_is_radial(self):
        result = clip_projected_bond(
            np.array([10.0, 20.0]),
            np.array([40.0, 60.0]),
            10.0,
            5.0,
        )
        self.assertIsNotNone(result)
        first, second, _t_first, _t_second = result
        self.assertAlmostEqual(float(np.linalg.norm(first - np.array([10.0, 20.0]))), 10.0)
        self.assertAlmostEqual(float(np.linalg.norm(second - np.array([40.0, 60.0]))), 5.0)

    def test_inset_polygon_moves_corners_towards_the_centre_by_requested_amount(self):
        polygon = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        shifted = inset_polygon(polygon, inset=1.0)
        centre = polygon.mean(axis=0)
        original_distances = np.linalg.norm(polygon - centre, axis=1)
        shifted_distances = np.linalg.norm(shifted - centre, axis=1)
        np.testing.assert_allclose(original_distances - shifted_distances, 1.0)

    def test_inset_polygon_accepts_one_radius_per_vertex(self):
        polygon = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        insets = np.array([1.0, 2.0, 3.0, 4.0])
        shifted = inset_polygon(polygon, inset=insets)
        centre = polygon.mean(axis=0)
        original_distances = np.linalg.norm(polygon - centre, axis=1)
        shifted_distances = np.linalg.norm(shifted - centre, axis=1)
        expected = np.minimum(insets, 0.49 * original_distances)
        np.testing.assert_allclose(original_distances - shifted_distances, expected)

    def test_inset_polygon_does_not_overshoot_the_centre(self):
        polygon = np.array([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]])
        shifted = inset_polygon(polygon, inset=1.0)
        centre = polygon.mean(axis=0)
        shifted_distances = np.linalg.norm(shifted - centre, axis=1)
        original_distances = np.linalg.norm(polygon - centre, axis=1)
        self.assertTrue(np.all(shifted_distances > 0.0))
        self.assertTrue(np.all(shifted_distances < original_distances))

    def test_clip_polygon_vertices_3d_cuts_triangle_corners_by_real_radii(self):
        polygon = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
        clipped = clip_polygon_vertices_3d(polygon, np.array([1.0, 1.0, 1.0]))
        expected = np.array([
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.2928932188, 0.7071067812, 0.0],
            [0.7071067812, 3.2928932188, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        np.testing.assert_allclose(clipped, expected)

    def test_clip_polygon_vertices_3d_limits_cut_to_short_edges(self):
        polygon = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        clipped = clip_polygon_vertices_3d(polygon, np.array([10.0, 10.0, 10.0]))
        self.assertEqual(clipped.shape[0], 6)
        # no clipped point may pass the midpoint of the short edge [0, 0] – [1, 0]
        self.assertTrue(np.all(clipped[:, 0] >= 0.0))
        self.assertTrue(np.all(clipped[:, 0] <= 1.0))


if __name__ == "__main__":
    unittest.main()
