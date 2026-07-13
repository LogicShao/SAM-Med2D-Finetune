import unittest

import numpy as np

from postprocess_3d import filter_connected_components, postprocess_brats_masks


class Postprocess3DTest(unittest.TestCase):
    def test_component_filter_keeps_only_largest_component(self):
        mask = np.zeros((5, 5, 5), dtype=np.uint8)
        mask[0:2, 0:2, 0:2] = 1
        mask[4, 4, 4] = 1

        filtered = filter_connected_components(mask, keep_largest=True)

        self.assertEqual(int(filtered.sum()), 8)
        self.assertEqual(int(filtered[4, 4, 4]), 0)

    def test_postprocess_enforces_et_tc_wt_hierarchy_and_records_counts(self):
        et = np.zeros((5, 5, 5), dtype=np.uint8)
        tc = np.zeros_like(et)
        wt = np.zeros_like(et)
        et[2, 2, 2] = 1
        tc[1, 1, 1] = 1
        wt[0, 0, 0] = 1

        processed, report = postprocess_brats_masks(
            {"ET": et, "TC": tc, "WT": wt},
            closing_radius=0,
            opening_radius=0,
            wt_keep_largest=False,
            keep_topk_tc=0,
            keep_topk_et=0,
            z_smooth_iterations=0,
        )

        self.assertTrue(np.all(processed["ET"] <= processed["TC"]))
        self.assertTrue(np.all(processed["TC"] <= processed["WT"]))
        self.assertEqual(report["hierarchy"]["after"], {"et_outside_tc": 0, "tc_outside_wt": 0})
        self.assertIn("after_hierarchy", report["classes"]["ET"])


if __name__ == "__main__":
    unittest.main()
