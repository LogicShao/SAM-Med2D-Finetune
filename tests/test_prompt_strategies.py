import unittest

from prompt_strategies import (
    ET_PROMPT_VARIANTS,
    _resolve_et_prompt_variant,
    build_class_specific_prompt_info,
)


class PromptStrategiesTest(unittest.TestCase):
    def test_every_declared_et_variant_resolves_to_its_own_configuration(self):
        for variant_name in ET_PROMPT_VARIANTS:
            config = _resolve_et_prompt_variant(variant_name)
            self.assertEqual(config["name"], variant_name)
            self.assertGreaterEqual(config["positive_points"], 0)
            self.assertGreaterEqual(config["negative_points"], 0)

    def test_baseline_prompt_does_not_require_case_or_prediction_data(self):
        prompt = build_class_specific_prompt_info(
            class_name="ET",
            slice_index=3,
            brats_case=None,
            image_size=256,
            base_prompt_info={"selected_box_xyxy": [1.0, 2.0, 10.0, 11.0]},
            predicted_masks={},
            class_prompt_variant="baseline",
        )

        self.assertEqual(prompt["source"], "yolo_box")
        self.assertEqual(prompt["box_xyxy"], [1.0, 2.0, 10.0, 11.0])

    def test_unknown_et_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            _resolve_et_prompt_variant("unknown")


if __name__ == "__main__":
    unittest.main()
