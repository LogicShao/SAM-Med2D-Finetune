import unittest
import argparse

from cli_utils import str_to_bool


class CliUtilsTest(unittest.TestCase):
    def test_str_to_bool_accepts_explicit_true_and_false_values(self):
        for value in (True, "true", "YES", "1", "on"):
            self.assertTrue(str_to_bool(value))
        for value in (False, "false", "NO", "0", "off"):
            self.assertFalse(str_to_bool(value))

    def test_str_to_bool_rejects_ambiguous_values(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            str_to_bool("sometimes")


if __name__ == "__main__":
    unittest.main()
