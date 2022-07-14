import io
import json
import os
import unittest
from unittest.mock import patch
import sys
import tempfile

from nllblangs.export import list_langs


import pandas as pd


class ExportTestSuite(unittest.TestCase):
    """Export language list test suite."""

    expected_list_length = 206

    def _test_df_with_columns(self, columns, expected_column_number):
        with patch("sys.stdout", new=io.StringIO()) as output:
            list_langs(export="simple", columns=columns)
            output.seek(0)
            reached_df = pd.read_csv(output, sep="\t", header=None)

            self.assertEqual(len(reached_df), self.expected_list_length)
            self.assertEqual(len(reached_df.columns), expected_column_number)

    def _export_to_file(self, filename):
        mock_file = os.path.join(tempfile.gettempdir(), filename)
        list_langs(export=mock_file, columns=None)

        return mock_file

    def test_export_columns(self):
        self._test_df_with_columns(None, 2)
        # Default columns "Language Name" and "Code" should not affect the
        # number of exported columns
        self._test_df_with_columns(["Language Name"], 2)
        self._test_df_with_columns(["Language Name", "BCP47", "Code"], 3)
        self._test_df_with_columns(["Language Name", "BCP47", "Family"], 4)

    def test_export_python(self):
        with patch("sys.stdout", new=io.StringIO()) as output:
            list_langs(export="python", columns=None)
            output.seek(0)
        reached_python_export = eval(output.getvalue())
        self.assertTrue(type(reached_python_export) is dict)
        self.assertEqual(len(reached_python_export.keys()), self.expected_list_length)

    def test_export_tsv(self):
        exported_file = self._export_to_file("result.tsv")
        reached_df = pd.read_csv(exported_file, sep="\t", header=0)
        self.assertEqual(len(reached_df), self.expected_list_length)

    def test_export_excel(self):
        exported_file = self._export_to_file("result.xlsx")
        reached_df = pd.read_excel(exported_file)
        self.assertEqual(len(reached_df), self.expected_list_length)


if __name__ == "__main__":
    unittest.main()
