import io
import os
import unittest
from unittest.mock import patch
import sys
import tempfile

from nllblangs.merge import merge_files

import pandas as pd


class MergeTestSuite(unittest.TestCase):
    mock_files = []

    def tearDown(self):
        [os.remove(a) for a in self.mock_files]

    def _create_mock_file(self, mock_content, name):
        mock_file = os.path.join(tempfile.gettempdir(), name)
        with open(mock_file, "w") as f:
            f.write(mock_content)
        self.mock_files.append(mock_file)
        return f

    def test_merge_how(self):
        left_content = """
apc\tArabic (Lebanon)
arb\tArabic (Modern Standard Arabic)
ary\tArabic (Morocco)
arz\tArabic (Egypt)
asm\tAssamese
ast\tAsturian
awa\tAwadhi
        """
        right_content = """
cat\tr_Catalan
ceb\tr_Cebuano
ast\tr_Asturian
ces\tr_Czech
cjk\tr_Chokwe
awa\tr_Awadhi
ckb\tr_Central (Sorani) Kurdish
        """

        expected_result_inner = """
ast\tAsturian\tr_Asturian
awa\tAwadhi\tr_Awadhi
        """
        expected_result_left = """
apc\tArabic (Lebanon)\t
arb\tArabic (Modern Standard Arabic)\t
ary\tArabic (Morocco)\t
arz\tArabic (Egypt)\t
asm\tAssamese\t
ast\tAsturian\tr_Asturian
awa\tAwadhi\tr_Awadhi
        """
        expected_result_right = """
cat\t\tr_Catalan
ceb\t\tr_Cebuano
ast\tAsturian\tr_Asturian
ces\t\tr_Czech
cjk\t\tr_Chokwe
awa\tAwadhi\tr_Awadhi
ckb\t\tr_Central (Sorani) Kurdish
        """
        expected_result_outer = """
apc\tArabic (Lebanon)\t
arb\tArabic (Modern Standard Arabic)\t
ary\tArabic (Morocco)\t
arz\tArabic (Egypt)\t
asm\tAssamese\t
ast\tAsturian\tr_Asturian
awa\tAwadhi\tr_Awadhi
cat\t\tr_Catalan
ceb\t\tr_Cebuano
ces\t\tr_Czech
cjk\t\tr_Chokwe
ckb\t\tr_Central (Sorani) Kurdish
        """

        left_file = self._create_mock_file(left_content, "left.tsv")
        right_file = self._create_mock_file(right_content, "right.tsv")

        merge_methods = [
            ("inner", expected_result_inner),
            ("left", expected_result_left),
            ("right", expected_result_right),
            ("outer", expected_result_outer),
        ]
        for merge_method, expected in merge_methods:
            reached_df = None
            with patch("sys.stdout", new=io.StringIO()) as output:
                with patch("sys.stderr", new=io.StringIO()) as _:
                    merge_files(
                        left=left_file.name,
                        right=right_file.name,
                        how=merge_method,
                        export="simple",
                        left_column="0",
                        right_column="0",
                        left_header=None,
                        right_header=None,
                    )
                    output.seek(0)
                    reached_df = pd.read_csv(output, sep="\t", header=None)
            expected_df = pd.read_csv(io.StringIO(expected), sep="\t", header=None)
            self.assertTrue(reached_df.equals(expected_df))


if __name__ == "__main__":
    unittest.main()
