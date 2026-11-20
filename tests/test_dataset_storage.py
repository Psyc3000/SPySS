import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dataset_storage import (
    load_shared_dataset,
    load_shared_python_files,
    save_shared_dataset,
    save_shared_python_file,
)


class SharedDatasetStorageTests(unittest.TestCase):
    def test_load_returns_no_dataset_before_first_save(self):
        with tempfile.TemporaryDirectory() as directory:
            dataframe, saved_at = load_shared_dataset(
                Path(directory) / "datasets.sqlite3"
            )

        self.assertIsNone(dataframe)
        self.assertIsNone(saved_at)

    def test_save_and_load_dataframe_and_replace_shared_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "datasets.sqlite3"
            initial = pd.DataFrame({"student": ["A", "B"], "score": [3, 5]})
            save_shared_dataset(initial, database_path)

            loaded, saved_at = load_shared_dataset(database_path)
            pd.testing.assert_frame_equal(loaded, initial)
            self.assertIsNotNone(saved_at)

            replacement = pd.DataFrame({"group": ["control"], "value": [7.5]})
            save_shared_dataset(replacement, database_path)
            loaded, _ = load_shared_dataset(database_path)

        pd.testing.assert_frame_equal(loaded, replacement)

    def test_save_rejects_duplicate_column_names(self):
        dataframe = pd.DataFrame([[1, 2]], columns=["score", "score"])

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "duplicate column names"):
                save_shared_dataset(
                    dataframe,
                    Path(directory) / "datasets.sqlite3",
                )

    def test_shared_python_files_can_be_saved_loaded_and_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "datasets.sqlite3"
            save_shared_python_file("analysis.py", "print('first')", database_path)
            save_shared_python_file("helper.py", "value = 1", database_path)
            save_shared_python_file("analysis.py", "print('updated')", database_path)

            shared_files = load_shared_python_files(database_path)

        self.assertEqual(
            {
                "analysis.py": "print('updated')",
                "helper.py": "value = 1",
            },
            shared_files,
        )

    def test_shared_python_file_requires_a_py_filename_without_a_path(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "datasets.sqlite3"
            for filename in ("", "folder/analysis.py", "folder\\analysis.py", "notes.txt"):
                with self.subTest(filename=filename):
                    with self.assertRaisesRegex(ValueError, "valid .py filename"):
                        save_shared_python_file(filename, "", database_path)


if __name__ == "__main__":
    unittest.main()
