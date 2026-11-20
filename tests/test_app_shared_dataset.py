import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from dataset_storage import (
    load_shared_dataset,
    load_shared_python_files,
    save_shared_dataset,
)


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class SharedDatasetAppTests(unittest.TestCase):
    def test_added_row_is_saved_and_restored_after_reload(self):
        initial = pd.DataFrame({"student": ["A"], "score": [3]})
        expected = pd.DataFrame({"student": ["A", "B"], "score": [3, 5]})

        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "shared.sqlite3"
            save_shared_dataset(initial, database_path)

            with patch.dict(os.environ, {"SPYSS_SQLITE_PATH": str(database_path)}):
                app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
                editor = app.data_editor[0]
                app.session_state[editor.key] = {
                    "edited_rows": {},
                    "added_rows": [{"student": "B", "score": 5}],
                    "deleted_rows": [],
                }
                app.sidebar.button[0].click().run()
                saved, _ = load_shared_dataset(database_path)
                reloaded_app = AppTest.from_file(
                    str(APP_PATH),
                    default_timeout=30,
                ).run()

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertEqual(
            [],
            [exception.message for exception in reloaded_app.exception],
        )
        self.assertTrue(app.sidebar.checkbox[0].value)
        pd.testing.assert_frame_equal(saved, expected)
        self.assertTrue(reloaded_app.sidebar.checkbox[0].value)
        pd.testing.assert_frame_equal(reloaded_app.data_editor[0].value, expected)

    def test_shared_python_file_is_saved_selected_and_loaded(self):
        source_code = "print('shared script')"

        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "shared.sqlite3"

            with patch.dict(os.environ, {"SPYSS_SQLITE_PATH": str(database_path)}):
                app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
                app.session_state["analysis_code"] = source_code
                app.session_state["code_editor_initial_value"] = source_code
                app.session_state["code_editor_revision"] += 1
                filename_input = next(
                    widget for widget in app.text_input
                    if widget.label == "Shared filename"
                )
                filename_input.set_value("shared.py")
                save_button = next(
                    widget for widget in app.button
                    if widget.label == "Save current code to shared SQL"
                )
                save_button.click().run()
                saved_files = load_shared_python_files(database_path)

                reloaded_app = AppTest.from_file(
                    str(APP_PATH),
                    default_timeout=30,
                ).run()
                file_picker = next(
                    widget for widget in reloaded_app.selectbox
                    if widget.label == "Saved Python files"
                )
                file_picker.select("shared.py").run()
                load_button = next(
                    widget for widget in reloaded_app.button
                    if widget.label == "Load selected file"
                )
                load_button.click().run()

        self.assertEqual([], [exception.message for exception in app.exception])
        self.assertEqual(
            [],
            [exception.message for exception in reloaded_app.exception],
        )
        self.assertEqual({"shared.py": source_code}, saved_files)
        shared_files_checkbox = next(
            widget for widget in reloaded_app.checkbox
            if widget.label == "Use shared Python files"
        )
        self.assertTrue(shared_files_checkbox.value)
        self.assertEqual(source_code, reloaded_app.session_state["analysis_code"])


if __name__ == "__main__":
    unittest.main()
