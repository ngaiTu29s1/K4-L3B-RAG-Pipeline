from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_starts_without_query():
    app = AppTest.from_file(Path(__file__).parent.parent / "app.py").run()

    assert not app.exception
    assert app.title[0].value == "⚔️ PatchLens"
    assert app.chat_input[0].placeholder.startswith("Ví dụ:")
