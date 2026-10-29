import pytest


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Keep tests from writing pose snapshots/history into the real data/ dir."""
    monkeypatch.setenv("TWFB_DATA_DIR", str(tmp_path / "data"))
