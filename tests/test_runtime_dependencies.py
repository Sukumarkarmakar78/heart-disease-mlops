from pathlib import Path


def test_numpy_core_compatibility_shim_is_present():
    main_source = Path("api/main.py").read_text(encoding="utf-8")

    assert 'sys.modules.setdefault("numpy._core"' in main_source
    assert "joblib.load" in main_source
