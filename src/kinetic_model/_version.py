try:
    # Python 3.8+
    from importlib.metadata import PackageNotFoundError, version
except Exception:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

# Try to get the installed package version; during editable/source builds
# fall back to VERSION file if present, else a safe default.
try:
    __version__ = version("microbesGYM")
except PackageNotFoundError:  # not installed yet
    _v = None
    try:
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        vf = root / "VERSION"
        if vf.is_file():
            _v = vf.read_text(encoding="utf-8").strip()
    except Exception:
        _v = None
    __version__ = _v or "0.0.0"
