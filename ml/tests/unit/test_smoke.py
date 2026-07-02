from __future__ import annotations

import app


def test_package_exposes_version():
    assert isinstance(app.__version__, str)
    assert app.__version__ != ""
