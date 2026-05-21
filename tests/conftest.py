"""Pytest 共享 fixture.

把仓库根目录加入 ``sys.path``，使所有测试无需 ``pip install -e .``
即可 ``import modeling`` / ``import datasets`` 等顶层包。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="session")
def device() -> str:
    """统一的测试设备：单元测试始终在 CPU 上运行，避免 CI 没有 GPU。"""
    return "cpu"


@pytest.fixture(scope="session")
def class_names() -> list[str]:
    return ["N", "D", "G", "C", "A", "H", "M", "O"]
