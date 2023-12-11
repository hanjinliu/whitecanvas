import numpy as np
from numpy.testing import assert_allclose
from whitecanvas.canvas._categorical import _plans as _p
import pytest

@pytest.mark.parametrize(
    "column, expected, policy",
    [
        ([1, 3, 2], [0, 1, 2], _p.NoMarginPolicy()),
        (["T", "TT", "Q"], [0, 1, 2], _p.NoMarginPolicy()),
        ([1, 3, 2], [0, 1.3, 2.6], _p.ConstMarginPolicy(0.3)),
    ]
)
def test_offset_one_level(column, expected, policy):
    plan = _p.OffsetPlan(("a",), [policy])
    out = plan.generate(np.array(column).reshape(-1, 1), by_all=("a",))
    assert_allclose(out, expected)

@pytest.mark.parametrize(
    "arr, expected, policy",
    [
        ([[1, 1], [1, 2], [1, 3], [2, 1]], [0, 0, 0, 1], _p.NoMarginPolicy()),
        ([[1, 1], [1, 2], [1, 3], [2, 3]], [0, 0, 0, 1], _p.NoMarginPolicy()),
        ([[1, 1], [1, 2], [1, 3], [2, 3]], [0, 0, 0, 1.3], _p.ConstMarginPolicy(0.3)),
    ]
)
def test_offset_subset(arr, expected, policy):
    plan = _p.OffsetPlan(("a",), [policy])
    out = plan.generate(np.array(arr), by_all=("a", "b"))
    assert_allclose(out, expected)

@pytest.mark.parametrize(
    "arr, expected, policy",
    [
        ([[1, 1], [1, 2], [1, 3], [2, 1]], [0, 1, 2, 3], [_p.NoMarginPolicy(), _p.NoMarginPolicy()]),
        ([[1, 1], [1, 2], [1, 3], [2, 3]], [0, 1, 2, 5], [_p.NoMarginPolicy(), _p.NoMarginPolicy()]),
        ([[1, 1], [1, 2], [2, 1], [2, 2]], [0, 1, 2.3, 3.3], [_p.NoMarginPolicy(), _p.ConstMarginPolicy(0.3)]),
    ]
)  # fmt: skip
def test_offset_multilevel(arr, expected, policy):
    plan = _p.OffsetPlan(("a", "b"), policy)
    out = plan.generate(np.array(arr), by_all=("a", "b"))
    assert_allclose(out, expected)
