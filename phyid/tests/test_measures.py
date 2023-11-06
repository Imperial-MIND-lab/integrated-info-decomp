"""For testing measures.py file."""
import pytest
import numpy as np
from phyid.measures import (
    local_entropy_mvn,
    local_entropy_binary,
    redundancy_mmi,
    double_redundacy_mmi,
    redundancy_ccs,
    double_redundacy_ccs,
)

@pytest.mark.parametrize(
    "test_input,expected", 
    [
        ((0, 0, 1), 0.9189385332046727),
        ((0, [0], [1]), 0.9189385332046727),
        (([0, 0], [0, 0], np.identity(2)), 1.8378770664093453)
    ]
)
def test_local_entropy_mvn(test_input, expected):
    """Test local_entropy_mvn function."""
    assert np.allclose(local_entropy_mvn(*test_input), expected)

@pytest.mark.xfail
def test_redundancy_mmi():
    """Test test_redundancy_mmi function."""
    assert False

@pytest.mark.xfail
def test_double_redundacy_mmi():
    """Test test_double_redundacy_mmi function."""
    assert False

@pytest.mark.xfail
def test_redundancy_ccs():
    """Test test_redundancy_ccs function."""
    assert False

@pytest.mark.xfail
def test_double_redundancy_ccs():
    """Test test_double_redundancy_ccs function."""
    assert False