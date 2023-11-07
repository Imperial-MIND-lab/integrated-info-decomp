"""For testing calculate.py file."""
import pytest
import urllib.request
import numpy as np
from scipy.io import loadmat
from phyid.calculate import calc_PhiID
from phyid.utils import PhiID_atoms_abbr

@pytest.fixture(scope="session")
def PhiID_test_simple_1(tmp_path_factory):
    """Load test data."""
    fn = tmp_path_factory.mktemp("test-data") / "PhiID_test_simple_1.mat"
    urllib.request.urlretrieve("https://osf.io/download/45u3y/", fn)
    return loadmat(fn)

calc_PhiID_test_params = [
    ("gaussian", "MMI", "PhiIDFull_MMI"),
    ("gaussian", "CCS", "PhiIDFull_CCS"),
    ("discrete", "MMI", "PhiIDFullDiscrete_MMI"),
    ("discrete", "CCS", "PhiIDFullDiscrete_CCS"),
]

@pytest.mark.parametrize("kind,redundancy,type", calc_PhiID_test_params)
def test_calc_PhiID_sample_1(PhiID_test_simple_1, kind, redundancy, type):
    """Test calc_PhiID function."""
    data = PhiID_test_simple_1
    atoms_res, calc_res = calc_PhiID(
        data["src"].squeeze(), 
        data["trg"].squeeze(), 
        int(data["tau"].squeeze()), 
        kind=kind, 
        redundancy=redundancy
    )
    calc_L = np.array([atoms_res[_] for _ in PhiID_atoms_abbr])
    calc_A = np.mean(calc_L, axis=1)
    assert np.allclose(data[f"{type}_L"], calc_L)
    assert np.allclose(data[f"{type}_A"].squeeze(), calc_A)

