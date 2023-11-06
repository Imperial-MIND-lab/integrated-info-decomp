"""For testing calculate.py file."""
import pytest
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from phyid.calculate import calc_PhiID
from phyid.utils import PhiID_atoms_abbr

@pytest.fixture
def PhiID_matlab_sample_1():
    temp_file = Path("/mnt/raid1/PycharmProjects/networks-2/project-20230803-pidt/data/luppi2023/PhiID_test_1.mat")
    return loadmat(temp_file)

calc_PhiID_test_params = [
    ("gaussian", "MMI", "PhiIDFull_MMI"),
    ("gaussian", "CCS", "PhiIDFull_CCS"),
    ("discrete", "MMI", "PhiIDFullDiscrete_MMI"),
    ("discrete", "CCS", "PhiIDFullDiscrete_CCS"),
]

@pytest.mark.parametrize("kind,redundancy,type", calc_PhiID_test_params)
def test_calc_PhiID_sample_1(PhiID_matlab_sample_1, kind, redundancy, type):
    """Test calc_PhiID function."""
    data = PhiID_matlab_sample_1
    calc_res = calc_PhiID(
        data["src"].squeeze(), 
        data["trg"].squeeze(), 
        int(data["tau"].squeeze()), 
        kind=kind, 
        redundancy=redundancy
    )
    calc_L = np.array([calc_res[_] for _ in PhiID_atoms_abbr])
    calc_A = np.mean(calc_L, axis=1)
    assert np.allclose(data[f"{type}_L"], calc_L)
    assert np.allclose(data[f"{type}_A"].squeeze(), calc_A)

