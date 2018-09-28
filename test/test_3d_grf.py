import numpy as np

import cosmotools.threedee.random_fields as threedee_rf

pi = np.pi

def test_pseudo_Pofk(plot=False):
    """Test of the 3D pseudo P(k) estimator works."""

    n_grid = 50
    L = 1.0

    m1 = np.random.randn(n_grid, n_grid, n_grid)
    m2 = np.random.randn(n_grid, n_grid, n_grid)

    Pk, k, n_mode = threedee_rf.pseudo_Pofk(m1, m2, L, 
                                        k_min=None, k_max=None, n_k_bin=None, logspaced_k_bins=True, 
                                        bin_edges=None, binning_mode=1, 
                                        verbose=False)

    k_min = 2*pi
    k_max = k_min*n_grid/2
    n_k_bin = 5

    Pk, k, Pk_var, n_mode = threedee_rf.pseudo_Pofk(m1, m2, L, 
                                        k_min=k_min, k_max=k_max, n_k_bin=n_k_bin, logspaced_k_bins=True, 
                                        bin_edges=None, binning_mode=1, 
                                        verbose=False)

if __name__ == "__main__":
    test_pseudo_Pofk()