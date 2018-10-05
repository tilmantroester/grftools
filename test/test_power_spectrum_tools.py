import numpy as np

import cosmotools.power_spectrum_tools

import cosmotools.onedee.random_fields

pi = np.pi

def test_1d_pseudo_Pofk():
    n = 512
    L = 1
    m = np.random.rand(n).astype(dtype=np.float32)

    Pk, k, Pk_var, n_mode = cosmotools.power_spectrum_tools.pseudo_Pofk(m, m, L, 
                                                                        k_min=(2*pi)/L, 
                                                                        k_max=(2*pi)/L*n/2, 
                                                                        n_k_bin=n//2, 
                                                                        logspaced_k_bins=False)

    Pk_1d, k_1d, Pk_err_1d = cosmotools.onedee.random_fields.pseudo_Pofk(m, m, L, 
                                                                         k_min=(2*pi)/L, 
                                                                         k_max=(2*pi)/L*n/2, 
                                                                         n_k_bin=n//2, 
                                                                         logspaced_k_bins=False)

    assert np.allclose(k, k_1d)
    assert np.allclose(Pk, Pk_1d)
    # assert np.allclose(Pk_var/n_mode, Pk_err_1d**2)

def test_2d_pseudo_Pofk():
    def calculate_pseudo_Cl(map1, map2, box_size, n_bin=None, ell_min=None, ell_max=None, logspaced=False):
        """Estimates the cross-power spectrum of two maps.
        
        Required arguments:
        map1            Array of size (N, M).
        map2            Array of same shape as map1.
        box_size        Physical size (L1, L2) of the maps.
        
        Optional arguments:
        n_bin           Number of ell bins. If None, no binning is performed.
        ell_min         Minimum ell.
        ell_max,        Maximum ell.
        logspaced       Log-spaced bins. Default is False.
        
        Returns:
        Tuple (pCl_real, pCl_real_err, ell_mean, bin_edges, n_mode) with
            pCl_real        Estimated cross-power spectrum,
            pCl_real_err    Error on the mean, estimated from the scatter of the 
                            individual modes,
            ell_mean        Mean ell per bin,
            bin_edges       Edges of the ell bins,
            n_mode          Number of modes per bin.
        """

        if map1.shape != map2.shape:
            raise ValueError("Map dimensions don't match: {}x{} vs {}x{}".format(*(map1.shape + map2.shape)))
        
        # This can be streamlined alot
        map1_ft = np.fft.fft2(map1) * (box_size[0]/map1.shape[0])*(box_size[1]/map1.shape[1])
        map1_ft = np.fft.fftshift(map1_ft, axes=0)
        map2_ft = np.fft.fft2(map2) * (box_size[0]/map1.shape[0])*(box_size[1]/map1.shape[1])
        map2_ft = np.fft.fftshift(map2_ft, axes=0)
        map_1x2_ft = map1_ft.conj()*map2_ft

        ell_x_min_box = 2.0*pi/box_size[0]
        ell_y_min_box = 2.0*pi/box_size[1]
        ell_x = np.fft.fftshift(np.fft.fftfreq(map1.shape[0], d=1.0/map1.shape[0]))*ell_x_min_box
        ell_y = np.fft.fftfreq(map1.shape[1], d=1.0/map1.shape[1])*ell_y_min_box
        x_idx, y_idx = np.meshgrid(ell_x, ell_y, indexing="ij")
        ell_grid = np.sqrt((x_idx)**2 + (y_idx)**2)
        
        if n_bin==None:
            bin_edges = np.arange(start=np.min([ell_x_min_box, ell_y_min_box])/1.00001, stop=np.max(ell_grid), step=np.min([ell_x_min_box, ell_y_min_box]))
            n_bin = len(bin_edges) - 1
        else:
            if ell_max > np.max(ell_grid):
                raise RuntimeWarning("Maximum ell is {}, where as ell_max was set as {}.".format(np.max(ell_grid), ell_max))
            if ell_min < np.min([ell_x_min_box, ell_y_min_box]):
                raise RuntimeWarning("Minimum ell is {}, where as ell_min was set as {}.".format(np.min([ell_x_min_box, ell_y_min_box]), ell_min))
            if logspaced:
                bin_edges = np.logspace(np.log10(ell_min), np.log10(ell_max), n_bin+1, endpoint=True)
            else:
                bin_edges = np.linspace(ell_min, ell_max, n_bin+1, endpoint=True)

        pCl_real = np.zeros(n_bin)
        pCl_imag = np.zeros(n_bin)
        pCl_real_err = np.zeros(n_bin)
        pCl_imag_err = np.zeros(n_bin)
        ell_mean = np.zeros(n_bin)
        n_mode = np.zeros(n_bin)
        bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
        
        ell_sort_idx = np.argsort(ell_grid.flatten())
        map_1x2_ft_sorted = map_1x2_ft.flatten()[ell_sort_idx]
        ell_grid_sorted = ell_grid.flatten()[ell_sort_idx]
        bin_idx = np.searchsorted(ell_grid_sorted, bin_edges)
        
        for i in range(n_bin):
            P = map_1x2_ft_sorted[bin_idx[i]:bin_idx[i+1]]/(box_size[0]*box_size[1])
            ell = ell_grid_sorted[bin_idx[i]:bin_idx[i+1]]
            pCl_real[i] = np.mean(P.real)
            pCl_imag[i] = np.mean(P.imag)
            pCl_real_err[i] = np.sqrt(np.var(P.real)/len(P))
            pCl_imag_err[i] = np.sqrt(np.var(P.imag)/len(P))
            ell_mean[i] = np.mean(ell)
            n_mode = len(P)
        
        return pCl_real, pCl_real_err, ell_mean, bin_edges, n_mode

    n = 128
    L = 1
    m = np.random.rand(n, n).astype(dtype=np.float32)

    pCl_real, pCl_real_err, ell_mean, _, n_mode = calculate_pseudo_Cl(m, m, (L, L),
                                                                      ell_min=(2*pi)/L, 
                                                                      ell_max=(2*pi)/L*n/2, 
                                                                      n_bin=n//2)

    Pk, k, Pk_var, n_mode = cosmotools.power_spectrum_tools.pseudo_Pofk(m, m, L, 
                                                                        k_min=(2*pi)/L, 
                                                                        k_max=(2*pi)/L*n/2, 
                                                                        n_k_bin=n//2, 
                                                                        logspaced_k_bins=False)

    assert np.allclose(pCl_real, Pk)
    assert np.allclose(ell_mean, k)
    assert np.allclose(pCl_real_err**2, Pk_var)

def test_3d_pseudo_Pofk(plot=False):
    """Test of the 3D pseudo P(k) estimator works."""

    n_grid = 50
    L = 1.0

    m1 = np.random.randn(n_grid, n_grid, n_grid)
    m2 = np.random.randn(n_grid, n_grid, n_grid)

    Pk, k, n_mode = cosmotools.power_spectrum_tools.pseudo_Pofk(m1, m2, L, 
                                        k_min=None, k_max=None, n_k_bin=None, logspaced_k_bins=True, 
                                        bin_edges=None, binning_mode=1, 
                                        verbose=False)

    k_min = 2*pi
    k_max = k_min*n_grid/2
    n_k_bin = 5

    Pk, k, Pk_var, n_mode = cosmotools.power_spectrum_tools.pseudo_Pofk(m1, m2, L, 
                                        k_min=k_min, k_max=k_max, n_k_bin=n_k_bin, logspaced_k_bins=True, 
                                        bin_edges=None, binning_mode=1, 
                                        verbose=False)

if __name__ == "__main__":
    test_1d_pseudo_Pofk()
    test_2d_pseudo_Pofk()
    test_3d_pseudo_Pofk()