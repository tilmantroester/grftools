import numpy as np

import cosmotools.onedee.stats

pi = np.pi

def test_xi_analytic():
    def xi_analytic_trapz(P, L, n_grid, r=None):
        """Calculate the correlation function given a power spectrum.
        
        Uses explicit integration using numpy.trapz."""
        k = np.fft.rfftfreq(n_grid)*2*pi/L*n_grid
        if r is None:
            r = np.arange(n_grid)/n_grid*L
        if isinstance(r, np.ndarray):
            integrand = P(k)*np.cos(k*r[...,np.newaxis])/pi
            return np.trapz(integrand, k)
        else:
            return np.trapz(Pk*np.cos(k*r)/pi, k)
    
    n_grid = 764
    L = 1.35

    # White noise
    P = lambda k: np.ones_like(k)
    
    xi_trapz = xi_analytic_trapz(P, L, n_grid)
    xi = cosmotools.onedee.stats.xi_analytic(P, L, n_grid)
    assert np.allclose(xi_trapz, xi)
    
    # Correlated fields
    sigma = 1.3
    beta = 5
    k0 = 150
    P = lambda k: np.where(k != 0, k/(1+k/k0)**(beta+1), sigma)
    
    xi_trapz = xi_analytic_trapz(P, L, n_grid)
    xi = cosmotools.onedee.stats.xi_analytic(P, L, n_grid)
    assert np.allclose(xi_trapz, xi)
    
    r = np.array([0, L/76.4, L/2])
    xi_trapz = xi_analytic_trapz(P, L, n_grid, r=r)
    xi = cosmotools.onedee.stats.xi_analytic(P, L, n_grid, r=r)
    assert np.allclose(xi_trapz, xi)

def test_var_mean_xi_L_analytic():
    # Explict numpy sums
    def _var_mean_xi_L_analytic_term_1_sum(xi, f, n_grid):
        n = int(n_grid*f)
        x = np.arange(n)
        y = np.arange(n)
        x_idx, y_idx = np.meshgrid(x, y, indexing="ij")

        return np.mean(xi[np.abs(x_idx-y_idx)%n_grid]**2)

    def _var_mean_xi_L_analytic_term_2_sum(xi, L, f, n_grid, r):
        n = int(n_grid*f)
        r_idx = np.array((r/L*n_grid,), dtype=int)
        x = np.arange(n)
        y = np.arange(n)
        x_idx, y_idx = np.meshgrid(x, y, indexing="ij")

        return np.mean(
                        xi[np.abs((x_idx-y_idx)[...,np.newaxis]-r_idx)%n_grid]
                        * xi[np.abs((x_idx-y_idx)[...,np.newaxis]+r_idx)%n_grid],
                    axis=(0,1)).squeeze()

    def var_mean_xi_L_analytic_numpy_sum(P, L, n_grid, f, r, xi=None):
        if xi is None:
            xi = cosmotools.onedee.stats.xi_analytic(P, L, n_grid)
        a = _var_mean_xi_L_analytic_term_1_sum(xi, f, n_grid)
        b = _var_mean_xi_L_analytic_term_2_sum(xi, L, f, n_grid, r)

        rounding_correction = int(n_grid*f)/(n_grid*f)
        return (a+b)*rounding_correction
        
    n_grid = 1000
    L = 1.35
    
    # White noise
    P = lambda k: np.ones_like(k)
    xi = cosmotools.onedee.stats.xi_analytic(P, L, n_grid)
    
    f = 0.1
    r = np.array([L/100, L/10, L/2])
    var1 = var_mean_xi_L_analytic_numpy_sum(None, L, n_grid, f, r, xi=xi)
    var2 = cosmotools.onedee.stats.var_mean_xi_L_analytic(L, n_grid, f, r, xi=xi)
    assert np.allclose(var1, var2)
    
    f = 1
    r = np.array([L/100, L/10, L/2])
    var1 = var_mean_xi_L_analytic_numpy_sum(None, L, n_grid, f, r, xi=xi)
    var2 = cosmotools.onedee.stats.var_mean_xi_L_analytic(L, n_grid, f, r, xi=xi)
    assert np.allclose(var1, var2)
    
    # Correlated field
    sigma = 1.3
    beta = 5
    k0 = 150
    P = lambda k: np.where(k != 0, k/(1+k/k0)**(beta+1), sigma)
    xi = cosmotools.onedee.stats.xi_analytic(P, L, n_grid)
    
    f = 0.1
    r = np.array([L/100, L/10, L/2])
    var1 = var_mean_xi_L_analytic_numpy_sum(None, L, n_grid, f, r, xi=xi)
    var2 = cosmotools.onedee.stats.var_mean_xi_L_analytic(L, n_grid, f, r, xi=xi)
    assert np.allclose(var1, var2)
    
    f = 1
    r = np.array([L/100, L/10, L/2])
    var1 = var_mean_xi_L_analytic_numpy_sum(None, L, n_grid, f, r, xi=xi)
    var2 = cosmotools.onedee.stats.var_mean_xi_L_analytic(L, n_grid, f, r, xi=xi)
    assert np.allclose(var1, var2)

def test_cov_mean_xi_L_analytic():
    # Explict numpy sums
    def cov_mean_xi_L_analytic_numpy_sum(P, L, n_grid, f, r1, r2):
        xi = cosmotools.onedee.stats.xi_analytic(P, L, n_grid)

        n = int(n_grid*f)
        r1_idx = np.array((r1/L*n_grid,), dtype=int)
        r2_idx = np.array((r2/L*n_grid,), dtype=int)
        x = np.arange(n)
        y = np.arange(n)
        x_idx, y_idx = np.meshgrid(x, y, indexing="ij")

        dxy = (x_idx - y_idx)[...,np.newaxis]
        dr = (r1_idx-r2_idx)
        rounding_correction = int(n_grid*f)/(n_grid*f)
        return np.mean(# Term 1
                       xi[np.abs(dxy)%n_grid]*xi[np.abs(dxy+dr)%n_grid]
                       # Term 2
                    +  xi[np.abs(dxy+r1_idx)%n_grid]*xi[np.abs(dxy-r2_idx)%n_grid], 
                       axis=(0,1)).squeeze()*rounding_correction

    
    n_grid = 1000
    L = 1.35
    
    # White noise
    P = lambda k: np.ones_like(k)
    
    f = 0.1
    r1 = np.array([L/100, L/10, L/2])
    r2 = np.array([L/50, L/5, L/4])
    
    var1 = cosmotools.onedee.stats.var_mean_xi_L_analytic(L, n_grid, f, r1, P=P)
    var2 = cosmotools.onedee.stats.cov_mean_xi_L_analytic(L, n_grid, f, r1, r1, P=P)
    assert np.allclose(var1, var2)
    
    cov1 = cov_mean_xi_L_analytic_numpy_sum(P, L, n_grid, f, r1, r2)
    cov2 = cosmotools.onedee.stats.cov_mean_xi_L_analytic(L, n_grid, f, r1, r2, P=P)
    assert np.allclose(cov1, cov2)
    
    # Correlated field
    sigma = 1.3
    beta = 5
    k0 = 150
    P = lambda k: np.where(k != 0, k/(1+k/k0)**(beta+1), sigma)
    
    f = 1.0
    r1 = np.array([L/100, L/10, L/2])
    r2 = np.array([L/50, L/5, L/4])
    cov1 = cov_mean_xi_L_analytic_numpy_sum(P, L, n_grid, f, r1, r2)
    cov2 = cosmotools.onedee.stats.cov_mean_xi_L_analytic(L, n_grid, f, r1, r2, P=P)
    assert np.allclose(cov1, cov2)
        
    
def test_cov_mean_xcorr_xi_L_analytic():
    def cov_mean_xcorr_xi_L_analytic_numpy_sum(P1, P2, P12, L, n_grid, f, r1, r2):
        xi_1 = cosmotools.onedee.stats.xi_analytic(P1, L, n_grid)
        xi_2 = cosmotools.onedee.stats.xi_analytic(P2, L, n_grid)
        xi_12 = cosmotools.onedee.stats.xi_analytic(P12, L, n_grid)

        n = int(n_grid*f)
        r1_idx = np.array((r1/L*n_grid,), dtype=int)
        r2_idx = np.array((r2/L*n_grid,), dtype=int)
        x = np.arange(n)
        y = np.arange(n)
        x_idx, y_idx = np.meshgrid(x, y, indexing="ij")

        dxy = (x_idx - y_idx)[...,np.newaxis]
        dr = r1_idx - r2_idx
        rounding_correction = int(n_grid*f)/(n_grid*f)

        return 1/4*np.mean(
                    # Term 1
                   (xi_1[np.abs(dxy)%n_grid]*xi_2[np.abs(dxy+dr)%n_grid] 
                  + xi_1[np.abs(dxy+dr)%n_grid]*xi_2[np.abs(dxy)%n_grid]
                  + xi_1[np.abs(dxy-r2_idx)%n_grid]*xi_2[np.abs(dxy+r1_idx)%n_grid]
                  + xi_1[np.abs(dxy+r1_idx)%n_grid]*xi_2[np.abs(dxy-r2_idx)%n_grid])
                    # Term 2
               + 2*(xi_12[np.abs(dxy-r2_idx)%n_grid]*xi_12[np.abs(dxy+r1_idx)%n_grid]
                  + xi_12[np.abs(dxy)%n_grid]*xi_12[np.abs(dxy+dr)%n_grid]),
                    axis=(0,1)).squeeze()*rounding_correction
    
    
    n_grid = 1000
    L = 1.35
    sigma = 1.3
    beta = 5
    k0 = 150
    P_AA = lambda k: np.where(k != 0, k/(1+k/k0)**(beta+1), sigma)
    P_BB = lambda k: np.ones_like(k)
    rho = 0.8
    P_AB = lambda k: np.sqrt(P_AA(k)*P_BB(k))*rho
    
    
    f = 0.5
    r1 = np.array([L/100, L/10, L/2])
    r2 = np.array([L/50, L/5, L/4])
    cov1 = cosmotools.onedee.stats.cov_mean_xi_L_analytic(L, n_grid, f, r1, r2, P=P_AA)
    cov2 = cosmotools.onedee.stats.cov_mean_xcorr_xi_L_analytic(L, n_grid, f, r1, r2, power_spectra=[P_AA, P_AA, P_AA])
    assert np.allclose(cov1, cov2)
        
    f = 0.1
    r1 = np.array([L/100, L/10, L/2])
    r2 = np.array([L/50, L/5, L/4])
    cov1 = cov_mean_xcorr_xi_L_analytic_numpy_sum(P_AA, P_BB, P_AB, L, n_grid, f, r1, r2)
    cov2 = cosmotools.onedee.stats.cov_mean_xcorr_xi_L_analytic(L, n_grid, f, r1, r2, power_spectra=[P_AA, P_BB, P_AB])
    assert np.allclose(cov1, cov2)
    
    f = 1
    r1 = np.array([L/100, L/10, L/2])
    r2 = np.array([L/50, L/5, L/4])
    cov1 = cov_mean_xcorr_xi_L_analytic_numpy_sum(P_AA, P_BB, P_AB, L, n_grid, f, r1, r2)
    cov2 = cosmotools.onedee.stats.cov_mean_xcorr_xi_L_analytic(L, n_grid, f, r1, r2, power_spectra=[P_AA, P_BB, P_AB])
    assert np.allclose(cov1, cov2)