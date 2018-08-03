import collections

import numpy as np
import numba

pi = np.pi

def var_mean_delta_L_analytic(P, L, f, n_grid):
    """Calculate the variance of the mean of a fraction f of the field."""
    k = np.fft.rfftfreq(n_grid)*2*pi/L*n_grid
    s = lambda x: np.sin(x)/x
    W = np.ones_like(k)
    W[1:] = s(k[1:]*L*f/2)
    return np.trapz(2/(2*pi)*P(k)*W**2, k)

def xi_analytic(P, L, n_grid, r=None):
    """Calculate the correlation function given a power spectrum."""
    if n_grid%2 != 0:
        warnings.warn("n_grid should be even.")
        
    k = np.fft.rfftfreq(n_grid)*2*pi/L*n_grid
    xi = np.fft.irfft(P(k))*n_grid/L
    if r is not None:
        r_idx = np.array(r/L*n_grid, dtype=int)
        if not np.allclose(r_idx/n_grid*L, r):
            warnings.warn("Requested r does not lie on grid.")
        xi = xi[r_idx]
    return xi

@numba.jit(nopython=True)
def _var_mean_auto_xi_L_analytic_sum(xi, n, r_idx):
    n_grid = len(xi)
    v = np.zeros_like(r_idx, dtype=np.float64)
    # Looping over r_idx seems to be faster in numba than advanced indexing.
    for i, r in enumerate(r_idx):
        for x_idx in range(n):
            for y_idx in range(n):
                v[i] +=   xi[abs(x_idx-y_idx)%n_grid] * xi[abs(x_idx-y_idx)%n_grid] \
                        + xi[abs(x_idx-y_idx+r)%n_grid] * xi[abs(x_idx-y_idx-r)%n_grid]
    return v/n**2

@numba.jit(nopython=True)
def _cov_mean_auto_xi_L_analytic_sum(xi, n, r1_idx, r2_idx):
    n_grid = len(xi)
    c = np.zeros_like(r1_idx, dtype=np.float64)
    # Looping over r_idx seems to be faster in numba than advanced indexing.
    for i, (r1, r2) in enumerate(zip(r1_idx, r2_idx)):
        dr = r1 - r2
        for x_idx in range(n):
            for y_idx in range(n):
                c[i] +=   xi[abs(x_idx-y_idx)%n_grid] * xi[abs(x_idx-y_idx+dr)%n_grid] \
                        + xi[abs(x_idx-y_idx+r1)%n_grid] * xi[abs(x_idx-y_idx-r2)%n_grid]
    return c/n**2

@numba.jit(nopython=True)
def _cov_mean_xcorr_xi_L_analytic_sum(xi_AA, xi_BB, xi_AB, n, r1_idx, r2_idx):
    n_grid = len(xi_AA)
    c = np.zeros_like(r1_idx, dtype=np.float64)
    for i, (r1, r2) in enumerate(zip(r1_idx, r2_idx)):
        dr = r1 - r2
        for x_idx in range(n):
            for y_idx in range(n):
                dxy = x_idx - y_idx
                c[i] +=  (xi_AA[abs(dxy)%n_grid]*xi_BB[abs(dxy+dr)%n_grid]      # Term 1 
                       + xi_AA[abs(dxy+dr)%n_grid]*xi_BB[abs(dxy)%n_grid]
                       + xi_AA[abs(dxy-r2)%n_grid]*xi_BB[abs(dxy+r1)%n_grid]
                       + xi_AA[abs(dxy+r1)%n_grid]*xi_BB[abs(dxy-r2)%n_grid]) \
                          \
                       + 2*(xi_AB[abs(dxy-r2)%n_grid]*xi_AB[abs(dxy+r1)%n_grid] # Term 2
                       + xi_AB[abs(dxy)%n_grid]*xi_AB[abs(dxy+dr)%n_grid])

    return c/(4*n**2)

def var_mean_xi_L_analytic(L, n_grid, f, r, xi=None, P=None):
    """Computes the variance of the average of a correlation function over a domain L."""
    if xi is None:
        if P is not None:
            xi = xi_analytic(P, L, n_grid)
        else:
            raise ValueError("Either xi or P must be specified.")
        
    n = int(n_grid*f)
    r_idx = np.array(r/L*n_grid, dtype=int)
    var = _var_mean_auto_xi_L_analytic_sum(xi, n, r_idx)
    
    rounding_correction = int(n_grid*f)/(n_grid*f)
    return var*rounding_correction

def cov_mean_xi_L_analytic(L, n_grid, f, r1, r2, xi=None, P=None):
    """Computes the covariance of the average of a correlation function over a domain L."""
    
    if xi is None:
        if P is not None:
            xi = xi_analytic(P, L, n_grid)
        else:
            raise ValueError("Either xi or P must be specified.")
        
    n = int(n_grid*f)
    r1_idx = np.array(r1/L*n_grid, dtype=int)
    r2_idx = np.array(r2/L*n_grid, dtype=int)
    cov = _cov_mean_auto_xi_L_analytic_sum(xi, n, r1_idx, r2_idx)
    
    rounding_correction = int(n_grid*f)/(n_grid*f)
    return cov*rounding_correction

def cov_mean_xcorr_xi_L_analytic(L, n_grid, f, r1, r2, xi=None, power_spectra=None):
    """Computes the covariance of the average of a cross-correlation function over a domain L."""

    if isinstance(r1, collections.Iterable) and isinstance(r2, collections.Iterable):
        if len(r1) != len(r2):
            raise ValueError("Shapes of r1 and r2 must match.")
            
    if xi is None:
        xi = []
        if power_spectra is not None:
            xi = [xi_analytic(P, L, n_grid) for P in power_spectra]
        else:
            raise ValueError("Either xi or P must be specified.")
    if len(xi) != 3:
        raise ValueError("Number of provided correlation functions or power spectra must be 3.")
        
    n = int(n_grid*f)
    r1_idx = np.array(r1/L*n_grid, dtype=int)
    r2_idx = np.array(r2/L*n_grid, dtype=int)
    cov = _cov_mean_xcorr_xi_L_analytic_sum(*xi, n, r1_idx, r2_idx)
    
    rounding_correction = int(n_grid*f)/(n_grid*f)
    return cov*rounding_correction
