import collections
import warnings

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
def _var_mean_auto_xi_L_analytic_sum(xi, n, r_idx, 
                                     w=np.empty(0)     # As of 0.39, numba doesn't deal well with None.
                                    ):
    n_grid = len(xi)
    var = np.zeros_like(r_idx, dtype=np.float64)
    # Looping over r_idx seems to be faster in numba than advanced indexing.
    for i, r in enumerate(r_idx):
        for x_idx in range(n):
            for y_idx in range(n):
                v =  (xi[abs(x_idx-y_idx)%n_grid] * xi[abs(x_idx-y_idx)%n_grid]
                    + xi[abs(x_idx-y_idx+r)%n_grid] * xi[abs(x_idx-y_idx-r)%n_grid])
                if len(w) == 0:
                    var[i] += v
                else:
                    var[i] += v*w[x_idx]*w[(x_idx+r)%n_grid]*w[y_idx]*w[(y_idx+r)%n_grid]
    return var/n**2


@numba.jit(nopython=True)
def _cov_mean_auto_xi_L_analytic_sum(xi, n, r1_idx, r2_idx, w=np.empty(0)):
    n_grid = len(xi)
    cov = np.zeros_like(r1_idx, dtype=np.float64)
    # Looping over r_idx seems to be faster in numba than advanced indexing.
    for i, (r1, r2) in enumerate(zip(r1_idx, r2_idx)):
        dr = r1 - r2
        for x_idx in range(n):
            for y_idx in range(n):
                c =   xi[abs(x_idx-y_idx)%n_grid] * xi[abs(x_idx-y_idx+dr)%n_grid] \
                    + xi[abs(x_idx-y_idx+r1)%n_grid] * xi[abs(x_idx-y_idx-r2)%n_grid]
                if len(w) == 0:
                    cov[i] += c
                else:
                    cov[i] += c*w[x_idx]*w[(x_idx+r1)%n_grid]*w[y_idx]*w[(y_idx+r2)%n_grid]
    return cov/n**2

@numba.jit(nopython=True)
def _cov_mean_xcorr_xi_L_analytic_sum(xi_AA, xi_BB, xi_AB, n, r1_idx, r2_idx):
    n_grid = len(xi_AA)
    cov = np.zeros_like(r1_idx, dtype=np.float64)
    for i, (r1, r2) in enumerate(zip(r1_idx, r2_idx)):
        dr = r1 - r2
        for x_idx in range(n):
            for y_idx in range(n):
                dxy = x_idx - y_idx
                c =  (xi_AA[abs(dxy)%n_grid]*xi_BB[abs(dxy+dr)%n_grid]      # Term 1 
                    + xi_AA[abs(dxy+dr)%n_grid]*xi_BB[abs(dxy)%n_grid]
                    + xi_AA[abs(dxy-r2)%n_grid]*xi_BB[abs(dxy+r1)%n_grid]
                    + xi_AA[abs(dxy+r1)%n_grid]*xi_BB[abs(dxy-r2)%n_grid]) \
                       \
                    + 2*(xi_AB[abs(dxy-r2)%n_grid]*xi_AB[abs(dxy+r1)%n_grid] # Term 2
                    + xi_AB[abs(dxy)%n_grid]*xi_AB[abs(dxy+dr)%n_grid])
                cov[i] += c

    return cov/(4*n**2)

@numba.jit(nopython=True)
def _cov_mean_xcorr_xi_L_analytic_sum_weights(xi_AA, xi_BB, xi_AB, n, r1_idx, r2_idx,
                                              w_A, w_B):
    n_grid = len(xi_AA)
    cov = np.zeros_like(r1_idx, dtype=np.float64)
    for i, (r1, r2) in enumerate(zip(r1_idx, r2_idx)):
        dr = r2 - r1
        for x_idx in range(n):
            w_x_p = w_A[x_idx]*w_B[(x_idx+r1)%n_grid]
            w_x_m = w_A[(x_idx+r1)%n_grid]*w_B[x_idx]
            for y_idx in range(n):
                w_y_p = w_A[y_idx]*w_B[(y_idx+r2)%n_grid]
                w_y_m = w_A[(y_idx+r2)%n_grid]*w_B[y_idx]
                dxy = y_idx - x_idx
                c =    w_x_p*w_y_p * (  xi_AA[abs(dxy)%n_grid]*xi_BB[abs(dxy+dr)%n_grid] 
                                      + xi_AB[abs(dxy+r2)%n_grid]*xi_AB[abs(dxy-r1)%n_grid]) \
                     + w_x_p*w_y_m * (  xi_AA[abs(dxy+r2)%n_grid]*xi_BB[abs(dxy-r1)%n_grid] 
                                      + xi_AB[abs(dxy)%n_grid]*xi_AB[abs(dxy+dr)%n_grid]) \
                     + w_x_m*w_y_p * (  xi_AA[abs(dxy-r1)%n_grid]*xi_BB[abs(dxy+r2)%n_grid] 
                                      + xi_AB[abs(dxy)%n_grid]*xi_AB[abs(dxy+dr)%n_grid]) \
                     + w_x_m*w_y_m * (  xi_AA[abs(dxy+dr)%n_grid]*xi_BB[abs(dxy)%n_grid]
                                      + xi_AB[abs(dxy+r2)%n_grid]*xi_AB[abs(dxy-r1)%n_grid])

                cov[i] += c

    return cov/(4*n**2)

def var_mean_xi_L_analytic(L, n_grid, f, r, xi=None, P=None, weights=None):
    """Computes the variance of the average of a correlation function over a domain L."""
    if xi is None:
        if P is not None:
            xi = xi_analytic(P, L, n_grid)
        else:
            raise ValueError("Either xi or P must be specified.")
        
    n = int(n_grid*f)
    r_idx = np.array(r/L*n_grid, dtype=int)
    if weights is None:
        var = _var_mean_auto_xi_L_analytic_sum(xi, n, r_idx)
    else:
        var = _var_mean_auto_xi_L_analytic_sum(xi, n, r_idx, weights)
        xi_w, _ = correlation_function(weights, L)
        var /= xi_w[r_idx]**2
    
    rounding_correction = int(n_grid*f)/(n_grid*f)
    return var*rounding_correction

def cov_mean_xi_L_analytic(L, n_grid, f, r1, r2, xi=None, P=None, weights=None):
    """Computes the covariance of the average of a correlation function over a domain L."""
    
    if xi is None:
        if P is not None:
            xi = xi_analytic(P, L, n_grid)
        else:
            raise ValueError("Either xi or P must be specified.")
        
    n = int(n_grid*f)
    r1_idx = np.array(r1/L*n_grid, dtype=int)
    r2_idx = np.array(r2/L*n_grid, dtype=int)
    if weights is None:
        cov = _cov_mean_auto_xi_L_analytic_sum(xi, n, r1_idx, r2_idx)
    else:
        cov = _cov_mean_auto_xi_L_analytic_sum(xi, n, r1_idx, r2_idx, weights)
        xi_w, _ = correlation_function(weights, L)
        cov /= xi_w[r1_idx]*xi_w[r2_idx]
     
    rounding_correction = int(n_grid*f)/(n_grid*f)
    return cov*rounding_correction

def cov_mean_xcorr_xi_L_analytic(L, n_grid, f, r1, r2, xi=None, power_spectra=None,
                                 weights1=None, weights2=None):
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
    if weights1 is None or weights2 is None:
        cov = _cov_mean_xcorr_xi_L_analytic_sum(*xi, n, r1_idx, r2_idx)
    else:
        cov = _cov_mean_xcorr_xi_L_analytic_sum_weights(*xi, n, r1_idx, r2_idx, weights1, weights2)
        xi_w, _ = cross_correlation_function(weights1, weights2, L)
        cov /= xi_w[r1_idx]*xi_w[r2_idx]

    rounding_correction = int(n_grid*f)/(n_grid*f)
    return cov*rounding_correction

def cross_correlation_function(d1, d2, L, bins=None, weights1=None, weights2=None):
    """Compute the cross-correlation function between d1 and d2 using FFTs.

    The cross-correlation is symmetrized between d1 and d2."""
    if d1.shape != d2.shape:
        raise RuntimeError("Shape mismatch between d1 and d2.")

    if weights1 is None and weights2 is None:
        d1_ft = np.fft.rfft(d1)
        d2_ft = np.fft.rfft(d2)
        P_k = d1_ft*np.conj(d2_ft)
        xi_raw = np.fft.irfft(P_k.real)
        xi_raw /= len(xi_raw)
        x = np.arange(len(xi_raw))*L/len(d1)

        if bins is None:
            return xi_raw, x
        else:
            return bin_data(x=x, y=xi_raw, bin_edges=bins, normalize=True)
    elif weights1 is not None and weights2 is not None:
        w1_ft = np.fft.rfft(weights1)
        w2_ft = np.fft.rfft(weights2)
        w_k = w1_ft*np.conj(w2_ft)
        xi_w_raw = np.fft.irfft(w_k.real)

        d1_ft = np.fft.rfft(d1*weights1)
        d2_ft = np.fft.rfft(d2*weights2)
        P_k = d1_ft*np.conj(d2_ft)
        xi_raw = np.fft.irfft(P_k.real)
        x = np.arange(len(xi_raw))*L/len(d1)

        if bins is None:
            return xi_raw/xi_w_raw, x
        else:
            xi, _ = bin_data(x=x, y=xi_raw, bin_edges=bins, normalize=False)
            xi_w, _ = bin_data(x=x, y=xi_w_raw, bin_edges=bins, normalize=False)

            xi[xi_w != 0] /= xi_w[xi_w != 0]

            x, _ = bin_data(x=x, y=x, bin_edges=bins, weights=xi_w_raw, normalize=True)

            return xi, x
    else:
        raise ValueError("Both weight1 and weight2 need to be specified.")

def correlation_function(d, L, bins=None, weights=None):
    """Compute the auto-correlation function of d using FFTs."""

    return cross_correlation_function(d1=d, d2=d, L=L, bins=bins, 
                                      weights1=weights, weights2=weights)

@numba.jit(nopython=True)
def cross_correlation_function_marks(d1, d2, L, x_max, marks_block_size=1, 
                                     x_bins=np.empty(0),      # Numba doesn't like None (as of 0.39)
                                     weights1=None, weights2=None):
    n_grid = len(d1)
    if len(d2) != n_grid:
        raise RuntimeError("Size mismatch between d1 and d2.")

    n_blocks = n_grid//marks_block_size
    
    n_x = int(x_max*n_grid/L)
    x = np.arange(n_x)/n_x*x_max
    if len(x_bins) == 0:
        n_x_bin = n_x
    else:
        n_x_bin = len(x_bins)-1
    
    xi_marks = np.zeros((n_blocks, n_x_bin))
    n_marks = np.zeros_like(xi_marks)

    w1 = np.ones_like(d1) if weights1 is None else weights1
    w2 = np.ones_like(d2) if weights2 is None else weights2
    xi = np.zeros(n_x)
    n = np.zeros_like(xi)

    for i in range(n_grid):
        if w1[i] == 0.0 and w2[i] == 0.0:
            continue
        xi *= 0.0
        n *= 0.0
        for j in range(i, i+n_x):
            idx = j-i
            if w1[i] != 0.0 and w2[j%n_grid] != 0.0:
                xi[idx] += d1[i]*d2[j%n_grid]*w1[i]*w2[j%n_grid]
                n[idx] += w1[i]*w2[j%n_grid]
            if w1[j%n_grid] != 0.0 and w2[i] != 0.0:
                xi[idx] += d1[j%n_grid]*d2[i]*w1[j%n_grid]*w2[i]
                n[idx] += w1[j%n_grid]*w2[i]

        if len(x_bins) == 0:
            xi_marks[i//marks_block_size] += xi
            n_marks[i//marks_block_size] += n
        else:
            xi_binned, x_binned = bin_data(x, xi, x_bins, weights=n, normalize=False)
            n_binned, _ = bin_data(x, n, x_bins, normalize=False)
            xi_marks[i//marks_block_size] += xi_binned
            n_marks[i//marks_block_size] += n_binned

    xi_marks = np.where(n_marks!=0, xi_marks/n_marks, xi_marks)
    return xi_marks, n_marks

def correlation_function_marks(d, L, x_max, marks_block_size=1, 
                               x_bins=np.empty(0),      # Numba doesn't like None (as of 0.39)
                               weights=None):
    return cross_correlation_function_marks(d1=d, d2=d, L=L, x_max=x_max,
                                            marks_block_size=marks_block_size,
                                            x_bins=x_bins,
                                            weights1=weights, weights2=weights)


# The following functions should be moved to a `utils` module or similar.

# Sort of replacement for numpy.searchsorted. Now that numba supports the
# side="right" keyword, this should be replaced by numpy.searchsorted.
@numba.jit(nopython=True)
def bin_search(x, l, r, bin_edges):
    if r-l == 1:
        return l
    pivot = int((l+r)/2)
    if x < bin_edges[pivot]:
        return bin_search(x, l, pivot, bin_edges)
    else:
        return bin_search(x, pivot, r, bin_edges)

@numba.jit(nopython=True)
def find_bin(x, bin_edges):
    if x < bin_edges[0] or x >= bin_edges[-1]:
        return -1
    return bin_search(x, 0, len(bin_edges)-1, bin_edges)

@numba.jit(nopython=True)
def bin_data(x, y, bin_edges, weights=None, normalize=True):
    bins = np.zeros(len(bin_edges)-1)
    mean_x = np.zeros_like(bins)
    n = np.zeros_like(bins)

    w = np.ones_like(x) if weights is None else weights

    for i in range(len(x)):
        idx = find_bin(x[i], bin_edges)
        if not idx == -1:
            bins[idx] += w[i]*y[i]
            mean_x[idx] += w[i]*x[i]
            n[idx] += w[i]

    mean_x[n==0] = ((bin_edges[:-1]+bin_edges[1:])/2)[n==0]
    if normalize:
        bins[n!=0] /= n[n!=0]
        mean_x[n!=0] /= n[n!=0]

    return bins, mean_x

