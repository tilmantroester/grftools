import numpy as np

import scipy.interpolate

from . import utils

pi = np.pi

def create_gaussian_random_field_1d(P, n_grid, L, fix_mean=None, 
                                    output_fourier=False):
    """Create 1D Gaussian random field from given power spectrum.

    Required Arguments:
    P               Callable. Power spectrum.
    n_grid          Number of grid points.
    L               Physical size of the box.
    fix_mean        Fix the mean of the output field. Default: sample from P(0).
    output_foutier  Return the Fourier transform of the field and k modes.
    """

    k_grid = np.fft.rfftfreq(n_grid)
    
    k_min = 2*pi/(L/n_grid)
    V = L/(n_grid)**2
    P_grid = np.atleast_1d(P(k_grid*k_min))
    m_ft = np.random.rayleigh(scale=np.sqrt(1/V*P_grid/2))*np.exp(2j*pi*np.random.random(k_grid.shape))
    if fix_mean is not None:
        m_ft[k_grid == 0] = fix_mean
    else:
        m_ft[k_grid == 0] = np.random.normal(scale=np.sqrt(1/V*P_grid[k_grid==0]))
    m = np.fft.irfft(m_ft)
    if output_fourier:
        return m, m_ft, k_grid*k_min
    else:
        return m

def pseudo_Pofk_1d(m1, m2, L, mask=None, mode_mixing_matrix=False,
                   k_min=None, k_max=None, n_k_bin=None, logspaced_k_bins=None,
                   bin_edges=None):
    if m1.shape != m2.shape:
        raise ValueError("Map dimensions don't match: {} vs {}".format(m1.size, m2.size))
    
    n = m1.size
    if mask is not None:
        Pofm = L*np.fft.fft(m1*mask)*np.fft.fft(m2*mask).conj()/n**2
    else:
        Pofm = L*np.fft.fft(m1)*np.fft.fft(m2).conj()/n**2
    k = np.fft.fftfreq(n)[:n//2]*2*pi/L*n
    if mask is not None:
        if isinstance(mode_mixing_matrix, bool) and mode_mixing_matrix:
            raise NotImplementedError("Mode mixing matrix not implemented yet.")
            #C = compute_mode_mixing_matrix(mask)
            #Pofm = (np.linalg.inv(C) @ Pofm)[:n//2]
        elif isinstance(mode_mixing_matrix, np.ndarray):
            Pofm = (np.linalg.inv(mode_mixing_matrix) @ Pofm)[:n//2]
        else:
            Pofm = Pofm[:n//2]/np.mean(mask)
    
    if bin_edges is not None or n_k_bin is not None:
        Pofm_binned, k_binned, Pofm_err = utils.bin_array(Pofm, k, 
                                                          x_min=k_min, x_max=k_max, 
                                                          n_bin=n_k_bin, logspaced=logspaced_k_bins, 
                                                          bin_edges=bin_edges)
        return Pofm_binned, k_binned, Pofm_err
    else:
        return Pofm[:n//2], k

def interpolated_powerspectrum_from_file(filename):
    k_grid, P_grid = np.loadtxt(filename, unpack=True)
    if k_grid[0] == 0:
        P0 = P_grid[0]
        k_grid = k_grid[1:]
        P_grid = P_grid[1:]
    else:
        P0 = 0

    log_P_intp = scipy.interpolate.InterpolatedUnivariateSpline(np.log(k_grid), np.log(P_grid), k=1, ext=0)
    def P(k):
        Pofk = np.piecewise(k, k > 0, [lambda k: np.exp(log_P_intp(np.log(k))), P0])
        return Pofk

    return P
# Alternative implementation of pseudo_Pofk_1d
# def calculate_pseudo_P_k_1d(m1, m2, box_size, n_k_bin=None, k_min=None, k_max=None, logspaced=False):
#     if m1.shape != m2.shape:
#         raise ValueError("Map dimensions don't match: {}x{} vs {}x{}".format(*(m1.shape + m2.shape)))
        
#     m1m2 = np.fft.rfft(m1)*np.conj(np.fft.rfft(m2))
    
#     k_grid = np.fft.rfftfreq(m1.shape[0])
#     k_min_box = 2*pi/(box_size/m1.shape[0])

#     if n_k_bin == None:
#         bin_edges = k_grid + k_min_box/2
#         Pk_real = m1m2[1:].real
#         Pk_imag = m1m2[1:].imag
#         Pk_err = np.zeros_like(Pk_real)
#         k_mean = k_grid[1:]
#         n_mode = np.ones(Pk_real.size, dtype=int)
#     else:
#         if logspaced:
#             bin_edges = np.logspace(np.log10(k_min/k_min_box), np.log10(k_max/k_min_box), n_k_bin+1, endpoint=True)
#         else:
#             bin_edges = np.linspace(k_min/k_min_box, k_max/k_min_box, n_k_bin+1, endpoint=True)
#         n_bin = n_k_bin
    
#         Pk_real = np.zeros(n_bin)
#         Pk_err = np.zeros(n_bin)
#         Pk_imag = np.zeros(n_bin)
#         k_mean = np.zeros(n_bin)
#         n_mode = np.zeros(n_bin)

#         bin_idx = np.searchsorted(k_grid, bin_edges)

#         for i in range(n_bin):
#             if bin_idx[i+1] - bin_idx[i] == 0:
#                 if logspaced:
#                     k_mean[i] = np.sqrt(bin_edges[i]*bin_edges[i+1])
#                 else:
#                     k_mean[i] = (bin_edges[i]+bin_edges[i+1])/2
#             else:
#                 P = m1m2[bin_idx[i]:bin_idx[i+1]]
#                 Pk_real[i] = np.mean(P.real)
#                 Pk_imag[i] = np.mean(P.imag)
#                 Pk_err[i] = np.sqrt(np.var(P.real)/len(P))
#                 k_mean[i] = np.mean(k_grid[bin_idx[i]:bin_idx[i+1]])
#                 n_mode[i] = len(P)
    
#     V = box_size/(m1.shape[0])**2
#     return Pk_real*V, Pk_err*V, k_mean*k_min_box, bin_edges*k_min_box, n_mode
