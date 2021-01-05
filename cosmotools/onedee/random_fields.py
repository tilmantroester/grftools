import collections

import numpy as np

import scipy.interpolate
import scipy.integrate
import scipy.optimize

from .. import utils

pi = np.pi

def variance_gamma_distribution(x, n, rho, sigma1=1, sigma2=1):
    """Computes the pdf of the variance-gamma distribution.

    The parameters follow the convention in https://en.wikipedia.org/wiki/Wishart_distribution#Marginal_distribution_of_matrix_elements.
    Note that  by reparameterising x -> sigma_1*sigma_2*x, the dependence on 
    sigma_1 and sigma_2 drops out of the variance-gamma distribution and only 
    the dependence on rho remains.
    """
    C = (1-rho**2)*sigma1*sigma2
    A = np.abs(x)**((n-1)/2)/(scipy.special.gamma(n/2)*np.sqrt(2**(n-1)*pi*C*(sigma1*sigma2)**n))
    B = scipy.special.kv((n-1)/2, np.abs(x)/C)
    D = np.exp(rho*x/C)
    
    return A*B*D

def variance_gamma_distribution_cdf(x, n, rho, sigma1=1, sigma2=1):
    """Computes the cdf of the variance-gamma distribution."""

    integrand = lambda y: variance_gamma_distribution(y, n, rho, sigma1, sigma2)
    if not isinstance(x, collections.Iterable):
        x = [x,]
    cdf = np.zeros(len(x))
    for i, y in enumerate(x):
        cdf[i] = scipy.integrate.quad(integrand, -np.inf, min(y, 0))[0]
        if y > 0:
            cdf[i] += scipy.integrate.quad(integrand, 0, y)[0]
    return cdf.squeeze()

def variance_gamma_distribution_ppf(q, n, rho, sigma1=1, sigma2=1):
    """Computes the ppf of the variance-gamma distribution."""

    if not isinstance(q, collections.Iterable):
        q = [q,]
    ppf = np.zeros(len(q))
    for i in range(len(q)):
        if q[i] <= 0:
            ppf[i] = -np.inf
        elif q[i] >= 1:
            ppf[i] = np.inf
            
        f = lambda x: q[i] - variance_gamma_distribution_cdf(x, n, rho, sigma1, sigma2)
        ppf[i] = scipy.optimize.brentq(f, -50, 50)
        
    return ppf.squeeze()

def gal_distribution(x, mu, sigma, s=1/2, d=1):
    if d == 1:
        if abs(sigma) < 1e-5:
            # Gamma distribution
            p = scipy.stats.gamma.pdf(x, a=s, scale=mu)
        else:
            K_nu = lambda x: scipy.special.kv(s-1/2, x)
            Gamma = scipy.special.gamma(s)
            Q = np.abs(x)/sigma
            C = np.sqrt(2+mu**2/sigma**2)
            p = 2*np.exp(mu*x/sigma**2)/((2*pi)**(1/2)*Gamma*sigma)*(Q/C)**(s-1/2)*K_nu(Q*C)
    else:
        Sigma_det = np.linalg.det(sigma)
        if abs(Sigma_det) < 1e-7:
            raise ValueError("I don't know what the limit Sigma->0 of the multivariate GAL distribution is.")
        else:
            K_nu = lambda x: scipy.special.kv(s-d/2, x)
            Gamma = scipy.special.gamma(s)
            Sigma_inv = np.linalg.inv(sigma)
            Q = np.sqrt(x.T @ Sigma_inv @ x)
            C = np.sqrt(2+mu.T @ Sigma_inv @ mu)
            p = 2*np.exp(mu.T @ Sigma_inv @ x)/((2*pi)**(d/2)*Gamma*np.sqrt(Sigma_det))*(Q/C)**(s-d/2)*K_nu(Q*C)
    return p
    
def sample_gal_distribution(size, mu, sigma, s=1/2, d=1):
    if d == 1:
        Z = scipy.stats.gamma.rvs(a=s, size=size)
        X = scipy.stats.norm.rvs(size=size)
        
        return mu*Z + sigma*np.sqrt(Z)*X
    else:
        raise NotImplementedError()

def create_gaussian_random_field(P, n_grid, L, fix_mean=None, 
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

def create_gaussian_random_fields(power_spectra, n_grid, L):
    """Create correlated 1D Gaussian random fields from given power spectra.

    Required Arguments:
    power_spectra   List of callables. For three fields A, B, C, the power spectra
                    should follow the order [AA, BB, CC, AB, AC, BC].
    n_grid          Number of grid points.
    L               Physical size of the box.
    """

    k_grid = np.fft.rfftfreq(n_grid)
    
    k_min = 2*pi/(L/n_grid)
    V = L/(n_grid)**2
    
    if callable(power_spectra) or len(power_spectra) == 1:
        if not callable(power_spectra):
            P = power_spectra[0]
        else:
            P = power_spectra
        P_grid = np.atleast_1d(P(k_grid*k_min))
        m_ft = np.random.rayleigh(scale=np.sqrt(1/V*P_grid/2))*np.exp(2j*pi*np.random.random(k_grid.shape))
    else:
        n_spectra = int(np.sqrt(8*len(power_spectra) + 1)/2 - 1/2)
        if n_spectra*(n_spectra+1)//2 != len(power_spectra):
            raise ValueError("Number of supplied power spectra invalid: {}".format(len(power_spectra)))
        
        cov = np.zeros((k_grid.size, n_spectra, n_spectra))
        for i in range(n_spectra):
            cov[:,i,i] = power_spectra[i](k_grid*k_min)/V/2
            for j in range(i+1, n_spectra):
                idx = n_spectra + i*n_spectra - (i*(i+1))//2 + (j-i-1)
                if callable(power_spectra[idx]):
                    cov[:,i,j] = power_spectra[idx](k_grid*k_min)/V/2
                    cov[:,j,i] = cov[:,i,j]
    
        L = np.linalg.cholesky(cov)
    
        real = np.einsum("kij, kj->ki", L, np.random.normal(size=(k_grid.size, n_spectra)))
        imag = np.einsum("kij, kj->ki", L, np.random.normal(size=(k_grid.size, n_spectra)))

        m_ft = real + 1j*imag
        
    m = np.fft.irfftn(m_ft, axes=[0,])
    return m.T

def pseudo_Pofk(m1, m2, L, mask=None, mode_mixing_matrix=False,
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
    else:
        Pofm = Pofm[:n//2]
    
    if bin_edges is not None or n_k_bin is not None:
        Pofm_binned, k_binned, Pofm_err = utils.bin_array(Pofm, k, 
                                                          x_min=k_min, x_max=k_max, 
                                                          n_bin=n_k_bin, logspaced=logspaced_k_bins, 
                                                          bin_edges=bin_edges)
        return Pofm_binned, k_binned, Pofm_err
    else:
        return Pofm, k

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
