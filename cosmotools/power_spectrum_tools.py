import numpy as np
import scipy.fftpack

pi = np.pi

def sinc(x):
    return np.where(x != 0, np.sin(x)/x, 1)

def pseudo_Pofk(m1, m2, L, 
                k_min=None, k_max=None, n_k_bin=None, logspaced_k_bins=True, 
                bin_edges=None,
                binning_mode=2, clean_caches=False, correct_pixel_window=False,
                verbose=False):
    """Compute the auto or cross-power spectrum between two maps.

    Arguments
    ---------
    m1 : numpy.array
        Field 1. Can be n-dimensional but the units haven't been checked for 
        anything greater than 3. Needs to have the same size in all dimensions.
    m2 : numpy.array
        Field 2. Needs to the same dimension as ``m1``.
    L : float
        Size of the box.
    k_min : float, optional
        Lowest bin edge. (default None).
    k_max : float, optional
        Highest bin edge. (default None).
    n_k_bin : int, optional
        Number of bins. (default None).
    logspaced_k_bins : bool, optional
        Log-spaced bins. (default True).
    bin_edges : numpy.array, optional
        Array with bin edges. (default None).
    binning_mode : int, optional
        Binning mode. Valid values are ``1`` and ``2``. Mode 1 selects bin members 
        with a Boolean mask, whereas mode 2 sorts the grid according to their 
        k modes. Mode 2 tends to be faster. (default 2).
    clean_caches : bool, optional
        Clean the ``scipy.fftpack`` caches after doing the FFTs. If memory usage
        is a concern this clears up memory as soon as possible. (default False).
    correct_pixel_window : bool, optional
        Correct for the pixel window function, i.e., divide by sinc. 
        (default False).
    verbose : bool, optional
        Verbose output. (default False)

    If neither ``n_k_bin`` nor ``bin_edges`` is specified, no binning is performed.
    
    Returns
    -------
    Pk : numpy.array
        Estimate of the power spectrum.
    k : numpy.array
        Wave numbers corresponding to ``Pk``. If binned, uses mean of k values
        in each bin.
    Pk_var : numpy.array
        The variance in each bin. Only present when binning.
    n_mode : numpy.array
        Multiplicity of modes or number of modes in each bin when binning.
    """
    n_dim = m1.ndim
    if m1.ndim != m2.ndim:
        raise RuntimeError(f"Dimensions of m1 and m2 do not match: {m1.ndim}, {m2.ndim}.")
    
    if m1.shape != m2.shape:
        raise RuntimeError(f"The two grids need to be the same shape: {m1.shape}, {m2.shape}.")

    if any([s != m1.shape[0] for s in m1.shape]):
        raise RuntimeError(f"All dimensions must be of the same size: {m1.shape}")
        
    n = m1.shape[0]
    if verbose: print("Computing FFT.")
    m1_m2_ft = scipy.fftpack.fftn(m1, overwrite_x=True)*scipy.fftpack.fftn(m2, overwrite_x=True).conj()
    
    if clean_caches:
        scipy.fftpack._fftpack.destroy_cfftnd_cache()

    if verbose: print("Computing k grid.")
    k_x = np.fft.fftfreq(n, d=1/n).astype(m1.dtype)*2*pi/L
    k_mesh = np.meshgrid(*([k_x]*n_dim), sparse=True, indexing="ij")
    k_grid = np.sqrt(sum(k**2 for k in k_mesh))
    
    if correct_pixel_window:
        m1_m2_ft *= np.prod([1/sinc(L/(2*n)*k) for k in k_mesh], axis=0)**(2*n_dim)
    
    if bin_edges is not None or n_k_bin is not None:
        if bin_edges is None:
            if logspaced_k_bins:
                bin_edges = np.logspace(np.log10(k_min), np.log10(k_max), n_k_bin+1, endpoint=True)
            else:
                bin_edges = np.linspace(k_min, k_max, n_k_bin+1, endpoint=True)
        else:
            bin_edges = np.copy(bin_edges)

        Pk = np.zeros(n_k_bin)
        k = np.zeros(n_k_bin)
        Pk_var = np.zeros(n_k_bin)
        n_mode = np.zeros(n_k_bin, dtype=int)
        
        if verbose: print("Calculating P(k).")
        if binning_mode == 1:
            for i in range(n_k_bin):
                idx =  (bin_edges[i] <= k_grid) & (k_grid < bin_edges[i+1])
                n_mode[i] = np.count_nonzero(idx)
                if n_mode[i] > 0:
                    Pk[i] = np.mean(m1_m2_ft[idx].real)
                    Pk_var[i] = np.var(m1_m2_ft[idx].real)
                    k[i] = np.mean(k_grid[idx])
                else:
                    if logspaced_k_bins:
                        k[i] = np.sqrt(bin_edges[i]*bin_edges[i+1])
                    else:
                        k[i] = (bin_edges[i]+bin_edges[i+1])/2
        elif binning_mode == 2:
            k_sorted_idx = np.argsort(k_grid.flatten())
            k_grid = k_grid.flatten()[k_sorted_idx]
            m1_m2_ft = m1_m2_ft.flatten()[k_sorted_idx]
            bin_idx = np.searchsorted(k_grid, bin_edges)
    
            for i in range(n_k_bin):
                if bin_idx[i] < bin_idx[i+1]:
                    Pk[i] = np.mean(m1_m2_ft[bin_idx[i]:bin_idx[i+1]].real)
                    k[i] = np.mean(k_grid[bin_idx[i]:bin_idx[i+1]])
                else:
                    if logspaced_k_bins:
                        k[i] = np.sqrt(bin_edges[i]*bin_edges[i+1])
                    else:
                        k[i] = (bin_edges[i]+bin_edges[i+1])/2
            
    else:
        if verbose: print("Getting unique k values.")
        k, idx, counts = np.unique(k_grid, return_inverse=True, return_counts=True)
        if verbose: print("k values: ", k)
        Pk = np.zeros_like(k)

        if verbose: print("Calculating P(k).")
        for i in range(len(k)):
            Pk[i] += np.sum(m1_m2_ft.flatten()[idx==i]).real
        counts = counts.astype(dtype=float)
        Pk /= counts
        Pk *= L**n_dim/n**(2*n_dim)
        counts /= 2
        return Pk, k, counts
            
    Pk *= L**n_dim/n**(2*n_dim)
    Pk_var *= (L**n_dim/n**(2*n_dim))**2
    n_mode = n_mode/2
    return Pk, k, Pk_var, n_mode