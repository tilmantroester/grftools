import numpy as np
import scipy.fftpack

pi = np.pi

def sinc(x):
    return np.where(x != 0, np.sin(x)/x, 1)

def pseudo_Pofk(m1, m2, L, 
                k_min=None, k_max=None, n_k_bin=None, logspaced_k_bins=True, 
                bin_edges=None, dtype=np.float32,
                binning_mode=2, clean_caches=False, correct_pixel_window=False,
                verbose=False):
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
    k_x = np.fft.fftfreq(n, d=1/n).astype(dtype)*2*pi/L
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