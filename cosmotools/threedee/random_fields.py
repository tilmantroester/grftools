import numpy as np
import scipy.fftpack

pi = np.pi

def sinc(x):
    return np.piecewise(x, (x != 0,), (lambda x: np.sin(x)/x, 1))

def pseudo_Pofk(m1, m2, L, 
                k_min=None, k_max=None, n_k_bin=None, logspaced_k_bins=True, 
                bin_edges=None, binning_mode=1, 
                verbose=False):
    if m1.shape[0] != m1.shape[1] or m1.shape[0] != m1.shape[2]:
        raise RuntimeError("Input grid needs to be cubic.")
    if m2.shape != m2.shape:
        raise RuntimeError("The two grids need to be the same size.")
        
    n = m1.shape[0]
    if verbose: print("Computing FFT.")
    m1_m2_ft = scipy.fftpack.fftn(m1, overwrite_x=True)*scipy.fftpack.fftn(m2, overwrite_x=True).conj()
    scipy.fftpack._fftpack.destroy_cfftnd_cache()

    
    if verbose: print("Computing k grid.")
    k_x = np.fft.fftfreq(n, d=1/n).astype(np.float32)*2*pi/L
    k_mesh = np.meshgrid(k_x, k_x, k_x, sparse=True, indexing="ij")
    k_grid = np.sqrt(k_mesh[0]**2 + k_mesh[1]**2 + k_mesh[2]**2)
    
    m1_m2_ft *= 1/(sinc(L/(2*n)*k_mesh[0])*sinc(L/(2*n)*k_mesh[1])*sinc(L/(2*n)*k_mesh[2]))**4
    
    if bin_edges is not None or n_k_bin is not None:
        if bin_edges is None:
            if logspaced_k_bins:
                bin_edges = np.logspace(np.log10(k_min), np.log10(k_max), n_k_bin+1, endpoint=True)
            else:
                bin_edges = np.linspace(k_min, k_max, n_k_bin+1, endpoint=True)
        else:
            bin_edges = np.copy(bin_edges)

        bin_edges[0] *= 0.999
        bin_edges[-1] *= 1.001
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
                    Pk[i] = np.mean(m1_m2_ft[bin_idx[i]:bin_idx[i+1]])
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
        Pk *= L**3/n**6
        counts /= 2
        return Pk, k, counts
            
    Pk *= L**3/n**6
    Pk_var *= (L**3/n**6)**2
    n_mode = n_mode/2
    return Pk, k, Pk_var, n_mode
