
import os
import numpy as np

def mkdirs(path):
    """Recursively create directories.

    Simply calls os.makedirs with exist_ok=True.
    """
    os.makedirs(path, exist_ok=True)

def weighted_var(a, w):
    """Compute weighted variance."""
    V1 = np.sum(w)
    V2 = np.sum(w**2)
    return 1/(V1-V2/V1)*np.sum(w*(a-np.average(a, weights=w))**2)

def bin_array(array, x, x_min=None, x_max=None, n_bin=None, logspaced=False, 
              bin_edges=None, weights=None, return_error_on_mean=True):
    """Bin array.

    Required arguments:
    array
    x

    Optional arguments:
    x_min
    x_max
    n_bin
    logspaced
    bin_edges
    weights

    Returns:
    Tuple (binned_array, mean_x, scatter)
    """
    if bin_edges is None:
        if logspaced:
            bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), n_bin+1, endpoint=True)
        else:
            bin_edges = np.linspace(x_min, x_max, n_bin+1, endpoint=True)
    else:
        n_bin = len(bin_edges)-1
        
    binned_array = np.zeros(n_bin, dtype=array.dtype)
    mean_x = np.zeros(n_bin, dtype=array.dtype)
    scatter = np.zeros(n_bin, dtype=array.dtype)
    
    w = np.ones_like(array) if weights is None else weights

    for i in range(n_bin):
        M = np.logical_and(bin_edges[i] <= x, x < bin_edges[i+1])
        if np.count_nonzero(M) == 0:
            # Empty bin
            binned_array[i] = 0
            scatter[i] = 0
            if logspaced:
                mean_x[i] = np.sqrt(bin_edges[i]*bin_edges[i+1])
            else:
                mean_x[i] = (bin_edges[i]+bin_edges[i+1])/2
        else:
            binned_array[i] = np.average(array[M], weights=w[M])
            if return_error_on_mean:
                scatter[i] = np.sqrt(weighted_var(array[M], w[M])/np.count_nonzero(M))
            else:
                scatter[i] = np.sqrt(weighted_var(array[M], w[M]))

            mean_x[i] = np.average(x[M], weights=w[M])

    return binned_array, mean_x, scatter

def rebin_2d(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def correlation_coefficient(cov):
    s = np.diag(1/np.sqrt(np.diag(cov)))
    return s @ cov @ s


def condition_number(M):
    u, s, v = np.linalg.svd(M)
    return np.max(s)/np.min(s)

def format_value_pm_error(val, err=None, precision=1, width=3):
    # Get exponent of value
    e = int(np.floor(np.log10(val)))
    val_fmt = "{val:>{width}.{precision}f}"
    err_fmt = "±{err:>{width}.{precision}f}"
    if abs(e) > 1:
        val = val/10**e
        if err is not None:
            err = err/10**e

    s = val_fmt.format(val=val, width=width, precision=precision)
    if err is not None:
        s += err_fmt.format(err=err, width=width, precision=precision)
    else:
        s += " "*(width+1)
        
    if abs(e) > 1:
        s += " 10^{exp:d}".format(exp=e)
        
    return s
