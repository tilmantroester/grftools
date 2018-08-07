import numpy as np

import cosmotools.onedee.stats

def _find_bin_np(x, bin_edges):
    if x < bin_edges[0] or x >= bin_edges[-1]:
        return -1
    return np.searchsorted(bin_edges, x, side="right") - 1

def _bin_data_np(x, y, bin_edges, weights=None, normalize=True):
    bins = np.zeros(len(bin_edges)+1)
    mean_x = np.zeros_like(bins)
    n = np.zeros_like(bins)

    w = np.ones_like(x) if weights is None else weights

    idx = np.digitize(x, bin_edges, right=False)
    np.add.at(bins, idx, w*y)
    np.add.at(mean_x, idx, w*x)
    np.add.at(n, idx, w)

    bins = bins[1:-1]
    mean_x = mean_x[1:-1]
    n = n[1:-1]

    # for i in range(len(x)):
    #     idx = find_bin_np(x[i], bin_edges)
    #     if not idx == -1:
    #         bins[idx] += y[i]
    #         n[idx] += 1
    mean_x[n==0] = ((bin_edges[:-1]+bin_edges[1:])/2)[n==0]
    if normalize:
        bins[n!=0] /= n[n!=0]
        mean_x[n!=0] /= n[n!=0]

    return bins, mean_x

def test_find_bin(n_sample=10000, n_bin=1000, seed=234, timing=False):
    np.random.seed(seed)
    x = np.random.rand(n_sample)
    bin_edges = np.linspace(0, 1, n_bin, endpoint=True)

    bins = np.zeros(n_sample)
    bins_np = np.zeros(n_sample)

    for i in range(n_sample):
        bins[i] = cosmotools.onedee.stats.find_bin(x[i], bin_edges)
        bins_np[i] = _find_bin_np(x[i], bin_edges)

    assert np.allclose(bins, bins_np)

    if timing:
        import timeit
        namespace = {"cosmotools" : cosmotools, "np" : np ,"_find_bin_np" : _find_bin_np, "bin_edges" : bin_edges}
        print("Timing find_bin")
        t = timeit.repeat("cosmotools.onedee.stats.find_bin(np.random.rand(1).squeeze(), bin_edges)", 
                        repeat=3, number=n_sample, globals=namespace)
        t_np = timeit.repeat("_find_bin_np(np.random.rand(1).squeeze(), bin_edges)", 
                            repeat=3, number=n_sample, globals=namespace)
        print("Numba: {} s".format(min(t)))
        print("Numpy: {} s".format(min(t_np)))

    

def test_bin_data(n_sample=1000, n_bin=10, seed=234, verbose=False, timing=False):
    bin_edges = np.linspace(0, 1, n_bin, endpoint=True)
    # Test left/right boundaries
    print("Test bin edges.")
    x = bin_edges
    y = np.random.rand(len(bin_edges))
    binned, mean_x = cosmotools.onedee.stats.bin_data(x, y, bin_edges)
    binned_np, mean_x_np = _bin_data_np(x, y, bin_edges)

    assert np.allclose(binned, binned_np)
    assert np.allclose(mean_x, mean_x_np)
    
    np.random.seed(seed)
    if verbose: print("Test with random data.")
    x = np.random.rand(n_sample)
    y = np.random.rand(n_sample)
    binned, mean_x = cosmotools.onedee.stats.bin_data(x, y, bin_edges)
    binned_np, mean_x_np = _bin_data_np(x, y, bin_edges)

    assert np.allclose(binned, binned_np)
    assert np.allclose(mean_x, mean_x_np)

    if verbose: print("Test with weights.")
    x = np.random.rand(n_sample)
    y = np.random.rand(n_sample)
    w = np.random.rand(n_sample)
    binned, mean_x = cosmotools.onedee.stats.bin_data(x, y, bin_edges, weights=w)
    binned_np, mean_x_np = _bin_data_np(x, y, bin_edges, weights=w)

    assert np.allclose(binned, binned_np)
    assert np.allclose(mean_x, mean_x_np)

    if timing:
        import timeit
        namespace = {"cosmotools" : cosmotools, "np" : np, "_bin_data_np" : _bin_data_np, "x" : x, "y" : y, "bin_edges" : bin_edges}
        print("Timing bin_data")
        t = timeit.repeat("cosmotools.onedee.stats.bin_data(x, y, bin_edges)", 
                        repeat=3, number=n_sample, globals=namespace)
        t_np = timeit.repeat("_bin_data_np(x, y, bin_edges)", 
                            repeat=3, number=n_sample, globals=namespace)
        print("Numba: {} s".format(min(t)))
        print("Numpy: {} s".format(min(t_np)))