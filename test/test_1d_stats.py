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


def test_xi_estimators():
    L = 0.5
    n_grid = 1000
    
    P = lambda k: np.ones_like(k)    
    d = cosmotools.onedee.random_fields.create_gaussian_random_field(P, n_grid, L)
    
    r_max = L/100
    
    xi1, r = cosmotools.onedee.stats.correlation_function(d, L)
    xi1 = xi1[r < r_max]
    xi2_marks, xi2_w = cosmotools.onedee.stats.correlation_function_marks(d, L, x_max=r_max)
    xi2 = np.average(xi2_marks, weights=xi2_w, axis=0)
    assert np.allclose(xi1, xi2)
    
    sigma = 1.3
    beta = 5
    k0 = 150
    P = lambda k: np.where(k != 0, k/(1+k/k0)**(beta+1), sigma)
    d = cosmotools.onedee.random_fields.create_gaussian_random_field(P, n_grid, L)
    
    w = np.ones(n_grid)
    w[:100] = 0
    w[-100:] = 0
    w[np.random.choice(n_grid, int(n_grid/4), replace=False)] = 0

    r_max = L/100
    
    xi1, r = cosmotools.onedee.stats.correlation_function(d, L, weights=w)
    xi1 = xi1[r < r_max]
    r = r[r < r_max]
    xi2_marks, xi2_w = cosmotools.onedee.stats.correlation_function_marks(d, L, x_max=r_max, weights=w)
    xi2 = np.average(xi2_marks, weights=xi2_w, axis=0)
    assert np.allclose(xi1, xi2)
    
def test_xcorr_xi_estimators():
    from cosmotools.onedee.stats import bin_data
    def correlation_function_fft(d, L, bins=None, weights=None):
        if weights is None:
            d_ft = np.fft.rfft(d)
            P_k = d_ft*np.conj(d_ft)
            xi_raw = np.fft.irfft(P_k)
            xi_raw /= len(xi_raw)
            x = np.arange(len(xi_raw))*L/len(d)

            if bins is None:
                return xi_raw, x
            else:
                return bin_data(x=x, y=xi_raw, bin_edges=bins, normalize=True)
        else:
            w_ft = np.fft.rfft(weights)
            w_k = w_ft*np.conj(w_ft)
            xi_w_raw = np.fft.irfft(w_k)

            d_ft = np.fft.rfft(d*weights)
            P_k = d_ft*np.conj(d_ft)
            xi_raw = np.fft.irfft(P_k)
            x = np.arange(len(xi_raw))*L/len(d)

            if bins is None:
                return xi_raw/xi_w_raw, x
            else:
                xi, _ = bin_data(x=x, y=xi_raw, bin_edges=bins, normalize=False)
                xi_w, _ = bin_data(x=x, y=xi_w_raw, bin_edges=bins, normalize=False)

                xi[xi_w != 0] /= xi_w[xi_w != 0]

                x, _ = bin_data(x=x, y=x, bin_edges=bins, weights=xi_w_raw, normalize=True)

                return xi, x

    n_grid = 1000
    L = 1.35
    
    sigma = 1.3
    beta = 5
    k0 = 150
    P = lambda k: np.where(k != 0, k/(1+k/k0)**(beta+1), sigma)
    d = cosmotools.onedee.random_fields.create_gaussian_random_field(P, n_grid, L)
    
    w = np.ones(n_grid)
    w[:100] = 0
    w[-100:] = 0
    w[np.random.choice(n_grid, int(n_grid/2), replace=False)] = 0

    
    xi1, r = correlation_function_fft(d, L, weights=w)
    xi2, r = cosmotools.onedee.stats.correlation_function(d, L, weights=w)
    assert np.allclose(xi1, xi2)
        
    r_max = L/100
    xi1_marks, xi1_w = cosmotools.onedee.stats.correlation_function_marks(d, L, x_max=r_max, weights=w)
    xi1 = np.average(xi1_marks, weights=xi1_w, axis=0)
    xi2_marks, xi2_w = cosmotools.onedee.stats.cross_correlation_function_marks(d, d, L, x_max=r_max, weights1=w, weights2=w)
    xi2 = np.average(xi2_marks, weights=xi2_w, axis=0)
    assert np.allclose(xi1, xi2)
            
    n_grid = 1000
    L = 1.0
    sigma = 1.3
    beta = 5
    k0 = 150
    P_AA = lambda k: np.where(k != 0, k/(1+k/k0)**(beta+1), sigma)
    P_BB = lambda k: np.ones_like(k)
    rho = 0.8
    P_AB = lambda k: np.sqrt(P_AA(k)*P_BB(k))*rho
    
    d1, d2 = cosmotools.onedee.random_fields.create_gaussian_random_fields([P_AA, P_BB, P_AB], n_grid, L)
    
    w1 = np.ones(n_grid)
    w1[:100] = 0
    w1[-100:] = 0
    w1[np.random.choice(n_grid, int(n_grid/2), replace=False)] = 0
    
    w2 = np.ones(n_grid)
    w2[np.random.choice(n_grid, int(n_grid/20), replace=False)] = 0
    
    r_max = L/20
    xi1, r = cosmotools.onedee.stats.cross_correlation_function(d1, d2, L, weights1=w1, weights2=w2)
    xi1 = xi1[r < r_max]
    r = r[r < r_max]
    xi2_marks, xi2_w = cosmotools.onedee.stats.cross_correlation_function_marks(d1, d2, L, x_max=r_max, weights1=w1, weights2=w2)
    xi2 = np.average(xi2_marks, weights=xi2_w, axis=0)
    assert np.allclose(xi1, xi2)

def test_xi_estimator_distribution(verbose=False, plot=False):
    np.random.seed(42)

    n_grid = 400
    L = 1.0
    sigma = 1.3
    beta = 5
    k0 = 150
    P = lambda k: np.where(k != 0, k/(1+k/k0)**(beta+1), sigma)
        
    w = np.ones(n_grid)    

    w[:n_grid//10] = 0
    w[-n_grid//10:] = 0
    w[np.random.choice(n_grid, int(n_grid/4), replace=False)] = 0
    
    r_max_idx = n_grid
    n_r = 40
    xis = np.zeros((n_r, n_r, r_max_idx))
    for i in range(n_r):
        for j in range(n_r):
            d = cosmotools.onedee.random_fields.create_gaussian_random_field(P, n_grid, L)
            _xi, r = cosmotools.onedee.stats.correlation_function(d, L, weights=w)
            xis[i, j] = _xi[:r_max_idx]    
    
    xi_var = xis.var(axis=1, ddof=1)
    
    r = r[:r_max_idx]
    
    xi_truth = cosmotools.onedee.stats.xi_analytic(P, L, n_grid)
    xi_var_truth = cosmotools.onedee.stats.var_mean_xi_L_analytic(L, n_grid, f=1, r=r, xi=xi_truth, weights=w)
    
    xi_truth = xi_truth[:r_max_idx]
    
    std = 3
    tolerance_xi = 0.01
    tolerance_xi_var = 0.01
    
    outliers_xi = np.abs(xi_truth - xis.mean(axis=(0,1)))/np.sqrt(xis.var(axis=(0,1))/n_r**2) > std
    if verbose: print("Number of xi points more than {} std away: {}".format(std, np.count_nonzero(outliers_xi)))
    assert np.count_nonzero(outliers_xi) < n_grid*tolerance_xi
    
    outliers_xi_var = np.abs(xi_var_truth - xi_var.mean(axis=0))/np.sqrt(xi_var.var(axis=0, ddof=1)/n_r) > std
    if verbose: print("Number of var of xi points more than {} std away: {}".format(std, np.count_nonzero(outliers_xi_var)))
    assert np.count_nonzero(outliers_xi_var) < n_grid*tolerance_xi_var
    
    if plot:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        fig.subplots_adjust(wspace=0.3)
        
        ax[0].plot(r, xis.reshape(n_r**2, -1).T, lw=0.5, color="C0", alpha=0.05, zorder=-1)
        ax[0].plot(r, xis.mean(axis=(0,1)), c="C0", label="Simulation", zorder=10)
        ax[0].plot(r, xi_truth, c="C1", label="Truth")
        
        ax[0].fill_between(r, xi_truth-np.sqrt(xi_var_truth), 
                              xi_truth+np.sqrt(xi_var_truth), lw=0, color="C1", alpha=0.5)
        ax[0].legend(fontsize="small")
        ax[0].set_ylabel(r"$\xi(r)$")
        ax[0].set_xlabel(r"$r$")
        
        ax[1].fill_between(r, xi_var.mean(axis=0) - xi_var.std(axis=0, ddof=1)/np.sqrt(n_r),
                              xi_var.mean(axis=0) + xi_var.std(axis=0, ddof=1)/np.sqrt(n_r), 
                           lw=0, color="C0", alpha=0.5, label="Simulation")
        
        ax[1].plot(r, xi_var_truth, c="C1", label="Truth")
        
        ax[1].legend(fontsize="small")
        ax[1].set_ylabel(r"Var[$\xi(r)$]")
        ax[1].set_xlabel(r"$r$")
        
        fig.savefig("plots/1d_grf_xi.png")

def test_xcorr_xi_estimator_distribution(verbose=False, plot=False):
    np.random.seed(42)

    n_grid = 400
    L = 1.0
    sigma = 1.3
    beta = 5
    k0 = 150
    P_AA = lambda k: np.where(k != 0, k/(1+k/k0)**(beta+1), sigma)
    P_BB = lambda k: np.ones_like(k)
    rho = 0.8
    P_AB = lambda k: np.sqrt(P_AA(k)*P_BB(k))*rho
        
    w1 = np.ones(n_grid)    
    w2 = np.ones(n_grid)

    w1[:n_grid//10] = 0
    w1[-n_grid//10:] = 0
    w1[np.random.choice(n_grid, int(n_grid/4), replace=False)] = 0
    w2 = w1
#     w2[np.random.choice(n_grid, int(n_grid/20), replace=False)] = 0
    
    r_max_idx = n_grid
    n_r = 40
    xis = np.zeros((n_r, n_r, r_max_idx))
    for i in range(n_r):
        for j in range(n_r):
            d1, d2 = cosmotools.onedee.random_fields.create_gaussian_random_fields([P_AA, P_BB, P_AB], n_grid, L)
            _xi, r = cosmotools.onedee.stats.cross_correlation_function(d1, d2, L, weights1=w1, weights2=w2)
            xis[i, j] = _xi[:r_max_idx]    
    
    xi_var = xis.var(axis=1, ddof=1)
    
    r = r[:r_max_idx]
    
    xi_truth = cosmotools.onedee.stats.xi_analytic(P_AB, L, n_grid)
    xi_var_truth = cosmotools.onedee.stats.cov_mean_xcorr_xi_L_analytic(L, n_grid, f=1, r1=r, r2=r, 
                                                                        power_spectra=[P_AA, P_BB, P_AB],
                                                                        weights1=w1, weights2=w2)
    
    xi_truth = xi_truth[:r_max_idx]
    
    std = 3
    tolerance_xi = 0.01
    tolerance_xi_var = 0.01
    
    outliers_xi = np.abs(xi_truth - xis.mean(axis=(0,1)))/np.sqrt(xis.var(axis=(0,1))/n_r**2) > std
    if verbose: print("Number of xi points more than {} std away: {}".format(std, np.count_nonzero(outliers_xi)))
    assert np.count_nonzero(outliers_xi) < n_grid*tolerance_xi
    
    outliers_xi_var = np.abs(xi_var_truth - xi_var.mean(axis=0))/np.sqrt(xi_var.var(axis=0, ddof=1)/n_r) > std
    if verbose: print("Number of var of xi points more than {} std away: {}".format(std, np.count_nonzero(outliers_xi_var)))
    assert np.count_nonzero(outliers_xi_var) < n_grid*tolerance_xi_var

    if plot:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        fig.subplots_adjust(wspace=0.3)
        
        ax[0].plot(r, xis.reshape(n_r**2, -1).T, lw=0.5, color="C0", alpha=0.05, zorder=-1)
        ax[0].plot(r, xis.mean(axis=(0,1)), c="C0", label="Simulation", zorder=10)
        ax[0].plot(r, xi_truth, c="C1", label="Truth")
        
        ax[0].fill_between(r, xi_truth-np.sqrt(xi_var_truth), 
                              xi_truth+np.sqrt(xi_var_truth), lw=0, color="C1", alpha=0.5)
        ax[0].legend(fontsize="small")
        ax[0].set_ylabel(r"$\xi(r)$")
        ax[0].set_xlabel(r"$r$")
        
        ax[1].fill_between(r, xi_var.mean(axis=0) - xi_var.std(axis=0, ddof=1)/np.sqrt(n_r),
                              xi_var.mean(axis=0) + xi_var.std(axis=0, ddof=1)/np.sqrt(n_r), 
                           lw=0, color="C0", alpha=0.5, label="Simulation")
        
        ax[1].plot(r, xi_var_truth, c="C1", label="Truth")
        
        ax[1].legend(fontsize="small")
        ax[1].set_ylabel(r"Var[$\xi(r)$]")
        ax[1].set_xlabel(r"$r$")

        fig.savefig("plots/1d_grf_xcorr_xi.png")