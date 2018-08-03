import numpy as np
import scipy.stats

import cosmotools.onedee.random_fields

def test_gaussian_random_field_variance(plot=False):
    """Test the variance of generated Gaussian random fields against the analytic prediction."""

    sigma = 1.3
    beta = 5
    k0 = 150
    P = lambda k: np.piecewise(k, k != 0, [lambda k: k/(1+k/k0)**(beta+1), sigma])
    
    n_r = 100
    n_grid = 400
    L = 1.0
    
    P_data = []
    for i in range(n_r):
        d = cosmotools.onedee.random_fields.create_gaussian_random_field(P=P, n_grid=n_grid, L=L)
        tmp, k_data = cosmotools.onedee.random_fields.pseudo_Pofk(d, d, L)
        P_data.append(tmp)
        
    P_data = np.array(P_data)
    
    tolerance_mean = 0.05
    assert np.abs(np.mean(np.var(P_data, axis=0)/P(k_data)**2-1)) < tolerance_mean
    
    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.subplots_adjust(hspace=0)
        plt.suptitle("Var[P(k)]")

        plt.subplot(211)
        plt.loglog(k_data, np.var(P_data, axis=0), label="Data")
        plt.loglog(k_data, P(k_data)**2, label="Prediction")
        plt.ylabel("Var[P(k)]")
        plt.legend()

        plt.subplot(212)
        plt.semilogx(k_data, np.var(P_data, axis=0)/P(k_data)**2-1)
        plt.ylabel("Fractional difference")
        plt.ylim(-0.5, 0.5)
        plt.xlabel("k")
        plt.savefig("plots/1d_grf_pofk_variance_test.png")
    
    
def test_gaussian_random_field_generator(plot=False):
    """Test generated Gaussian random fields against the input power spectrum."""
    
    sigma = 1.3
    beta = 5
    k0 = 150
    P = lambda k: np.piecewise(k, k != 0, [lambda k: k/(1+k/k0)**(beta+1), sigma])
    
    n_grid = 500
    L = 1.0
    
    d = cosmotools.onedee.random_fields.create_gaussian_random_field(P=P, n_grid=n_grid, L=L)
    P_data, k_data = cosmotools.onedee.random_fields.pseudo_Pofk(d, d, L)
    P_data = P_data.real
    
    P_var = P(k_data)**2
    tolerace_p_value = 0.03
    assert scipy.stats.kstest(P_data.real/np.sqrt(P_var/4), scipy.stats.chi2(df=2).cdf)[1] > tolerace_p_value

    if plot:
        import matplotlib.pyplot as plt
        q = scipy.stats.chi2(df=2).ppf([0.16, 0.84])

        plt.figure()
        plt.subplots_adjust(hspace=0)
        plt.suptitle("P(k)")

        plt.subplot(211)
        plt.loglog(k_data, P_data, label="Data")
        plt.loglog(k_data, P(k_data), label="Prediction")
        plt.fill_between(k_data, q[0]*np.sqrt(P_var/4), q[1]*np.sqrt(P_var/4), facecolor="C1", alpha=0.5, label="68% CI")

        plt.ylabel("P(k)")
        plt.legend()

        plt.subplot(212)
        plt.semilogx(k_data, P_data/P(k_data)-1)
        plt.fill_between(k_data, q[0]*np.sqrt(P_var/4)/P(k_data)-1, q[1]*np.sqrt(P_var/4)/P(k_data)-1, facecolor="C1", alpha=0.5)
        plt.ylabel("Fractional difference")
        plt.xlabel("k")
        plt.savefig("plots/1d_grf_pofk_test.png")

def test_multiple_gaussian_random_field_generator(plot=False):
    """Test multiple, correlated generated Gaussian random fields against the input power spectra."""
    
    P = {}
    sigma = 1.3
    beta = 5
    k0 = 150
    rho = 0.5
    P["AA"] = lambda k: np.where(k != 0, k/(1+k/k0)**(beta+1), sigma)
    P["BB"] = lambda k: np.ones_like(k)*sigma/3
    P["AB"] = lambda k: np.sqrt(P["AA"](k)*P["BB"](k))*rho

    L = 1.0
    n_grid = 1000
    
    d1, d2 = cosmotools.onedee.random_fields.create_gaussian_random_fields(power_spectra=[P["AA"], P["BB"], P["AB"]], 
                                                                       n_grid=n_grid, L=L)
    P_data = {}
    P_data["AA"], k_data = cosmotools.onedee.random_fields.pseudo_Pofk(d1, d1, L)
    P_data["BB"], k_data = cosmotools.onedee.random_fields.pseudo_Pofk(d2, d2, L)
    P_data["AB"], k_data = cosmotools.onedee.random_fields.pseudo_Pofk(d1, d2, L)
    
    P_var = {}
    for key in ["AA", "BB", "AB"]:
        P_data[key] = P_data[key].real
        if key[0] == key[1]:
            tolerace_p_value = 0.03
            assert scipy.stats.kstest(P_data[key].real/(P[key](k_data)/2), scipy.stats.chi2(df=2).cdf)[1] > tolerace_p_value
        else:
            tolerace_p_value = 0.03
            cdf = lambda x: cosmotools.onedee.random_fields.variance_gamma_distribution_cdf(x, n=2, sigma1=1, sigma2=1, rho=rho)
            assert scipy.stats.kstest(P_data[key].real/(P[key](k_data)/2/rho), cdf)[1] > tolerace_p_value

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, len(P), figsize=(len(P)*4, 4))
        fig.subplots_adjust(hspace=0, wspace=0.3, bottom=0.2)
        
        for i, key in enumerate(["AA", "BB", "AB"]):
            if key[0] == key[1]:
                q = scipy.stats.chi2(df=2).ppf([0.16, 0.84])
            else:
                q = cosmotools.onedee.random_fields.variance_gamma_distribution_ppf([0.16, 0.84], n=2, rho=rho)
                
            ax[0,i].loglog(k_data, P_data[key], label="Data")
            ax[0,i].loglog(k_data, P[key](k_data), label="Prediction")
            ax[0,i].fill_between(k_data, q[0]*P[key](k_data)/2, q[1]*P[key](k_data)/2, facecolor="C1", alpha=0.5, label="68% CI")

            ax[0,i].set_ylabel("P(k)")
            ax[0,i].set_title(key)
            ax[0,i].legend(fontsize="small")

            ax[1,i].semilogx(k_data, P_data[key]/P[key](k_data)-1)
            ax[1,i].fill_between(k_data, q[0]/2-1, q[1]/2-1, facecolor="C1", alpha=0.5)
            ax[1,i].set_ylabel("Fractional\ndifference")
            ax[1,i].set_xlabel("k")
            
        fig.savefig("plots/1d_grf_multiple_pofk_test.png")

        # Plot distribution
        fig, ax = plt.subplots(1, len(P), figsize=(len(P)*4, 3))
        fig.subplots_adjust(hspace=0, wspace=0.3, bottom=0.2)
        
        for i, key in enumerate(["AA", "BB", "AB"]):
            if key[0] == key[1]:
                q = scipy.stats.chi2(df=2).ppf([0.16, 0.84])
                b = ax[i].hist(P_data[key]/(P[key](k_data)/2), bins=50, density=True, label="Data")
                x = np.linspace(b[1][0], b[1][-1], 100)
                ax[i].plot(x, scipy.stats.chi2(df=2).pdf(x), label=r"$\chi^2_{\nu=2}$")
                ax[i].axvline(q[0], ls="--", c="k", alpha=0.2, label="68% CI")
                ax[i].axvline(q[1], ls="--", c="k", alpha=0.2)
            else:
                q = cosmotools.onedee.random_fields.variance_gamma_distribution_ppf([0.16, 0.84], n=2, rho=rho)
                b = ax[i].hist((P_data[key]/(P[key](k_data)/2/rho)), bins=50, density=True, label="Data")
                x = np.linspace(b[1][0], b[1][-1], 100)
                distr = cosmotools.onedee.random_fields.variance_gamma_distribution(x, n=2, rho=rho)
                ax[i].plot(x, distr, label="Variance-gamma\ndistribution")
                ax[i].axvline(q[0], ls="--", c="k", alpha=0.2, label="68% CI")
                ax[i].axvline(q[1], ls="--", c="k", alpha=0.2)
                
            ax[i].legend(fontsize="small")
            ax[i].set_xlabel("P(k)")
            ax[i].set_ylabel("Probablilty")
            ax[i].set_title(key)

        fig.savefig("plots/1d_grf_multiple_pofk_distribution_test.png")

if __name__ == "__main__":
    test_gaussian_random_field_variance(plot=True)
    test_gaussian_random_field_generator(plot=True)
    test_multiple_gaussian_random_field_generator(plot=True)