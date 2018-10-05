import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1

pi = np.pi

def subplot_colorbar(im, axes, **kwargs):
    """Adds colorbar to subplot."""
    cax = mpl_toolkits.axes_grid1.make_axes_locatable(axes).append_axes("right", size = "5%", pad = 0.05)
    plt.colorbar(im, cax=cax, **kwargs)

def plot_power_spectrum(ax, modes, spectrum, 
                        label="", 
                        dimensionless=True, mode="Pofk", n_dim=3, 
                        x_units=None, y_units=None, h_factors=True,
                        x_axis_label=True, y_axis_label=True,
                        x_scale="log", y_scale="log",
                        **kwargs):
    """Plot a power spectrum.
    
    Arguments
    ---------
    ax : matplotlib.Axes
        Axes to be used for plotting.
    modes : numpy.array:
        k or ell modes of the power spectrum ``spectrum``.
    spectrum : numpy.array
        Power spectrum to be plotted.
    label : str, optional
        Plot label. (default "").
    dimensionless : bool, optional
        Plot the dimensionless power spectrum. (default True).
    mode : str, optional
        The kind of power spectrum. Valid values are ``"Cl"`` and ``"Pofk"``. 
        (default "Pofk").
    n_dim : int, optional
        Number of dimensions of the underlying field. (default 3).
    x_units : str, optional
        Units to be used in the x-label. (default None).
    y_units : str, optional
        Units to be used in the y-label. (default None).
    h_factors : bool, optional
        Use units of h Mpc^-1 for k units. (default True).
    x_axis_label : bool, optional
        Print x-axis label. (default True).
    y_axis_label : bool, optional
        Print y-axis label. (default True).
    x_scale : str, optional
        Scale of the x axis. (default "log).
    y_scale : str, optional
        Scale of the y axis. (default "log).
    kwargs : dict
        kwargs that are passed on the ``ax.plot``.
        
    The ``mode`` parameter decides on the normalization factor when using 
    dimensionless spectra, as well as the default unit label. 
    
    """
    
    
    if mode.lower() == "cl":
        x_label = r"$\ell$"
        if dimensionless:
            y_label = r"$\ell(\ell+1)/2\pi\ C_\ell$"
            u = modes*(modes+1)/(2*pi)
        else:
            y_label = r"$C_\ell$"
            u = 1
    elif mode.lower() == "pofk":
        x_label = r"$k$"
        if x_units is None:
            h_term = "$h$" if h_factors else ""
            x_units = "[" + h_term + " " + "Mpc$^{-1}$" + "]"
        x_label += " " + x_units
        if y_units is None:
            if h_factors:
                h_term = f"$h^{{-{n_dim}}}$"
            else:
                h_term = ""
            mpc_term = "Mpc" if n_dim == 1 else f"Mpc$^{{{n_dim}}}$"
            y_units = "[" + h_term + " " + mpc_term + "]"
            
        if dimensionless:
            y_label = r"$\Delta^2(k)$"
            u = modes**n_dim / (2*pi)**n_dim * 4*pi
        else:
            y_label = r"$P(k)$"
            y_label += " " + y_units
            u = 1
    else:
        raise ValueError(f"Mode {mode} not supported.")
    
    
    ax.plot(modes, u*spectrum, label=label, **kwargs)
    
    
    if x_axis_label:
        ax.set_xlabel(x_label)
    if y_axis_label:
        ax.set_ylabel(y_label)

    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)