import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1

def subplot_colorbar(im, axes, **kwargs):
    """Adds colorbar to subplot."""
    cax = mpl_toolkits.axes_grid1.make_axes_locatable(axes).append_axes("right", size = "5%", pad = 0.05)
    plt.colorbar(im, cax=cax, **kwargs)