import numpy as np
import astropy
import scipy.ndimage
import astropy.io.fits

import utils

pi = np.pi

def bin_C_ell(C_ell, ell, ell_1, ell_2, weighted=False):
    mask = np.logical_and(ell >= ell_1, ell < ell_2)
    if weighted:
        # TODO weighted variance
        return np.sum((ell*C_ell)[mask])/np.sum(ell[mask]), np.sum(ell[mask]**2)/np.sum(ell[mask]), np.var(C_ell[mask])/np.count_nonzero(mask)
    else:
        return np.mean(C_ell[mask]), np.mean(ell[mask]), np.var(C_ell[mask])/np.count_nonzero(mask)

def create_Gaussian_field(Cl, shape, box_size, mean=0):
    """Creates a random Gaussian fied.

    Required arguments:
    Cl              Callable f(ell) returning the power spectrum for multipole
                    ell.
    shape           Tuple (N, M) of the requested output size.
    box_size        Tuple (L1, L2) of the physical dimension of the created 
                    field. The units have to be consistent with those returned
                    by Cl.

    Optional arguments:
    mean            Mean of the created field. Default is 0.

    Returns:
    Array m with m.shape == shape.
    """

    ell_x_min_box = 2.0*pi/box_size[0]
    ell_y_min_box = 2.0*pi/box_size[1]
    ell_x = np.fft.fftfreq(shape[0], d=1.0/shape[0])*ell_x_min_box
    ell_y = np.fft.rfftfreq(shape[1], d=1.0/shape[1])*ell_y_min_box
    x_idx, y_idx = np.meshgrid(ell_x, ell_y, indexing="ij")
    ell_grid = np.sqrt((x_idx)**2 + (y_idx)**2)
    
    Cl_grid = np.zeros_like(ell_grid)
    Cl_grid[ell_grid != 0] = Cl(ell_grid[ell_grid != 0])
    #if np.any(Cl_grid <= 0):
    #    m_ft = np.zeros(ell_grid.shape, dtype=np.complex64)
    #    m_ft[Cl_grid>0] = np.random.rayleigh(scale=np.sqrt((shape[0]/box_size[0])*(shape[1]/box_size[1])*shape[0]*shape[1]*Cl_grid[Cl_grid>0]/2))*np.exp(2j*pi*np.random.random(ell_grid.shape)[Cl_grid>0])
    #else:
    m_ft = np.random.rayleigh(scale=np.sqrt((shape[0]/box_size[0])*(shape[1]/box_size[1])*shape[0]*shape[1]*Cl_grid/2))*np.exp(2j*pi*np.random.random(ell_grid.shape))
    m_ft[ell_grid == 0] = mean
    
    m = np.fft.irfft2(m_ft)
    return m

def calculate_pseudo_Cl(map1, map2, box_size, n_bin=None, ell_min=None, ell_max=None, logspaced=False):
    """Estimates the cross-power spectrum of two maps.
    
    Required arguments:
    map1            Array of size (N, M).
    map2            Array of same shape as map1.
    box_size        Physical size (L1, L2) of the maps.
    
    Optional arguments:
    n_bin           Number of ell bins. If None, no binning is performed.
    ell_min         Minimum ell.
    ell_max,        Maximum ell.
    logspaced       Log-spaced bins. Default is False.
    
    Returns:
    Tuple (pCl_real, pCl_real_err, ell_mean, bin_edges, n_mode) with
        pCl_real        Estimated cross-power spectrum,
        pCl_real_err    Error on the mean, estimated from the scatter of the 
                        individual modes,
        ell_mean        Mean ell per bin,
        bin_edges       Edges of the ell bins,
        n_mode          Number of modes per bin.
    """

    if map1.shape != map2.shape:
        raise ValueError("Map dimensions don't match: {}x{} vs {}x{}".format(*(map1.shape + map2.shape)))
    
    # This can be streamlined alot
    map1_ft = np.fft.fft2(map1) * (box_size[0]/map1.shape[0])*(box_size[1]/map1.shape[1])
    map1_ft = np.fft.fftshift(map1_ft, axes=0)
    map2_ft = np.fft.fft2(map2) * (box_size[0]/map1.shape[0])*(box_size[1]/map1.shape[1])
    map2_ft = np.fft.fftshift(map2_ft, axes=0)
    map_1x2_ft = map1_ft.conj()*map2_ft

    ell_x_min_box = 2.0*pi/box_size[0]
    ell_y_min_box = 2.0*pi/box_size[1]
    ell_x = np.fft.fftshift(np.fft.fftfreq(map1.shape[0], d=1.0/map1.shape[0]))*ell_x_min_box
    ell_y = np.fft.fftfreq(map1.shape[1], d=1.0/map1.shape[1])*ell_y_min_box
    x_idx, y_idx = np.meshgrid(ell_x, ell_y, indexing="ij")
    ell_grid = np.sqrt((x_idx)**2 + (y_idx)**2)
    
    if n_bin==None:
        bin_edges = np.arange(start=np.min([ell_x_min_box, ell_y_min_box])/1.00001, stop=np.max(ell_grid), step=np.min([ell_x_min_box, ell_y_min_box]))
        n_bin = len(bin_edges) - 1
    else:
        if ell_max > np.max(ell_grid):
            raise RuntimeWarning("Maximum ell is {}, where as ell_max was set as {}.".format(np.max(ell_grid), ell_max))
        if ell_min < np.min([ell_x_min_box, ell_y_min_box]):
            raise RuntimeWarning("Minimum ell is {}, where as ell_min was set as {}.".format(np.min([ell_x_min_box, ell_y_min_box]), ell_min))
        if logspaced:
            bin_edges = np.logspace(np.log10(ell_min), np.log10(ell_max), n_bin+1, endpoint=True)
        else:
            bin_edges = np.linspace(ell_min, ell_max, n_bin+1, endpoint=True)

    pCl_real = np.zeros(n_bin)
    pCl_imag = np.zeros(n_bin)
    pCl_real_err = np.zeros(n_bin)
    pCl_imag_err = np.zeros(n_bin)
    ell_mean = np.zeros(n_bin)
    n_mode = np.zeros(n_bin)
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
    
    ell_sort_idx = np.argsort(ell_grid.flatten())
    map_1x2_ft_sorted = map_1x2_ft.flatten()[ell_sort_idx]
    ell_grid_sorted = ell_grid.flatten()[ell_sort_idx]
    bin_idx = np.searchsorted(ell_grid_sorted, bin_edges)
    
    for i in range(n_bin):
        P = map_1x2_ft_sorted[bin_idx[i]:bin_idx[i+1]]/(box_size[0]*box_size[1])
        ell = ell_grid_sorted[bin_idx[i]:bin_idx[i+1]]
        pCl_real[i] = np.mean(P.real)
        pCl_imag[i] = np.mean(P.imag)
        pCl_real_err[i] = np.sqrt(np.var(P.real)/len(P))
        pCl_imag_err[i] = np.sqrt(np.var(P.imag)/len(P))
        ell_mean[i] = np.mean(ell)
        n_mode = len(P)
    
    return pCl_real, pCl_real_err, ell_mean, bin_edges, n_mode

def calculate_shear_Cl_corrections(Cl, ell, theta, theta_min, theta_max, pixel_size):
    xi_pCl_func = lambda theta: np.trapz(ell*scipy.special.jn(2, ell*theta)*Cl, ell)/(2.0*pi)
    xi_pCl = np.zeros_like(theta)
    for i in range(xi_pCl.size):
        xi_pCl[i] = xi_pCl_func(theta[i]/180*pi)
        
    theta_min_idx = np.searchsorted(theta, theta_min)
    theta_max_idx = np.searchsorted(theta, theta_max)

    Cl_low_correction_func = lambda ell: np.trapz((theta/180*pi*xi_pCl)[:theta_min_idx]*scipy.special.jn(2, ell*theta[:theta_min_idx]/180*pi), theta[:theta_min_idx]/180*pi)*2.0*pi
    Cl_high_correction_func = lambda ell: np.trapz((theta/180*pi*xi_pCl)[theta_max_idx:]*scipy.special.jn(2, ell*theta[theta_max_idx:]/180*pi), theta[theta_max_idx:]/180*pi)*2.0*pi
    Cl_low_correction = np.zeros_like(Cl)
    Cl_high_correction = np.zeros_like(Cl)
    Cl_pixel_correction = (np.sin(ell*pixel_size/2/180*pi)/(ell*pixel_size/2/180*pi))**2

    for i in range(Cl.size):
        Cl_low_correction[i] = Cl_low_correction_func(ell[i])
        Cl_high_correction[i] = Cl_high_correction_func(ell[i])

    return xi_pCl, Cl_low_correction, Cl_high_correction, Cl_pixel_correction

def process_map(m, processes, map_size, verbose=False):
    """Process an input map.

    Required arguments:
    m               Input map.
    processes       List of dicts describing the processing. 
    map_size        Physical size of the map.

    Optional Arguments:
    verbose         Verbosity. Default is False.

    Returns:
    Tuple (m, size) with m the processed map and size the new physical size.

    The dicts used for specifying the processing have the common key "type",
    which specifies the processing to be done. Supported processes are
        "set-const"         Sets the map to a constant value. Required field:
                            "value" (float).
        "crop"              Crops the map. Required field: "slice" (slice).
        "gaussian_noise"    Adds random Gaussian noise to the map. Required
                            field: "spectrum" (callable).
        "smoothing"         Smoothes the map with a gaussian kernel. Required 
                            field: "sigma" (float). Optional field: "mode" 
                            (default "wrap).
        "zoom"              Resamples the map. Required field: "zoom_factor" 
                            (float).
        "poisson_noise"     Add Poisson noise to the map. Required field: 
                            "lambda" (float).
        "scale"             Rescale the map by a constant. Required field:
                            "normalization" (float).
    """
    new_map = np.copy(m)
    new_map_size = map_size
    map_shape = m.shape
    pixel_size = map_size[0]/map_shape[0]
    for process in processes:
        # Set to constant
        if process["type"] == "set-const":
            new_map = np.ones_like(m)*process["value"]
        # Crop
        if process["type"] == "crop":
            new_map = new_map[process["slice"], process["slice"]]
            if verbose: print("Cropping: {} -> {}".format(map_shape, new_map.shape))
            map_shape = new_map.shape
            new_map_size = pixel_size*new_map.shape[0], pixel_size*new_map.shape[1]
        # Add Gaussian noise
        if process["type"] == "gaussian_noise":
            if verbose: print("Adding Gaussian noise.")
            new_map += create_Gaussian_field(process["spectrum"], new_map.shape, map_size)
        # Smooth maps
        if process["type"] == "smoothing":
            mode = "wrap" if not "mode" in process else process["mode"]
            new_map = scipy.ndimage.gaussian_filter(new_map, sigma=process["sigma"]/pixel_size, mode=mode)
            if verbose: print("Smoothing: sigma = {}, mode = {}.".format(process["sigma"], mode))
        # Downsample maps
        if process["type"] == "zoom":
            new_map = utils.rebin_2d(new_map, 
                                    (int(new_map.shape[0]*process["zoom_factor"]), 
                                     int(new_map.shape[1]*process["zoom_factor"])))
            if verbose: print("Zoom: {} -> {}".format(map_shape, new_map.shape))
            map_shape = new_map.shape
            pixel_size /= process["zoom_factor"]
        #Add poisson noise
        if process["type"] == "poisson_noise":
            if verbose: print("Adding Poisson noise.")
            m_min = np.min(new_map)
            new_map = np.random.poisson((new_map-m_min)*process["lambda"], new_map.shape)
            new_map = 1.0*new_map/process["lambda"] + m_min
        if process["type"] == "scale":
            if verbose: print("Rescale: noralization = {}".format(process["normalization"]))
            new_map *= process["normalization"]
    return new_map, new_map_size

class Pseudo_Cl(object):
    def __init__(self, map1_paths, map2_paths, map_size, map_shape, n_LOS, 
                 ell_min=None, ell_max=None, n_bin_Cl=None, logspaced_Cl=False, 
                 verbose=True, 
                 map1_processing=[], map2_processing=[], 
                 filetype="fits", fits_hdu=0):
        self.map_size = (map_size[0]/180*pi, map_size[1]/180*pi)
        self.map_shape = map_shape
        self.pixel_size = self.map_size[0]/self.map_shape[0]
        if self.map_size[1]/self.map_shape[1] != self.pixel_size:
            print("Warning: non-square pixels.")
        
        self.n_LOS = n_LOS
        
        self.ell_min = ell_min
        self.ell_max = ell_max
        self.n_bin_Cl = n_bin_Cl
        self.logspaced_Cl = logspaced_Cl

        self.Cl = []
        self.Cl_scatter = []
        self.Cl_imag = []
        self.Cl_imag_scatter = []
        
        self.verbose = verbose
        
        if filetype != "fits" and filetype != "raw":
            raise NotImpemented("Filetype not supported.")
        
        if verbose: print("Computing Cl.")
        self.compute_Cl(map1_paths, map2_paths, 
                        map1_processing, map2_processing,
                        filetype, fits_hdu, verbose)
        if verbose: print("Computing band Cl.")
        self.compute_Cl_band()

    def compute_Cl(self, map1_paths, map2_paths, 
                         map1_processing, map2_processing,
                         filetype, fits_hdu, verbose):
        for i in range(self.n_LOS):
            if self.verbose: print("LOS {}".format(i+1))
            filename1 = map1_paths[i]
            if filetype == "fits":
                hdu = astropy.io.fits.open(filename1)
                map1 = hdu[fits_hdu].data
                hdu.close()
            elif filetype == "raw":
                map1 = np.fromfile(filename1, dtype=np.float32).reshape(*self.map_shape)
            if map2_paths != None:
                filename2 = map2_paths[i]
                if filetype == "fits":
                    hdu = astropy.io.fits.open(filename2)
                    map2 = hdu[fits_hdu].data
                    hdu.close()
                elif filetype == "raw":
                    map2 = np.fromfile(filename2, dtype=np.float32).reshape(*self.map_shape)
            
            map1, map1_size = process_map(map1, map1_processing, self.map_size, verbose)
            if map2_paths == None:
                map2 = map1
            else:
                map2, map2_size = process_map(map2, map2_processing, self.map_size, verbose)
                if map1.shape != map2.shape or map1_size != map2_size:
                    raise RuntimeError("Map shapes or sizes do not match: shapes = {}, {}; sizes = {}, {}.".format(map1.shape, map2.shape, map1_size, map2_size))
                
            tmp_Cl, tmp_Cl_err, self.mean_ell, self.ell_bin_edges, _ = calculate_pseudo_Cl(map1, map2, self.map_size, 
                                                                                                                 self.n_bin_Cl, ell_min=self.ell_min, ell_max=self.ell_max, logspaced=self.logspaced_Cl)
            if self.n_bin_Cl == None:
                self.n_bin_Cl = len(self.ell_bin_centers)
                self.ell_min = self.ell_bin_edges[0]
                self.ell_max = self.ell_bin_edges[-1]
                
            self.Cl.append(tmp_Cl)
            self.Cl_scatter.append(tmp_Cl_err)

        self.Cl = np.array(self.Cl)
        self.Cl_scatter = np.array(self.Cl_scatter)
        self.Cl_mean = np.mean(self.Cl, axis=0)
        self.Cl_error = np.sqrt(np.var(self.Cl, axis=0)/self.n_LOS)
        self.Cl_cov = np.cov(self.Cl.T)
       
    def compute_Cl_band(self):
        self.Cl_band = np.zeros((self.n_LOS, self.n_bin_Cl))
        self.Cl_band_scatter = np.zeros((self.n_LOS, self.n_bin_Cl))
        for i in range(self.n_LOS):
            for j in range(self.n_bin_Cl):
                self.Cl_band[i,j], mean_ell, self.Cl_band_scatter[i,j] = bin_C_ell(self.Cl[i], self.mean_ell, self.ell_bin_edges[j], self.ell_bin_edges[j+1], weighted=True)
                        
        self.Cl_band_mean = np.mean(self.Cl_band, axis=0)
        self.Cl_band_error = np.sqrt(np.var(self.Cl_band, axis=0)/self.n_LOS)
        self.Cl_band_cov = np.cov(self.Cl_band.T)
        
    def compute_corrections(self, theta_min, theta_max, pixel_size, Cl=None, ell=None):
        if Cl==None and ell==None:
            Cl = self.Cl_mean
            ell = self.mean_ell
        self.theta_fine = np.logspace(-1, np.log10(self.map_size[0]/pi*180*3600*np.sqrt(2)), 400)/3600

        self.xi_pCl, self.Cl_low_correction, self.Cl_high_correction, self.Cl_pixel_correction = calculate_shear_Cl_corrections(Cl, ell, self.theta_fine, theta_min, theta_max, pixel_size)

        self.Cl_low_correction_band = np.zeros(self.n_bin_Cl)
        self.Cl_high_correction_band = np.zeros(self.n_bin_Cl)
        self.Cl_pixel_correction_band = np.zeros(self.n_bin_Cl)

        for i in range(self.n_bin_Cl):
            self.Cl_low_correction_band[i], _, _ = bin_C_ell(self.Cl_low_correction, ell, self.ell_bin_edges[i], self.ell_bin_edges[i+1])
            self.Cl_high_correction_band[i], _, _ = bin_C_ell(self.Cl_high_correction, ell, self.ell_bin_edges[i], self.ell_bin_edges[i+1])
            self.Cl_pixel_correction_band[i], _, _ = bin_C_ell(self.Cl_pixel_correction, ell, self.ell_bin_edges[i], self.ell_bin_edges[i+1])

