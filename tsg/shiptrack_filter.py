import xarray as xr
import numpy as np
import numba
import pandas as pd


@numba.jit
def _shiptrack_filter(data, mask, time, lon, lat, cutoff=10e3, 
                      win_dt=1., max_break=0.5):
    """
    Private low-level function written using numba to speed-up the computation
    
    Parameters
    ----------
    data : 1darray
        Input observational data to filter
    t : 1darray
        Time record 
    lon : 1darray
        Longitudinal coordinates 
    lat : 1darray
        Latitudinal coordinates 
    cutoff : 1darray
        Cutoff wave period
    
    Returns
    -------
    data_meso : 1darray
        Data filtered to retain only mesoscale variations
    """
    def distance(lon1, lat1, lon2, lat2):
        EARTH_RADIUS = 6371 * 1e3
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        dy = np.pi / 180. * EARTH_RADIUS * dlat
        dx = (np.cos(np.pi / 180. * lat2) * np.pi / 180. * EARTH_RADIUS * dlon)
        distance = (dx ** 2 + dy ** 2 ) ** 0.5
        return distance
    
    cutoff /= (2 * np.pi) # Scale the cutoff as a wavelength
    nobs = len(data) # Total number of observation
    data_meso = np.zeros(nobs) # Initialize outputs
    max_break *= (3600 * 1e9)
    win_dt *= (3600 * 24 * 1e9)    
    # Loop on every data point
    for ic in range(1, nobs - 1):
        # Reference point at the center of the window
        t_ref, lon_ref, lat_ref = time[ic], lon[ic], lat[ic]
        # Next data point: r for "right"
        jr = ic + 1
        dist_r = distance(lon_ref, lat_ref, lon[jr], lat[jr])
        dt_r = abs(time[jr] - t_ref)
        # Previous data point: l for "left"
        jl = ic - 1         
        dist_l = distance(lon_ref, lat_ref, lon[jl], lat[jl])       
        dt_l = abs(time[jl] - t_ref)
        # Initialize convolution sums, weight sums and window size
        conv_r, conv_l, wsum_r, wsum_l = 0, 0, 0, 0
        nwin_r, nwin_l = 0, 0
        # Right side of the window
        while (dt_r < win_dt) and (jr < nobs - 1):
            w_r = np.exp(- dist_r ** 2 / (2. * cutoff ** 2)) # Gaussian weight
            conv_r += w_r * data[jr]
            wsum_r += w_r * mask[jr]
            nwin_r += 1
            # Go to next observation
            jr += 1
            dist_r = distance(lon_ref, lat_ref, lon[jr], lat[jr])
            dt_r = abs(time[jr] - t_ref)
        # Left side of the window    
        while (dt_l < win_dt) and (jl > 1):
            w_l = np.exp(- dist_l ** 2 / (2. * cutoff ** 2))  # Gaussian weight
            conv_l += w_l * data[jl]
            wsum_l += w_l * mask[jl]
            nwin_l += 1
            # Go to previous observation
            jl -= 1
            dist_l = distance(lon_ref, lat_ref, lon[jl], lat[jl])
            dt_l = abs(time[jl] - t_ref)
        # Make some tests to exclude unvalid data
        norm_factor = (wsum_l + mask[ic] + wsum_r)
        if ((nwin_r >= 5) and (nwin_l >= 5) and 
            (abs(time[ic] - time[ic + 1]) < max_break) and 
            (abs(time[ic] - time[ic - 1]) < max_break) and
            (norm_factor != 0.)
           ): 
            data_meso[ic] = (conv_r + data[ic] + conv_l) / norm_factor
        else:
            data_meso[ic] = np.nan
    return data_meso


def shiptrack_filter(data, cutoff=10e3, win_dt=1, max_break=0.5):
    """
    Perform a gaussian filter based on the local deformation radius to 
    remove scales smaller than mesoscale motions
    
    Parameters
    ----------
    data : xr.DataArray
        Input observational data to filter
    
    Returns
    -------
    data_meso : xr.DataArray
        Data filtered to retain only mesoscale variations
    """
    mask = 1. - np.isnan(data)
    data_filled = np.nan_to_num(data)
    res = _shiptrack_filter(data_filled, mask.data, pd.to_numeric(data['time'].data),
                            data['lon'].data, data['lat'].data, cutoff=cutoff,
                            win_dt=win_dt, max_break=max_break)
    data_meso = xr.DataArray(res, 
                             name=data.name + '_ME', 
                             dims=data.dims, 
                             coords=data.coords)
    return data_meso