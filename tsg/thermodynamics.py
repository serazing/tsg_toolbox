import xarray as xr
import gsw 


def compute_buoyancy(data, temperature_var='SST', salinity_var='SSS'):
    """
    Compute buoyancy from a dataset containing salinity and temperature 
    variables
    
    Parameters
    ----------
    data : xarray.Dataset
        A Dataset containing salinity and temperature variables
    temperature_var : str, optional
        The name of the temperature variable. Default is "SST".
    salinity_var : str, optional
        The name of the salinity variable Default is "SSS"
    
    Returns
    -------
    res : xarray.Dataset
        A Dataset similar to the input but with a new variable 
        "buoyancy"     
    """
    
    p = 0 * data['lon']
    SA = gsw.SA_from_SP(data[salinity_var], p, data['lon'], data['lat'])
    rho = gsw.rho_t_exact(SA, data[temperature_var], p)
    b = 9.81 * (1 - rho / 1025.)
    return data.assign(buoyancy=xr.DataArray(b, dims='time'))


def compute_alpha_beta(data, temperature_var='SST', salinity_var='SSS'):
    p = 0 * data['lon']
    SA = gsw.SA_from_SP(data[salinity_var], p, data['lon'], data['lat'])
    CT = gsw.CT_from_t(SA, data[temperature_var], p)
    rho, alpha, beta = gsw.rho_alpha_beta(SA, CT, p)
    return data.assign(alpha=xr.DataArray(alpha, dims='time'),
                       beta=xr.DataArray(beta, dims='time'))