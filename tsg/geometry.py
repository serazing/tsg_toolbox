import xarray as xr
import numpy as np

EARTH_RADIUS = 6371 * 1e3

def latlon2dydx(lat, lon, dim, label='upper'):
    """
    Convert latitude and longitude arrays to elementary displacements in dy
    and dx

    Parameters
    ----------
    lat : array-like
        Latitudinal spherical coordinates
    lon : array-like
        Longitudinal spherical coordinates
    dim : str
        Dimension along which the differentiation is performed, generally
        associated with the time dimension.
    label : {'upper', 'lower'}, optional
        The new coordinate in dimension dim will have the values of
        either the minuend's or subtrahend's coordinate for values `upper`
        and `lower`, respectively.

    Returns
    -------
    dy : array-like
        Zonal elementary displacement in cartesian coordinates
    dx : array-like
        Meridional elementary displacement in cartesian coordinates
    """
    dlat = lat.diff(dim, label=label)
    dlon = lon.diff(dim, label=label)
    dy = np.pi / 180. * EARTH_RADIUS * dlat
    # Need to slice the latitude data when there are duplicate values
    if label is 'upper':
        dx = (np.cos(np.pi / 180. * lat.isel(**{dim: slice(1, None)})) *
              np.pi / 180. * EARTH_RADIUS * dlon)
    elif label is 'lower':
        dx = (np.cos(np.pi / 180. * lat.isel(**{dim: slice(None, -1)})) *
              np.pi / 180. * EARTH_RADIUS * dlon)
    return dy, dx


def compute_horizontal_gradient(data):
    """
    Compute horizontal gradients from latitude and longitude
    """
    dy, dx = latlon2dydx(data['lat'], data['lon'], dim='time')
    dl = np.sqrt(dx ** 2 + dy ** 2)
    dl = dl.where(dl != 0)
    dvar = data.diff('time')
    dvardl = dvar / dl
    return dvardl