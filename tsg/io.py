#!/usr/bin/env python

import xarray as xr
import pandas as pd
from pandas.errors import OutOfBoundsDatetime
import numpy as np


def check_data(ds, max_SST_rate=10, max_SSS_rate=5, drop=True):   
    #Remove duplicated time values
    ds = ds.sel(time=~ds.indexes['time'].duplicated())
    # Uniformize the longitude coordinates to be between 0 and 360
    ds['lon'] = (ds['lon'] + 360) % 360
    dSST = np.abs(ds['SST'].diff('time'))
    dSSS = np.abs(ds['SSS'].diff('time'))
    ds = ds.where(dSST < max_SST_rate, drop=drop)
    ds = ds.where(dSSS < max_SSS_rate, drop=drop)
    return ds


def open_tsg_from_legos(filename):
    """
    Open thermosalinograph (TSG) transect from the LEGOS dataset, 
    and homogenize the coordinates
    
    Parameters
    ----------
    filename : str
        Name of the file to open
    
    Returns
    -------
    ds : xarray.Dataset
        The TSG transect under the form of a Dataset
    """
    renamed_var = {'TIME': 'time', 'LON': 'lon', 'LAT': 'lat'}    
    ds = (xr.open_dataset(filename, autoclose=True)
            .rename(renamed_var)
            .set_coords(('lon', 'lat'))
         )
    ds['lon'] = (ds['lon'] + 360) % 360
    # Sort by time to avoid strange behaviours
    ds = ds.sortby('time')
    for var in ds.variables:
        try:
            del(ds[var].attrs['coordinates'])
        except(KeyError):
            pass
    return ds


def open_tsg_from_fresh(filename, quality=None):
    ds = xr.open_dataset(filename, decode_times=False,      
                         autoclose=True)
    ref_time = ds.REFERENCE_DATE_TIME.astype('str').data.tolist()
    origin = pd.Timestamp(ref_time, unit='s')
    time = pd.to_datetime(ds.DAYD, unit='D', origin=origin)
    ds = ds.rename({'DAYD': 'time', 'LATX': 'lat', 'LONX': 'lon'})
    ds = ds.set_coords(('lon', 'lat'))
    ds['time'] = time
    ds['lon'] = (ds['lon'] + 360) % 360
    new_ds = xr.Dataset({'SSS': ds['SSPS_ADJUSTED'],
                         'SSS_QC': ds['SSPS_ADJUSTED_QC'],
                         'SST': ds['SSTP_ADJUSTED'],
                         'SST_QC': ds['SSTP_ADJUSTED_QC']})
    if quality is 'good':
        new_ds['SSS'] = new_ds['SSS'].where(new_ds['SSS_QC'] == 1)
        new_ds['SST'] = new_ds['SST'].where(new_ds['SST_QC'] == 1)
    elif quality is 'probably_good':
        new_ds['SSS'] = new_ds['SSS'].where((new_ds['SSS_QC'] == 1) | (new_ds['SSS_QC'] == 2))
        new_ds['SST'] = new_ds['SST'].where((new_ds['SST_QC'] == 1) | (new_ds['SST_QC'] == 2))
    new_ds.attrs = ds.attrs
    return new_ds
 
    
def open_tsg_from_gosud(filename, quality=None, drop=True):
    ds = xr.open_dataset(filename, decode_times=False,      
                         autoclose=True)
    ref_time = ds.REFERENCE_DATE_TIME.astype('str').data.tolist()
    origin = pd.Timestamp(ref_time, unit='s')
    time = pd.to_datetime(ds.DAYD, unit='D', origin=origin)
    ds = ds.rename({'DAYD': 'time', 'LATX': 'lat', 'LONX': 'lon'})
    ds = ds.set_coords(('lon', 'lat'))
    ds['time'] = time
    if np.all(np.isnan(ds['SSTP'])):
        temp_var = 'SSJT'
    else:
        temp_var = 'SSTP'        
    new_ds = xr.Dataset({'SSS': ds['SSPS'],
                         'SSS_QC': ds['SSPS_QC'],
                         'SST': ds[temp_var],
                         'SST_QC': ds[temp_var + '_QC']})
    if quality is 'good':
        new_ds = new_ds.where(new_ds['SSS_QC'] == 1, drop=drop)
        new_ds = new_ds.where(new_ds['SST_QC'] == 1, drop=drop)
        #new_ds['SSS'] = new_ds['SSS'].where(new_ds['SSS_QC'] == 1, drop=drop)
        #new_ds['SST'] = new_ds['SST'].where(new_ds['SST_QC'] == 1, drop=drop)
    elif quality is 'probably_good':
        new_ds = new_ds.where((new_ds['SSS_QC'] == 1) | 
                              (new_ds['SSS_QC'] == 2), drop=drop)
        new_ds = new_ds.where((new_ds['SST_QC'] == 1) | 
                              (new_ds['SST_QC'] == 2), drop=drop)
        #new_ds['SSS'] = new_ds['SSS'].where((new_ds['SSS_QC'] == 1) | (new_ds['SSS_QC'] == 2), drop=drop)
        #new_ds['SST'] = new_ds['SST'].where((new_ds['SST_QC'] == 1) | (new_ds['SST_QC'] == 2), drop=drop)
    new_ds.attrs = ds.attrs
    new_ds = check_data(new_ds, drop=drop)
    return new_ds    
    

def open_tsg_from_noaa(filename):
    """
    Open thermosalinograph (TSG) transect from NOAA dataset, 
    and homogenize the coordinates
    
    Parameters
    ----------
    filename : str
        Name of the file to open
    
    Returns
    -------
    ds : xarray.Dataset
        The TSG transect under the form of a Dataset
    """
    ds = xr.open_dataset(filename, autoclose=True).drop('trajectory')
    ds = (ds.rename({'time': 'time_counter'})
            .rename({'obs': 'time', 'time_counter': 'time'})
            .squeeze()
         ) 
    new_ds = xr.Dataset({'SST': ds['sea_surface_temperature'], 
                         'SST_QC':ds['sea_surface_temperature_qc'].astype('i4'), 
                         'SSS': ds['sea_surface_salinity'], 
                         'SSS_QC': ds['sea_surface_salinity_qc'].astype('i4')})
    return new_ds


def open_tsg_from_imos(filename, quality=None):
    """
    Open thermosalinograph (TSG) transect from NOAA dataset, 
    and homogenize the coordinates
    
    Parameters
    ----------
    filename : str
        Name of the file to open
    
    Returns
    -------
    ds : xarray.Dataset
        The TSG transect under the form of a Dataset
    """
    ds = xr.open_dataset(filename, autoclose=True)
    ds = ds.rename({'TIME': 'time', 'LATITUDE': 'lat', 'LONGITUDE': 'lon'})
    ds = ds.set_coords(('lon', 'lat'))
    ds['lon'] = (ds['lon'] + 360) % 360
    new_ds = xr.Dataset({'SST': ds['TEMP'],
                         'SST_QC': ds['TEMP_quality_control'],
                         'SSS': ds['PSAL'],
                         'SSS_QC': ds['PSAL_quality_control'],
                         })
    if quality is 'good':
        if new_ds['SSS_QC'].dtype == 'float':
            new_ds['SSS'] = new_ds['SSS'].where(new_ds['SSS_QC'] == 2)
            new_ds['SST'] = new_ds['SST'].where(new_ds['SST_QC'] == 2)
        elif new_ds['SSS_QC'].dtype == 'S1':
            new_ds['SSS'] = new_ds['SSS'].where(new_ds['SSS_QC'] == b'Z')
            new_ds['SST'] = new_ds['SST'].where(new_ds['SST_QC'] == b'Z')
    elif quality is 'probably_good':
        if new_ds['SSS_QC'].dtype == 'float':
            new_ds['SSS'] = new_ds['SSS'].where((new_ds['SSS_QC'] == 2) | (new_ds['SSS_QC'] == 3))
            new_ds['SST'] = new_ds['SST'].where((new_ds['SST_QC'] == 2) | (new_ds['SST_QC'] == 3))
    return new_ds


def open_sea_mammals_from_meop(filename):
    ds = xr.open_dataset(filename)
    renamed_var = {'N_LEVELS': 'depth', 'N_PROF': 'time'}
    coords = ('time', 'depth', 'lat', 'lon')
    ds = ds.rename({'N_LEVELS': 'depth', 'N_PROF': 'time'}).squeeze()
    new_ds = xr.Dataset({'Temperature': ds['TEMP_ADJUSTED'], 
                         'Temperature_QC': ds['TEMP_ADJUSTED_QC'].astype('i4'), 
                         'Temperature_ERROR': ds['TEMP_ADJUSTED_ERROR'],
                         'Salinity': ds['PSAL_ADJUSTED'], 
                         'Salinity_QC': ds['PSAL_ADJUSTED_QC'].astype('i4'),
                         'Salinity_ERROR': ds['PSAL_ADJUSTED_ERROR'],
                         'Pressure': ds['PRES_ADJUSTED'], 
                         'Pressure_QC': ds['PRES_ADJUSTED_QC'].astype('i4'),
                         'Pressure_ERROR': ds['PRES_ADJUSTED_ERROR'].astype('i4')})
    new_ds = new_ds.assign_coords(lat=ds['LATITUDE'], 
                                  lon=ds['LONGITUDE'], 
                                  time=ds['JULD'])
    # Check the longitude vector 
    new_ds['lon'] = (new_ds['lon'] + 360) % 360
    new_ds = new_ds.sortby('time')
    return new_ds


def open_rossby_climatology(filename):
    """
    Open the filtered Rossby climatology from the GLORYS reanalysis
    
    Parameters
    ----------
    filename : str
        Name of the file to open
    
    Returns
    -------
    ds : xarray.DataArray
        The Rossby climatology under the form of a DataArray
    """
    ds = xr.open_dataset(filename).set_coords(('nav_lat', 'nav_lon')).squeeze()          
    ds['nav_lon'] = (ds['nav_lon'] + 360) % 360
    ds = ds.rename({'nav_lon': 'lon', 'nav_lat': 'lat', 'sorosrad': 'Rd'}) 
    return ds['Rd']