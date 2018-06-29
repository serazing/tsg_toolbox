#!/usr/bin/env python
# Useful python module
import os
import glob
import sys
import xarray as xr
from mpi4py import MPI

# Open local modules
import data_io as io
import geometry
import mesoscale_filter as mesofilt
import thermodynamics as thermo
import plot

# Define the parser for input parameters
import argparse
description = ("Compute horizontal gradients of temperature, salinity "
               "and buoyancy on different in situ dataset using MPI. "
               "A climatology of the baroclinic Rossby radius is used "
               "to filter out submesoscale features."
              )
parser = argparse.ArgumentParser(description=description)
parser.add_argument('infiles', nargs='+', help='Input files')
parser.add_argument('--dataset', choices=['LEGOS', 'FRESH', 'GOSUD', 'IMOS', 'MEOP'], 
                                 required=True,
                                 help="Dataset origin of the input files")
parser.add_argument('--rossby', required=True,
                                help='The Rossby file climatology')
parser.add_argument('--outpath', nargs='?', default='./',
                                 help='Path to store ouput_files')
args = parser.parse_args()

# MPI initialization
comm = MPI.COMM_WORLD
nb_proc = comm.Get_size()
proc_indx = comm.Get_rank()

Rd = io.open_rossby_climatology(args.rossby)
make_plots = False

for filename in args.infiles:
    file_index = args.infiles.index(filename)
    outname = "%s/%s_track_%04i_horizontal_gradients.nc" %(args.outpath, args.dataset, file_index)    
    if ((file_index % nb_proc) == proc_indx) and (not os.path.isfile(outname)):
        print("Processing %s on core %s" %(filename, proc_indx))
        try:
            if args.dataset == 'LEGOS':
                data = io.open_tsg_from_legos(filename)
            elif args.dataset == 'FRESH':
                data = io.open_tsg_from_fresh(filename, quality='probably_good')
            elif args.dataset == 'GOSUD':
                data = io.open_tsg_from_gosud(filename, quality='probably_good')
            elif args.dataset == 'IMOS':
                data = io.open_tsg_from_imos(filename, quality='good')
            elif args.dataset == 'MEOP':
                data = io.open_sea_mammals_from_meop(filename).isel(depth=0)
                data = data.rename({'Temperature': 'SST', 'Salinity': 'SSS',
                                    'Temperature_QC': 'SST_QC', 
                                    'Salinity_QC': 'SSS_QC',})
                data = data.where(((data['SST_QC'] == 1) | 
                                   (data['SST_QC'] == 2)
                                  ) &
                                  ((data['SSS_QC'] == 1) | 
                                   (data['SSS_QC'] == 2)
                                  )
                                 )
        except (KeyError, ValueError):
            print("Error when opening %s." %filename)
        if data.sizes['time'] == 0:
            print("Skipping processing because data has size 0")
            continue
        try:
            data = thermo.compute_buoyancy(data, temperature_var='SST', salinity_var='SSS')
            Rd_nearest = mesofilt.interpolate_rossby_radius(Rd, data['lon'], data['lat'],
                                                            interp_type='nearest', 
                                                            radius_of_influence=10e3)
            SST_filtered = mesofilt.mesoscale_filter(data['SST'], Rd_nearest, 
                                                     win_dt=3, max_break=24)
            SSS_filtered = mesofilt.mesoscale_filter(data['SSS'], Rd_nearest, 
                                                     win_dt=3, max_break=24)
            SSb_filtered = mesofilt.mesoscale_filter(data['buoyancy'], Rd_nearest, 
                                                          win_dt=3, max_break=24)
            data_me = xr.Dataset({'SST_ME': SST_filtered, 
                                  'SSS_ME': SSS_filtered,
                                  'SSb_ME': SSb_filtered}
                                )
            if make_plots:
                plot.plot_filtered_vars(data, data_me, 
                                        output_name=("%s/PLOTS/%s_%04i_filtered_variables.png" 
                                                     %(args.outpath, args.dataset, file_index)
                                                    )
                                        )
            horizontal_gradients = geometry.compute_horizontal_gradient(data_me)
            if make_plots:
                plot.plot_horizontal_gradients(horizontal_gradients,
                                               output_name= ("%s/PLOTS/%s_%04i_horizontal_gradients.png" 
                                                             %(args.outpath, args.dataset, file_index)
                                                            )
                                              )
            if horizontal_gradients.sizes['time'] > 0:
                horizontal_gradients.to_netcdf(outname)
        except (ValueError, NameError):
            print("Error when processing %s." %filename)
