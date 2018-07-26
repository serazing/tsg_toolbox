#!/usr/bin/env python
# Useful python module
import os
import glob
import sys
import xarray as xr
from mpi4py import MPI
import tsg


# Define the parser for input parameters
import argparse
description = ("Compute horizontal gradients of temperature, salinity "
               "and buoyancy on different in situ dataset using MPI. "
               "A climatology of the baroclinic Rossby radius may be used "
               "to filter out submesoscale features. If not given, a fixed "
               "spatial cutoff is used to arbitrary separate small and large "
               "spatial scales. "
              ) 
parser = argparse.ArgumentParser(description=description)
parser.add_argument('infiles', nargs='+', help='Input files')
parser.add_argument('--dataset', choices=['LEGOS', 'FRESH', 'GOSUD', 'IMOS'], 
                                 required=True,
                                 help="Dataset origin of the input files")
parser.add_argument('--rossby', required=False,
                                help='The Rossby file climatology')
parser.add_argument('--cutoff', required=False, default=10e3,
                                help='The cutoff value used for the filter')
parser.add_argument('--outpath', nargs='?', default='./',
                                 help='Path to store ouput_files')
args = parser.parse_args()

# MPI initialization
comm = MPI.COMM_WORLD
nb_proc = comm.Get_size()
proc_indx = comm.Get_rank()

for filename in args.infiles:
    file_index = args.infiles.index(filename)
    outname = ("%s/%s_track_%04i_horizontal_gradients.nc" 
               %(args.outpath, args.dataset, file_index)
              )
    if ((file_index % nb_proc) == proc_indx) and (not os.path.isfile(outname)):
        print("Processing %s on core %s" %(filename, proc_indx))
        try:
            if args.dataset == 'LEGOS':
                data = tsg.open_tsg_from_legos(filename)
            elif args.dataset == 'FRESH':
                data = tsg.open_tsg_from_fresh(filename, quality='probably_good')
            elif args.dataset == 'GOSUD':
                data = tsg.open_tsg_from_gosud(filename, quality='probably_good')
            elif args.dataset == 'IMOS':
                data = tsg.open_tsg_from_imos(filename, quality='good')
        except (KeyError, ValueError):
            print("Error when opening %s." %filename)
        if data.sizes['time'] == 0:
            print("Skipping processing because data has size 0")
            continue
        try:
            data = tsg.compute_buoyancy(data, 
                                        temperature_var='SST', 
                                        salinity_var='SSS')
            SST_filtered = tsg.shiptrack_filter(data['SST'], cutoff=float(args.cutoff), 
                                                win_dt=3, max_break=24)
            SSS_filtered = tsg.shiptrack_filter(data['SSS'], cutoff=float(args.cutoff), 
                                                win_dt=3, max_break=24)
            SSb_filtered = tsg.shiptrack_filter(data['buoyancy'], cutoff=float(args.cutoff), 
                                                win_dt=3, max_break=24)
            data_me = xr.Dataset({'SST_LS': SST_filtered, 
                                  'SSS_LS': SSS_filtered,
                                  'SSb_LS': SSb_filtered}
                                )
            horizontal_gradients = tsg.geometry.compute_horizontal_gradient(data_me)
            if horizontal_gradients.sizes['time'] > 0:
                horizontal_gradients.to_netcdf(outname)
        except (ValueError, NameError):
            import pdb
            pdb.set_trace()
            print("Error when processing %s." %filename)
