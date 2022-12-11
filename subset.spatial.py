#! /usr/bin/env python3
#
# Extract the subset of tracks within the lat/lon box
# and output both to pared down NetCDF
#
# N.B. This does not handle crossing the anti-merdian!
#
# Dec-2022, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
import xarray as xr
import numpy as np
import os

parser = ArgumentParser()
parser.add_argument("input", nargs="+", type=str, help="Input GeoJSON files with AVISO data.")
parser.add_argument("--output", "-o", type=str, default="tpw",
                    help="Directory for the output files")
parser.add_argument("--latmin", type=float, default=-5,
                    help="Southern latitude limit in decimal degrees")
parser.add_argument("--latmax", type=float, default=40,
                    help="Northern latitude limit in decimal degrees")
parser.add_argument("--lonmin", type=float, default=116,
                    help="Eastern longitude limit in decimal degrees")
parser.add_argument("--lonmax", type=float, default=166,
                    help="Western longitude limit in decimal degrees")
args = parser.parse_args()

args.output = os.path.abspath(os.path.expanduser(args.output))
if not os.path.isdir(args.output):
    print("Making", args.output)
    os.makedirs(args.output, exist_ok=True, mode=0o755)

latmin = min(args.latmin, args.latmax)
latmax = max(args.latmin, args.latmax)
lonmin = min(args.lonmin, args.lonmax)
lonmax = max(args.lonmin, args.lonmax)

encoding = {}
encoding["time"] = dict(zlib=True, complevel=9)
encoding["track"] = encoding["time"]
encoding["latitude"] = encoding["time"]
encoding["longitude"] = encoding["time"]

for fn in args.input:
    fn = os.path.abspath(os.path.expanduser(fn))
    (ofn, ext) = os.path.splitext(os.path.basename(fn))
    qCyclonic = False if ofn.find("nticyclonic") >= 0 else True
    ofn  = os.path.join(args.output, ofn + ".subset.spatial.nc")
    with xr.open_dataset(fn) as ds:
        ds.longitude[ds.longitude <    0] += 360 # Walked across merdian westward
        ds.longitude[ds.longitude >= 360] -= 360 # Walked across merdian eastward
        ds.longitude[ds.longitude >= 180] -= 360 # Wrap to [-180,+180)
        # Select based on lat limits
        es = ds.sel(obs=ds.obs[np.logical_and(ds.latitude  >= latmin, ds.latitude  <= latmax)])
        # Select on lon limits
        es = es.sel(obs=es.obs[np.logical_and(es.longitude >= lonmin, es.longitude <= lonmax)])
        # Retain the full track trajectory for any track that was ever within the lat/lon box
        es = ds.sel(obs=ds.obs[np.isin(ds.track, np.unique(es.track))])
        # Prune the dataset down
        a = xr.Dataset(
                data_vars=dict(
                    qCyclonic=qCyclonic,
                    time=es.time,
                    track=es.track,
                    latitude=es.latitude,
                    longitude=es.longitude,
                    ),
                coords=dict(
                    obs=es.obs,
                    ),
                attrs=es.attrs,
                )
        print("Writing", os.path.basename(ofn), ds.obs.size, "->", a.obs.size)
        a.to_netcdf(ofn, encoding=encoding)
