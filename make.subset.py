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
import time

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
encoding["latitude_max"] = encoding["time"]
encoding["longitude_max"] = encoding["time"]
encoding["amplitude"] = encoding["time"]
encoding["effective_area"] = encoding["time"]
encoding["effective_contour_height"] = encoding["time"]
encoding["effective_contour_shape_error"] = encoding["time"]
encoding["effective_radius"] = encoding["time"]
encoding["inner_contour_height"] = encoding["time"]
encoding["observation_flag"] = encoding["time"]
encoding["speed_area"] = encoding["time"]
encoding["speed_average"] = encoding["time"]
encoding["speed_contour_height"] = encoding["time"]
encoding["speed_contour_shape_error"] = encoding["time"]
encoding["speed_radius"] = encoding["time"]

for fn in args.input:
    fn = os.path.abspath(os.path.expanduser(fn))
    (ofn, ext) = os.path.splitext(os.path.basename(fn))
    qCyclonic = False if ofn.find("nticyclonic") >= 0 else True
    ofn  = os.path.join(args.output, ofn + ".subset.spatial.nc")
    stime = time.time()
    with xr.open_dataset(fn) as ds:
        ds.longitude[ds.longitude <    0] += 360 # Walked across the prime merdian westward
        ds.longitude[ds.longitude >= 360] -= 360 # Walked across the prime merdian eastward
        ds.longitude[ds.longitude >= 180] -= 360 # Wrap to [-180,+180)
        # within lat/lon box
        q = np.logical_and(
                np.logical_and( # Lat limits
                    ds.latitude >= latmin,
                    ds.latitude <= latmax,
                    ),
                np.logical_and( # Lon limits
                    ds.longitude >= lonmin,
                    ds.longitude <= lonmax,
                    )
                )
        tracks = np.unique(ds.track[q]) # Unique track ids that have presence within lat/lon box
        q = np.isin(ds.track, tracks) # Which observations to retain
        # Prune the dataset down
        a = xr.Dataset(
                data_vars=dict(
                    qCyclonic=qCyclonic,
                    ),
                coords=dict(
                    obs=ds.obs[q],
                    ),
                attrs=ds.attrs,
                )
        for key in encoding:
            a[key] = ds[key][q]

        print("Writing", os.path.basename(ofn), ds.obs.size, "->", a.obs.size, 
              "dt {:.2f}".format(time.time() - stime))
        a.to_netcdf(ofn, encoding=encoding)
        print("Took {:.2f} seconds".format(time.time() - stime))
