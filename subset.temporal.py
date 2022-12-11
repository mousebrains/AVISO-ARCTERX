#! /usr/bin/env python3
#
# Extract the subset of tracks that existed on a set of month/days ignoring the calendar year
# and output both to pared down NetCDF
#
# The input is the output of track.subset.spatial.nc
#
# Dec-2022, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
import xarray as xr
import numpy as np
import re
import os
import sys

parser = ArgumentParser()
parser.add_argument("input", nargs="+", type=str, help="Input GeoJSON files with AVISO data.")
parser.add_argument("--output", "-o", type=str, default="tpw",
                    help="Directory for the output files")
parser.add_argument("--monthDOM", type=str, action="append", required=True,
                    help="List of month/day strings that eddies must have existed on")
args = parser.parse_args()

args.output = os.path.abspath(os.path.expanduser(args.output))
if not os.path.isdir(args.output):
    print("Making", args.output)
    os.makedirs(args.output, exist_ok=True, mode=0o755)

dates = []
for md in args.monthDOM:
    matches = re.fullmatch(r"(\d+)/(\d+)", md)
    if not matches:
        parser.error(f"Invalid month/day string, {md}")
    month = int(matches[1])
    dom = int(matches[2])
    if month < 1 or month > 12:
        parser.error(f"Invalid month/day string, {md}")
    if dom < 1 or dom > 31:
        parser.error(f"Invalid month/day string, {md}")
    dates.append((month, dom))

encoding = {}
encoding["time"] = dict(zlib=True, complevel=9)
encoding["track"] = encoding["time"]
encoding["latitude"] = encoding["time"]
encoding["longitude"] = encoding["time"]

for fn in args.input:
    fn = os.path.abspath(os.path.expanduser(fn))
    (ofn, ext) = os.path.splitext(os.path.basename(fn))
    ofn  = os.path.join(args.output, ofn + ".subset.temporal.nc")
    with xr.open_dataset(fn) as ds:
        print("INIT", np.unique(ds.track).size)
        tracks = None
        t  = ds.time.to_numpy().astype("datetime64[D]")
        tm = t.astype("datetime64[M]")
        tMonth = (tm - t.astype("datetime64[Y]")).astype(int) + 1 # Month [1,12]
        tDOM = (t - tm).astype(int) + 1 # Day of month [1,31]
        for md in dates:
            q = np.logical_and(md[0] == tMonth, md[1] == tDOM)
            trks = np.unique(ds.track[q])
            if tracks is None:
                tracks = trks
                print(md, trks.size)
            else:
                tracks = np.intersect1d(tracks, trks)
                print(md, trks.size, tracks.size)
        
        a = ds.sel(obs=ds.obs[np.isin(ds.track, tracks)])
        print("Writing", os.path.basename(ofn))
        print("Reduced", ds.obs.size, "->", a.obs.size)
        a.to_netcdf(ofn, encoding=encoding)
