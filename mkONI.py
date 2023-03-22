#! /usr/bin/env python3
#
# Grab the latest ONI data and extract the values for January of each year
# For use as a regressor in the classifier
#
# March-2023, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
import numpy as np
import pandas as pd
import xarray as xr
import requests
import sys

parser = ArgumentParser()
parser.add_argument("--url", type=str, default="https://psl.noaa.gov/data/correlation/oni.data",
                    help="Where to fetch oni.data from")
parser.add_argument("--nc", type=str, default="oni.nc", help="Output NetCDF filename")
args = parser.parse_args()

years = []
oni = []

with requests.get(args.url) as r:
    if not r.ok:
        print(f"Error fetching '{url}'", r.reason)
        sys.exit(1)
    for line in r.text.split("\n"):
        fields = line.split()
        if not len(fields): continue
        if len(fields) == 13:
            years.append(fields[0]) # Calendar year
            oni.append(fields[1]) # January
        if fields[0] == "-99.9": break

ds = xr.Dataset(
        data_vars=dict(
            oni=("year", np.array(oni).astype(float)),
            ),
        coords=dict(
            year=np.array(years).astype(int),
            ),
        )

print(ds)
ds.to_netcdf(args.nc)
print("Wrote", args.nc)
