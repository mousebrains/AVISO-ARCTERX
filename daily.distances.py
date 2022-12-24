#! /usr/bin/env python3
#
# Calculate the quantiles of distance travelled per day for the AVISO Mesoscale Eddy Product data
#
# The input is the output of subset.spatial.nc
#
# Dec-2022, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re
import os
import time
import sys

def greatCircleDistance(lat1:np.array, lon1:np.array, lat2:np.array, lon2:np.array) -> np.array:
    # Calculate the great circle distance between lat0/lon0 and lat1/lon1
    # The expected error due to the oblate spheroid aspect of the earth is less than 1%

    rMajor = 6378137          # WGS-84 semi-major axis in meters
    f = 1/298.257223563       # WGS-84 flattening of the ellipsoid
    rMinor = (1 - f) * rMajor # WGS-84 semi-minor axis in meters

    lat1 = np.deg2rad(lat1.to_numpy()) # Degrees to radians as an numpy array instead of pandas
    lat2 = np.deg2rad(lat2.to_numpy())
    lon1 = np.deg2rad(lon1.to_numpy())
    lon2 = np.deg2rad(lon2.to_numpy())

    tanU1 = (1 - f) * np.tan(lat1) # Tangent of reduced latitude
    tanU2 = (1 - f) * np.tan(lat2)
    cosU1 = 1 / np.sqrt(1 + tanU1**2) # Cosine of reduced latitude
    cosU2 = 1 / np.sqrt(1 + tanU2**2)
    sinU1 = tanU1 * cosU1 # Sine of reduced latitude
    sinU2 = tanU2 * cosU2

    L = lon2 - lon1 # difference of longitudes

    lambdaTerm = L # Initial guess of the lambda term

    for cnt in range(10): # Iteration loop through Vincenty's inverse problem to get the distance
        sinLambda = np.sin(lambdaTerm)
        cosLambda = np.cos(lambdaTerm)
        sinSigma = np.sqrt((cosU2 * sinLambda)**2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda)**2)
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = np.arctan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * sinLambda / np.sin(sigma)
        cosAlpha2 = 1 - sinAlpha**2
        cos2Sigma = cosSigma - 2 * sinU1 * sinU2 / cosAlpha2
        C = f / 16 * cosAlpha2 * (4 + f * (4 - 3 * cosAlpha2))
        lambdaPrime = L + (1 - C) * f * sinAlpha * (
                sigma + 
                C * sinAlpha * (cos2Sigma + C * cosSigma * (-1 + 2 * cos2Sigma**2)))
        delta = np.abs(lambdaTerm - lambdaPrime)
        lambdaTerm = lambdaPrime
        if delta.max() < 1e-8: break

    u2 = cosAlpha2 * (rMajor**2 - rMinor**2) / rMinor**2
    A = 1 + u2/16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2/1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    deltaSigma = B * sinAlpha * \
            (cos2Sigma + B / 4 * \
            (cosSigma * \
             (-1 + 2 * cos2Sigma**2) - \
             B/6 * cos2Sigma * (-3 + 4 * sinSigma**2) * (-3 + 4 * cos2Sigma**2))
             )
    return rMinor * A * (sigma - deltaSigma) # Distance on the elipsoid
    
def adjustLongitude(ds:xr.Dataset) -> xr.Dataset:
        ds.longitude[ds.longitude <    0] += 360 # Walked across the prime merdian westward
        ds.longitude[ds.longitude >= 360] -= 360 # Walked across the prime merdian eastward
        ds.longitude[ds.longitude >= 180] -= 360 # Wrap to [-180,+180)
        return ds

def pruneBox(ds:xr.Dataset,
             latMin:float, latMax:float,
             lonMin:float, lonMax:float) -> xr.Dataset:
    q = np.logical_and(
            np.logical_and(
                ds.latitude >= min(latMin, latMax),
                ds.latitude <= max(latMin, latMax)),
            np.logical_and(
                ds.longitude >= min(lonMin, lonMax),
                ds.longitude <= max(lonMin, lonMax)),
            )
    q = np.isin(ds.track, np.unique(ds.track[q])) # Which observations to retain
    return ds.sel(obs=ds.obs[q])

def calcDistances(df:pd.DataFrame, sDates:np.array, eDates:np.array) -> pd.DataFrame:
    items = []
    for i in range(sDates.size):
        q = np.logical_and(df.time >= sDates[i], df.time <= eDates[i])
        if not q.any(): continue
        a = df[q].sort_values("time")
        if a.size < 10: continue
        dist = greatCircleDistance(a.latitude.iloc[:-1], a.longitude.iloc[:-1],
                                   a.latitude.iloc[1:], a.longitude.iloc[1:])
        items.append(
                pd.DataFrame(
                    dict(
                        track = a.track.iloc[1:],
                        time = a.time.iloc[1:],
                        dist = dist)
                    )
                )

    if len(items) == 0: return None
    if len(items) == 1: return items[0]
    return pd.concat(items) # More than one item, probably rare

parser = ArgumentParser()
parser.add_argument("input", nargs="+", type=str, help="Input GeoJSON files with AVISO data.")
parser.add_argument("--latmin", type=float, default=-5,
                    help="Southern latitude limit in decimal degrees")
parser.add_argument("--latmax", type=float, default=40,
                    help="Northern latitude limit in decimal degrees")
parser.add_argument("--lonmin", type=float, default=116,
                    help="Eastern longitude limit in decimal degrees")
parser.add_argument("--lonmax", type=float, default=166,
                    help="Western longitude limit in decimal degrees")
parser.add_argument("--monthDOM", type=int, default=505,
                    help="Calendar day/month tracks must exist over.")
parser.add_argument("--duration", type=int, default=60,
                    help="Number of days to look in after monthDOM for distance data")
parser.add_argument("--quantile", type=float, action="append",
                    help="Quantile(s) to calculate distance travelled")
args = parser.parse_args()

if not args.quantile: args.quantile = np.linspace(0.9, 1, 11) # [90,91,...100]% quantiles

monthOffset = np.timedelta64(np.floor(args.monthDOM / 100).astype(int) - 1, "M")
domOffset = np.timedelta64(np.mod(args.monthDOM, 100) - 1, "D")
duration = np.timedelta64(args.duration, "D")

todo = []
for fn in args.input:
    fn = os.path.abspath(os.path.expanduser(fn))
    with xr.open_dataset(fn) as ds:
        print("Working on", os.path.basename(fn), ds.sizes)
        stime = time.time()
        ds = adjustLongitude(ds) # prime meridian crossover and wrap lon to [-180,180)
        ds = pruneBox(ds, args.latmin, args.latmax, args.lonmin, args.lonmax)
        print(time.time()-stime, "Seconds to prune", ds.sizes)
        a = pd.DataFrame( # Pandas' groupby is ~30 times faster than on xarray's
                dict(
                    track = ds.track.data,
                    time = ds.time.data,
                    latitude = ds.latitude.data,
                    longitude = ds.longitude.data,
                    ))
        sDates = np.unique(ds.time.data.astype("datetime64[Y]")) + monthOffset + domOffset
        eDates = sDates + duration
        b = a.groupby("track").apply(lambda x: calcDistances(x, sDates, eDates))
        todo.append(b)
        print("Number of points", b.shape[0], "number of tracks", np.unique(b.track).shape[0])
        bq = pd.DataFrame(
                dict(
                    quant = args.quantile * 100,
                    dist = np.quantile(np.abs(b.dist), args.quantile),
                    ),
                )
        print(bq)

if todo:
    b = pd.concat(todo)
    print("Total number of poitns", b.shape[0], "Number of tracks", np.unique(b.track).shape[0])
    bq = pd.DataFrame(
            dict(
                quant = args.quantile * 100,
                dist = np.quantile(np.abs(b.dist), args.quantile),
                ),
            )
    print(bq)

    plt.hist(np.abs(b.dist)/1000, np.linspace(0,80,81), 
             density=True, cumulative=True, histtype="step")
    plt.grid(True)
    plt.xlabel("Distance/day (km)")
    plt.show()

