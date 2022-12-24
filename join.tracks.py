#! /usr/bin/env python3
#
# Due to storms, AVISO sometimes looses track of an eddy and there is a 1-2 day gap
# in between two tracks, which are really the same track.
#
# I'll use spatial and temporal correlation to reassign track numbers, skipping these gaps.
# I won't interpolate dates, so there may be gaps in the dates within a track!
#
# The input is the output of subset.spatial.nc
#
# Dec-2022, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
import xarray as xr
from cartopy  import crs as ccrs
from shapely.geometry import Point, MultiLineString
import geopandas as gpd
import pandas as pd
import numpy as np
import re
import os
import time
import sys

def greatCircleDistance(lat1:np.array, lon1:np.array, lat2:np.array, lon2:np.array,
                        criteria:float=1e-8) -> np.array:
    # Calculate the great circle distance between lat0/lon0 and lat1/lon1
    # The expected error due to the oblate spheroid aspect of the earth is less than 1%

    rMajor = 6378137          # WGS-84 semi-major axis in meters
    f = 1/298.257223563       # WGS-84 flattening of the ellipsoid
    rMinor = (1 - f) * rMajor # WGS-84 semi-minor axis in meters

    if isinstance(lat1, pd.Series):
        lat1 = np.deg2rad(lat1.to_numpy()) # Degrees to radians as an numpy array instead of pandas
        lat2 = np.deg2rad(lat2.to_numpy())
        lon1 = np.deg2rad(lon1.to_numpy())
        lon2 = np.deg2rad(lon2.to_numpy())
    else:
        lat1 = np.deg2rad(lat1) # Degrees to radians as an numpy array instead of pandas
        lat2 = np.deg2rad(lat2)
        lon1 = np.deg2rad(lon1)
        lon2 = np.deg2rad(lon2)

    tanU1 = (1 - f) * np.tan(lat1) # Tangent of reduced latitude
    tanU2 = (1 - f) * np.tan(lat2)
    cosU1 = 1 / np.sqrt(1 + tanU1**2) # Cosine of reduced latitude
    cosU2 = 1 / np.sqrt(1 + tanU2**2)
    sinU1 = tanU1 * cosU1 # Sine of reduced latitude
    sinU2 = tanU2 * cosU2

    dLon = lon2 - lon1 # difference of longitudes

    lambdaTerm = dLon # Initial guess of the lambda term

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
        lambdaPrime = dLon + \
                (1 - C) * f * sinAlpha * (
                        sigma +
                        C * sinSigma * (
                            cos2Sigma + 
                            C * cosSigma * (-1 + 2 * cos2Sigma**2)
                            )
                        )
        delta = np.abs(lambdaTerm - lambdaPrime)
        lambdaTerm = lambdaPrime
        if delta.max() < criteria: break

    u2 = cosAlpha2 * (rMajor**2 - rMinor**2) / rMinor**2
    A = 1 + u2/16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2/1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    deltaSigma = B * sinAlpha * (
            cos2Sigma + 
            B / 4 * (
                cosSigma * (-1 + 2 * cos2Sigma**2) -
                B/6 * cos2Sigma * (-3 + 4 * sinSigma**2) * (-3 + 4 * cos2Sigma**2)
                )
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

def mkTrackEndPoints(df:pd.DataFrame) -> pd.DataFrame:
    imin = df.time.argmin()
    imax = df.time.argmax()
    return pd.Series(
            dict(
                track = df.track.iloc[0],
                t0 = df.time.iloc[imin],
                t1 = df.time.iloc[imax],
                lon0 = df.longitude.iloc[imin], 
                lat0 = df.latitude.iloc[imin],
                lon1 = df.longitude.iloc[imax],
                lat1 = df.latitude.iloc[imax],
                ))

def findClosest(df:gpd.GeoDataFrame) -> pd.Series:
    imin = np.argmin(df.distPerDay)
    return pd.Series(
            dict(
                # trackLHS = df.trackLHS.iloc[imin],
                trackRHS = df.trackRHS.iloc[imin],
                ),
            )

def joinTracks(ds:xr.Dataset, monthOffset:np.timedelta64, domOffset: np.timedelta64,
               gapLength:np.timedelta64, maxRadius:float) -> xr.Dataset:
    dates = np.unique(ds.time.data.astype("datetime64[Y]")) + monthOffset + domOffset

    stime = time.time()
    df = pd.DataFrame( # Using pandas groupby is ~30 times faster than xarray's
            dict(
                track = ds.track.data,
                time = ds.time.data,
                latitude = ds.latitude.data,
                longitude = ds.longitude.data,
                ),
            )
    print(time.time()-stime, "Seconds to build dataframe")

    stime = time.time()
    trks = df.groupby(by="track", sort=False, group_keys=True).apply(mkTrackEndPoints)
    print(time.time()-stime, "Seconds to build trks", trks.shape)

    for d in dates:
        (ds, trks, cnt) = appendToTracks(ds, trks, d, gapLength, maxRadius)
        print(" append tracks", trks.shape[0], "iterations", cnt)
        (ds, trks, cnt) = prependToTracks(ds, trks, d, gapLength, maxRadius)
        print("prepend tracks", trks.shape[0], "iterations", cnt)

    return ds

def appendToTracks(ds:xr.Dataset, trks:pd.DataFrame, d:np.datetime64,
                   gapLength:np.timedelta64, maxRadius:float,
                   extra:np.timedelta64=np.timedelta64(1,"Y")) -> list:
    minGapLength = np.timedelta64(1,"D") # a delta time > 1, since 1 is no gap 
    maxGapLength = gapLength + minGapLength # delta time <= gapLength+1

    lhs = trks[np.logical_and(trks.t0 <= d+maxGapLength, 
                              trks.t1 >= d-maxGapLength).to_numpy()] # Spans d, more or less
    rhs = trks[np.logical_and(trks.t0 >= lhs.t1.min() + minGapLength,
                              trks.t0 <= lhs.t1.max() + maxGapLength + extra).to_numpy()]

    for cnt in range(10): # Avoid potential infinite loops while appending tracks
        # Look at appending, i.e. tracks that end just after tgt tracks
        nLHS = lhs.shape[0]
        nRHS = rhs.shape[0]
        # Interesting times are where rhs.t0 > lhs.t1
        dt = np.tile(rhs.t0, (nLHS,1)).T - np.tile(lhs.t1, (nRHS,1)) # nRHSxnLHS matrix
        (iRHS, iLHS) = np.where(np.logical_and(dt > minGapLength, dt <= maxGapLength))
        if not iRHS.size: break # Nothing close in time, so we're done 
        pre = lhs.iloc[iLHS]
        post = rhs.iloc[iRHS]

        dist = greatCircleDistance(pre.lat1, pre.lon1, post.lat0, post.lon0)
        dt = (post.t0.to_numpy() - pre.t1.to_numpy()).astype("timedelta64[D]").astype(int)
        distPerDay = dist / dt
        q = distPerDay <= maxRadius
        if not q.any(): break # Nothing close enough in distance, so we're done
        a = pd.DataFrame( # All track pairs whose endpoints are close in time and space
                dict(
                    trackLHS = pre.track.to_numpy()[q],
                    trackRHS = post.track.to_numpy()[q],
                    dist = dist[q],
                    dt = dt[q],
                    distPerDay = distPerDay[q],
                    ),
                )
        trackMap = a.groupby("trackLHS",  # A pandas Series is produced
                             group_keys=True, 
                             sort=False
                             ).apply(lambda x: x.trackRHS.iloc[np.argmin(x.distPerDay)])
        for (trackLHS, trackRHS) in trackMap.items():
            ds.track[ds.track == trackRHS] = trackLHS
            ll = lhs.loc[trackLHS].copy()
            rr = rhs.loc[trackRHS]
            ll.t1 = rr.t1
            ll.lat1 = rr.lat1
            ll.lon1 = rr.lon1
            lhs.loc[trackLHS] = ll
            trks.loc[trackLHS] = ll
        toDrop = trackMap.to_list()
        rhs = rhs.drop(index=toDrop)
        trks = trks.drop(index=toDrop)

    return (ds, trks, cnt)

def prependToTracks(ds:xr.Dataset, trks:pd.DataFrame, d:np.datetime64,
                    gapLength:np.timedelta64, maxRadius:float,
                    extra:np.timedelta64=np.timedelta64(1,"Y")) -> list:
    minGapLength = np.timedelta64(1,"D") # a delta time > 1, since 1 is no gap 
    maxGapLength = gapLength + minGapLength # delta time <= gapLength+1

    rhs = trks[np.logical_and(trks.t0 <= d+maxGapLength, 
                              trks.t1 >= d-maxGapLength).to_numpy()] # Spans d, more or less
    lhs = trks[np.logical_and(trks.t0 <= rhs.t0.max() - minGapLength,
                              trks.t0 >= rhs.t0.min() - maxGapLength - extra).to_numpy()]

    for cnt in range(10): # Avoid potential infinite loops while prepending tracks
        # Look at prepending, i.e. tracks that end just before tgt tracks
        nLHS = lhs.shape[0]
        nRHS = rhs.shape[0]
        # Interesting times are where rhs.t1 < lhs.t0
        dt = np.tile(rhs.t0, (nLHS,1)).T - np.tile(lhs.t1, (nRHS,1)) # nRHSxnLHS matrix
        (iRHS, iLHS) = np.where(np.logical_and(dt > minGapLength, dt <= maxGapLength))
        if not iRHS.size: break # Nothing close in time, so we're done 
        pre = lhs.iloc[iLHS]
        post = rhs.iloc[iRHS]

        dist = greatCircleDistance(pre.lat1, pre.lon1, post.lat0, post.lon0)
        dt = (post.t0.to_numpy() - pre.t1.to_numpy()).astype("timedelta64[D]").astype(int)
        distPerDay = dist / dt
        q = distPerDay <= maxRadius
        if not q.any(): break # Nothing close enough in distance, so we're done
        a = pd.DataFrame( # All track pairs whose endpoints are close in time and space
                dict(
                    trackLHS = pre.track.to_numpy()[q],
                    trackRHS = post.track.to_numpy()[q],
                    dist = dist[q],
                    dt = dt[q],
                    distPerDay = distPerDay[q],
                    ),
                )
        trackMap = a.groupby("trackRHS",  # A pandas Series is produced
                             group_keys=True, 
                             sort=False
                             ).apply(lambda x: x.trackLHS.iloc[np.argmin(x.distPerDay)])
        for (trackRHS, trackLHS) in trackMap.items():
            ds.track[ds.track == trackLHS] = trackRHS
            ll = lhs.loc[trackLHS]
            rr = rhs.loc[trackRHS].copy()
            rr.t0 = ll.t0
            rr.lat0 = ll.lat0
            rr.lon0 = ll.lon0
            rhs.loc[trackRHS] = rr
            trks.loc[trackRHS] = rr
        toDrop = trackMap.index.to_list()
        rhs = rhs.drop(index=toDrop)
        trks = trks.drop(index=toDrop)

    return (ds, trks, cnt)

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
parser.add_argument("--monthDOM", type=int, default=505,
                    help="Calendar day/month trucks must exist over.")
parser.add_argument("--gapLength", type=int, default=2,
                    help="Maximum gap length to consider for connecting tracks")
parser.add_argument("--radius", type=float, default=45, # ~98 percentile km/day
                    help="How far apart endpoints can be in km/day")
args = parser.parse_args()

args.output = os.path.abspath(os.path.expanduser(args.output))
if not os.path.isdir(args.output):
    print("Making", args.output)
    os.makedirs(args.output, exist_ok=True, mode=0o755)

monthOffset = np.timedelta64(np.floor(args.monthDOM / 100).astype(int) - 1, "M")
domOffset = np.timedelta64(np.mod(args.monthDOM, 100) - 1, "D")
gapLength = np.timedelta64(args.gapLength, "D")

encoding = {}
encoding["time"] = dict(zlib=True, complevel=9)

for fn in args.input:
    fn = os.path.abspath(os.path.expanduser(fn))
    fnBase = os.path.basename(fn)
    (ofn, ext) = os.path.splitext(fnBase)
    ofn  = os.path.join(args.output, ofn + ".joined.nc")
    with xr.open_dataset(fn) as ds:
        ds = ds.drop(labels=[ # Drop two dimensional variables
            "effective_contour_latitude", 
            "effective_contour_longitude",
            "speed_contour_latitude",
            "speed_contour_longitude",
            "uavg_profile",
            ])
        print("Working on", fnBase, ds.sizes)
        stime = time.time()
        ds = adjustLongitude(ds) # prime meridian crossover and wrap lon to [-180,180)
        ds = pruneBox(ds, args.latmin, args.latmax, args.lonmin, args.lonmax)
        print(time.time()-stime, "seconds to prune", ds.sizes, np.unique(ds.track).size)
        print(ds.speed_radius.data[-10:])
        stime = time.time()
        ds = joinTracks(ds, monthOffset, domOffset, gapLength, args.radius * 1000)
        print(time.time()-stime, "seconds to join tracks", np.unique(ds.track).size)
        stime = time.time()
        ds.to_netcdf(ofn)
        print(time.time()-stime, "seconds to write", os.path.basename(ofn))
        print(ds.speed_radius.data[-10:])
