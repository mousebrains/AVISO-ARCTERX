#! /usr/bin/env python3
#
# Due to storms, AVISO sometimes looses track of an eddy and there is a 1-2 day gap
# in between two tracks, which are really the same track.
#
# I'll use spatial and temporal correlation to reassign track numbers, skipping these gaps.
# I won't interpolate dates, so there may be gaps in the dates within a track!
#
# The input is the output of make.subset.nc
#
# Dec-2022, Pat Welch, pat@mousebrains.com

from GreatCircleDistance import greatCircleDistance
from argparse import ArgumentParser
import xarray as xr
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, MultiPoint
import geopandas as gpd
import re
import yaml
import os
import time

def mkOutputFilename(dirname:str, fn:str, monthDOM:int) -> str:
    idirname = os.path.dirname(fn)
    if idirname != dirname: return os.path.join(dirname, os.path.basename(fn));
    suffix = f".joined.{args.monthDOM:04d}.nc"
    matches = re.match(r"^(.*).subset.spatial.nc$", fn)
    if not matches:
        (ofn, ext) = os.path.splitext(os.path.basename(fn))
        return os.path.join(dirname, ofn + suffix)
    return os.path.join(dirname, matches[1] + suffix)

def adjustLongitude(ds:xr.Dataset) -> xr.Dataset:
        ds.longitude[ds.longitude <    0] += 360 # Walked across the prime merdian westward
        ds.longitude[ds.longitude >= 360] -= 360 # Walked across the prime merdian eastward
        ds.longitude[ds.longitude >= 180] -= 360 # Wrap to [-180,+180)
        return ds

def prunePolygon(ds:xr.Dataset, polygon:gpd.GeoDataFrame) -> xr.Dataset:
    bb = polygon.bounds

    # Prune to course box
    ds = ds.sel(obs=ds.obs[
        np.logical_and(
            np.logical_and(ds.latitude  >= bb.miny[0], ds.latitude  <= bb.maxy[0]),
            np.logical_and(ds.longitude >= bb.minx[0], ds.longitude <= bb.maxx[0]),
            )])

    b = gpd.GeoDataFrame(
            dict(
                track=ds.track.data,
                geometry=MultiPoint(np.array([ds.longitude.data, ds.latitude.data]).T).geoms,
                ),
            crs=4326)
    tracks = np.unique(b.sjoin(polygon, how="right").track) # Tracks inside polygon
    return ds.sel(obs=ds.obs[np.isin(ds.track, tracks)])

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
        print(d, " append tracks", trks.shape[0], "iterations", cnt)
        (ds, trks, cnt) = prependToTracks(ds, trks, d, gapLength, maxRadius)
        print(d, "prepend tracks", trks.shape[0], "iterations", cnt)

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
parser.add_argument("--polygon", type=str, default="subset.polygon.yaml",
                    help="YAML file with a polygon of lat/lon points")
parser.add_argument("--monthDOM", type=int, default=424,
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

with open(args.polygon, "r") as fp: polygon = yaml.safe_load(fp.read())
if "polygon" not in polygon:
    raise ValueError("polygon not defined in " + args.polygon)
polygon = np.array(polygon["polygon"]) # Lat/Lon
polygon = gpd.GeoDataFrame({"geometry": Polygon(polygon)}, index=[0], crs=4326) # WGS84
bb = polygon.bounds

print("Limits Lat", bb.miny[0], bb.maxy[0], "Lon", bb.minx[0], bb.maxx[0])

for fn in args.input:
    fn = os.path.abspath(os.path.expanduser(fn))
    ofn  = mkOutputFilename(args.output, fn, args.monthDOM)
    with xr.open_dataset(fn) as ds:
        print("Working on", os.path.basename(fn), ds.sizes)
        stime = time.time()
        ds = adjustLongitude(ds) # prime meridian crossover and wrap lon to [-180,180)
        ds = prunePolygon(ds, polygon)
        print(time.time()-stime, "seconds to prune", ds.sizes, np.unique(ds.track).size)
        stime = time.time()
        ds = joinTracks(ds, monthOffset, domOffset, gapLength, args.radius * 1000)
        print(time.time()-stime, "seconds to join tracks", np.unique(ds.track).size)
        toDrop = set()
        for key in sorted(ds.keys()):
            if len(ds[key].dims) > 1:
                toDrop.add(key)
                continue
            ds[key].encoding = dict(zlib=True, complevel=3)
        ds = ds.drop(toDrop)
        stime = time.time()
        ds.to_netcdf(ofn)
        print(time.time()-stime, "seconds to write", os.path.basename(ofn), ds.obs.size)
