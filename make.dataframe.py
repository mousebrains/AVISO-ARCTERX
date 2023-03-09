#! /usr/bin/env python3
#
# From the output of the make.joined.py,
# Build track level information,
# which will be used by the classify.py to build a Pandas dataframe
# for the classifiers
#
# Dec-2022, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
from GreatCircleDistance import greatCircleDistance as gcDist
import xarray as xr
import pandas as pd
import numpy as np
import os

def mkTrackEndPoints(ds:xr.Dataset, dates:np.array) -> xr.Dataset:
    df = pd.DataFrame( # groupby on Pandas DataFrames is ~30 times faster than on xarray
            dict(
                track = ds.track.data,
                time = ds.time.data,
                ))

    grps = df.groupby("track")
    t0 = grps.apply(lambda x: x.time.min()) # earliest time for each track
    t1 = grps.apply(lambda x: x.time.max()) # latest time for each track

    d  = np.tile(dates, (t0.size, 1)).T
    q = np.logical_and(
            np.tile(t0, (dates.size, 1)) <= d,
            np.tile(t1, (dates.size, 1)) >= d,
            ).any(axis=0)

    return pd.DataFrame(
            dict(
                track = t0.index.to_numpy()[q],
                t0 = t0.to_numpy()[q],
                t1 = t1.to_numpy()[q],
                ))

def calcDist(aLat:xr.DataArray, aLon:xr.DataArray, 
             bLat:xr.DataArray, bLon:xr.DataArray, 
             diag:float=None) -> np.array:
    if not aLat.size or not bLat.size: return None
    dist = gcDist(
            np.tile(aLat, (bLat.size,1)).T, np.tile(aLon, (bLat.size,1)).T,
            np.tile(bLat, (aLat.size,1)),   np.tile(bLon, (aLat.size,1)))
    if diag is not None:
        np.fill_diagonal(dist, diag)

    return dist

def findNeighbors(df:pd.DataFrame) -> pd.DataFrame:
    # Now find nearest neighbor of each flavor for each date
    cDist = np.full(df.shape[0], 1e40)
    aDist = np.full(df.shape[0], 1e40)

    cycl = df[df.qCyclonic.astype(bool)]
    anti = df[np.logical_not(df.qCyclonic)]

    for d in np.unique(df.date):
        qDate = df.date == d
        qAnti = np.logical_and(qDate, np.logical_not(df.qCyclonic))
        qCycl = np.logical_and(qDate, df.qCyclonic)

        aLat = df.latitude[qAnti]
        cLat = df.latitude[qCycl]
        aLon = df.longitude[qAnti]
        cLon = df.longitude[qCycl]

        # Anti-Anti distance
        dist = calcDist(aLat, aLon, aLat, aLon, 1e40)
        if dist is not None: aDist[qAnti] = dist.min(axis=0) # Which axis doesn't matter
       
        # Cycl-Cycl distance
        dist = calcDist(cLat, cLon, cLat, cLon, 1e40)
        if dist is not None: cDist[qCycl] = dist.min(axis=0) # Which axis does not matter

        # Anti-Cycl distance
        dist = calcDist(aLat, aLon, cLat, cLon)
        if dist is not None:
            aDist[qCycl] = dist.min(axis=0)
            cDist[qAnti] = dist.min(axis=1)

    df["cyclDistance"] = cDist
    df["antiDistance"] = aDist
    return df

def mkDataSeries(ds:xr.Dataset, d:np.datetime64, preDays:np.timedelta64,
                 qCyclonic:bool) -> pd.DataFrame:
    nPrevious = d - ds.time.data.min()
    nFuture = ds.time.data.max() - d

    d0 = d - preDays

    ds = ds.sel(obs=ds.obs[
        np.logical_and(
            ds.time >= (d0 - np.timedelta64(1,"D")), # An extra day before hand
            ds.time <= d)])

    if ds.obs.size < 2: return None

    ds = ds.sortby("time", ascending=True) # Ascending in time

    dt = np.diff(ds.time.data).astype("timedelta64[D]").astype(float)
    distPerDay = gcDist(
            ds.latitude.data[:-1], ds.longitude.data[:-1],
            ds.latitude.data[1:],  ds.longitude.data[1:]) / dt

    ds = ds.assign({"distPerDay": ("obs", np.full(ds.obs.size, np.nan))})
    ds.distPerDay.data[1:] = distPerDay

    ds = ds.sel(obs=ds.obs[ds.time >= d0]) # Prune off extra preday if needed
    if not ds.obs.size: return None

    keys = list(ds.keys())
    for key in ("num_contours", "num_point_e", "num_point_s", "observation_number",
                "qCyclonic", "qDelayedTime", "time", "track", "observation_flag"):
        if key in keys: keys.remove(key)

    dt = (ds.time.data - ds.time.data[0]).astype("timedelta64[D]").astype(float)
    sumX = dt.sum()
    sumXX = (dt * dt).sum()
    N = dt.size
    denom = N * sumXX - np.square(sumX)

    items = {
             "track": ds.track.data[0],
             "qCyclonic": qCyclonic, 
             "date": d,
             "preDays": nPrevious,
             "duration": nFuture,
             }

    for key in sorted(keys): 
        data = ds[key].data
        sumY = data.sum()
        sumXY = (data * dt).sum()
        m = (N * sumXY - sumX * sumY) / denom if denom != 0 else 0

        items[key] = data[-1] # Latest entry
        items[key + "_median"] = np.nanmedian(data)
        items[key + "_mean"] = np.nanmean(data)
        items[key + "_sigma"] = np.nanstd(data)
        items[key + "_slope"] = m
    return pd.DataFrame(items, index=[0])

def mkDataFrames(ds:xr.Dataset, trks:pd.DataFrame, 
                 dates:np.array, preDays:np.timedelta64, qCyclonic:bool) -> list:
    items = []
    for d in dates:
        for trk in trks.track[np.logical_and(trks.t0 <= d, trks.t1 >= d)]:
            series = mkDataSeries(ds.sel(obs=ds.obs[ds.track == trk]), d, preDays, qCyclonic)
            if series is not None: items.append(series)
    return pd.concat(items, ignore_index=True) if items else None

parser = ArgumentParser()
parser.add_argument("input", nargs="+", type=str,
                    help="Input NetCDF files with subsetted AVISO data.")
parser.add_argument("--output", "-o", type=str, default="tkw/joint.nc",
                    help="Directory for the output files")
parser.add_argument("--monthDOM", type=int, default=424, help="Observation Month/Day")
parser.add_argument("--preDays", type=int, default=10, help="Days prior to monthDOM to consider")
parser.add_argument("--nrtMinYear", type=int, default=2019,
                    help="Use NRT data after this year, else DT data")
args = parser.parse_args()

ofn = os.path.abspath(os.path.expanduser(args.output))
if not os.path.isdir(os.path.dirname(ofn)):
    print("Making", os.path.dirname(ofn))
    os.makedirs(os.path.dirname(ofn), exist_ok=True, mode=0o755)

monthOffset = np.timedelta64(int(args.monthDOM/100) - 1, "M")
domOffset = np.timedelta64((args.monthDOM % 100) - 1, "D")
preDays = np.timedelta64(args.preDays, "D")
nrtYearMin = np.datetime64(args.nrtMinYear-1970, "Y")

attributes = {}
items = []
cnt = 0
for fn in args.input:
    basename = os.path.basename(fn)
    qDelayedTime = basename.startswith("META")
    qCyclonic = basename.find("nticycl") < 0
    with xr.open_dataset(fn) as ds:
        if "cost_association" in ds:
            ds = ds.drop("cost_association")

        for key in ds.keys(): attributes[key] = ds[key].attrs

        print("Started with", ds.obs.size, basename)
        dates = np.unique(ds.time.data.astype("datetime64[Y]")) + monthOffset + domOffset

        if qDelayedTime:
            dates = dates[dates < nrtYearMin]
        else: # NRT
            dates = dates[dates >= nrtYearMin]

        print("Dates", dates.size, dates.min(), dates.max())
        trks = mkTrackEndPoints(ds, dates)
        print("Tracks", trks.track.size)
        ds = ds.sel(obs=ds.obs[np.isin(ds.track, trks.track)])
        print("Pruned", ds.obs.size)
        df = mkDataFrames(ds, trks, dates, preDays, qCyclonic)
        if df is not None:
            items.append(df)
            print("DataFrame", df.shape[0])

df = items[0] if len(items) == 1 else pd.concat(items, ignore_index=True)

df = findNeighbors(df)
ds = xr.Dataset.from_dataframe(df).drop(("track"))

for key in sorted(ds.keys()):
    ds[key].encoding = {"zlib": True, "complevel": 3}
    if key in attributes:
        ds[key].attrs = attributes[key]
    elif (key in ("antiDistance", "cyclDistance")):
        ds[key].attrs = {"units": "meters", "min": 0}
    elif key.startswith("distPerDay"):
        ds[key].attrs = {"units": "meters/day", "min": 0}
    else:
        index = key.rfind("_")
        if index >= 0:
            alt = key[:index]
            if alt in attributes: ds[key].attrs = attributes[alt]
    if "min" in ds[key].attrs:
        ds[key].attrs["min"] = ds[key].data.min()
        ds[key].attrs["max"] = ds[key].data.max()
    elif key in ("duration", "preDays"):
        ds[key].attrs["min"] = ds[key].data.min().astype("timedelta64[D]").astype(str)
        ds[key].attrs["max"] = ds[key].data.max().astype("timedelta64[D]").astype(str)

print("Writing", ds.index.size, "to", ofn)
ds.to_netcdf(ofn)
