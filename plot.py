#! /usr/bin/env python3
#
# From subsetted data, make a plot of the tracks
#
# Dec-2022, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
import xarray as xr
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPoint
import geopandas as gpd
import matplotlib.pyplot as plt
from cartopy  import crs as ccrs
import os

def dropSinglePoints(df:pd.DataFrame) -> pd.DataFrame:
    (tracks, cnts) = np.unique(df.track, return_counts=True)
    q = cnts > 1
    if not q.all():
        df = df[np.isin(df.track, tracks[q])]

    return df

def qEEZ(df:gpd.GeoDataFrame, name:str) -> np.array:
    qs = np.logical_or(df.SOVEREIGN1 == name, df.SOVEREIGN2 == name)
    qt = np.logical_or(df.TERRITORY1 == name, df.TERRITORY2 == name)
    q = np.logical_and(
            np.logical_or(qs, qt),
            df.LINE_TYPE != "Archipelagic Baseline")
    return q

def mkDataFrame(fn:str, latmin:float, latmax:float, lonmin:float, lonmax:float) -> pd.DataFrame:
    fn = os.path.abspath(os.path.expanduser(fn))
    qCyclonic = False if fn.find("nticyclonic") >= 0 else True

    with xr.open_dataset(fn) as ds:
        # Prune to the lat/lon box
        qLat = np.logical_and(ds.latitude >= latmin, ds.latitude <= latmax)
        qLon = np.logical_and(ds.longitude >= lonmin, ds.longitude <= lonmax)
        tracks = np.unique(ds.track[np.logical_and(qLat, qLon)])
        ds = ds.sel(obs=ds.obs[np.isin(ds.track, tracks)])

        return pd.DataFrame(
                dict(
                    track=ds.track,
                    time=ds.time,
                    latitude = ds.latitude,
                    longitude = ds.longitude,
                    qCyclonic = (np.zeros(ds.latitude.size) + qCyclonic).astype(bool),
                    ),
                )

def mkDates(df:pd.DataFrame, year:int, dMonth:np.array, dDOM:np.array,
            durations:np.array=None) -> tuple:
    if args.year:
        dates = (np.array(args.year) - 1970).astype("datetime64[Y]")
    else:
        dates = np.unique(df.time.to_numpy().astype("datetime64[Y]"))
    dates = dates + dMonth + dDOM

    if durations is None or not durations.size:
        return np.reshape(dates, (dates.size,1))

    durations = np.unique(durations) 
    zeroDays = np.timedelta64(0, "D")
    if not (durations == zeroDays).any():
        durations = np.insert(durations, 0, zeroDays)
    return np.tile(dates, (durations.size,1)).T + np.tile(durations, (dates.size,1))

def pruneTime(df:pd.DataFrame, year:int, dMonth:np.array, dDOM:np.array,
              durations:np.array) -> pd.DataFrame:
    dates = mkDates(df, args.year, dMonth, dDOM, durations)
    dMin = dates.min(axis=1)
    dMax = dates.max(axis=1)

    grps = df.groupby("track")
    t0 = grps.apply(lambda x: x.time.min())
    t1 = grps.apply(lambda x: x.time.max())

    q = np.logical_and(
            np.tile(t0, (dMin.size,1)).T <= np.tile(dMin, (t0.size,1)),
            np.tile(t1, (dMax.size,1)).T >= np.tile(dMax, (t1.size,1)),
            ).any(axis=1)

    tracks = t0.index[q]
    return df[np.isin(df.track, t0.index[q])]

def mkTracks(df:pd.DataFrame) -> gpd.GeoDataFrame:
    lines = df.groupby("track", group_keys=True).apply(
            lambda x: LineString(x.sort_values("time")[["longitude", "latitude"]].to_numpy()))
    return gpd.GeoDataFrame(
            dict(
                geometry = lines,
                ),
            crs=4326,
            )

parser = ArgumentParser()
parser.add_argument("input", nargs="+", type=str,
                    help="Input NetCDF files with subsetted AVISO data.")
parser.add_argument("--latmin", type=float, default=5,
                    help="Southern latitude limit in decimal degrees")
parser.add_argument("--latmax", type=float, default=30,
                    help="Northern latitude limit in decimal degrees")
parser.add_argument("--lonmin", type=float, default=116,
                    help="Eastern longitude limit in decimal degrees")
parser.add_argument("--lonmax", type=float, default=146,
                    help="Western longitude limit in decimal degrees")
parser.add_argument("--monthDOM", type=int, default=505, help="Month/Day a track must exist on")
parser.add_argument("--duration", type=int, action="append", help="Additional date constraints")
parser.add_argument("--year", type=int, action="append", help="Year(s) to filter on")
parser.add_argument("--eez", type=str, default="eez_boundaries_v11.gpkg",
                    help="EEZ geopackage file to read")
parser.add_argument("--png", type=str, help="Output png filename")
parser.add_argument("--dpi", type=int, default=300, help="DPI of the plot for --png")
args = parser.parse_args()

monthOffset = (np.floor(args.monthDOM / 100).astype(int) - 1).astype("timedelta64[M]")
domOffset = (np.mod(args.monthDOM, 100) - 1).astype("timedelta64[D]")
durations = np.array(args.duration).astype("timedelta64[D]")

latmin = min(args.latmin, args.latmax)
latmax = max(args.latmin, args.latmax)
lonmin = min(args.lonmin, args.lonmax)
lonmax = max(args.lonmin, args.lonmax)

eez = gpd.read_file(args.eez) # Exclusive Economoic Zone boundaries

fig = plt.figure(figsize=[6,5])
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_extent([lonmin, lonmax, latmin, latmax])
ax.gridlines(draw_labels=True)
# eez.plot(ax=ax, color="black") # Everybody
eez[qEEZ(eez, "United States")].plot(ax=ax, color="gray")
eez[qEEZ(eez, "Palau")].plot(ax=ax, color="cyan")
eez[qEEZ(eez, "Micronesia")].plot(ax=ax, color="goldenrod")
eez[qEEZ(eez, "Japan")].plot(ax=ax, color="purple")
eez[qEEZ(eez, "China")].plot(ax=ax, color="Pink")
eez[qEEZ(eez, "Taiwan")].plot(ax=ax, color="olive")
eez[qEEZ(eez, "Philippines")].plot(ax=ax, color="orange")
eez[qEEZ(eez, "Overlapping claim area Taiwan / Japan / China")].plot(ax=ax, color="red")

nCyclonic = 0
nAnticyclonic = 0

allDates = []
qDelayedTime = False
qNearRealTime = False

for fn in args.input:
    df = mkDataFrame(fn, latmin, latmax, lonmin, lonmax) # NetCDF -> pandas dataframe
    print("Started with", df.shape[0])
    if df.empty: continue
    df = pruneTime(df, args.year, monthOffset, domOffset, durations)
    print("After month/dom/durations filter", df.size)
    if df.empty: continue
    df = dropSinglePoints(df)
    print("After dropping single points", df.size)
    if df.empty: continue
    tracks = mkTracks(df) # Create LineString for each track
    print("Tracks", tracks.size)

    qDelayedTime |= fn.find("_DT_") >= 0
    qNearRealTime |= fn.find("_nrt_") >= 0

    if df.qCyclonic.any():
        nCyclonic += tracks.size
        color = "blue"
    else:
        nAnticyclonic += tracks.size
        color = "red"
    print("Working with", os.path.basename(fn), tracks.size)

    tracks.plot(ax=ax, color=color, linewidth=0.3, alpha=0.8)

    dates = mkDates(df, args.year, monthOffset, domOffset, durations)
    allDates.append(dates)

    for col in range(dates.shape[1]):
        delta = (dates[0,col] - dates[0,0]).astype("timedelta64[D]").astype(int)
        if delta == 0:
            marker = "*"
            markersize = 20
        elif delta > 0:
            marker = "p"
            markersize = 30
        else:
            marker = "o"
            markersize = 10

        pts = MultiPoint(df[np.isin(df.time, dates[:,col])][["longitude", "latitude"]].to_numpy())
        gpd.GeoDataFrame({"geometry": list(pts.geoms)}, crs=4326).plot(
                ax=ax,
                marker = marker,
                markersize = markersize,
                )

tit = f"blue-cyclonic({nCyclonic}), red-anticyclonic({nAnticyclonic})\n"

if qNearRealTime:
    if qDelayedTime:
        tit += "NRT+DT "
    else:
        tit += "NRT "
elif qDelayedTime:
    tit += "DT "

allDates = np.concatenate(allDates)

years = np.unique(allDates[:,0].astype("datetime64[Y]"))
if years.size == 1:
    tit += allDates[0,0].astype(str) + " "
elif years.size > 1:
    years.sort()
    if max(np.diff(years).astype(int)) == 1: # Contiguous
        tit += years[0].astype(str) + "-" + years[-1].astype(str)
    else: # Not contiguous
        tit += ",".join(years.astype(str).tolist())
    tit += " " + (monthOffset.astype(int) + 1).astype(str)
    tit += "/" + (domOffset.astype(int) + 1).astype(str)

if durations.size:
    tit += " days " + ",".join(np.sort(durations.astype(int)).astype(str))

ax.set_title(tit)
ax.coastlines()
if args.png:
    fn = os.path.abspath(os.path.expanduser(args.png))
    dirname = os.path.dirname(fn)
    if not os.path.isdir(dirname):
        print("Creating", dirname)
        os.makedirs(dirname, exist_ok=True, mode=0o755)

    print("Saving", fn)
    plt.savefig(fname=fn, dpi=args.dpi)
else:
    plt.show()
