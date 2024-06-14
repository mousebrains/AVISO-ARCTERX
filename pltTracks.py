#! /usr/bin/env python3
#
# Show tracks for eddies which last at least n days from a reference day/month
#
# Dec-2022, Pat Welch, pat@mousebrains.com
# Mar-2023, Pat Welch, pat@mousebrains.com, derivation from plot.py

from argparse import ArgumentParser
import xarray as xr
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPoint
import geopandas as gpd
import matplotlib.pyplot as plt
from cartopy  import crs as ccrs
import os

def qEEZ(df:gpd.GeoDataFrame, name:str) -> np.array:
    qs = np.logical_or(df.SOVEREIGN1 == name, df.SOVEREIGN2 == name)
    qt = np.logical_or(df.TERRITORY1 == name, df.TERRITORY2 == name)
    q = np.logical_and(
            np.logical_or(qs, qt),
            df.LINE_TYPE != "Archipelagic Baseline")
    return q

def mkTracks(df:pd.DataFrame) -> gpd.GeoDataFrame:
    grps = df.groupby(["track", "qCyclonic", "qDT"], group_keys=True)
    lines = grps.apply(
            lambda x: MultiPoint(x.sort_values("time")[["longitude", "latitude"]].to_numpy()))
    qCyclonic = grps.apply(lambda x: np.unique(x.qCyclonic))
    qDT = grps.apply(lambda x: np.unique(x.qDT))
    return gpd.GeoDataFrame(
            dict(
                qCyclonic = qCyclonic,
                qDT = qDT,
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
parser.add_argument("--lonmin", type=float, default=115,
                    help="Eastern longitude limit in decimal degrees")
parser.add_argument("--lonmax", type=float, default=150,
                    help="Western longitude limit in decimal degrees")
parser.add_argument("--monthDOM", type=int, default=120, help="Month/Day a track must exist on")
parser.add_argument("--duration", type=int, default=75, help="End date constraints")
parser.add_argument("--year", type=int, action="append", help="Year(s) to filter on")
parser.add_argument("--eez", type=str, default="eez_boundaries_v11.gpkg",
                    help="EEZ geopackage file to read")
parser.add_argument("--png", type=str, help="Output png filename")
parser.add_argument("--dpi", type=int, default=300, help="DPI of the plot for --png")
args = parser.parse_args()

monthOffset = (np.floor(args.monthDOM / 100).astype(int) - 1).astype("timedelta64[M]")
domOffset = (np.mod(args.monthDOM, 100) - 1).astype("timedelta64[D]")
duration = np.floor(args.duration).astype("timedelta64[D]")

latmin = min(args.latmin, args.latmax)
latmax = max(args.latmin, args.latmax)
lonmin = min(args.lonmin, args.lonmax)
lonmax = max(args.lonmin, args.lonmax)

frames = []

for fn in args.input:
    with xr.open_dataset(fn) as ds:
        # First prune full dataset to tracks that existed on both the start end end dates
        yr = ds.time.data.astype("datetime64[Y]")
        t0 = yr + monthOffset + domOffset # Starting date
        t1 = t0 + duration # Ending date
        tracks = np.intersect1d(ds.track.data[ds.time == t0], ds.track.data[ds.time == t1])
        qTrk = np.isin(ds.track.data, tracks)
        qRng = np.logical_and(ds.time.data >= t0, ds.time.data <= t1)
        ds = ds.sel(obs=ds.obs[np.logical_and(qTrk, qRng)])
        ds = ds[["latitude", "longitude", "time", "track"]]
        yr = ds.time.data.astype("datetime64[Y]")
        t0 = yr + monthOffset + domOffset # Starting date
        ds["dt"] = ds.time - t0
        ds["qCyclonic"] = False if fn.find("nticyclonic") >= 0 else True
        ds["qDT"] = fn.find("_DT_") > 0
        frames.append(ds.to_dataframe())

df = pd.concat(frames)
trks = mkTracks(df)
trksCycl = trks[trks.qCyclonic == True]
trksAnti = trks[trks.qCyclonic == False]
nCyclonic = trksCycl.shape[0]
nAnticyclonic = trksAnti.shape[0]
nDT  = sum(trks.qDT == True)
nNRT = sum(trks.qDT == False)

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

trksCycl.plot(ax=ax, color="blue", linewidth=0.3, alpha=0.8, markersize=0.2)
trksAnti.plot(ax=ax, color="red",  linewidth=0.3, alpha=0.5, markersize=0.2)

tit = f"blue-cyclonic({nCyclonic}), red-anticyclonic({nAnticyclonic})\n"

if nNRT:
    tit += "NRT+DT " if nDT else "NRT "
elif nDT:
    tit += "DT "

years = np.unique(df.time[df.dt == np.timedelta64(0, "D")]).astype("datetime64[Y]")
if years.size == 1:
    tit += years[0].astype(str)
elif years.size > 1:
    years.sort()
    if max(np.diff(years).astype(int)) == 1: # Contiguous
        tit += years[0].astype(str) + "-" + years[-1].astype(str)
    else: # Not contiguous
        tit += ",".join(years.astype(str).tolist())
    tit += " " + (monthOffset.astype(int) + 1).astype(str)
    tit += "/" + (domOffset.astype(int) + 1).astype(str)

tit += " days " + duration.astype(int).astype(str)

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
