#! /usr/bin/env python3
#
# From subsetted data, make a plot of the tracks
#
# Dec-2022, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
from Utils import qEEZ, mkGDF, pruneMonthDOM, mkTracks, qMonthDOM, pruneYear, dropSinglePoints
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from cartopy  import crs as ccrs
import os
import sys

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
parser.add_argument("--month", type=int, default=5, help="Month a track must exist on")
parser.add_argument("--dom", type=int, default=5, help="Day-of-month a track must exist on")
parser.add_argument("--duration", type=int, action="append", help="Additional date constraints")
parser.add_argument("--year", type=int, action="append", help="Year(s) to filter on")
parser.add_argument("--eez", type=str, default="eez_boundaries_v11.gpkg",
                    help="EEZ geopackage file to read")
parser.add_argument("--png", type=str, help="Output png filename")
parser.add_argument("--dpi", type=int, default=300, help="DPI of the plot for --png")
args = parser.parse_args()

sDOM = args.dom
sMonth = args.month
# N.B. sign flip since we are adjusting dates to sMonth/sDOM
durations = [] if args.duration is None else -np.array(args.duration).astype("timedelta64[D]")

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

for fn in args.input:
    df = mkGDF(fn, latmin, latmax, lonmin, lonmax) # NetCDF -> collection of points
    print("Started with", df.size)
    if df.empty: continue
    df = pruneYear(df, args.year)
    print("After --year filter", df.size)
    if df.empty: continue
    df = pruneMonthDOM(df, sMonth, sDOM)
    print("After month/dom filter", df.size)
    if df.empty: continue
    print(np.unique(df.track))
    for duration in durations:
        # duration >0 -> alive afterwards, <0 -> beforehand
        df = pruneMonthDOM(df, sMonth, sDOM, duration) # Alive n days afterwords
        print("Duration", duration, "->", df.size)
        if df.empty: break
        print(np.unique(df.track))
    if df.empty: continue
    df = dropSinglePoints(df)
    if df.empty: continue
    tracks = mkTracks(df) # Create LineString for each track
    if df.qCyclonic.any():
        nCyclonic += tracks.size
        color = "blue"
    else:
        nAnticyclonic += tracks.size
        color = "red"
    print("Working with", os.path.basename(fn), tracks.size)
    tracks.plot(ax=ax, color=color, linewidth=0.3, alpha=0.8)
    q = qMonthDOM(df.time.to_numpy(), sMonth, sDOM)
    if q.any(): df[q].plot(ax=ax, edgecolor=color, markersize=10, marker='o')
    for duration in durations:
        q = qMonthDOM(df.time.to_numpy(), sMonth, sDOM, duration)
        if q.any(): df[q].plot(ax=ax, edgecolor=color, markersize=25, marker='p')

tit = f"blue-cyclonic({nCyclonic}), red-anticyclonic({nAnticyclonic})"
if args.year:
    tit += f" sdate {args.year:04d}{sMonth:02d}{sDOM:02d}"
else:
    tit += f" monDOM {sMonth:02d}{sDOM:02d}"

if args.duration:
    tit += " " + ",".join(map(str, args.duration))

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
