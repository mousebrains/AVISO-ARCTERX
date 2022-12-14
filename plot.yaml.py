#! /usr/bin/env python3
#
# From subsetted data output by tracks.subset.py, do the following:
#
# Identify tracks that are within a lat/lon polygon on a specified date.
# Identify tracks that are outside a specified EEZ boundary during a specified period.
# plot these tracks
#
# Dec-2022, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
from shapely.geometry import LineString, Polygon, MultiPoint
import matplotlib.pyplot as plt
from cartopy  import crs as ccrs
import geopandas as gpd
import xarray as xr
import numpy as np
import yaml
import json
import gzip
import os
import sys

def qEEZ(df:gpd.GeoDataFrame, name:str) -> np.array:
    return np.logical_or(
            df.TERRITORY1 == name,
            df.TERRITORY2 == name,
            )

def qCyclonic(fn:str) -> bool:
    (name, ext) = os.path.splitext(os.path.basename(fn))
    return name.find("nticyclonic") < 0

def mkGDF(fn:str, latmin:float, latmax:float, lonmin:float, lonmax:float) -> gpd.GeoDataFrame:
    fn = os.path.abspath(os.path.expanduser(fn))

    with xr.open_dataset(fn) as ds:
        # Prune to the lat/lon box
        qLat = np.logical_and(ds.latitude >= latmin, ds.latitude <= latmax)
        qLon = np.logical_and(ds.longitude >= lonmin, ds.longitude <= lonmax)
        ds = ds.sel(obs=ds.obs[np.logical_and(qLat, qLon)])

        df = gpd.GeoDataFrame(
                dict(
                    track=ds.track,
                    time=ds.time,
                    geometry=MultiPoint(np.array((ds.longitude, ds.latitude)).T),
                    ),
                crs="EPSG:4326",
                )
        return df

def mkTracks(df:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    b = df.groupby("track", group_keys=True)
    return b.geometry.apply(lambda x: LineString(x.tolist()))

def qMonthDOM(t:gpd.GeoDataFrame, month:int, DOM:int,
              dOffset:np.timedelta64=np.timedelta64(0, "D")) -> np.array:
    t = (df.time.to_numpy() + dOffset).astype("datetime64[D]")
    months = t.astype("datetime64[M]")
    tm = (months - t.astype("datetime64[Y]")).astype(int) + 1
    td = (t - months).astype(int) + 1
    return np.logical_and(month == tm, DOM == td)

def pruneMonthDOM(df:gpd.GeoDataFrame, month:int, DOM:int,
                  dOffset:np.timedelta64=np.timedelta64(0, "D")) -> gpd.GeoDataFrame:
    q = qMonthDOM(df.time, Month, DOM, dOffset)
    tracks = np.unique(df.track[q])
    q = np.isin(df.track, tracks)
    return df[q]

def pruneSPoly(df:gpd.GeoDataFrame, poly:Polygon,
               month:int, DOM:int,
               dOffset:np.timedelta64=np.timedelta64(0,"D")) -> gpd.GeoDataFrame:
    q = qMonthDOM(df, month, DOM, dOffset)
    a = df[q]
    b = gpd.GeoDataFrame(
            dict(
                row=a.index,
                geometry=poly,
                ),
            crs = a.crs,
            index=a.index,
            )
    q = a.within(b) # Points in df which are within the polygon
    tracks = np.unique(a.track[q])
    return df[np.isin(df.track, tracks)]

parser = ArgumentParser()
parser.add_argument("input", nargs="+", type=str,
                    help="Input geojson files with AVISO data.")
parser.add_argument("--config", type=str, default="5may.yaml",
                    help="YAML file containing a start date, end date, and the initial polygon")
parser.add_argument("--eez", type=str, default="eez_boundaries_v11.gpkg",
                    help="EEZ geopackage file to read")
args = parser.parse_args()

crs = "EPSG:4326"

with open(os.path.abspath(os.path.expanduser(args.config)), "rb") as fp: config = yaml.safe_load(fp)

polygon = Polygon(np.array(config["startPoly"])[:,0:2])

eDuration = np.timedelta64(config["eDuration"], "D")
bDuration = np.timedelta64(config["bDuration"], "D")

sDOM   = config["sDOM"]
sMonth = config["sMonth"]

eez = gpd.read_file(args.eez)

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([config["lonmin"], config["lonmax"], config["latmin"], config["latmax"]])
ax.gridlines(draw_labels=True)
eez[qEEZ(eez, "Japan")].plot(ax=ax, color="c")
eez[qEEZ(eez, "Taiwan")].plot(ax=ax, color="y")
eez[qEEZ(eez, "Philippines")].plot(ax=ax, color="m")
eez[qEEZ(eez, "Overlapping claim area Taiwan / Japan / China")].plot(ax=ax, color="m")

for fn in args.input:
    fn = os.path.abspath(os.path.expanduser(fn))
    print("Reading on", fn)
    if fn.endswith(".gz"):
        with gzip.open(fn, "rb") as fp: df = gpd.read_file(fp)
        pass
    else:
        df = gpd.read_file(fn)
        df = df.to_crs(crs)

    print("Initial", df.track.size, np.unique(df.track).size)
    df = pruneMonthDOM(df, sDOM, sMonth)
    print("sDOM/Month", df.track.size, np.unique(df.track).size)
    df = pruneMonthDOM(df, sDOM, sMonth, bDuration)
    print("bDuration", df.track.size, np.unique(df.track).size)
    df = pruneMonthDOM(df, sDOM, sMonth, -eDuration)
    print("eDuration", df.track.size, np.unique(df.track).size)
    df = pruneSPoly(df, polygon, sDOM, sMonth)
    print("sPoly", df.track.size, np.unique(df.track).size)
    qCyclonic = df.qCyclonic.iloc[0]
    color = "blue" if qCyclonic else "red"

    qStart = qMonthDOM(df.time, config["sDOM"], config["sMonth"])
    qBefore = qMonthDOM(df.time, config["sDOM"], config["sMonth"], bDuration)

    df[qStart].plot(ax=ax, color=color, markersize=40, marker='o', edgecolor='b')
    df[qBefore].plot(ax=ax, color=color, markersize=20.5, marker='*', edgecolor='b')
    df2 = df.groupby("track", group_keys=True).geometry.apply(lambda x: LineString(x.tolist()))
    df2.plot(ax=ax, color=color, linewidth=1)
    break
    # df.plot(ax=ax, color="blue" if qCyclonic else "red", markersize=0.005)

ax.coastlines()
plt.show()
