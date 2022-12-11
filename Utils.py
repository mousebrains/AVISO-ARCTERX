#! /usr/bin/env python3
#
# Utilities for my use
#
# Dec-2022, Pat Welch, pat@mousebrains.com

import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import MultiPoint, LineString
import os.path

def qEEZ(df:gpd.GeoDataFrame, name:str) -> np.array:
    qs = np.logical_or(df.SOVEREIGN1 == name, df.SOVEREIGN2 == name)
    qt = np.logical_or(df.TERRITORY1 == name, df.TERRITORY2 == name)
    q = np.logical_and(
            np.logical_or(qs, qt),
            df.LINE_TYPE != "Archipelagic Baseline")
    return q

def mkGDF(fn:str, latmin:float, latmax:float, lonmin:float, lonmax:float) -> gpd.GeoDataFrame:
    fn = os.path.abspath(os.path.expanduser(fn))

    with xr.open_dataset(fn) as ds:
        # Prune to the lat/lon box
        qLat = np.logical_and(ds.latitude >= latmin, ds.latitude <= latmax)
        qLon = np.logical_and(ds.longitude >= lonmin, ds.longitude <= lonmax)
        ds = ds.sel(obs=ds.obs[np.logical_and(qLat, qLon)])

        df = gpd.GeoDataFrame(
                dict(
                    qCyclonic = ds.qCyclonic.data,
                    track=ds.track,
                    time=ds.time,
                    geometry=MultiPoint(np.array((ds.longitude, ds.latitude)).T),
                    ),
                crs="EPSG:4326",
                )
        return df

def mkTracks(df:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    b = df.groupby("track")
    return b.geometry.apply(lambda x: LineString(x.tolist()))

def qMonthDOM(t:gpd.GeoDataFrame, month:int, DOM:int,
              dOffset:np.timedelta64=np.timedelta64(0, "D")) -> np.array:
    t = (t + dOffset).astype("datetime64[D]")
    months = t.astype("datetime64[M]")
    tm = (months - t.astype("datetime64[Y]")).astype(int) + 1
    td = (t - months).astype(int) + 1
    return np.logical_and(month == tm, DOM == td)

def pruneMonthDOM(df:gpd.GeoDataFrame, month:int, DOM:int, 
                  dOffset:np.timedelta64=np.timedelta64(0, "D")) -> gpd.GeoDataFrame:
    q = qMonthDOM(df.time.to_numpy(), month, DOM, dOffset)
    tracks = np.unique(df.track[q])
    q = np.isin(df.track, tracks)
    return df[q]

def pruneYear(df:gpd.GeoDataFrame, years:list) -> gpd.GeoDataFrame:
    if not years: return df # Nothing to do

    ty = df.time.to_numpy().astype("datetime64[Y]").astype(int) + 1970
    tracks = np.unique(df.track[np.isin(ty, years)])
    return df[np.isin(df.track, tracks)]
