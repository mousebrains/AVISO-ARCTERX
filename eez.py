#! /usr/bin/env python3

from argparse import ArgumentParser
import numpy as np
import geopandas as gpd
from cartopy  import crs as ccrs
import matplotlib.pyplot as plt
import sys

def mkUnion(df:gpd.GeoDataFrame, *names) -> np.array:
    items = None
    for name in names:
        a = df[name].astype(str)
        a = a[a != "None"]
        items = a if items is None else np.union1d(a, items)
    return items

parser = ArgumentParser()
parser.add_argument("--eez", type=str, default="eez_boundaries_v11.gpkg",
        help="Exclusive Economic Zone GPKG filename")
grp = parser.add_mutually_exclusive_group()
grp.add_argument("--plot", "-p", action="store_true", help="Plot EEZs")
grp.add_argument("--territories", "-t", action="store_true", help="Print territory names")
grp.add_argument("--sovereign", "-s", action="store_true", help="Print soverein names")
grp.add_argument("--keys", "-k", action="store_true", help="Print keys")

parser.add_argument("--country", "-c", type=str, action="append", help="Country filter")

args = parser.parse_args()

df = gpd.read_file(args.eez)

if args.keys:
    print(df.iloc[0])
    sys.exit(0)

if args.territories:
    for name in mkUnion(df, "TERRITORY1", "TERRITORY2"):
        print(name)
    sys.exit(0)

if args.sovereign:
    for name in mkUnion(df, "SOVEREIGN1", "SOVEREIGN2"):
        print(name)
    sys.exit(0)

if args.country: # Filter on country names
    print(df)
    for name in args.country:
        q = np.logical_or(df.SOVEREIGN1 == name, df.SOVEREIGN2 == name)
        df = df[q]
        print(name)
        print(df)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.gridlines(draw_labels=True)
ax.coastlines()
df.plot(ax=ax)
plt.show()
