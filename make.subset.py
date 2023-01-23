#! /usr/bin/env python3
#
# Extract the subset of tracks within the lat/lon polygon
# and output both to pared down NetCDF
#
# N.B. This does not handle crossing the anti-merdian!
#
# Dec-2022, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
import yaml
import xarray as xr
import numpy as np
from shapely.geometry import Polygon, MultiPoint
import geopandas as gpd
import os
import time

parser = ArgumentParser()
parser.add_argument("input", nargs="+", type=str, help="Input GeoJSON files with AVISO data.")
parser.add_argument("--output", "-o", type=str, default="tpw",
                    help="Directory for the output files")
parser.add_argument("--polygon", type=str, default="subset.polygon.yaml",
                    help="YAML file with a polygon of lat/lon points")
args = parser.parse_args()

args.output = os.path.abspath(os.path.expanduser(args.output))
if not os.path.isdir(args.output):
    print("Making", args.output)
    os.makedirs(args.output, exist_ok=True, mode=0o755)

with open(args.polygon, "r") as fp: polygon = yaml.safe_load(fp.read())
if "polygon" not in polygon:
    raise ValueError("polygon not defined in " + args.polygon)
polygon = np.array(polygon["polygon"]) # Lat/Lon
polygon = gpd.GeoDataFrame({"geometry": Polygon(polygon)}, index=[0], crs=4326) # WGS84
bb = polygon.bounds

print("Limits Lat", bb.miny[0], bb.maxy[0], "Lon", bb.minx[0], bb.maxx[0])

for fn in args.input:
    stime = time.time()
    fn = os.path.abspath(os.path.expanduser(fn))
    (basename, ext) = os.path.splitext(os.path.basename(fn))
    qCyclonic = basename.find("nticyclonic") < 0
    qDelayedTime = basename.startswith("META")
    ofn  = os.path.join(args.output, basename + ".subset.spatial.nc")
    stime = time.time()
    with xr.open_dataset(fn) as ds:
        print("Initial", ds.obs.size, basename)
        # Get the tracks that are inside the crude lat/lon box
        a = ds.sel(obs=ds.obs[np.logical_and(ds.latitude >= bb.miny[0], ds.latitude <= bb.maxy[0])])
        print("BB lat limits", a.obs.size)

        a.longitude[a.longitude <    0] += 360 # Walked across the prime merdian westward
        a.longitude[a.longitude >= 360] -= 360 # Walked across the prime merdian eastward
        a.longitude[a.longitude >= 180] -= 360 # Wrap to [-180,+180)

        a = a.sel(obs=a.obs[np.logical_and(a.longitude >= bb.minx[0], a.longitude <= bb.maxx[0])])
        print("BB lon limits", a.obs.size)

        b = gpd.GeoDataFrame(
                dict(
                    track=a.track.data,
                    geometry=MultiPoint(np.array([a.longitude.data, a.latitude.data]).T).geoms,
                    ),
                crs=4326)
        tracks = np.unique(b.sjoin(polygon, how="right").track) # Tracks inside polygon
        print("Tracks inside polygon", tracks.size)
        ds = ds.sel(obs=ds.obs[np.isin(ds.track, tracks)])
        print("Post track selection", ds.obs.size)

        ds = ds.drop(("effective_contour_latitude", "effective_contour_longitude",
                      "speed_contour_latitude", "speed_contour_longitude",
                      "uavg_profile"))
        ds = ds.assign({"qCyclonic": qCyclonic, "qDelayedTime": qDelayedTime})

        for key in ds.keys(): 
            ds[key].encoding = dict(zlib=True, complevel=3)

        print("{:.2f}".format(time.time()-stime), "seconds to prune dataset")
        print("Writing", os.path.basename(ofn), ds.obs.size)
        ds.to_netcdf(ofn)
        print("Took {:.2f} seconds to write file".format(time.time() - stime))
