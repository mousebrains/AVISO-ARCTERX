# AVISO-ARCTERX
Identifying tracks for the ARCTERX cruises

- `make -j4` handles the next two steps in parallel
- `./make.joined.py --monthDOM=424 --polygon=subset.polygon.yaml data/Ed*.nc data/ME*long*.nc` reads the AVISO files and prunes the eddies to a polygon, then and joins tracks together that have 1 or 2 day gaps between their start/end points and are within a specified distance/day. This handles data gaps due to storms.
- `make.dataframe.py` takes the output of `make.joined.py` and builds track summary information for all tracks that existed on a specified observation date.
-`classify.py` takes the output of `make.dataframe.py` and runs a classifier on the tracks to identify tracks that existed between two specified days of the year.

You should download the data from AVISO, but there is a set for testing available at [https://arcterx.ceoas.oregonstate.edu/AVISO/data/](ARCTERX-AVISO.)
