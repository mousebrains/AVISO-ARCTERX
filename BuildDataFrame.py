#! /usr/bin/env python3
#
# Take the output of make.dataframe.py and build a Pandas dataframe
# that can be used in classification.
#
# Jan-2023, Pat Welch, pat@mousebrains.com

import numpy as np
import pandas as pd
import xarray as xr

def mkDataFrame(fn:str, minDuration:float) -> pd.DataFrame:
    omega = 2 * np.pi / (23 * 3600 + 56 * 60 + 4.1) # radians/sec for earth's rotation

    with xr.open_dataset(fn) as ds: df = ds.to_dataframe()
    df.preDays = df.preDays.astype("timedelta64[D]").astype(int) # classifiers don't like datetime
    minDuration = np.timedelta64(minDuration, "D");
    df["qPersistent"] = minDuration <= df.duration # Will it live at least this duration
    print("Persistent", 
          df.qPersistent.sum(), 
          df.shape[0], 
          df.qPersistent.sum() / df.shape[0] * 100)

    df["f"] = 2 * omega * np.sin(np.deg2rad(df.latitude)) # Coriolis parameter

    # We don't know the thickness of the eddy, so we don't know the mass.
    # We'll assume a circular disk of speed_radius_mean.
    # The moment of inertial is then 0.5 m r^2
    # and the kinetic energy is 0.75 m r^2 omega^2
    # where omega is the angular speed, speed_average_mean / speed_radius_mean
    # So we have KE ~= 3/4 * speed_average_mean^2
    df["KE"] = 3/4 * df.speed_average_mean**2

    # Potential energy of a disk
    df["PE"] = df.effective_area_mean * df.effective_contour_height

    return df

def shuffleDataFrame(rng:np.random.Generator, df:pd.DataFrame) -> pd.DataFrame:
    # random sample without replacment
    return df.sample(n=df.shape[0], replace=False, random_state=rng)

def splitDataFrame(rng:np.random.Generator, df:pd.DataFrame, fracTesting:float) -> tuple:
    # Split into training/testing sets by year
    year = df.date.to_numpy().astype("datetime64[Y]").astype(int) + 1970
    years = rng.permutation(np.unique(year)) # Unique years randomly permuted
    nTest = np.round(years.size * fracTesting).astype(int)
    return (df[np.isin(year, years[nTest:])],
            df[np.isin(year, years[:nTest])])

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("fn", type=str, nargs="+", help="Input file(s)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Minimum duration for eddies in days")
    parser.add_argument("--seed", type=int, default=1234567890, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    for fn in args.fn:
        df = mkDataFrame(fn, args.duration)
        print(fn)
        print(df)
        df = shuffleDataFrame(rng, df)
        print("Shuffled"); print(df)
        (tst, trn) = splitDataFrame(rng, df, 0.25)
        print("Test"); print(tst)
        print("Training"); print(trn)
