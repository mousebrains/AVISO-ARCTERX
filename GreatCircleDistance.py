#! /usr/bin/env python3
#
# Calculate the great circle distance on the oblate spheroid of the Earth
# using Vicenty's method
#
# Dec-2022, Pat Welch, pat@mousebrains.com

import numpy as np
import pandas as pd

def greatCircleDistance(lat1:np.array, lon1:np.array, lat2:np.array, lon2:np.array,
                        criteria:float=1e-8) -> np.array:
    # Calculate the great circle distance between lat0/lon0 and lat1/lon1
    # The expected error due to the oblate spheroid aspect of the earth is less than 1%

    rMajor = 6378137          # WGS-84 semi-major axis in meters
    f = 1/298.257223563       # WGS-84 flattening of the ellipsoid
    rMinor = (1 - f) * rMajor # WGS-84 semi-minor axis in meters

    if isinstance(lat1, pd.Series):
        lat1 = np.deg2rad(lat1.to_numpy()) # Degrees to radians as an numpy array instead of pandas
        lat2 = np.deg2rad(lat2.to_numpy())
        lon1 = np.deg2rad(lon1.to_numpy())
        lon2 = np.deg2rad(lon2.to_numpy())
    else:
        lat1 = np.deg2rad(lat1) # Degrees to radians as an numpy array instead of pandas
        lat2 = np.deg2rad(lat2)
        lon1 = np.deg2rad(lon1)
        lon2 = np.deg2rad(lon2)

    # Avoid a division problem when the points are identical
    qSame = np.logical_and(lat1 == lat2, lon1 == lon2) # Identical, zero distance
    qDiff = np.logical_not(qSame)

    lat1 = lat1[qDiff]
    lon1 = lon1[qDiff]
    lat2 = lat2[qDiff]
    lon2 = lon2[qDiff]

    tanU1 = (1 - f) * np.tan(lat1) # Tangent of reduced latitude
    tanU2 = (1 - f) * np.tan(lat2)
    cosU1 = 1 / np.sqrt(1 + tanU1**2) # trig ident from 1 = sin^2+cos^2, up to +-
    cosU2 = 1 / np.sqrt(1 + tanU2**2)
    sinU1 = tanU1 * cosU1 # Sine of reduced latitude
    sinU2 = tanU2 * cosU2

    dLon = lon2 - lon1 # difference of longitudes

    lambdaTerm = dLon # Initial guess of the lambda term

    for cnt in range(10): # Iteration loop through Vincenty's inverse problem to get the distance
        sinLambda = np.sin(lambdaTerm)
        cosLambda = np.cos(lambdaTerm)
        sinSigma = np.sqrt(
                (cosU2 * sinLambda)**2 + 
                (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda)**2
                )
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = np.arctan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * sinLambda / np.sin(sigma)
        cosAlpha2 = 1 - sinAlpha**2 # Trig identity
        cos2Sigma = cosSigma - 2 * sinU1 * sinU2 / cosAlpha2
        C = f / 16 * cosAlpha2 * (4 + f * (4 - 3 * cosAlpha2))
        lambdaPrime = dLon + \
                (1 - C) * f * sinAlpha * (
                        sigma +
                        C * sinSigma * (
                            cos2Sigma + 
                            C * cosSigma * (-1 + 2 * cos2Sigma**2)
                            )
                        )
        delta = np.abs(lambdaTerm - lambdaPrime)
        lambdaTerm = lambdaPrime
        if delta.max() < criteria: break

    u2 = cosAlpha2 * (rMajor**2 - rMinor**2) / rMinor**2
    A = 1 + u2/16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2/1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    deltaSigma = B * sinSigma * (
            cos2Sigma + 
            B / 4 * (
                cosSigma * (-1 + 2 * cos2Sigma**2) -
                B/6 * cos2Sigma * (-3 + 4 * sinSigma**2) * (-3 + 4 * cos2Sigma**2))
            )

    a = np.zeros(qSame.shape)
    a[qDiff] = rMinor * A * (sigma - deltaSigma) # Distance on the elipsoid
    return a

if __name__ == "__main__":
    from numpy.random import default_rng

    n = 10

    rng = default_rng(seed=123456789) # Random number generator with a fixed seed
    aLat = rng.random(n) * 180 - 90
    aLon = rng.random(n) * 360 - 180
    bLat = rng.random(n) * 180 - 90
    bLon = rng.random(n) * 360 - 180

    bLat[3:5] = aLat[3:5] # Force the same
    bLon[3:5] = aLon[3:5]

    print(aLat)
    print(bLat)
    print(aLon)
    print(bLon)
    print(greatCircleDistance(aLat, aLon, bLat, bLon))

