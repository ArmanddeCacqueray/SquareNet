import numpy as np
import matplotlib.pyplot as plt
import os
import requests
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/ArmanddeCacqueray/SquareNet/main/src/squarenet/data"
CACHE_DIR = Path.home() / ".sqdata_demo" 
AVAILABLE_METHODS = {"realdata": ["barbara", "lena", "france", "germany", "everest"],
                     "synthetic": ["square", "ball", "ring", "gaussian", "spiky", "holy", "4Dloop"]} 

def list_methods():
    """Return available sampling methods."""
    return AVAILABLE_METHODS


def plotpoints(points):
    """ plot a 2D projection of the point cloud"""
    plt.figure(figsize = (10, 10))
    plt.scatter(points[:, 0], points[:, 1], color = "black", s = 10000/len(points))
    plt.axis("equal")
    plt.axis("off")
    plt.show()

def place_at(points, position=(0, 0)):
    """
    Normalize point cloud to [0,1]^D and shift it at the specified position

    Parameters
    ----------
    points : numpy.ndarray
        Input point cloud.

    position : Target position of the minimum corner after normalization.

    Returns
    -------
    numpy.ndarray
        point cloud normalized and placed at the 
    """
    points = points.copy().astype(float) - points.min(axis = 0, keepdims = True)
    points = points/(points.max(axis = 0, keepdims = True) + 0.00001)
    points += np.array(position)[None, :]
    return points

def samplepoints(method="gaussian", size=(1_000_000, 2), plot_points=True):
    """
    Generate point clouds using various sampling strategies.

    Parameters
    ----------
    method : str
        Sampling method. Supported values:

        General methods (any dimension D):
        - "square"   : uniform in [-1, 1]^D
        - "ball"     : uniform in the unit ball
        - "ring"     : Gaussian with radial offset
        - "gaussian" : standard normal distribution
        - "spiky"    : L^alpha ball (alpha < 1)
        - "holy"     : hypercube with spherical holes

        Dataset-based methods (fixed dimensions):
        - "barbara", "lena", "france", "germany" : 2D datasets
        - "everest" : 3D dataset (elevation-based sampling)

    size : tuple of int
        (N, D) where N is the number of points and D the dimension.
        Note: D must match the dataset dimension for dataset-based methods.

    plot_points : bool, optional
        If True, display the generated points.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, D) containing sampled points.
    """

    N, D = size

    if method == "square":
        points = np.random.uniform(-1, 1, size=(N, D))

    elif method == "ball":
        # uniforme dans la boule unité en dimension D
        x = np.random.randn(N, D)
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        directions = x / norms
        radii = np.random.rand(N, 1) ** (1.0 / D)
        points = directions * radii

    elif method == "ring":
        # généralisation directe de ton idée
        x = np.random.randn(N, D)
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        points = x + x / norms

    elif method == "gaussian":
        points = np.random.randn(N, D)

    elif method == "spiky":
        alpha = 0.5
        c = 1.0

        points = np.empty((N, D), dtype=np.float32)
        idx = 0

        while idx < N:
            batch = int((N - idx) * 1.5)

            candidate = np.random.uniform(-1, 1, (batch, D))
            vals = np.sum(np.abs(candidate) ** alpha, axis=1)

            keep = candidate[vals <= c]

            n = min(len(keep), N - idx)
            if n > 0:
                points[idx:idx+n] = keep[:n]
                idx += n

    elif method == "holy":
        n_holes = 10
        centers = np.random.uniform(-0.7, 0.7, (n_holes, D))
        radii = np.random.uniform(0.1, 0.25, n_holes)

        points = np.empty((N, D), dtype=np.float32)
        idx = 0

        while idx < N:
            batch = int((N - idx) * 1.5)

            candidate = np.random.uniform(-1, 1, (batch, D))
            keep = np.ones(batch, dtype=bool)

            for c, r in zip(centers, radii):
                d = np.sum((candidate - c) ** 2, axis=1)
                keep &= (d > r ** 2)

            valid = candidate[keep]

            n = min(len(valid), N - idx)
            if n > 0:
                points[idx:idx+n] = valid[:n]
                idx += n
    
    elif method == "4Dloop":
        points = samplepoints("ball", size = (N, 2))
        points = np.concatenate([points, np.zeros((N, 1))], axis = 1)
        points = np.concatenate([points, np.zeros((N, 1))], axis = 1)
        points[:, 2] = 3*np.exp(-(points[:, 0]**2 + points[:, 1]**2)*3)
        points[:, 3] = 3*np.exp(-((points[:, 0] + points[:, 1])**2)*1)
        points = points[:, [2, 3, 1, 0]]

    elif method in AVAILABLE_METHODS["realdata"]:
        fpath = CACHE_DIR / f"{method}.npy"
        if not fpath.exists():
            loaddatasets()        
        data = np.load(fpath).astype(float)
        data -= data.min(keepdims = True)
        data /= data.max(keepdims = True)
        Nmax, D = data.shape
        idx = np.random.choice(Nmax, N)
        noise_ratio = 0.1/(Nmax**(1/D))
        noise = noise_ratio * np.random.randn(N, D)
        points = data[idx] + noise

    else:
        raise ValueError(f"Unknown method: {method}, availables: {AVAILABLE_METHODS}")

    if plot_points:
        plotpoints(points)
    return points

def loaddatasets(force=False):
    """
    Download dataset files from GitHub into a local cache directory.

    Parameters
    ----------
    force : bool
        If True, re-download files even if they already exist.
    """

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for name in AVAILABLE_METHODS["realdata"]:
        fname = f"{name}.npy"
        fpath = CACHE_DIR / fname

        if force or not fpath.exists():
            url = f"{BASE_URL}/{fname}"
            print(f"Downloading {fname}...")

            r = requests.get(url)
            r.raise_for_status()

            with open(fpath, "wb") as f:
                f.write(r.content)

    print(f"Datasets available in: {CACHE_DIR}")