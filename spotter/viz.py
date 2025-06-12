"""
Visualization utilities for spherical maps and stellar surfaces.

This module provides functions for plotting HEALPix maps, graticules, and
generating videos of rotating stars.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from spotter import core

_DEFAULT_CMAP = "magma"


def lon_lat_lines(n: int = 6, pts: int = 100, radius: float = 1.0):
    """
    Generate latitude and longitude lines on a sphere.

    Parameters
    ----------
    n : int or tuple, optional
        Number of latitude lines (and longitude lines if tuple).
    pts : int, optional
        Number of points per line.
    radius : float, optional
        Sphere radius.

    Returns
    -------
    lat : ndarray
        Latitude lines.
    lon : ndarray
        Longitude lines.
    """

    assert isinstance(n, int) or len(n) == 2

    if isinstance(n, int):
        n = (n, 2 * n)

    n_lat, n_lon = n

    sqrt_radius = radius

    _theta = np.linspace(0, 2 * np.pi, pts)
    _phi = np.linspace(0, np.pi, n_lat + 1)
    lat = np.array(
        [
            (r * np.cos(_theta), r * np.sin(_theta), np.ones_like(_theta) * h)
            for (h, r) in zip(sqrt_radius * np.cos(_phi), sqrt_radius * np.sin(_phi))
        ]
    )

    _theta = np.linspace(0, np.pi, pts // 2)
    _phi = np.linspace(0, 2 * np.pi, n_lon + 1)[0:-1]
    radii = np.sin(_theta)
    lon = np.array(
        [
            (
                sqrt_radius * radii * np.cos(p),
                sqrt_radius * radii * np.sin(p),
                sqrt_radius * np.cos(_theta),
            )
            for p in _phi
        ]
    )

    return lat, lon


def rotation(inc, obl, theta):
    """
    Compute the rotation for given inclination, obliquity, and phase.

    Parameters
    ----------
    inc : float
        Inclination in radians.
    obl : float
        Obliquity in radians.
    theta : float
        Rotation phase in radians.

    Returns
    -------
    R : scipy.spatial.transform.Rotation
        Rotation object.
    """

    u = [np.cos(obl), np.sin(obl), 0]
    u /= np.linalg.norm(u)
    u *= inc

    R = Rotation.from_rotvec(np.array(u))
    R *= Rotation.from_rotvec([0, 0, obl])
    R *= Rotation.from_rotvec([np.pi / 2, 0, 0])
    R *= Rotation.from_rotvec([0, 0, -theta])
    return R


def rotate_lines(lines, inc, obl, theta):
    """
    Rotate lines by given inclination, obliquity, and phase.

    Parameters
    ----------
    lines : ndarray
        Input lines.
    inc : float
        Inclination in radians.
    obl : float
        Obliquity in radians.
    theta : float
        Rotation phase in radians.

    Returns
    -------
    rotated_lines : ndarray
        Rotated lines.
    """

    R = rotation(inc, obl, theta)

    rotated_lines = np.array([R.apply(l.T) for l in lines]).T
    rotated_lines = np.swapaxes(rotated_lines.T, -1, 1)

    return rotated_lines


def plot_lines(lines, axis=(0, 1), ax=None, **kwargs):
    """
    Plot lines on a matplotlib axis.

    Parameters
    ----------
    lines : ndarray
        Lines to plot.
    axis : tuple, optional
        Axes to plot (default (0, 1)).
    ax : matplotlib axis, optional
        Axis to plot on.
    **kwargs
        Additional plot arguments.
    """

    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
        if ax is None:
            ax = plt.subplot(111)

    # hide lines behind
    other_axis = list(set(axis).symmetric_difference([0, 1, 2]))[0]
    behind = lines[:, other_axis, :] < 0
    _xyzs = lines.copy().swapaxes(1, 2)
    _xyzs[behind, :] = np.nan
    _xyzs = _xyzs.swapaxes(1, 2)

    for i, j in _xyzs[:, axis, :]:
        ax.plot(i, j, **kwargs)


def graticule(
    inc: float,
    obl: float = 0.0,
    theta: float = 0.0,
    pts: int = 100,
    radius: float = 1.0,
    n=6,
    ax=None,
    **kwargs,
):
    """
    Plot a graticule (latitude/longitude grid) on a sphere.

    Parameters
    ----------
    inc : float
        Inclination in radians.
    obl : float, optional
        Obliquity in radians.
    theta : float, optional
        Rotation phase in radians.
    pts : int, optional
        Number of points per line.
    radius : float, optional
        Sphere radius.
    n : int or tuple, optional
        Number of latitude/longitude lines.
    ax : matplotlib axis, optional
        Axis to plot on.
    **kwargs
        Additional plot arguments.
    """

    import matplotlib.pyplot as plt

    _inc = np.pi / 2 - inc

    if ax is None:
        ax = plt.gca()
        if ax is None:
            ax = plt.subplot(111)

    kwargs.setdefault("c", kwargs.pop("color", "k"))
    kwargs.setdefault("lw", kwargs.pop("linewidth", 1))
    kwargs.setdefault("alpha", 0.2)

    # plot lines
    lat, lon = lon_lat_lines(pts=pts, radius=radius, n=n)
    lat = rotate_lines(lat, _inc, obl, theta)
    plot_lines(lat, ax=ax, **kwargs)
    lon = rotate_lines(lon, _inc, obl, theta)
    plot_lines(lon, ax=ax, **kwargs)
    theta = np.linspace(0, 2 * np.pi, 2 * pts)


def show(
    y,
    inc=np.pi / 2,
    obl=0.0,
    u=None,
    xsize=800,
    phase=0.0,
    ax=None,
    radius=None,
    period=None,
    rv=False,
    **kwargs,
):
    """
    Show a rendered map with graticule.

    Parameters
    ----------
    y : array_like
        HEALPix map.
    inc : float, optional
        Inclination in radians.
    obl : float, optional
        Obliquity in radians.
    u : array_like or None, optional
        Limb darkening coefficients.
    xsize : int, optional
        Output image size.
    phase : float, optional
        Rotation phase in radians.
    ax : matplotlib axis, optional
        Axis to plot on.
    **kwargs
        Additional plot arguments.
    """

    import matplotlib.pyplot as plt

    kwargs.setdefault("cmap", "RdBu_r" if rv else _DEFAULT_CMAP)
    kwargs.setdefault("origin", "lower")
    # kwargs.setdefault("vmin", 0.0)
    # kwargs.setdefault("vmax", 1.0)
    ax = ax or plt.gca()

    img = core.render(
        y,
        inc,
        u,
        phase,
        obl,
        xsize=xsize,
        radius=radius if rv else None,
        period=period if rv else None,
    )
    plt.setp(ax.spines.values(), visible=False)
    ax.tick_params(left=False, labelleft=False)
    ax.tick_params(bottom=False, labelbottom=False)
    ax.patch.set_visible(False)
    ax.imshow(img, extent=(-1, 1, -1, 1), **kwargs)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    graticule(inc, obl, phase, ax=ax)


def video(
    y,
    inc=None,
    obl=0.0,
    u=None,
    duration=4,
    fps=10,
    radius=None,
    period=None,
    rv=False,
    **kwargs,
):
    """
    Create an HTML video of a rotating map (for Jupyter notebooks).

    Parameters
    ----------
    y : array_like
        HEALPix map.
    inc : float, optional
        Inclination in radians.
    obl : float, optional
        Obliquity in radians.
    u : array_like or None, optional
        Limb darkening coefficients.
    duration : int, optional
        Duration of the video in seconds.
    fps : int, optional
        Frames per second.
    render_fun: callable, optional
        A function of phase that renders the star. Default is core.render
    **kwargs
        Additional plot arguments.

    Returns
    -------
    None
    """

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from IPython import display

    kwargs.setdefault("cmap", "RdBu_r" if rv else _DEFAULT_CMAP)
    kwargs.setdefault("origin", "lower")

    def render_fun(phase):
        return core.render(
            y,
            inc,
            u,
            phase,
            obl,
            period=period if rv else None,
            radius=radius if rv else None,
        )

    inc = inc or 0.0

    fig, ax = plt.subplots(figsize=(3, 3))
    im0 = render_fun(0.0)
    im = plt.imshow(im0, extent=(-1, 1, -1, 1), **kwargs)
    plt.axis("off")
    plt.tight_layout()
    ax.set_frame_on(False)
    fig.patch.set_alpha(0.0)
    frames = duration * fps

    def update(frame):
        a = im.get_array()
        phase = np.pi * 2 * frame / frames
        a = render_fun(phase)
        for art in list(ax.lines):
            art.remove()
        graticule(inc, ax=ax, theta=phase, obl=obl)

        im.set_array(a)
        return [im]

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=frames, interval=1000 / fps
    )
    video = ani.to_jshtml(embed_frames=True)
    html = display.HTML(video)
    plt.close()
    return display.display(html)
