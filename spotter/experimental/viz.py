import numpy as np
from scipy.spatial.transform import Rotation

from spotter.experimental import core

_DEFAULT_CMAP = "magma"


def lon_lat_lines(n: int = 6, pts: int = 100, radius: float = 1.0):
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
    u = [np.cos(obl), np.sin(obl), 0]
    u /= np.linalg.norm(u)
    u *= inc

    R = Rotation.from_rotvec(u)
    R *= Rotation.from_rotvec([0, 0, obl])
    R *= Rotation.from_rotvec([np.pi / 2, 0, 0])
    R *= Rotation.from_rotvec([0, 0, -theta])
    return R


def rotate_lines(lines, inc, obl, theta):
    R = rotation(inc, obl, theta)

    rotated_lines = np.array([R.apply(l.T) for l in lines]).T
    rotated_lines = np.swapaxes(rotated_lines.T, -1, 1)

    return rotated_lines


def plot_lines(lines, axis=(0, 1), ax=None, **kwargs):
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
    white_contour=True,
    radius: float = 1.0,
    n=6,
    ax=None,
    **kwargs,
):
    import matplotlib.pyplot as plt

    _inc = core.inclination_convention(inc)

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

    # contour
    sqrt_radius = radius
    ax.plot(sqrt_radius * np.cos(theta), sqrt_radius * np.sin(theta), **kwargs)
    if white_contour:
        ax.plot(sqrt_radius * np.cos(theta), sqrt_radius * np.sin(theta), c="w", lw=3)


def show(y, inclination=0.0, u=None, phase=0.0, ax=None, **kwargs):
    import matplotlib.pyplot as plt

    kwargs.setdefault("cmap", _DEFAULT_CMAP)
    kwargs.setdefault("origin", "lower")
    kwargs.setdefault("vmin", 0.0)
    kwargs.setdefault("vmax", 1.0)
    ax = ax or plt.gca()

    img = core.render(y, inclination, u, phase)
    ax.axis(False)
    ax.imshow(img, extent=(-1, 1, -1, 1), **kwargs)
    graticule(inclination, 0.0, phase, ax=ax)


def video(y, inclination=None, u=None, duration=4, fps=10, **kwargs):
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from IPython import display

    kwargs.setdefault("cmap", _DEFAULT_CMAP)
    kwargs.setdefault("origin", "lower")
    kwargs.setdefault("vmin", 0.0)
    kwargs.setdefault("vmax", 1.0)

    fig, ax = plt.subplots(figsize=(3, 3))
    im = plt.imshow(
        core.render(y, inclination, u, 0.0), extent=(-1, 1, -1, 1), **kwargs
    )
    plt.axis("off")
    plt.tight_layout()
    ax.set_frame_on(False)
    fig.patch.set_alpha(0.0)
    frames = duration * fps

    def update(frame):
        a = im.get_array()
        phase = np.pi * 2 * frame / frames
        a = core.render(y, inclination, u, phase)
        for art in list(ax.lines):
            art.remove()
        graticule(inclination, ax=ax, theta=phase, white_contour=False)

        im.set_array(a)
        return [im]

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=frames, interval=1000 / fps
    )
    video = ani.to_jshtml(embed_frames=True)
    html = display.HTML(video)
    plt.close()
    return display.display(html)
