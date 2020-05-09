import numpy as np

import matplotlib
# matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.collections
import matplotlib.colors

from matplotlib import pyplot as plt
import matplotlib.animation
from dataclasses import dataclass


def make_trajectories_circle(n):
    """
    Returns a tensor of shape (N trajectories x T time x 2 coord)
    where coord is X or Y coorinates
    :param n:
    :return:
    """

    t = np.linspace(0, 1, 101)[:-1]
    offsets = np.linspace(.5, 1, n)
    t, offsets = np.meshgrid(t, offsets)

    x = np.sin(t * np.pi * 2) * offsets * .5 + .5
    y = np.cos(t * np.pi * 2) * offsets * .5 + .5

    return np.stack([x, y], axis=-1)


def make_trajectories_sine(n):
    """
    Returns a tensor of shape (N trajectories x T time x 2 coord)
    where coord is X or Y coorinates
    :param n:
    :return:
    """

    t = np.linspace(0., 1., 101)
    phases = np.arange(0, n) * np.pi / 16.
    x, phases = np.array(np.meshgrid(t, phases))
    y = np.sin(x * np.pi * 2 + phases) * .5 + .5

    return np.stack([x, y], axis=-1)


def make_anim_func(fig, lc, trajectories, step_size):

    def animate(step):
        # TODO there are 2 ways we can plot traces:
        #  - Rolling, which looks good if start meets end (but not otherwise
        #  - Jumping, the other way round. In this case, we need to make sure we enter/exit the endpoints.

        rolled = np.roll(trajectories, -step, axis=1)
        section = rolled[:, :step_size, :]

        # section = trajectories[:, step:step+step_size, :]

        lc.set_segments(list(section))
        print(step)

    anim = matplotlib.animation.FuncAnimation(
        fig, animate, interval=1000 / 30, frames=trajectories.shape[1],
    )

    return anim




@dataclass
class Field:
    """A simple representation of a 2D force field"""
    col_bins: np.ndarray
    row_bins: np.ndarray
    arrows: np.ndarray

    def __init__(self, col_bins, row_bins, arrows):
        assert arrows.shape[0] == 2
        assert len(col_bins) - 1 == arrows.shape[2]
        assert len(row_bins) - 1 == arrows.shape[1]
        self.col_bins = col_bins
        self.row_bins = row_bins
        self.arrows = arrows

    @property
    def xbin_centers(self):
        return (self.col_bins[1:] + self.col_bins[:-1]) * .5

    @property
    def ybin_centers(self):
        return (self.row_bins[1:] + self.row_bins[:-1]) * .5

    def random_init_points(self, count):
        return np.array([
            np.random.random(count) * (np.max(bins) - np.min(bins)) + np.min(bins)
            for bins in [self.col_bins, self.row_bins]])

    def simulate_field(self, init, step_count, dt=.1):
        """
        generate trajectories in this field using simple euler method
        :param init:
        :param step_count:
        :param dt:

        :return: a tensor of shape (N trajectories x T time x 2 coord)
        where N=len(init), T=step_count
        """

        pos = np.empty((init.shape[1], step_count, 2))

        step = 0
        pos[:, step, :] = init.T

        for step in range(1, step_count):
            yidx = np.digitize(pos[:, step-1, 0], self.col_bins) - 1
            xidx = np.digitize(pos[:, step-1, 1], self.row_bins) - 1

            valid = (
                    (0 <= xidx) & (xidx < self.arrows.shape[1]) &
                    (0 <= yidx) & (yidx < self.arrows.shape[2])
            )

            pos[~valid, step, :] = np.nan

            dxy = self.arrows[:, xidx[valid], yidx[valid]]

            new_pos = pos[valid, step - 1, :] + dxy.T * dt
            pos[valid, step, :] = new_pos

        return pos


def example_field(xrange, yrange):
    """
    :return: 1 matrix of shape size(X) x size(Y)
    """
    xbins = np.linspace(*xrange, 41)
    ybins = np.linspace(*yrange, 31)

    x = (xbins[:-1] + xbins[1:]) * .5
    y = (ybins[:-1] + ybins[1:]) * .5

    x, y = np.meshgrid(x, y)
    z = x * np.exp(-np.square(x) - np.square(y))
    v, u = np.gradient(z, .2, .2)

    return Field(xbins, ybins, np.array([u, v]))


def example_quiver():

    field = example_field((-2, 2), (-1, 2.))

    init = field.random_init_points(count=2000)

    step_count = 40
    trajectories = field.simulate_field(init, step_count=step_count)

    fig, axs = plt.subplots(ncols=2, sharex='all', sharey='all')

    ax = axs[0]
    ax.set(aspect='equal')
    q = ax.quiver(field.xbin_centers, field.ybin_centers, field.arrows[0], field.arrows[1], color='xkcd:silver')

    offsets = np.random.random_integers(0, step_count-1, len(trajectories))

    step_size = 5

    # we use nan to represent the end of a trajectory
    # padding as many nans as a trace hides the "jump" as the trace rolls
    padded = np.pad(trajectories, pad_width=[(0, 0), (0, step_size), (0, 0)], constant_values=np.nan)
    rolled = np.array([np.roll(tr, off, axis=0) for off, tr in zip(offsets, padded)])
    trajectories = rolled

    ax = axs[1]
    ax.set(aspect='equal')
    lc = matplotlib.collections.LineCollection(list(trajectories), linewidths=.25, colors='xkcd:teal')
    ax.add_collection(lc)
    anim = make_anim_func(fig, lc, trajectories, step_size=step_size)

    plt.show()


def main():

    trajectories_0 = make_trajectories_sine(4)
    trajectories_1 = make_trajectories_circle(4)
    all_trajectories = [trajectories_0, trajectories_1]

    fig, axs = plt.subplots(ncols=len(all_trajectories), figsize=(4, 3), dpi=100)

    anims = []

    for i, ax in enumerate(axs):
        trajectories = all_trajectories[i]

        ax.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1), aspect='equal')
        lc = matplotlib.collections.LineCollection(list(trajectories), colors='xkcd:silver', linestyles=':')
        ax.add_collection(lc)

        colors = matplotlib.colors.to_rgba_array(['xkcd:teal', 'xkcd:blood', 'xkcd:charcoal', 'xkcd:orange'])

        colors[:, -1] = np.linspace(0.25, 1., 4)

        lc = matplotlib.collections.LineCollection(list(trajectories), linewidths=2, colors=colors)
        ax.add_collection(lc)

        anim = make_anim_func(fig, lc, trajectories, 10)

        anims.append(anim)
    plt.show()


if __name__ == '__main__':
    # main()
    example_quiver()
