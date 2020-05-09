import numpy as np

import matplotlib
# matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.collections
import matplotlib.colors

from matplotlib import pyplot as plt
import matplotlib.animation


def make_trajectories_circle(n):
    """
    Returns a tensor of shape (N trajectories x T time x 2 coord)
    where coord is X or Y coorinates
    :param n:
    :return:
    """

    t = np.linspace(0, 1, 101)
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

    x = np.linspace(0., 1., 101)
    phases = np.arange(0, n) * np.pi / 16.
    x, phases = np.array(np.meshgrid(x, phases))
    y = np.sin(x * np.pi * 2 + phases) * .5 + .5

    return np.stack([x, y], axis=-1)


def make_anim_func(trajectories, lc):

    def animate(step):
        # TODO there are 2 ways we can plot traces:
        #  - Rolling, which looks good if start meets end (but not otherwise
        #  - Jumping, the other way round. In this case, we need to make sure we enter/exit the endpoints.
        rolled = np.roll(trajectories, -step, axis=1)
        section = rolled[:, :10, :]
        lc.set_segments(list(section))
        print(step)

    return animate


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

        animate = make_anim_func(trajectories, lc)

        anim = matplotlib.animation.FuncAnimation(
            fig, animate, interval=1000 / 30, frames=trajectories.shape[1],
        )
        anims.append(anim)

    plt.show()


if __name__ == '__main__':
    main()
