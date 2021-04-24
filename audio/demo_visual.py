import numpy as np
# import matplotlib as mpl
# mpl.rc("savefig", dpi=200)
import matplotlib.pyplot as plt
import seaborn as sns

# categories = ['Mean duration of silence', 'Unique number of words',
#               'Unique number of words divided by total number of words', 'Number of words per second']
categories = ['', '', '', '']
data_eg = np.asarray([0.3194, 329, 0.2486, 5.002])
data_std = np.asarray([0.3581, 374, 0.2494, 4.6267])
data_eg = 1 - np.abs(data_std - data_eg) / data_std
ranges = [(0.0, 1), (0, 1), (0, 1), (0, 1)]


def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])


def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d - y1) / (y2 - y1)
                     * (x2 - x1) + x1)
    return sdata


class ComplexRadar():
    def __init__(self, fig, variables, ranges, n_ordinate_levels=6):
        angles = np.arange(0, 360, 360. / len(variables))
        axes = [fig.add_axes([0.1, 0.1, 0.9, 0.9], polar=True, label="axes{}".format(i))
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, labels=variables)
        [txt.set_rotation(angle - 90) for txt, angle in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x, 2)) for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1]  # hack to invert grid
                # gridlabels aren't reversed
            gridlabel[0] = ""  # clean up origin
            ax.set_rgrids(grid, labels=gridlabel, angle=angles[i])
            # ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]

    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)


# plotting
fig1 = plt.figure(figsize=(10, 10))
radar = ComplexRadar(fig1, categories, ranges)
radar.plot(data_eg)
radar.fill(data_eg, alpha=0.2)
radar.plot(np.ones(4))
radar.fill(np.ones(4), alpha=0.2)
print(data_eg)
plt.savefig('visualization.png')
