import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_fairness_radar(fobject, fairness_metrics=['acc', 'tpr', 'ppv', 'fpr', 'stp']) :
    # /!\ plotting with plotnine in polar coordinates is not possible,
    # so we have to make a matplotlib figure.

    data = fobject.parity_loss_metric_data[fairness_metrics].copy()
    data[data == np.inf] = np.NaN
    data = data.dropna(axis=1)


    colors = color_cycle(plt.cm.viridis) + color_cycle(plt.cm.plasma)[::-1]
    # np.random.shuffle(colors)
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler("color", colors)
    mpl.rcParams['grid.color'] = "#dfdfdf"
    mpl.rcParams['axes.edgecolor'] = "#444444"
    # mpl.rcParams['text.usetex'] = True
    # mpl.rcParams['text.latex.preamble'] = "\\usepackage{color}"
    # mpl.rcParams["font.sans-serif"].insert(0, 'Helvetica')

    fig, ax = plt.subplots(subplot_kw={"polar": True})

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    angles = np.linspace(0, 360, len(fairness_metrics)+1)
    # labels = ["\\textcolor{gray}{" + m + "}" for m in fairness_metrics]
    labels = fairness_metrics
    ax.set_thetagrids(angles[:-1], labels)
    ticks = np.linspace(0, np.max(data.to_numpy())*1.05, 5, endpoint=False)
    ticks = ticks.round(2)
    ax.set_rticks(ticks)
    ax.set_rlabel_position(0)

    for model in data.index :
        metrics = data.loc[model].to_numpy().tolist()
        ax.plot(np.radians(angles), metrics + metrics[:1], '-o', markersize=4)

    ax.legend(data.index)
    ax.set_title("Parity loss metric radar plot")

    return fig


def color_cycle(colormap, n=3) :
    color = colormap(np.linspace(0.25, 0.95, n))
    hexcolor = map(lambda rgb: '#%02x%02x%02x' % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)),
                   tuple(color[:, 0:-1]))
    return list(hexcolor)