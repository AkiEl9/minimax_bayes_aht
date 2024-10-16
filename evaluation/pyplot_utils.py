from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mc
import numpy as np
from matplotlib.transforms import offset_copy

label_dict = {
    "random": "Random",
    "maximin_utility": "MU", #"Maximin Utility",#r"$\max_\pi \min_\beta U(\pi, \beta)$",
    "minimax_regret": "MR", #"Minimax Regret", #r"$\min_\pi \max_\beta R(\pi, \beta)$",
    "uniform": "PBR",#"Uniform", #r"$\max_\pi U(\pi, \mathcal{U}(\{\Sigma(\mathcal{B}^\text{train})\}))$",
    "main_test_set": r"$\Sigma(\mathcal{B}^\text{test})$",
    "train_set_maximin_utility": r"$\beta^*_U$",
    "train_set_minimax_regret": r"$\beta^*_R$",
    "train_set_maximin_utility_average": r"$\beta^*_U$",
    "train_set_minimax_regret_average": r"$\beta^*_R$",
    "train_set_uniform": r"$\Sigma(\mathcal{B}^\text{train})$",
    "meltingpot_test_set": r"$\Sigma(\mathcal{B}^\text{Melting Pot})$",
    "minimax_auto_regret": r"Minimax $\bar R$",
    "auto_sgda": r"Minimax $\bar R$ Fictitious Play",
    "self_play": "SP", #"Self-play",
    "fictitious_play": "FP", #"Fictitious Play",
    "vmpo": "V-MPO",
    "acb": "ACB",
    "opre": "OPRE",

    "tft": "Tit-for-Tat",
    "cud": "Cooperate-until-Defected",

    "uniform_utility": r"$U_\text{avg}$",#r"$p(\pi, \Sigma(\mathcal{B}^\text{train}))$",
    "worst_case_utility": r"$U_\text{min}$", #r"$U^-(\pi, \Sigma(\mathcal{B}^\text{train}))$"
    "worst_case_regret": r"$R_\text{max}$",  # r"$U^-(\pi, \Sigma(\mathcal{B}^\text{train}))$"

}

color_dict = {
    # regret based
    "minimax_regret": "#54a8a8" ,#"#85e0e0",
    "minimax_auto_regret": "#33cccc",

    "auto_sgda": "#248f8f",


    # Utility based
    "maximin_utility": "#3a853a",#"#85e085",

    # uniform
    "uniform": "#c68c53",


    # self-play to PSRO
    "self_play": "#ff8080",
    "fictitious_play": "#c90202",#"#ff0000",

    "random": "#8a8a8a",

    "tft": "#ae1f97",
    "cud": "#d5bfd2",
}

plt.rc('text', usetex=True)
plt.rc("font", size=20)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def hex_to_rgb(hexcode):
    # Remove '#' if present
    hexcode = hexcode.lstrip('#')

    # Convert hex to RGB
    r = int(hexcode[0:2], 16)
    g = int(hexcode[2:4], 16)
    b = int(hexcode[4:6], 16)

    return np.array([r, g, b])


def set_box_color(bp, color):
    #color = "#555555"
    color = hex_to_rgb(color) * 0.6 / 255
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['means'], color=color)



def make_grouped_barplot(data: dict, name="grouped_barplot", plot_type="utility"):
    # data[appraoch]
    num_runs = len(data)
    plt.figure()

    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightgray', 'gray']
    widths = 1. / (num_runs * 1.2)

    spacings = np.linspace(-0.7 + widths * 0.5, 0.7 - widths * 0.5, num=num_runs)
    datamin = np.inf
    datamax = -np.inf
    for i, (approach, all_values) in enumerate(data.items()):
        n_runs = len(all_values)
        values = np.mean(all_values, axis=0)
        ste = np.std(all_values, axis=0) / np.sqrt(n_runs)

        x = np.arange(len(values))
        datamin = min([datamin, np.min(values)])
        datamax = max([datamax,  np.max(values)])

        values[values==0.] = 1.

        bars = plt.bar(
            x + (i - len(data)//2) * widths, values, width=widths*0.9, label=label_dict.get(approach, approach),
            color=color_dict.get(approach, "#8a8a8a"),
            #edgecolor="gray"
        )
        plt.errorbar(x + (i - len(data)//2) * widths, values, yerr=ste, fmt='None', ecolor='black')

        #plt.bar_label(bars, )

    if "meltingpot" in name:
        ticks = [
                    f"Scenario {i}" for i in range(len(values))
                ]
    else:
        ticks = [
            f"Scenario {i}" for i in range(len(values) - 1)
        ] + ["Self-play"]

    plt.xticks(x, ticks, rotation=90)

    norm = (datamax - datamin) * 0.01
    plt.ylim(bottom=datamin - norm)

    plt.ylabel(plot_type.capitalize())


    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{name}')

    plt.clf()


def make_grouped_boxplot(data, name="grouped_boxplot", whiskers=(0, 100), plot_type="regret"):
    # data[approach][metric (run type)]

    plt.figure(figsize=(5.5, 4))

    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightgray', 'gray']

    num_approaches = len(data)

    widths = 1. / (num_approaches)

    spacings = np.linspace(-0.6 + widths * 0.5, 0.6 - widths * 0.5, num=num_approaches)

    datamin = np.inf
    datamax = -np.inf
    for (label, all_values), spacing in zip(data.items(), spacings):

        # simple 1 run data
        if len(np.array(list(all_values.values())[0]).shape) < 2:
            values = [
                all_set_values
                for all_set_values in all_values.values()
            ]
        else:
            values = [
                np.mean(all_set_values, axis=0)
                for all_set_values in all_values.values()
            ]

        datamin = min([v for sublist in values for v in sublist] + [datamin])
        datamax = max([v for sublist in values for v in sublist] + [datamax])

        boxplot = plt.boxplot(
            values, positions=np.arange(len(values))*2.0+spacing, sym='', widths=widths,
            whis=whiskers,
            patch_artist=True,
            meanline=True, showmeans=True,
        )

        set_box_color(boxplot, color_dict.get(label, "#8a8a8a"))
        for patch in boxplot["boxes"]:
            patch.set_facecolor(color=color_dict.get(label, "#8a8a8a"))

        plt.fill_between([], [], color=color_dict.get(label, "#8a8a8a"), label=label_dict.get(label, label))

    plt.xticks(range(0, len(all_values) * 2, 2), [label_dict.get(k, k) for k in all_values], rotation=0)
    plt.xlim(-1, len(all_values)*2-1)

    plt.tick_params(axis='both', which='major', labelsize=12)
    # Customize the spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 5))

    norm = (datamax - datamin) * 0.06

    plt.ylabel(plot_type.capitalize())
    if plot_type == "regret":
        pass
        #plt.ylim(np.maximum(datamin - norm, -1e-3), datamax + norm * 8)
    else:
        plt.ylim(datamin - norm, datamax + norm)
        plt.legend(fontsize=12)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{name}', dpi=300, bbox_inches='tight')

    plt.clf()


def make_grouped_plot(data, name):
    plt.figure()

    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'lightcoral'] + [
         tuple(np.random.random(3)) for _ in range(100)
     ]
    #colors = [(x, x, x) for x in np.linspace(0, 0.9, len(data))]
    markers = ['o', 's', '^', '*', 'v'] + [''] * 1000

    data_per_run = defaultdict(dict)


    for approach, metrics in data.items():
        for metric, values in metrics.items():
            data_per_run[metric][approach] = values

    for (metric, approaches) in data_per_run.items():

        for (approach, values), marker in zip(approaches.items(), markers):

            if isinstance(values, dict):
                xlabel = "Environment steps"
                timesteps = np.mean(values["timesteps"], axis=0)
                np_values = np.asarray(values["values"])
                #print(np_values, np_values.shape)
                average = np.mean(np_values, axis=0)
                ste = np.std(np_values, axis=0) / np.sqrt(len(values))
                plt.fill_between(timesteps, average - ste, average + ste, alpha=0.3, label=label_dict.get(approach, approach), color=color_dict.get(approach, "#8a8a8a"))
                plt.plot(timesteps, average, label=label_dict.get(approach, approach), color=color_dict.get(approach, "#8a8a8a"))

            else:
                xlabel = "Iterations"
                plt.plot(values, label=label_dict.get(approach, approach), color=color_dict.get(approach, "#8a8a8a"))

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=16)
        plt.xlabel(xlabel, fontsize=26)
        plt.ylabel(label_dict.get(metric, metric), fontsize=34)
        plt.grid(axis="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{name}_{metric}')
        print("saved ", f'{name}_{metric}.png')
        plt.clf()



def plot_distribution(prior_overtime, timesteps=None, name="", scenarios=None):

    all_data = np.stack(prior_overtime, axis=1)
    np_data = np.mean(all_data, axis=1).T
    stes = np.std(all_data, axis=1).T / np.sqrt(len(all_data[0]))
    colors = plt.cm.rainbow(np.linspace(0, 1, all_data.shape[-1])) #mc.XKCD_COLORS
    for i, color in enumerate(colors):
        if color[0] > 0.8 and color[1] < 0.4 and color[2] <0.2:
            colors[i] = (0.7, color[1], color[2], color[3])

    for i, (scenario_prob_overtime, ste, color) in enumerate(zip(np_data, stes, colors)):
        if len(np_data) > 7:
            if i < 2:
                label = scenarios[i] if scenarios is not None else f"Scenario {i}"
            elif i == 3:
                label = "..."
            elif i == len(np_data) - 1:
                label = "Self-play"
                color = "r"
            else:
                label = None
        else:

            if i < len(np_data) - 1:
                label = scenarios[i] if scenarios is not None else f"Scenario {i}"
            else:
                label = "Self-play"
                color = "r"
        if timesteps is not None:
            print(scenario_prob_overtime.shape, timesteps)
            plt.plot(timesteps, scenario_prob_overtime, color=color, label=label, alpha=0.8)
            if len(all_data) > 1:
                plt.fill_between(timesteps, scenario_prob_overtime - ste, scenario_prob_overtime + ste, alpha=0.15, color=color)
        else:
            dv = (i - len(np_data) / 2) * 1.5e-3
            x = np.arange(len(scenario_prob_overtime)) + dv * len(scenario_prob_overtime) * 2
            y = scenario_prob_overtime + dv
            plt.plot(x, y, color=color, label=label, alpha=0.8)


    plt.legend()
    if timesteps is not None:
        plt.xlabel("Environment steps")
    else:
        plt.xlabel("Iterations")
    plt.ylabel("Scenario probability")
    plt.title("Scenario Distribution Over Time")
    plt.grid(axis="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(name)
    print("saved ", f'{name}')
    plt.clf()


if __name__ == '__main__':

    np.random.seed(0)
    data  = {
        f"approach_{approach}": {
            f"run_type_{run_type}": np.random.random(10) for run_type in range(4)
        }
        for approach in range(3)
    }

    make_grouped_boxplot(data, name="test")