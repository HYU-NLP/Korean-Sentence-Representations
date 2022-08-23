import json

import numpy as np
from matplotlib import pyplot as plt


def plot_align_uniform(filepath):
    f = open(filepath)  # assume file exists
    data = json.load(f)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Uniform-Align plot just like Figure 2 in SimCSE paper')
    ax.set_xlabel('Uniform loss')
    ax.set_ylabel('Align loss')
    ax.set_xlim(-2.6, -1.6)
    ax.set_ylim(0.2, 0.4)

    x = []
    y = []

    for log in data['log_history']:
        if 'eval_uniform_loss' in log:
            x.append(log['eval_uniform_loss'])
            y.append(log['eval_align_loss'])

    alphas = np.linspace(0.3, 1, len(x))
    ax.scatter(x, y, alpha=alphas)
    plt.show()


if __name__ == '__main__':
    plot_align_uniform('./output_dir/trainer_state.json')
