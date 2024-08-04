import numpy as np
from gym import LEGGED_GYM_ROOT_DIR
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def load_data(file_path):
    return np.load(file_path)

def calculate_mean_std(data):
    mean_std_info = {}
    for key in data.files:
        mean_std_info[key] = [(np.mean(data[key][:,:,i]), np.std(data[key][:,:,i])) for i in range(data[key].shape[2])]
    return mean_std_info

def print_mean_std(mean_std_info):
    for key, stats in mean_std_info.items():
        print(f"\n{key.upper()}:")
        for i, (mean, std) in enumerate(stats):
            print(f"  Dimension {i+1} - Mean: {mean:.4f}, Std: {std:.4f}")

def load_csv_data(file_path):
    df = pd.read_csv(file_path)
    return df

def plot_reward_graph():
    methods = ['random', 'fixed', 'linear']
    # methods = ['random']
    colors = {'random': 'r', 'fixed': 'g', 'linear': 'b'}
    labels = {'random': 'random', 'fixed': 'CP', 'linear': 'CP+CR'}

    for method in methods:
        file_path = f'/home/hjlee/git/legacygym/analysis/rewards/{method}_frt.csv'
        df = load_csv_data(file_path)
        df_np = df.to_numpy()
        # data = np.vstack((df.random1, df.random2, df.random3, df.random4, df.random5))
        data = df_np[:, [1,4,7,10,13]]
        step = df.Step
        # mean = np.nanmean(data, axis=0)
        # std = np.nanstd(data, axis=0)
        mean = data.mean(axis=1)
        std = data.std(axis=1)
        plt.plot(step, mean, color=colors[method], label=labels[method])
        plt.fill_between(step, mean+std, mean-std, facecolor=colors[method], alpha=0.3)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.legend(loc='upper right', fontsize=12, markerscale=5)

    plt.show()

if __name__ == "__main__":
    # file_path = f'{LEGGED_GYM_ROOT_DIR}/analysis/logs/play_log_10000.npz'  # Replace with your file path
    # data = load_data(file_path)
    # mean_std_info = calculate_mean_std(data)
    # print_mean_std(mean_std_info)
    # plot_reward_graph()
    # fig, ax = plt.subplots(figsize=(6, 1))
    # fig.subplots_adjust(bottom=0.5)

    # cmap = mpl.cm.viridis
    # bounds = np.arange(5)
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    #             cax=ax, orientation='horizontal',
    #             label="Discrete intervals with extend='both' keyword",
    #             ticks=np.arange(4))
    # df = pd.DataFrame({"x" : np.linspace(0,1,20),
    #                "y" : np.linspace(0,1,20),
    #                "cluster" : np.tile(np.arange(4),5)})
    # cmap = mpl.colors.ListedColormap(["navy", "crimson", "limegreen", "gold"])
    # norm = mpl.colors.BoundaryNorm(np.arange(-0.5,4), cmap.N) 
    # fig, ax = plt.subplots()
    # scatter = ax.scatter(x='x', y='y', c='cluster', marker='+', data=df,
    #                 cmap=cmap, norm=norm, s=100, edgecolor ='none', alpha=0.70)

    # cbar = fig.colorbar(scatter, ticks=np.arange(4))
    # cbar.ax.set_yticklabels(['a', 'b', 'c', 'd'])


    # plt.show()
    # import matplotlib.pyplot as plt

    # x = np.linspace(0, 2*np.pi, 64)
    # y = np.cos(x) 

    # plt.figure()
    # plt.plot(x,y)

    # n = 20
    # colors = plt.cm.jet(np.linspace(0,1,n))

    # for i in range(n):
    #     plt.plot(x, i*y, color=colors[i])
    # plt.show()
    n_lines = 5
    x = np.linspace(0, 10, 100)
    y = np.sin(x[:, None] + np.pi * np.linspace(0, 1, n_lines))
    c = np.arange(1, n_lines + 1)

    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
    cmap.set_array([])

    fig, ax = plt.subplots(dpi=100)
    for i, yi in enumerate(y.T):
        ax.plot(x, yi, c=cmap.to_rgba(i + 1))
    cbar = fig.colorbar(cmap, ticks=c)
    cbar.ax.set_yticklabels(['a', 'b', 'c', 'd', 'e'])
    plt.show()
