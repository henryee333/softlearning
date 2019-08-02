import numpy as np
import os
from softlearning.replay_pools import SimpleReplayPool
import gym
from softlearning.environments.adapters.gym_adapter import GymAdapter
import glob
import sys
import pickle
import gzip
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.colors import ListedColormap

# from softlearning.environments.gym import register_environments
# register_environments()


def replay_pool_pickle_path(checkpoint_dir):
    return os.path.join(checkpoint_dir, 'replay_pool.pkl')


def plot_position_heatmap(positions, xy_max, save_path, title=""):
    num_points = positions.shape[0]
    plt.figure()
    ax = plt.gca()
    hexbins = plt.hexbin(positions[:, 0], positions[:, 1],
                         linewidths=0.2, vmax=num_points / 100)
    hexbins = hexbins.get_array()
    thresholds = [1, num_points // 10000]
    support_metric = [np.sum(hexbins >= i) / hexbins.shape[0] for i in thresholds]

    ax.set_xlim([-xy_max, xy_max])
    ax.set_ylim([-xy_max, xy_max])
    ax.ticklabel_format(axis='both', style='sci')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title + "Prop of bins with >x visits:\n(x=1, {0:.2f}), (x={1}, {2:.2f})".format(
        support_metric[0], thresholds[1], support_metric[1]))
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return support_metric, thresholds


def plot_angle_distribution(angles, save_path, title=""):
    num_angles = angles.shape[0]
    plt.figure()
    ax = plt.gca()
    if num_angles > 1e6:
        alpha = 0.01
        s = 0.05
    else:
        alpha = 0.1
        s = 0.05
    std = 0.1
    rad = np.random.normal(1, std, size=angles.shape)
    x = rad * np.cos(angles)
    y = rad * np.sin(angles)
    plt.scatter(
        x,
        y,
        alpha=alpha,
        s=s)
    ax.set_xlim([-1 - 5*std, 1 + 5*std])
    ax.set_ylim([-1 - 5*std, 1 + 5*std])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    plt.title(title)
    plt.close()

    bins = np.histogram(angles, bins=100, range=(-np.pi, np.pi))[0]
    thresholds = [1, num_angles // 500]
    support_metric = [np.sum(bins >= i) / bins.shape[0] for i in thresholds]
    return support_metric, thresholds


def get_checkpoint_number(path):
    checkpoint_num = os.path.basename(os.path.normpath(os.path.dirname(path)))
    return int(checkpoint_num.split('checkpoint_')[1])


experiment_dir = sys.argv[1]
xy_max = float(sys.argv[2])
for experiment_root in sorted(glob.iglob(
        os.path.join(experiment_dir, 'id=*'))):
    print(experiment_root)
    experience_paths = [
        replay_pool_pickle_path(checkpoint_dir)
        for checkpoint_dir in sorted(glob.iglob(
                os.path.join(experiment_root, 'checkpoint_*')))
    ]
    experience_paths.sort(key=get_checkpoint_number)

    position_heatmap_directory = experiment_root + '/object_position_heatmaps/'
    orient_dist_directory = experiment_root + '/object_orientation_heatmaps/'
    if not os.path.exists(position_heatmap_directory):
        os.mkdir(position_heatmap_directory)
    if not os.path.exists(orient_dist_directory):
        os.mkdir(orient_dist_directory)

    screw_positions_total = []
    object_angles_total = []
    pools = []
    support_metrics, thresholds = [], []
    checkpoint_nums = [0]
    i = 0

    for experience_path in experience_paths:
        # if i % 2 == 0:
        #     two_set_angles = []
        #     two_set_positions = []
        with gzip.open(experience_path, 'rb') as f:
            pool = pickle.load(f)
            pools.append(pool)
            obs = pool['observations']
            screw_positions = obs['object_position'][:, :2]
            screw_positions_total.append(screw_positions)
            object_angles = np.arctan2(
                obs['object_orientation_sin'][:, 2],
                obs['object_orientation_cos'][:, 2],
            )
            object_angles_total.append(object_angles)
            # two_set_angles.append(object_angles)
            # two_set_positions.append(screw_positions)
        # if i % 2 == 1:
        #     save_path = position_heatmap_directory + checkpoint_num + '.png'
        #     support_metric, threshold = plot_position_heatmap(np.concatenate(two_set_positions, axis=0), xy_max, save_path)

        #     save_path = orient_dist_directory + checkpoint_num + '.png'
        #     plot_angle_distribution(np.concatenate(two_set_angles, axis=0), save_path)
        #     support_metrics.append(support_metric)
        #     thresholds.append(threshold)
        #     checkpoint_nums.append(int(checkpoint_num.split("checkpoint_")[1]))

        checkpoint_num = os.path.basename(os.path.normpath(os.path.dirname(experience_path)))
        checkpoint_nums.append(int(checkpoint_num.split("checkpoint_")[1]))
        title = "Steps {}-{}k\n".format(
            checkpoint_nums[-2],
            checkpoint_nums[-1])
        last_checkpoint_num = checkpoint_num
        save_path = position_heatmap_directory + checkpoint_num + '.png'
        support_metric, threshold = plot_position_heatmap(screw_positions, xy_max, save_path, title)

        save_path = orient_dist_directory + checkpoint_num + '.png'
        angular_support_metric, angular_threshold = plot_angle_distribution(object_angles, save_path, title)

        support_metrics.append(support_metric + angular_support_metric)
        thresholds.append(threshold + angular_threshold)
        i += 1

    screw_positions_total = np.concatenate(screw_positions_total, axis=0)
    object_angles_total = np.concatenate(object_angles_total, axis=0)

    save_path = position_heatmap_directory + '/total.png'
    plot_position_heatmap(screw_positions_total, xy_max, save_path, "Total Pool\n")

    save_path = orient_dist_directory + '/total.png'
    plot_angle_distribution(object_angles_total, save_path, "Total Pool")

    support_metrics = np.array(support_metrics)
    sorted_checkpoint_nums, m1, m2, ang_m1, ang_m2 = zip(
        *sorted(zip(
            checkpoint_nums[1:],
            support_metrics[:, 0],
            support_metrics[:, 1],
            support_metrics[:, 2],
            support_metrics[:, 3])))

    plt.figure()
    plt.plot(sorted_checkpoint_nums, m1)
    plt.plot(sorted_checkpoint_nums, m2)
    plt.xlabel('Epochs [1 epoch = 1000 steps]')
    plt.ylabel('Proportion of bins with >= n visits')
    plt.legend(['n=1', 'n={}'.format(thresholds[0][1])])
    plt.title('Support of Object XY Position Distribution Over Time')
    plt.ylim([0, 1.1])
    plt.savefig(position_heatmap_directory + '/support_metric.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(sorted_checkpoint_nums, ang_m1)
    plt.plot(sorted_checkpoint_nums, ang_m2)
    plt.xlabel('Epochs [1 epoch = 1000 steps]')
    plt.ylabel('Proportion of bins with >= n visits')
    plt.legend(['n=1', 'n={}'.format(thresholds[0][3])])
    plt.title('Support of Object Z-Angle Distribution Over Time')
    plt.ylim([0, 1.1])
    plt.savefig(orient_dist_directory + '/support_metric.png', bbox_inches='tight')
    plt.close()



    # nbins = 21
    # heatmap, xedges, yedges = np.histogram2d(init_object_xy[:, 0],
    #                                          init_object_xy[:, 1], bins=nbins,
    #                                          range=[[-0.175, 0.175], [-0.175, 0.175]])
    # grid = np.round(np.linspace(-0.175, 0.175, num=nbins), 3)
    # ax = sns.heatmap(heatmap, xticklabels=grid, yticklabels=grid)
    # ax.invert_yaxis()
    # plt.show()
