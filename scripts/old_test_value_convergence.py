
# standard
import numpy as np 
import time as timer
import os
import itertools as it
import multiprocessing as mp
import tqdm 
import glob 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# custom
import plotter 
from util import util 
from build.bindings import get_mdp, get_dots_mdp, get_uct, RNG, UCT, MDP, \
    run_uct, Trajectory, rollout_action_sequence, Tree
from learning.feedforward import Feedforward

from test_simple2 import test_simple2, extract_from_tree_statistics

plt.rcParams.update({'font.size': 12})
plt.rcParams['lines.linewidth'] = 2.0


def _test_value_convergence(args):
    return test_value_convergence(*args)

def test_value_convergence(process_count, config_dict, seed, H, N, desired_depth, timeout, parallel_on, initial_state=None):

    config_dict = config_dict.copy()

    config_dict["rollout_mode"] = "uct"
    config_dict["dots_decision_making_horizon"] = H
    if config_dict["dots_dynamics_horizon"] > H:
        config_dict["dots_dynamics_horizon"] = H
    config_dict["uct_N"] = N
    config_dict["uct_heuristic_mode"] = "shuffled"
    config_dict["uct_mode"] = "no_cbds"
    config_dict["uct_max_depth"] = int(max(1, desired_depth / config_dict["dots_decision_making_horizon"]))
    config_dict["uct_wct"] = timeout

    print('config_dict["dots_dynamics_horizon"]',config_dict["dots_dynamics_horizon"])
    print('config_dict["dots_decision_making_horizon"]',config_dict["dots_decision_making_horizon"])
    print('config_dict["uct_max_depth"]',config_dict["uct_max_depth"])
    print('config_dict["uct_N"]',config_dict["uct_N"])

    test_simple2_result = test_simple2(process_count, config_dict, seed, parallel_on, initial_state=None)
    # total_visit_counts_per_depth, visit_counts_per_depth = extract_from_tree_statistics(test_simple2_result["uct_trees"][0])
    value_convergence_result = {
        "seed" : seed,
        "xs" : test_simple2_result["rollout_xs"],
        "test_simple2_config_dict" : test_simple2_result["config_dict"],
        "H" : H, 
        "vs" : np.array(test_simple2_result["uct_vss"][0]),  
        "ns" : 1 + np.arange(len(test_simple2_result["uct_vss"][0])),
    }
    # todo: save data
    util.save_pickle(value_convergence_result, "../data/test_value_convergence_result_cast_game2_{}.pkl".format(process_count))
    return value_convergence_result


def plot_value_convergence_results(results):
    fig, ax = plotter.make_fig()
    
    # get data structure size
    seeds = set()
    Hs = set()
    num_vs = 0
    for result in results:
        seeds.add(result["seed"])
        Hs.add(result["H"])
        num_vs = np.max((num_vs, result["vs"].shape[0]))
    num_seeds = len(seeds)
    num_Hs = len(Hs)
    seeds = list(seeds)
    Hs = list(Hs)

    # put into data 
    data = np.nan * np.ones((num_seeds, num_Hs, num_vs))
    for result in results:
        ii_seed = seeds.index(result["seed"])
        ii_H = Hs.index(result["H"])
        # this factor of H is because we average value by horizon at some point in uct 
        data[ii_seed, ii_H, 0:result["vs"].shape[0]] = result["H"] * result["vs"]

    # plot 
    for ii_H in range(data.shape[1]):

        # ii_k is the minimum number of simulations that have been run for all seeds
        ii_k = np.inf
        for ii_seed in range(data.shape[0]):
            if np.isnan(data[ii_seed,ii_H,:]).any():
                # print("np.where(np.isnan(data[ii_seed,ii_H,:]))",np.where(np.isnan(data[ii_seed,ii_H,:])))
                ii_k = np.min((ii_k, np.where(np.isnan(data[ii_seed,ii_H,:]))[0][0]))

        if not np.isfinite(ii_k):
            ii_k = data.shape[2]
        
        if ii_k == 0:
            continue

        print("ii_k",ii_k)
        ii_k = int(ii_k)

        mean_vs = np.nanmean(data[:,ii_H,0:ii_k], axis=0)
        std_vs = np.nanstd(data[:,ii_H,0:ii_k], axis=0)

        idxs = np.logspace(np.log10(1), np.log10(mean_vs.shape[0]), num=np.min((10000,mean_vs.shape[0])), endpoint=False, base=10.0, dtype=int)

        print("idxs",idxs)
        print("mean_vs.shape[0]",mean_vs.shape[0])
        
        mean_vs_idx = mean_vs[idxs]
        std_vs_idx = std_vs[idxs]
        ns_idx = np.arange(mean_vs.shape[0])[idxs]

        ax.plot([0.999, *list(ns_idx)], [0.0, *list(mean_vs_idx)], label="H: {}".format(Hs[ii_H]))
        ax.fill_between(ns_idx, mean_vs_idx - std_vs_idx, mean_vs_idx + std_vs_idx, alpha=0.1)

    ax.set_xlabel("Number of Simulations")
    ax.set_ylabel("Value Estimate")
    ax.set_yticklabels([])
    ax.set_xscale("log")
    # ax.set_title("Value Convergence")
    ax.grid(True, which="both")
    # ax.legend()

    # # https://stackoverflow.com/questions/22263807/how-is-order-of-items-in-matplotlib-legend-determined
    handles, labels = ax.get_legend_handles_labels()
    order = list(np.argsort([np.float32(l[3:]) for l in labels]))
    sorted_handles, sorted_labels = [handles[idx] for idx in order],[labels[idx] for idx in order]

    ax.axvspan(1000, 10000, alpha=0.5, color="gray")
    import matplotlib.patches as mpatches
    patch = mpatches.Patch(color='grey', alpha=0.5)
    sorted_handles.append(patch)
    sorted_labels.append("WCT Budget")
    ax.legend(sorted_handles, sorted_labels)


    for result in results: 
        render_fig, render_ax = plotter.make_3d_fig()
        if "xs" in result.keys():
            xs_np = np.array(result["xs"])
            if xs_np.shape[0] > 0:
                plotter.render_xs_sixdofaircraft_game(xs_np, result["test_simple2_config_dict"]["ground_mdp_name"], result["test_simple2_config_dict"], 
                    obstacles_on=True, thermals_on=True, color="blue", alpha=0.5, fig=render_fig, ax=render_ax)
                render_ax.set_title("H={}".format(result["H"]))


def main():

    # num_seeds = 4
    # parallel_on = True
    # only_plot = True

    # config_path = util.get_config_path("fixed_wing")
    # Hs = [100, 500, 1000, 1500]
    # N = 10000000
    # desired_depth = 20000
    # timeout = 3600

    num_seeds = 4
    # parallel_on = True
    parallel_on = False
    only_plot = False

    config_path = util.get_config_path("fixed_wing")
    Hs = [50, 100, 200, 500, 1000]
    # N = 10000000
    N = 100000
    desired_depth = 20000
    timeout = 3600


    # N = 50000
    # desired_depth = 50000
    # timeout = 3000

    if only_plot:
        fns = glob.glob("../data/test_value_convergence_result_cast_game2_*.pkl")
        # fns = glob.glob("../data/test_value_convergence_result_cast_game2_4.pkl")
        print("fns",fns)
        results = [util.load_pickle(fn) for fn in fns]
    else:
        start_time = timer.time()
        args = list(it.product([util.load_yaml(config_path)], range(num_seeds), Hs, [N], [desired_depth]))
        args = [[ii, *arg, timeout, parallel_on] for ii, arg in enumerate(args)] 
        if parallel_on:
            # num_workers = mp.cpu_count() - 1
            # num_workers = 16
            num_workers = 4
            pool = mp.Pool(num_workers)
            results = [x for x in tqdm.tqdm(pool.imap(_test_value_convergence, args), total=len(args))]
        else:
            results = [_test_value_convergence(arg) for arg in tqdm.tqdm(args)]
        results = [result for result in results if result is not None]   
        print("total time: {}s".format(timer.time() - start_time))

    plot_value_convergence_results(results)

    plotter.save_figs("../plots/test_value_convergence.pdf")
    plotter.open_figs("../plots/test_value_convergence.pdf")

    print("done!")



if __name__ == '__main__':
    main()
