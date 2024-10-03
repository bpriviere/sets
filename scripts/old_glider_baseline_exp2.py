
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
from build.bindings import get_mdp, get_dots_mdp, get_uct, RNG, UCT, UCT2, MDP, SCP2, \
    run_uct, run_scp2, Trajectory, rollout_action_sequence, wrapper_aero_model, Tree
from learning.feedforward import Feedforward

from test_simple2 import test_simple2, extract_from_tree_statistics


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


def main():
    path_to_data = "/home/ben/projects/current/dots/saved/glider_experiment/dots/test_value_convergence_result_cast_game2_*.pkl"
    # path_to_data = "/home/ben/projects/current/dots/saved/glider_experiment/baseline_eta3/test_value_convergence_result_cast_game2_*.pkl"
    # path_to_data = "/home/ben/projects/current/dots/saved/glider_experiment/baseline_eta5/test_value_convergence_result_cast_game2_*.pkl"
    # path_to_data = "/home/ben/projects/current/dots/saved/glider_experiment/baseline_eta7/test_value_convergence_result_cast_game2_*.pkl"
    # path_to_data = "/home/ben/projects/current/dots/saved/glider_experiment/baseline_eta9/test_value_convergence_result_cast_game2_*.pkl"

    des_num_sims = -1
    print("des_num_sims",des_num_sims)

    ave_terminal_value = {}

    results = []
    for fn in glob.glob(path_to_data):
        # print("fn",fn)
        result = util.load_pickle(fn)
        # for key, value in result.items():
            # print("key",key)
        results.append(result)

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
    data = np.nan * np.ones((num_seeds, num_Hs, 1))
    for result in results:
        ii_seed = seeds.index(result["seed"])
        ii_H = Hs.index(result["H"])
        if result["vs"].shape[0]==0: 
            data[ii_seed, ii_H, 0] = np.nan
        else:
            if result["vs"].shape[0] < des_num_sims:
                data[ii_seed, ii_H, 0] = np.nan
            elif des_num_sims == -1:
                # this factor of H is because we average value by horizon at some point in uct 
                data[ii_seed, ii_H, 0] = result["H"] * result["vs"][-1]
            else:
                # this factor of H is because we average value by horizon at some point in uct 
                data[ii_seed, ii_H, 0] = result["H"] * result["vs"][des_num_sims-1]

    # average across seeds 
    data_mean = np.nanmean(data, axis=0)
    data_std = np.nanstd(data, axis=0)

    print("Hs", Hs)
    print("data_mean", data_mean)
    print("data_std", data_std)


if __name__ == '__main__':
    main()