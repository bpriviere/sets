
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing as mp
import tqdm 
import glob 
import time as timer
import itertools as it
import os 
import matplotlib.cm as cm

import sys 
import plotter 
from util import util 
from build.bindings import \
    get_mdp, get_dots_mdp, get_uct, get_uct2, get_ud_mcts, get_ud_ps, get_dpw_mcts, get_se_ps, \
    run_uct, run_uct2, run_ud_mcts, run_dpw_mcts, run_ud_ps, run_se_ps, \
    RNG, MDP, Trajectory, Tree

# from learning.feedforward import Feedforward

plt.rcParams.update({'font.size': 12})
plt.rcParams['lines.linewidth'] = 1.0


def _run_sim(args):
    return run_sim(*args)


def get_solver_name(solver_param):
    solver_string = ""
    if solver_param["solver_name"] == "uct2":
        solver_string = r"SETS $H=${}".format(solver_param["dots_H"])
    elif solver_param["solver_name"] == "ud_mcts":
        solver_string = r"UD-MCTS $(H, \eta)=$({},{})".format(solver_param["dots_H"],solver_param["num_points_per_dimension"])
    elif solver_param["solver_name"] == "ud_ps":
        solver_string = r"UD-PS $(H, \eta)=$({},{})".format(solver_param["dots_H"],solver_param["num_points_per_dimension"])
    elif solver_param["solver_name"] == "dpw_mcts":
        solver_string = r"DPW-MCTS $H=${}".format(solver_param["dots_H"])
    elif solver_param["solver_name"] == "se_ps":
        solver_string = r"SE-PS $H=${}".format(solver_param["dots_H"])
    else:
        exit("solver name not recognized")
    return solver_string


def get_solver(solver_param, config_dict, tree_depth_H):
    if solver_param["solver_name"] == "uct":
        solver = get_uct() 
        solver.set_param(
            config_dict["uct_N"],         
            tree_depth_H,         
            config_dict["uct_wct"],         
            config_dict["uct_c"],         
            config_dict["uct_export_topology"],         
            config_dict["uct_export_node_states"],         
            config_dict["uct_export_trajs"],         
            config_dict["uct_export_cbdsbds"],
            config_dict["uct_export_tree_statistics"],
            config_dict["uct_heuristic_mode"],         
            config_dict["uct_tree_exploration"],         
            config_dict["uct_downsample_traj_on"],         
            config_dict["uct_verbose"],         
            )
    elif solver_param["solver_name"] == "uct2":
        solver = get_uct2() 
        solver.set_param(
            config_dict["uct_N"],         
            tree_depth_H,         
            config_dict["uct_wct"],         
            config_dict["uct_c"],         
            config_dict["uct_export_topology"],         
            config_dict["uct_export_node_states"],         
            config_dict["uct_export_trajs"],         
            config_dict["uct_export_cbdsbds"],
            config_dict["uct_export_tree_statistics"],
            config_dict["uct_heuristic_mode"],         
            config_dict["uct_tree_exploration"],         
            config_dict["uct_downsample_traj_on"],         
            config_dict["uct_verbose"],         
            )
    elif solver_param["solver_name"] == "ud_mcts":
        solver = get_ud_mcts()
        solver.set_param(
            config_dict["uct_N"],         
            tree_depth_H,         
            config_dict["uct_wct"],         
            config_dict["uct_c"],         
            config_dict["uct_export_topology"],         
            config_dict["uct_export_node_states"],         
            config_dict["uct_export_trajs"],         
            config_dict["uct_export_cbdsbds"],
            config_dict["uct_export_tree_statistics"],
            config_dict["uct_heuristic_mode"],         
            config_dict["uct_tree_exploration"],         
            config_dict["uct_downsample_traj_on"],    
            solver_param["num_points_per_dimension"], 
            solver_param["dots_H"], 
            config_dict["uct_verbose"],         
            )
    elif solver_param["solver_name"] == "dpw_mcts":
        # void set_param(int N, int max_depth, double wct, double c, bool export_topology, 
        #         bool export_states, bool export_trajs, bool export_cbdsbds, bool export_tree_statistics, 
        #         std::string heuristic_mode, std::string tree_exploration, bool downsample_trajs, 
        #         int alpha, int init_num_children, int num_timesteps_hold, bool verbose) {
        solver = get_dpw_mcts()
        solver.set_param(
            config_dict["uct_N"],         
            tree_depth_H,         
            config_dict["uct_wct"],         
            config_dict["uct_c"],         
            config_dict["uct_export_topology"],         
            config_dict["uct_export_node_states"],         
            config_dict["uct_export_trajs"],         
            config_dict["uct_export_cbdsbds"],
            config_dict["uct_export_tree_statistics"],
            config_dict["uct_heuristic_mode"],         
            config_dict["uct_tree_exploration"],         
            config_dict["uct_downsample_traj_on"],    
            solver_param["dpw_alpha"], 
            solver_param["dpw_init_num_children"], 
            solver_param["dots_H"], 
            config_dict["uct_verbose"],         
            )
    elif solver_param["solver_name"] == "ud_ps":
        # void set_param(int N, int max_depth, double wct, double c, bool export_topology, 
        #         bool export_states, bool export_trajs, bool export_cbdsbds, bool export_tree_statistics, 
        #         std::string heuristic_mode, std::string tree_exploration, bool downsample_trajs, 
        #         int num_points_per_dimension, int num_timesteps_hold, bool verbose) {
        solver = get_ud_ps()
        solver.set_param(
            config_dict["uct_N"],         
            tree_depth_H,         
            config_dict["uct_wct"],         
            config_dict["uct_c"],         
            config_dict["uct_export_topology"],         
            config_dict["uct_export_node_states"],         
            config_dict["uct_export_trajs"],         
            config_dict["uct_export_cbdsbds"],
            config_dict["uct_export_tree_statistics"],
            config_dict["uct_heuristic_mode"],         
            config_dict["uct_tree_exploration"],         
            config_dict["uct_downsample_traj_on"],    
            solver_param["num_points_per_dimension"], 
            solver_param["dots_H"], 
            config_dict["uct_verbose"],         
            )
    elif solver_param["solver_name"] == "se_ps":
        # void set_param(int N, int max_depth, double wct, double c, bool export_topology, 
        #         bool export_states, bool export_trajs, bool export_cbdsbds, bool export_tree_statistics, 
        #         std::string heuristic_mode, std::string tree_exploration, bool downsample_trajs, 
        #         int num_points_per_dimension, int num_timesteps_hold, bool verbose) {
        solver = get_se_ps()
        solver.set_param(
            config_dict["uct_N"],         
            tree_depth_H,         
            config_dict["uct_wct"],         
            config_dict["uct_c"],         
            config_dict["uct_export_topology"],         
            config_dict["uct_export_node_states"],         
            config_dict["uct_export_trajs"],         
            config_dict["uct_export_cbdsbds"],
            config_dict["uct_export_tree_statistics"],
            config_dict["uct_heuristic_mode"],         
            config_dict["uct_tree_exploration"],         
            config_dict["uct_downsample_traj_on"],    
            config_dict["uct_verbose"],         
            )
    return solver


def run_solver(ground_mdp, dots_mdp, solver, solver_param, curr_state, rng):
    if solver_param["solver_name"] == "uct":
        solver_result = run_uct(dots_mdp, solver, curr_state, rng)
    elif solver_param["solver_name"] == "uct2":
        solver_result = run_uct2(dots_mdp, solver, curr_state, rng)
    elif solver_param["solver_name"] == "ud_mcts":
        solver_result = run_ud_mcts(ground_mdp, solver, curr_state, rng)
    elif solver_param["solver_name"] == "dpw_mcts":
        solver_result = run_dpw_mcts(ground_mdp, solver, curr_state, rng)
    elif solver_param["solver_name"] == "ud_ps":
        solver_result = run_ud_ps(ground_mdp, solver, curr_state, rng)
    elif solver_param["solver_name"] == "se_ps":
        solver_result = run_se_ps(dots_mdp, solver, curr_state, rng)
    return solver_result


def run_sim(process_count, initial_state, config_dict, solver_param, seed, parallel_on):

    np.set_printoptions(precision=3)

    mdp_H = config_dict["ground_mdp_H"]
    dots_H = solver_param["dots_H"]
    tree_depth_H = int(np.floor(config_dict["ground_mdp_H"]/solver_param["dots_H"]))
    if dots_H % config_dict["dots_dynamics_horizon"] == 0:
        dynamics_H = config_dict["dots_dynamics_horizon"]
    else:
        dynamics_H = dots_H

    solver_name = get_solver_name(solver_param)

    # set seeds 
    rng = RNG()
    rng.set_seed(seed)

    config_name = config_dict["config_name"]
    config_path = config_dict["config_path"]

    # ground MDP 
    ground_mdp = get_mdp(config_dict["ground_mdp_name"], config_path)

    # get dots 
    dots_mdp = get_dots_mdp()
    dots_mdp.set_param(ground_mdp, 
        config_dict["dots_expansion_mode"], 
        config_dict["dots_initialize_mode"],
        config_dict["dots_special_actions"], 
        config_dict["dots_num_branches"], 
        dots_H, 
        dynamics_H,
        config_dict["dots_spectral_branches_mode"], 
        config_dict["dots_control_mode"], 
        config_dict["dots_scale_mode"], 
        config_dict["dots_modal_damping_mode"], 
        config_dict["dots_modal_damping_gains"], 
        config_dict["dots_rho"] * np.diag(np.ones(ground_mdp.state_dim()-2)), 
        config_dict["dots_greedy_gain"], 
        config_dict["dots_greedy_rate"], 
        config_dict["dots_greedy_min_dist"], 
        config_dict["dots_baseline_mode_on"],
        config_dict["dots_num_discrete_actions"],
        config_dict["dots_verbose"])
    
    # get solver 
    solver = get_solver(solver_param, config_dict, tree_depth_H)
    
    if initial_state is None:
        initial_state = ground_mdp.sample_state(rng)

    result = {
        "initial_state" : initial_state,
        "config_name" : config_name,
        "config_dict" : config_dict,
        "config_path" : config_path,
        "solver_param" : solver_param,
        "process_count" : process_count,
        "planned_xss" : [], # save full planned trajectory every planner call 
        "planned_uss" : [], 
        "planned_rss" : [], 
        "nss" : [],
        "vss" : [],
        "rollout_xs" : [], 
        "rollout_us" : [], 
        "rollout_rs" : [], 
        "success" : False
    }

    curr_state = initial_state
    curr_action = np.zeros((4,))
    curr_reward = 0.0
    mpc_horizon = config_dict["dots_decision_making_horizon"] * config_dict["uct_mpc_depth"]

    # only one static thermal 
    X_thermal = config_dict["Xs_thermal"][0]
    V_thermal = config_dict["Vs_thermal"][0]
    X_thermal = np.reshape(X_thermal, (13,2), order="F")
    V_thermal = np.reshape(V_thermal, (6,))
    ground_mdp.add_thermal(X_thermal, V_thermal)

    if config_dict["uct_export_trajs"]:
        ii_trajs = 0
    
    if config_dict["uct_downsample_traj_on"] and config_dict["rollout_mode"] == "mpc":
        exit("NotImplementedError: these two modes together")

    while curr_state[ground_mdp.timestep_idx()] < ground_mdp.H():

        # if not parallel_on: print("k/H: {}/{}, curr_state: {}".format(curr_state[ground_mdp.timestep_idx()], ground_mdp.H(), curr_state))
        if not parallel_on: 
            print("\n process_count: {}".format(process_count))
            print(" k/H: {}/{}".format(curr_state[ground_mdp.timestep_idx()], ground_mdp.H()))
            print("\t p_(x,y,z): {}".format(np.array(curr_state)[0:3]))
            print("\t v^body_(x,y,z): {}".format(np.array(curr_state)[3:6]))
            print("\t (roll, pitch, yaw) [deg]: {}".format(180 / np.pi * np.array(curr_state)[6:9]))
            print("\t (rollrate, pitchrate, yawrate) [deg/s]: {}".format(180 / np.pi * np.array(curr_state)[9:12]))
            ground_mdp.R_verbose(curr_state, np.zeros((3,)), True)

        print("running solver_name: {}...".format(solver_name))
        
        start_time = timer.time()
        solver_result = run_solver(ground_mdp, dots_mdp, solver, solver_param, curr_state, rng)
        tree_wct = timer.time() - start_time

        if solver_result.success:
            # print("solver_name: {} complete".format(solver_name))
            # print("\t tree size: {}".format(len(solver_result.vs)))
            # print("\t tree wct: {}".format(tree_wct))

            vs = np.array([0, *solver_result.vs]) * solver_param["dots_H"] / config_dict["ground_mdp_H"]
            ns = np.arange(1, vs.shape[0]+1)
            # to_plot_idxs = np.logspace(0, np.log(ns.shape[0]), num=5000, dtype=int)
            # to_plot_idxs = np.minimum(to_plot_idxs, ns.shape[0]-1)
            # vs = vs[to_plot_idxs]

            # print("\t vs[-1]: {}".format(vs[-1]))


            result["nss"].append(ns)
            result["vss"].append(vs)

            planned_xs = np.array(solver_result.planned_traj.xs).tolist()
            planned_us = np.array(solver_result.planned_traj.us).tolist()
            planned_rs = np.array(solver_result.planned_traj.rs).tolist()

            last_timestep = planned_xs[-1][ground_mdp.timestep_idx()]
            last_value = vs[-1]
            num_trajs = vs.shape[0]
            print("completed (solver_name, last_value, last_timestep, num_trajs, tree_wct): ({}, {}, {}, {}, {})".format(
                solver_name, last_value, last_timestep, num_trajs, tree_wct))

            # print('\t len(planned_xs) * solver_param["dots_H"]: {}'.format(len(planned_xs) * solver_param["dots_H"]))

            if config_dict["uct_downsample_traj_on"]:
                h = config_dict["uct_mpc_depth"]
            else:
                h = mpc_horizon 

            to_step_xs = planned_xs[0:h]
            to_step_us = planned_us[0:h]
            to_step_rs = planned_rs[0:h]

            result["planned_xss"].append(planned_xs)
            result["planned_uss"].append(planned_us)
            result["planned_rss"].append(planned_rs) 
            result["rollout_xs"].extend(to_step_xs)
            result["rollout_us"].extend(to_step_us)
            result["rollout_rs"].extend(to_step_rs)

            curr_state = to_step_xs[-1]
            curr_action = np.mean(to_step_us, axis=0)
            curr_reward = np.mean(to_step_rs)

            if config_dict["uct_export_trajs"] and len(solver_result.tree.trajs) != 0:
                trajs_cpp = solver_result.tree.trajs
                # max_length = dots_H * tree_depth_H
                max_length = max([len(traj.xs) for traj in trajs_cpp])

                tree_memory_bytes = np.ones((1,)).itemsize * len(trajs_cpp) * max_length * ground_mdp.state_dim() 
                bens_laptop_mem = 30.0 * 10e9 
                print("tree_memory_bytes", tree_memory_bytes)
                if tree_memory_bytes < (bens_laptop_mem / 5.0):
                    trajs_np = np.nan * np.ones((len(trajs_cpp), max_length, ground_mdp.state_dim()))
                    for jj, traj in enumerate(trajs_cpp):
                        xs = np.array(traj.xs) # (timesteps, n)
                        if xs.shape[0] != 0:
                            trajs_np[jj,0:xs.shape[0],:] = xs
                    trajs_np = sparsify_trajs(trajs_np, 1000)
                    fn = "../data/test_{}_trajs_pc{}_ii{}".format(result["config_name"], result["process_count"], ii_trajs)
                    util.save_npy(trajs_np, fn)
                    ii_trajs += 1

            del solver_result, solver, ground_mdp, dots_mdp, rng 

        else: 
            print("solver_name: {} failed @ {} for process_count: {} and config_name: {}".format(solver_name, np.array(curr_state), process_count, config_name))
            break

        if config_dict["rollout_mode"] == "uct":
            result["rollout_xs"] = planned_xs
            result["rollout_us"] = planned_us
            result["rollout_rs"] = planned_rs
            break 

    if len(result["rollout_xs"]) == 0:
        return None

    # print("result",result)

    result["success"] = True
    util.save_pickle(result, "../data/test_{}_{}.pkl".format(config_name, process_count))

    # print("process count: {} complete!".format(process_count))
    return None


def load_trajss(result):
    fns = glob.glob("../data/test_{}_trajs_pc{}_ii{}".format(result["config_name"], result["process_count"], "*"))
    iis = [int(find_between(fn, "ii", ".npy")) for fn in fns]
    # print(iis)
    fns = [fn for _,fn in sorted(zip(iis,fns))]
    print(fns)
    trajss = [util.load_npy(trajs_fn) for trajs_fn in fns]
    return trajss


def find_between(s, start, end):
    return s.split(start)[1].split(end)[0]


def plot_glider_result(result):

    ground_mdp = get_mdp(result["config_dict"]["ground_mdp_name"], result["config_path"])
    config_dict = result["config_dict"]
    solver_name = get_solver_name(result["solver_param"])

    uct_xs = np.array([result["initial_state"] , *result["rollout_xs"]]) # (num timesteps, state_dim)
    uct_us = np.array(result["rollout_us"]) # (num timesteps, action_dim)
    uct_rs = np.array(result["rollout_rs"]) # (num timesteps, )
    uct_times = uct_xs[:,ground_mdp.timestep_idx()] # in timesteps

    state_lims = np.array(config_dict["ground_mdp_X"])
    state_lims = state_lims.reshape((state_lims.shape[0] // 2, 2), order="F")
    xlim = [state_lims[0,0], state_lims[0,1]]
    ylim = [state_lims[1,0], state_lims[1,1]]
    zlim = [-1*state_lims[2,1], -1*state_lims[2,0]]

    if result["config_dict"]["uct_export_trajs"]:
        if result["config_dict"]["uct_mode"] == "no_cbds" and result["config_dict"]["uct_downsample_traj_on"]:
            mpc_horizon = config_dict["uct_mpc_depth"]
        else:
            mpc_horizon = config_dict["dots_decision_making_horizon"] * config_dict["uct_mpc_depth"]
        trajss = load_trajss(result)
        for tree_idx, trajs in enumerate(trajss):
            time_idx = tree_idx * mpc_horizon
            fig, axs = plotter.make_fig_3d_1x3_ax()
            views = [[30, -60], [90, -90], [0, -90]] 
            for ii, (ax, view) in enumerate(zip(axs, views)):
                # if tree_idx == 0:
                #     plotter.render_xs_sixdofaircraft_game(uct_xs[0:1,:], "GameSixDOFAircraft", config_dict, fig, ax, "blue", 0.5, view=view)
                # else:
                #     plotter.render_xs_sixdofaircraft_game(uct_xs[0:time_idx,:], "GameSixDOFAircraft", config_dict, fig, ax, "blue", 0.5, view=view)
                plotter.render_xs_sixdofaircraft_game(uct_xs, "GameSixDOFAircraft", config_dict, fig, ax, "blue", 1.0, view=view)
                render_tree_glider(ground_mdp, trajs, config_dict, result["solver_param"], fig, ax, "black", 0.5)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)
            fig.suptitle("solver {} tree at time idx: {}".format(solver_name, time_idx))
            
    fig, axs = plotter.make_fig_3d_1x3_ax()
    views = [[30, -60], [90, -90], [0, -90]] 
    for ii, (ax, view) in enumerate(zip(axs, views)):
        plotter.render_xs_sixdofaircraft_game(uct_xs, "GameSixDOFAircraft", config_dict, fig, ax, "blue", 0.5, view=view)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    fig.suptitle("solver {}".format(solver_name))

    # plot: state trajs over time 
    if uct_xs.shape[0] > 0:
        fig, ax = plotter.make_fig(nrows=3, ncols=4)
        default_figsize = [6.4, 4.8] # (width, height)
        fig.set_size_inches(1.5*default_figsize[0], 2.0*default_figsize[1])

        colors = plotter.get_n_colors(3)
        fig.suptitle("State for solver: {}".format(solver_name))
        ax[0,0].plot(uct_times, uct_xs[:,0]) # pos
        ax[1,0].plot(uct_times, uct_xs[:,1])
        ax[2,0].plot(uct_times, -1 * uct_xs[:,2])
        ax[0,1].plot(uct_times, uct_xs[:,3]) # vel
        ax[1,1].plot(uct_times, uct_xs[:,4])
        ax[2,1].plot(uct_times, -1 * uct_xs[:,5])
        ax[0,2].plot(uct_times, uct_xs[:,6]) # angle
        ax[1,2].plot(uct_times, uct_xs[:,7])
        ax[2,2].plot(uct_times, uct_xs[:,8])
        ax[0,3].plot(uct_times, uct_xs[:,9]) # rates
        ax[1,3].plot(uct_times, uct_xs[:,10])
        ax[2,3].plot(uct_times, uct_xs[:,11])

        # ax[0,3].plot(uct_xs[:,12]) # time since seen target 

        ax[0,0].set_ylim(xlim)
        ax[1,0].set_ylim(ylim)
        ax[2,0].set_ylim(zlim)

        ax[0,1].set_ylim(state_lims[3,:]) # vel 
        ax[1,1].set_ylim(state_lims[4,:])
        ax[2,1].set_ylim([-1 * state_lims[5,1], -1 * state_lims[5,0]])
        ax[0,2].set_ylim(state_lims[6,:]) # angle
        ax[1,2].set_ylim(state_lims[7,:])
        ax[2,2].set_ylim(state_lims[8,:])
        ax[0,3].set_ylim(state_lims[9,:]) # angle rate
        ax[1,3].set_ylim(state_lims[10,:])
        ax[2,3].set_ylim(state_lims[11,:])

        ax[0,0].set_title("p_x")
        ax[1,0].set_title("p_y")
        ax[2,0].set_title("p_z")
        ax[0,1].set_title("v_x")
        ax[1,1].set_title("v_y")
        ax[2,1].set_title("v_z")
        ax[0,2].set_title("roll")
        ax[1,2].set_title("pitch")
        ax[2,2].set_title("yaw")
        ax[0,3].set_title("roll rate")
        ax[1,3].set_title("pitch rate")
        ax[2,3].set_title("yaw rate")

    # plot: control trajs over time
    if uct_xs.shape[0] > 0:
        fig, ax = plotter.make_fig(nrows=1, ncols=3)
        fig.suptitle("Control for solver: {}".format(solver_name))
        ax[0,0].plot(uct_times[1:], uct_us[:,0])
        ax[0,1].plot(uct_times[1:], uct_us[:,1])
        ax[0,2].plot(uct_times[1:], uct_us[:,2])
        ax[0,0].set_title("u1")
        ax[0,1].set_title("u2")
        ax[0,2].set_title("u3")

    # plot: reward trajs over time 
    if uct_xs.shape[0] > 0:
        fig, ax = plotter.make_fig(nrows=1, ncols=1)
        fig.suptitle("Reward for solver: {}".format(solver_name))
        ax[0,0].plot(uct_times[1:], uct_rs)

    # plot: num traj in tree over time
    fig, ax = plotter.make_fig()
    ax.plot([len(vs) for vs in result["vss"]], marker="o", color="blue")
    if result["config_dict"]["uct_export_trajs"]:
        # trajss already exists 
        num_valid_trajs = []
        for trajs in trajss:
            count = 0
            for traj in trajs:
                if not np.isnan(np.sum(np.sum(traj))):
                    count += 1
            num_valid_trajs.append(count)
        ax.plot(num_valid_trajs, marker="o", color="orange")
    ax.set_xlabel("time")
    ax.set_ylabel("num tree trajs")


def plot_results(results):
    if results[0]["config_dict"]["rollout_mode"] != "uct":
        return None 

    linewidth = 2.0

    # we want value vs number sims, mean+-std for each solver 
    data_dict = {}
    for result in results:
        solver_name = get_solver_name(result["solver_param"])
        ns = np.array(result["nss"][0])
        vs = np.array(result["vss"][0])
        # we multiply by factor of $dots_H$ because we compute vs averaged by horizon in uct 
        # we divide by factor of $ground_mdp_H$ to normalize in [0,1]
        # vs = vs * result["solver_param"]["dots_H"] / result["config_dict"]["ground_mdp_H"]
        if solver_name in data_dict.keys():
            data_dict[solver_name].append(vs)
        else:
            data_dict[solver_name] = [vs]

    colors = plotter.get_n_colors(len(data_dict.keys()))
    solver_names_sorted = sorted(data_dict.keys())

    data_summary = {}

    fig, ax = plotter.make_fig()
    for jj, solver_name in enumerate(solver_names_sorted):
        data = data_dict[solver_name] # data is list of vs
        max_length = max([len(vs) for vs in data])
        data_np = np.nan * np.ones((max_length, len(data)))
        for ii, vs in enumerate(data):
            len_vs = len(vs)
            if len_vs < max_length:
                data_np[0:len_vs, ii] = vs
                data_np[len_vs:, ii] = vs[-1]
            else:
                data_np[:,ii] = vs

        data_mean = np.nanmean(data_np, axis=1) 
        data_std = np.nanstd(data_np, axis=1) 
        ns = np.arange(1, data_mean.shape[0]+1)

        data_summary[solver_name] = "(mean, std) = ({},{})".format(data_mean[-1], data_std[-1])

        # ax.plot(ns, data_mean, color=colors[jj], label=solver_name, linewidth=linewidth)
        # ax.fill_between(ns, data_mean - data_std, data_mean + data_std, color=colors[jj], label=None, alpha=0.5)

        to_plot_idxs = np.logspace(0, np.log(ns.shape[0]), num=5000, dtype=int)
        to_plot_idxs = np.minimum(to_plot_idxs, ns.shape[0]-1)
        ax.plot(ns[to_plot_idxs], data_mean[to_plot_idxs], color=colors[jj], label=solver_name, linewidth=linewidth)
        ax.fill_between(ns[to_plot_idxs], data_mean[to_plot_idxs] - data_std[to_plot_idxs], data_mean[to_plot_idxs] + data_std[to_plot_idxs], color=colors[jj], label=None, alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Number of Simulations")
    ax.set_ylabel("Value Estimate")
    ax.legend()
    ax.grid(True)

    # print("data_summary",data_summary)
    for key, value in data_summary.items():
        print("{}: {}".format(key, value))
    util.save_pickle(data_summary, "../data/data_summary_{}.pkl".format(results[0]["config_name"]))



def sparsify_trajs(trajs, N_traj):
    # trajs is list of "xs" numpy arrays 
    sparsified_trajs = []
    steps = np.linspace(0, len(trajs)-1, num=N_traj, dtype=int)
    steps = list(set(list(steps)))
    steps.sort()
    for step in steps:
        sparsified_trajs.append(trajs[step])
    return sparsified_trajs


def render_tree_glider(mdp, trajs, config_dict, solver_param, fig, ax, color, alpha):
    # tree is list of trajectory objects 

    # trajs = sparsify_trajs(trajs, 1000)
    linewidth = 1.0
    nodesize = 1.0
    segments = []

    if config_dict["uct_downsample_traj_on"]:
        branch_length = 1
    else:
        branch_length = solver_param["dots_H"]

    for traj in trajs: 
        xs = np.array(traj) # (H, n)
        for kk in range(xs.shape[0] // branch_length - 1):
            if not np.isnan(np.sum(np.sum(xs[kk*branch_length:(kk+1)*branch_length,:]))):
                segments.append([
                    [xs[kk*branch_length,0], xs[kk*branch_length,1], -1*xs[kk*branch_length,2]],
                    [xs[(kk+1)*branch_length,0], xs[(kk+1)*branch_length,1], -1*xs[(kk+1)*branch_length,2]]
                    ])

    cmap = cm.get_cmap("viridis")
    colors = [cmap(ii / len(segments)) for ii in range(len(segments))]
    plotter.add_3d_segments(segments, fig, ax, linewidth, colors, alpha)
    return fig, ax


def animate_render(result, config_dict):
    fig, ax = plotter.make_3d_fig()
    state_lims = np.array(config_dict["ground_mdp_X"])
    state_lims = state_lims.reshape((state_lims.shape[0] // 2, 2), order="F")
    xlim = [state_lims[0,0], state_lims[0,1]]
    ylim = [state_lims[1,0], state_lims[1,1]]

    duration = 10
    num_frames = max(len(result["rollout_xs"])//duration,1)

    def update(frame):
        ax.clear()
        ax.set_title("timestep: {}".format(frame * 10))
        xs = np.array(result["rollout_xs"][10*frame]) # (n,)? 
        plotter.render_xs_sixdofaircraft_game(xs[np.newaxis,:], "GameSixDOFAircraft", config_dict, fig, ax, "blue", 0.5)
        return
    ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=True, interval=1)
    
    fn = "../plots/{}_trial{}".format(config_dict["config_name"], result["process_count"])
    start_time = timer.time()
    ani.save(filename="{}.mp4".format(fn), writer="ffmpeg", fps=num_frames/duration)
    print("saving ffmpeg mp4 time: {}".format(timer.time() - start_time))
        
    plt.show()



def main():

    run_sim_on = True
    max_num_workers = 4
    num_seeds = 2
    parallel_on = True
    config_name = "value_convergence"
    config_path = util.get_config_path(config_name)
    config_dict = util.load_yaml(config_path)

    mdp = get_mdp("GameSixDOFAircraft", config_path)
    initial_state = mdp.initial_state()

    solver_params = [
        # # UCT2
        { 
            "solver_name": "uct2", 
            "dots_H" : 500,
        },
        { 
            "solver_name": "uct2", 
            "dots_H" : 1000,
        },
        { 
            "solver_name": "uct2", 
            "dots_H" : 2000,
        },
        { 
            "solver_name": "ud_mcts", 
            "dots_H" : 1000,
            "num_points_per_dimension" : 7, 
        },
        { 
            "solver_name": "dpw_mcts", 
            "dots_H" : 500,
            "dpw_alpha" : 0.5, 
            "dpw_init_num_children" : 50, 
        },
        { 
            "solver_name": "ud_ps", 
            "dots_H" : 1000,
            "num_points_per_dimension" : 7, 
        },
        { 
            "solver_name": "se_ps", 
            "dots_H" : 1000,
        },
    ]

    if run_sim_on:

        # remove old files
        print("removing old files...")
        for fn in glob.glob("../data/test_{}_{}.pkl".format(config_dict["config_name"], "*")):
            os.remove(fn)
        for fn in glob.glob("../data/test_{}_trajs_pc{}_ii{}".format(config_dict["config_name"], "*", "*")):
            os.remove(fn)

        start_time = timer.time()
        args = list(it.product([initial_state], [util.load_yaml(config_path)], solver_params, range(num_seeds)))
        args = [[ii, *arg, parallel_on] for ii, arg in enumerate(args)] 
        if parallel_on:
            pool = mp.Pool(max_num_workers)
            _ = [x for x in tqdm.tqdm(pool.imap(_run_sim, args), total=len(args))]
            # pool.imap(_run_sim, args)
        else:
            # for arg in tqdm.tqdm(args):
            #     _run_sim(arg)
            _ = [_run_sim(arg) for arg in tqdm.tqdm(args)]
        
        # results = [result for result in results if result is not None]   
        print("total time: {}s".format(timer.time() - start_time))

    results = []
    for fn in glob.glob("../data/test_{}_{}.pkl".format(config_dict["config_name"], "*")):
    # for fn in glob.glob("/home/ben/projects/current/dots/saved/2024_05_11/experiment/test_{}_{}.pkl".format(config_dict["config_name"], "*")):
        results.append(util.load_pickle(fn))
    results = [result for result in results if result is not None]   

    count = 0
    for result in results:
        if result is not None:
            print("plot count: {}".format(count))
            plot_glider_result(result)
            count += 1
            # if count > 10:
            #     break 
    
    plot_results(results)

    plotter.save_figs("../plots/glider.pdf")
    plotter.open_figs("../plots/glider.pdf")
    # plotter.show_figs()
    plotter.close_figs()

    # if len(results) == 1:
    if False:
    # if True:
        for result in results:
            animate_render(result, config_dict)

    return

if __name__ == "__main__":
    main()
