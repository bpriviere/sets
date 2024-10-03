
import glob
from util import util 
import numpy as np 


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