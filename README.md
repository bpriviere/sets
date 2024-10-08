
## Motivation

Implementation of Monte Carlo Tree Search with Spectral Expansion from Benjamin Riviere and John Lathrop in Caltech's Computational and Mathematical Sciences Department. If you use our software in an academic publication, please cite our paper: 


B. Rivière*, J. Lathrop*, and S.-J. Chung, “Monte Carlo Tree Search with Spectral Expansion for Planning with Dynamical Systems,” (In review at Science Robotics). Equal Contribution.


## Installation

Basic dependencies 
```bash
sudo apt install build-essential
sudo apt install libeigen3-dev
sudo apt install libyaml-cpp0.5v5 libyaml-cpp-dev libspdlog-dev libfmt-dev
```

Use conda for most dependencies. 
```bash
conda env create --file environment.yml
```

Add src to path:
```
conda develop /home/ben/projects/dots/src/
```

## Build
from project directory, 
```bash
mkdir build
cd build 
cmake -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release ..
make 
```

## Scripts

<!-- To make the rollout plot (fig 5a/b)
```
cd ~/scripts/
python rollout.py
``` -->


To make the value convergence plot (fig 5c)
```
cd ~/scripts/
python value_convergence.py
```

<!-- To make the policy convergence plot in the paper (fig 5d)
```
cd ~/scripts/
python value_convergence.py
``` -->


## Notes

- Because SETS is an anytime algorithm with a wall-clock timeout condition, the results will depend on your machine. We use XX. 

- To run value and policy convergence scripts at their highest level of resolution, you will need a large RAM (>32GB) to store the tree data structure in memory. 



## License
XX