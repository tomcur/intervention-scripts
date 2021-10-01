# Intervention launch scripts

These scripts are intended to help use the
[Intervention](https://github.com/tomcur/intervention) code on University of
Tartu's Slurm cluster, or on a single (multi-)GPU machine.

- [./slurm](./slurm): Slurm helper scripts
- [./neuron](./neuron): single-machine helper scripts

## Example Slurm usage

This collects episodes of an expert teacher driving.
It can be run multiple times in parallel---either across nodes or on the same node.
It merges the collected data into a specifiable dataset.

E.g., to run a single job:

```shell
$ cd ./slurm
$ sbatch job-collect-teacher-examples.sh
```

Or to run multiple jobs:

```shell
$ cd ./slurm
$ for i in {1..10}; do
>     sbatch job-collect-teacher-examples.sh
> done
```

Environment variables:

- `NUMBER_OF_EPSIODES` configures the number of episodes desired
- `INTERVENTION_DATASET_DIRECTORY` configures the directory the data should be synchronized to

## Example single-machine usage

This collects episodes of driving. It can parallelize over multiple GPUs on a single machine.

Configure the collection you want to run:

```shell
$ cd ./neuron
$ $EDITOR ./config.py
```

Then start collection:

```shell
$ ./parallel-collect.py
```

## Setup

The scripts assume Conda is installed and available in the current environment.
Further, the scripts assume the existence of a conda environment called
`intervention`, providing some dependencies as well as making the
[intervention learning executable](https://github.com/tomcur/intervention/tree/master/scripts/intervention-learning)
available.

To set this up, run, e.g.:

```shell
$ conda env create -f environment.yml
$ conda activate intervention
$ cd path/to/main/intervention/repository
$ python3 -m pip install -r requirements.txt
$ easy_install path/to/carla/PythonAPI/carla/dist/carla-*-py3.7-linux-x86_64.egg
```

Also see the [intervention learning repository](https://github.com/tomcur/intervention) for specific installation instructions.

Change exported variables in `./slurm/prepare-carla-env.sh` and/or
`./neruron/prepare-carla-env.sh` to point to a Carla directory.
