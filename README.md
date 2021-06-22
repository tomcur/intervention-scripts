# Intervention launch scripts

These scripts are intended to help use the [Intervention](https://github.com/tomcur/intervention) code on University of Tartu's Slurm cluster.

## Example usage

### `job-collect-teacher-examples.sh`

This collects episodes of an expert teacher driving.
It can be run multiple times in parallel---either across nodes or on the same node.
It merges the collected data into a specifiable dataset.

E.g. to run a single job:

```shell
$ sbatch job-collect-teacher-examples.sh
```

To run multiple jobs:

```shell
$ for i in {1..10}; do
>     sbatch job-collect-teacher-examples.sh
> done
```

Environment variables:

- `NUMBER_OF_EPSIODES` configures the number of episodes desired
- `INTERVENTION_DATASET_DIRECTORY` configures the directory the data should be synchronized to

## Setup

The scripts assume Conda is installed and available in the current environment.
Further, the scripts assume the existence of a conda environment called `intervention`,
providing some dependencies as well as making the [intervention learning executable](https://github.com/tomcur/intervention/tree/master/scripts/intervention-learning) available.

To set this up, run e.g.:

```shell
$ conda env create -f environment.yml
$ conda activate intervention
$ cd path/to/main/intervention/repository
$ python3 -m pip install -r requirements.txt
$ easy_install path/to/carla/PythonAPI/carla/dist/carla-*-py3.7-linux-x86_64.egg
```

Also see the [intervention learning repository](https://github.com/tomcur/intervention) for specific installation instructions.

Change exported variables in `./prepare-carla-env.sh` to point to a Carla directory.
