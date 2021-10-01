from pathlib import Path


#: Ids of CUDA devices to use.
CUDA_DEVICES = [0]  # , 1, 2, 3]
PROCESSES_PER_CUDA_DEVICE = 4

INTERVENTION_CARLA_DIRECTORY = (
    Path.home() / "carla/CARLA_0.9.10.1_RSS"
)
INTERVENTION_LBC_BIRDVIEW_CHECKPOINT = (
    Path.home() / "checkpoints/lbc-birdview/model-128.th"
)
OUT_DATA_PATH = Path.home() / "datasets"
TEMPORARY_DIRECTORY = Path.home() / "datasets" / "temp"

#: The temporary directory location, or None to automatically pick a location
STUDENT_CHECKPOINTS_PATH = Path.home() / "checkpoints"

EPISODES_PER_CHECKPOINT: int = 30

TOWNS = ["Town01", "Town02"]
WEATHERS = ["ClearNoon"]

#: One of "teacher", "student", "intervention"
# COLLECT_TYPE = "student"
# COLLECT_TYPE = "teacher"
COLLECT_TYPE = "intervention"

# INTERVENTION_TYPE = "regular"
INTERVENTION_TYPE = "only-student"

#: List of tuples of checkpoint directories and lists of checkpoint numbers to
# use of those directories.
STUDENT_CHECKPOINTS = [
    ("2021-01-25-intervention-ce-di0.0-dt10.0", [25, 27]),
    ("2021-01-25-intervention-ce-di0.0-dt10.0", [25, 27]),
    ("2021-01-25-intervention-ce-di0.0-dt10.0", [25, 27]),
]
