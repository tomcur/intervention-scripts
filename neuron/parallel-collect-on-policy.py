#!/usr/bin/env python3

from typing import List, Union

import os
import sys
import signal
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path
import tempfile

#: Ids of CUDA devices to use.
CUDA_DEVICES = [0]  # , 1, 2, 3]

INTERVENTION_CARLA_DIRECTORY = (
    Path.home() / "code/autonomous-driving/carla/CARLA_0.9.10.1_RSS"
)
INTERVENTION_LBC_BIRDVIEW_CHECKPOINT = (
    Path.home() / "intervention-models/lbc-birdview/model-128.th"
)
OUT_DATA_PATH = Path.home() / "datasets"
CHECKPOINTS_PATH = Path.home() / "checkpoints"

EPISODES_PER_CHECKPOINT: int = 30

#: List of tuples of checkpoint directories and lists of checkpoint numbers to
# use of those directories.
CHECKPOINTS = [
    ("2021-01-25-intervention-ce-di0.0-dt10.0", [25, 27]),
    ("2021-01-25-intervention-ce-di0.0-dt10.0", [25, 27]),
    ("2021-01-25-intervention-ce-di0.0-dt10.0", [25, 27]),
]

def spawn_carla(cuda_device: int, carla_world_port: int) -> asyncio.subprocess.Process:
    """Spawns CARLA simulator in the background. Returns the process handle."""

    return asyncio.create_subprocess_shell(
        "DISPLAY= SDL_VIDEODRIVER=offscreen "
        f"SDL_HINT_CUDA_DEVICE={cuda_device} "
        f"{INTERVENTION_CARLA_DIRECTORY}/CarlaUE4.sh "
        "-opengl -nosound -ResX=800 -ResY=600 -windowed "
        f"-carla-world-port={carla_world_port}",
        stdout=sys.stdout,
    )


async def execute(checkpoint_file: Path, data_path: Path, cuda_device: int) -> None:
    print(f"Spawning job #{episode_num+1} for {checkpoint_file}")

    carla_world_port = 5000 + cuda_device * 10

    carla_process = spawn_carla(cuda_device, carla_world_port)
    print(f"Spawned CARLA, pid: {carla_process.pid}")

    await asyncio.sleep(5.0)

    with tempfile.TemporaryDirectory(prefix="intervention-on-policy-") as temp_path:
        collection_process = await asyncio.create_subprocess_shell(
            "LD_LIBRARY_PATH=$CONDA_PREFIX "
            f"CARLA_WORLD_PORT={carla_world_port} "
            "xvfb-run "
            "intervention-learning collect-on-policy "
            f'-s "{checkpoint_file}" '
            f'-t "{INTERVENTION_LBC_BIRDVIEW_CHECKPOINT}" '
            "-n 1"
            f"-d {temp_path}"
        )
        await collection_process.wait()

        merge_process = await asyncio.create_subprocess_shell(
            f'../merge-datasets.sh "{temp_path}" "{data_path}"'
        )
        await merge_process.wait()

    carla_process.terminate()
    await carla_process.wait()


async def executor(
    checkpoints_and_names: List[Union[Path, str]], cuda_device: int
) -> None:
    while len(checkpoints_and_names) > 0:
        checkpoint_file, name = checkpoints_and_names.pop(0)
        data_path = OUT_DATA_PATH / name

        await execute(checkpoint_file, data_path, cuda_device)


async def run(checkpoints_and_names: List[Union[Path, str]]) -> None:
    await asyncio.gather(
        *[executor(checkpoints_and_names, cuda_device) for cuda_device in CUDA_DEVICES]
    )


if __name__ == "__main__":
    checkpoints_and_names = []
    for (checkpoint_directory, checkpoints) in CHECKPOINTS:
        for checkpoint in checkpoints:
            for episode_num in range(EPISODES_PER_CHECKPOINT):
                checkpoint_file = (
                    CHECKPOINTS_PATH / checkpoint_directory / f"{checkpoint}.pth"
                )
                checkpoints_and_names.append(
                    (
                        checkpoint_file,
                        f"{datetime.now().isoformat()}-on-policy-{checkpoint_directory}-{checkpoint}",
                    )
                )

    loop = asyncio.new_event_loop()
    loop.run_until_complete(run(checkpoints_and_names))
