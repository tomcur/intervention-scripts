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

sys.path.append(os.getcwd())
import config


async def spawn_carla(
    cuda_device: int, carla_world_port: int
) -> asyncio.subprocess.Process:
    """Spawns CARLA simulator in the background. Returns the process handle."""

    environ = os.environ.copy()
    environ["DISPLAY"] = ""
    environ["SDL_HINT_CUDA_DEVICE"] = f"{cuda_device}"
    environ["UE4_PROJECT_ROOT"] = f"{config.INTERVENTION_CARLA_DIRECTORY}"

    return await asyncio.create_subprocess_exec(
        f"{config.INTERVENTION_CARLA_DIRECTORY}/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping",
        "-opengl",
        "-nosound",
        "-ResX=800",
        "-ResY=600",
        "-windowed",
        f"-carla-world-port={carla_world_port}",
        env=environ,
        stdout=sys.stdout,
    )


async def spawn_intervention(
    cuda_device: int, start_port_range: int, checkpoint_file: Path, data_path: Path
) -> asyncio.subprocess.Process:
    """Spawns CARLA simulator in the background. Returns the process handle."""
    environ = os.environ.copy()
    environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_device}"
    environ["CARLA_TRAFFIC_MANAGER_PORT"] = f"{start_port_range}"
    environ["CARLA_WORLD_PORT"] = f"{start_port_range+1}"

    return await asyncio.create_subprocess_exec(
        "xvfb-run",
        "--auto-servernum",
        "intervention-learning",
        "collect-on-policy",
        "-s",
        f"{checkpoint_file}",
        "-t",
        f"{config.INTERVENTION_LBC_BIRDVIEW_CHECKPOINT}",
        "-n",
        "1",
        "-d",
        f"{data_path}",
        env=environ,
        stdout=sys.stdout,
    )


async def execute(checkpoint_file: Path, data_path: Path, cuda_device: int) -> None:
    print(f"Handling job for {checkpoint_file}")

    start_port_range = 5000 + cuda_device * 10

    carla_process = await spawn_carla(cuda_device, start_port_range + 1)
    print(f"Spawned CARLA, pid: {carla_process.pid}")

    await asyncio.sleep(5.0)

    with tempfile.TemporaryDirectory(prefix="intervention-on-policy-", dir=config.TEMPORARY_DIRECTORY) as temp_path:
        collection_process = await spawn_intervention(
            cuda_device, start_port_range, checkpoint_file, Path(temp_path)
        )
        await collection_process.wait()

        merge_process = await asyncio.create_subprocess_exec(
            "../merge-datasets.sh",
            f"{temp_path}",
            f"{data_path}",
        )
        await merge_process.wait()

    carla_process.terminate()
    await carla_process.wait()


async def executor(
    checkpoints_and_names: List[Union[Path, str]], cuda_device: int
) -> None:
    while len(checkpoints_and_names) > 0:
        checkpoint_file, name = checkpoints_and_names.pop(0)
        data_path = config.OUT_DATA_PATH / name

        await execute(checkpoint_file, data_path, cuda_device)


async def run(checkpoints_and_names: List[Union[Path, str]]) -> None:
    await asyncio.gather(
        *[
            executor(checkpoints_and_names, cuda_device)
            for cuda_device in config.CUDA_DEVICES
        ]
    )


if __name__ == "__main__":
    os.setpgrp()

    checkpoints_and_names = []
    for (checkpoint_directory, checkpoints) in config.CHECKPOINTS:
        for checkpoint in checkpoints:
            for episode_num in range(config.EPISODES_PER_CHECKPOINT):
                checkpoint_file = (
                    config.CHECKPOINTS_PATH / checkpoint_directory / f"{checkpoint}.pth"
                )
                checkpoints_and_names.append(
                    (
                        checkpoint_file,
                        f"{datetime.now().isoformat()}-on-policy-{checkpoint_directory}-{checkpoint}",
                    )
                )

    loop = asyncio.new_event_loop()
    asyncio.get_child_watcher().attach_loop(loop)
    try:
        loop.run_until_complete(run(checkpoints_and_names))
    finally:
        os.killpg(0, signal.SIGKILL)
