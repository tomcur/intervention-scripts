#!/usr/bin/env python3

from typing import List, Union, BinaryIO

import os
import sys
import signal
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path
import tempfile
import random

sys.path.append(os.getcwd())
import config


async def spawn_carla(
    cuda_device: int, carla_world_port: int, log_file: BinaryIO
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
        stdout=log_file,
    )


async def spawn_intervention(
    cuda_device: int,
    start_port_range: int,
    checkpoint_file: Path,
    data_path: Path,
    log_file: BinaryIO,
) -> asyncio.subprocess.Process:
    """Spawns CARLA simulator in the background. Returns the process handle."""
    environ = os.environ.copy()
    environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_device}"
    environ["CARLA_TRAFFIC_MANAGER_PORT"] = f"{start_port_range}"
    environ["CARLA_WORLD_PORT"] = f"{start_port_range+1}"

    if config.COLLECT_TYPE == "teacher":
        return await asyncio.create_subprocess_exec(
            "xvfb-run",
            "--auto-servernum",
            "intervention-learning",
            "collect-teacher",
            "-t",
            f"{config.INTERVENTION_LBC_BIRDVIEW_CHECKPOINT}",
            "-n",
            "1",
            "-d",
            f"{data_path}",
            env=environ,
            stdout=log_file,
            stderr=log_file,
        )
    elif config.COLLECT_TYPE == "student":
        return await asyncio.create_subprocess_exec(
            "xvfb-run",
            "--auto-servernum",
            "intervention-learning",
            "collect-student",
            "-s",
            f"{checkpoint_file}",
            "-n",
            "1",
            "-d",
            f"{data_path}",
            env=environ,
            stdout=log_file,
            stderr=log_file,
        )
    elif config.COLLECT_TYPE == "intervention":
        return await asyncio.create_subprocess_exec(
            "xvfb-run",
            "--auto-servernum",
            "intervention-learning",
            "collect-intervention",
            "-s",
            f"{checkpoint_file}",
            "-t",
            f"{config.INTERVENTION_LBC_BIRDVIEW_CHECKPOINT}",
            "-n",
            "1",
            "-d",
            f"{data_path}",
            env=environ,
            stdout=log_file,
            stderr=log_file,
        )
    else:
        raise Exception("unknown collect type")


async def soft_kill(process: asyncio.subprocess.Process) -> None:
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        process.kill()


async def execute(
    checkpoint_file: Path, data_path: Path, cuda_device: int, process_num: int
) -> None:
    """
    :return: whether the collection process finished succesfully
    """
    print(f"{cuda_device}.{process_num}: Handling job for {checkpoint_file}")

    log_dir = Path("logs")
    log_path = (
        log_dir
        / f"log-{datetime.now().isoformat()}-cuda-device-{cuda_device}-process-{process_num}.out"
    )

    log_file = open(str(log_filename), "w")

    start_port_range = (
        5000 + (process_num + cuda_device * config.PROCESSES_PER_CUDA_DEVICE) * 10
    )

    carla_process = await spawn_carla(cuda_device, start_port_range + 1, log_file)
    print(
        f"{cuda_device}.{process_num}: Spawned CARLA, pid: {carla_process.pid} (start port range {start_port_range})"
    )

    await asyncio.sleep(15.0)

    success = False
    with tempfile.TemporaryDirectory(
        prefix="intervention-collect-", dir=config.TEMPORARY_DIRECTORY
    ) as temp_path:
        collection_process = await spawn_intervention(
            cuda_device, start_port_range, checkpoint_file, Path(temp_path), log_file
        )
        print(
            f"{cuda_device}.{process_num}: Spawned collection, pid: {collection_process.pid}"
        )
        try:
            await asyncio.wait_for(collection_process.wait(), timeout=10.0 * 60.0)

            if collection_process.returncode == 0:
                success = True

                merge_process = await asyncio.create_subprocess_exec(
                    "../merge-datasets.sh",
                    f"{temp_path}",
                    f"{data_path}",
                    stdout=log_file,
                )
                print(
                    f"{cuda_device}.{process_num}: Spawned data merging, pid: {collection_process.pid}"
                )
                await merge_process.wait()
        except asyncio.TimeoutError:
            print(
                f"{cuda_device}.{process_num}: Collection timed out, killing pid: {collection_process.pid}"
            )
            await soft_kill(collection_process)

    await soft_kill(carla_process)
    log_file.close()

    return success


async def executor(
    checkpoints_and_names: List[Union[Path, str]], cuda_device: int, process_num: int
) -> None:
    while len(checkpoints_and_names) > 0:
        checkpoint_file, name = checkpoints_and_names.pop(0)
        data_path = config.OUT_DATA_PATH / name

        await asyncio.sleep(random.random() * 15)
        success = await execute(checkpoint_file, data_path, cuda_device, process_num)
        if not success:
            print(
                f"{cuda_device}.{process_num}: Collection was unsuccessful, rescheduling {name}"
            )
            checkpoints_and_names.append((checkpoint_file, name))


async def run(checkpoints_and_names: List[Union[Path, str]]) -> None:
    await asyncio.gather(
        *[
            executor(checkpoints_and_names, cuda_device, process_num)
            for cuda_device in config.CUDA_DEVICES
            for process_num in range(config.PROCESSES_PER_CUDA_DEVICE)
        ]
    )


if __name__ == "__main__":
    os.setpgrp()

    iso_time_str = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    if config.COLLECT_TYPE not in ["teacher", "student", "intervention"]:
        print(
            "config.COLLECT_TYPE must be one of "
            "'teacher', 'student', or 'intervention'"
        )

    print(f"Running collection type {config.COLLECT_TYPE}")

    checkpoints_and_names = []

    if config.COLLECT_TYPE == "teacher":
        for episode_num in range(config.EPISODES_PER_CHECKPOINT):
            checkpoints_and_names.append(
                (
                    config.INTERVENTION_LBC_BIRDVIEW_CHECKPOINT,
                    f"{iso_time_str}-{config.COLLECT_TYPE}",
                )
            )
    else:
        for (checkpoint_directory, checkpoints) in config.STUDENT_CHECKPOINTS:
            for checkpoint in checkpoints:
                for episode_num in range(config.EPISODES_PER_CHECKPOINT):
                    checkpoint_file = (
                        config.STUDENT_CHECKPOINTS_PATH
                        / checkpoint_directory
                        / f"{checkpoint}.pth"
                    )
                    checkpoints_and_names.append(
                        (
                            checkpoint_file,
                            f"{iso_time_str}-{config.COLLECT_TYPE}-{checkpoint_directory}-{checkpoint}",
                        )
                    )

    loop = asyncio.new_event_loop()
    asyncio.get_child_watcher().attach_loop(loop)
    try:
        loop.run_until_complete(run(checkpoints_and_names))
    finally:
        os.killpg(0, signal.SIGKILL)
