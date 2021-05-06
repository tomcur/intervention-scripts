#!/usr/bin/env python3

import asyncio
import itertools
import os
import random
import signal
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, List, Union

sys.path.append(os.getcwd())
import config


@dataclass
class EpisodeSetup:
    name: str
    checkpoint: Path
    town: str
    weather: str


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
    town: str,
    weather: str,
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
            "--town",
            f"{town}",
            "--weather",
            f"{weather}",
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
            "--town",
            f"{town}",
            "--weather",
            f"{weather}",
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
            "--town",
            f"{town}",
            "--weather",
            f"{weather}",
            env=environ,
            stdout=log_file,
            stderr=log_file,
        )
    else:
        raise Exception("unknown collect type")


async def soft_kill(process: asyncio.subprocess.Process) -> None:
    # First try terminating...
    try:
        process.terminate()
        await asyncio.wait_for(process.wait(), timeout=10.0)
        return
    except ProcessLookupError:
        # (can be thrown e.g. if the process has exited in the meantime)
        return
    except asyncio.TimeoutError:
        pass

    # ... then try killing
    try:
        process.kill()
        await process.wait()
    except ProcessLookupError:
        return


async def execute(setup: EpisodeSetup, cuda_device: int, process_num: int) -> None:
    """
    :return: whether the collection process finished succesfully
    """
    print(f"{cuda_device}.{process_num}: Handling job for {setup.checkpoint}")

    data_path = config.OUT_DATA_PATH / setup.name

    log_dir = Path("collect-logs")
    log_dir.mkdir(exist_ok=True)

    log_path = (
        log_dir
        / f"log-{datetime.now().isoformat()}-cuda-device-{cuda_device}-process-{process_num}.out"
    )

    log_file = open(str(log_path), "w")

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
            setup.town,
            setup.weather,
            cuda_device,
            start_port_range,
            setup.checkpoint,
            Path(temp_path),
            log_file,
        )
        print(
            f"{cuda_device}.{process_num}: Spawned collection ({setup.town}, {setup.weather}), pid: {collection_process.pid}"
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
    episode_setups: List[EpisodeSetup], cuda_device: int, process_num: int
) -> None:
    while len(episode_setups) > 0:
        setup = episode_setups.pop(0)

        await asyncio.sleep(random.random() * 15)
        success = await execute(setup, cuda_device, process_num)
        if not success:
            print(
                f"{cuda_device}.{process_num}: Collection was unsuccessful, rescheduling {setup.name}"
            )
            episode_setups.append(setup)


async def run(episode_setups: List[EpisodeSetup]) -> None:
    await asyncio.gather(
        *[
            executor(episode_setups, cuda_device, process_num)
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

    episode_setups = []

    if config.COLLECT_TYPE == "teacher":
        for episode_num, town, weather in zip(
            range(config.EPISODES_PER_CHECKPOINT),
            itertools.cycle(config.TOWNS),
            itertools.cycle(config.WEATHERS),
        ):
            episode_setups.append(
                EpisodeSetup(
                    name=f"{iso_time_str}-{config.COLLECT_TYPE}",
                    checkpoint=config.INTERVENTION_LBC_BIRDVIEW_CHECKPOINT,
                    town=town,
                    weather=weather,
                )
            )
    else:
        for (checkpoint_directory, checkpoints) in config.STUDENT_CHECKPOINTS:
            for checkpoint in checkpoints:
                for episode_num in zip(
                    range(config.EPISODES_PER_CHECKPOINT),
                    itertools.cycle(config.TOWNS),
                    itertools.cycle(config.WEATHERS),
                ):
                    checkpoint_file = (
                        config.STUDENT_CHECKPOINTS_PATH
                        / checkpoint_directory
                        / f"{checkpoint}.pth"
                    )
                    episode_setups.append(
                        EpisodeSetup(
                            name=f"{iso_time_str}-{config.COLLECT_TYPE}-{checkpoint_directory}-{checkpoint}",
                            checkpoint=checkpoint_file,
                            town=town,
                            weather=weather,
                        )
                    )

    loop = asyncio.new_event_loop()
    asyncio.get_child_watcher().attach_loop(loop)
    try:
        loop.run_until_complete(run(episode_setups))
    except Exception as e:
        print(traceback.format_exc())
    finally:
        os.killpg(0, signal.SIGKILL)
