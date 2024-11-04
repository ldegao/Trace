import subprocess
import time
import select
from types import SimpleNamespace

import torch
import config
import states
import os
import os
import shutil
from datetime import datetime

current_dir = os.getcwd()
os.chdir('..')
config.set_carla_api_path()

try:
    import carla
except ModuleNotFoundError as e:
    print("[-] Carla module not found. Make sure you have built Carla.")
    proj_root = config.get_proj_root()
    print("    Try `cd {}/carla && make PythonAPI' if not.".format(proj_root))
    exit(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client, world, G, blueprint_library, town_map = None, None, None, None, None
# model = cluster.FeatureExtractor().to(device)
accumulated_trace_graphs = []
autoware_container = None
exec_state = states.ExecState()

import fuzzer

os.chdir(current_dir)


def run_command(command, wait=True):
    """Runs a shell command and captures its output in real-time without blocking."""
    print(f"Running command: {command}")  # Debugging output
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True, bufsize=1)

    # Real-time output capturing
    if wait:
        stdout_lines = []
        stderr_lines = []
        try:
            while True:
                # Use select to avoid blocking on readline
                ready_to_read, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)

                if process.stdout in ready_to_read:
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        print(f"Standard Output: {stdout_line.strip()}")
                        stdout_lines.append(stdout_line)

                if process.stderr in ready_to_read:
                    stderr_line = process.stderr.readline()
                    if stderr_line:
                        print(f"Standard Error: {stderr_line.strip()}")
                        stderr_lines.append(stderr_line)

                # Check if process has finished
                if process.poll() is not None:
                    # Process has finished, make sure to flush remaining output
                    stdout_remaining = process.stdout.read()
                    stderr_remaining = process.stderr.read()
                    if stdout_remaining:
                        stdout_lines.append(stdout_remaining)
                    if stderr_remaining:
                        stderr_lines.append(stderr_remaining)
                    break

        except Exception as e:
            print(f"Error occurred: {e}")

        return ''.join(stdout_lines), ''.join(stderr_lines)

    return process


def init_environment():
    """Equivalent to init.sh functionality."""
    fuzzerdata_dir = f"/tmp/fuzzerdata/{os.getlogin()}"
    docker_name = f"carla-{os.getlogin()}"

    # Create fuzzerdata_dir if it doesn't exist
    if not os.path.exists(fuzzerdata_dir):
        os.makedirs(fuzzerdata_dir)
        print(f"Created directory {fuzzerdata_dir}")

    stop_autoware()
    # Check if Docker container is running
    docker_status, _ = run_command(f"docker inspect -f '{{{{.State.Status}}}}' {docker_name} 2>/dev/null")
    if "running" not in docker_status:
        print(f"Docker container {docker_name} is not in running state. Running stop_carla()...")
        stop_carla()

    # Check if Docker container exists
    docker_exists, _ = run_command(f"docker ps -a --filter name={docker_name} --format '{{{{.Names}}}}'")
    if not docker_exists:
        print(f"Docker container {docker_name} doesn't exist. Running run_carla()...")
        run_carla()

    # Remove files in fuzzerdata_dir
    for filename in os.listdir(fuzzerdata_dir):
        file_path = os.path.join(fuzzerdata_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Call save_files functionality
    save_files()

    # Remove directories
    if os.path.exists("../data/output"):
        shutil.rmtree("../data/output")
        print("Removed ../data/output directory")

    if os.path.exists("../data/seed-artifact"):
        shutil.rmtree("../data/seed-artifact")
        print("Removed ../data/seed-artifact directory")

    # Remove specific Docker containers
    containers, _ = run_command("docker ps -a --filter ancestor=carla-autoware:improved-record --format='{{.ID}}'")
    if containers:
        run_command(f"docker rm -f {containers}")
        print(f"Removed Docker containers: {containers}")


def run_carla():
    """Equivalent to run_carla.sh functionality."""
    # idle_gpu = 0
    port = 4000
    carla_cmd = f"./CarlaUE4.sh -RenderOffScreen -carla-rpc-port={port} -quality-level=Epic && /bin/bash"
    docker_name = f"carla-{os.getlogin()}"

    # Run CARLA Docker
    command = f"docker run --name='carla-{os.getlogin()}' -d --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:0.9.13 {carla_cmd}"
    # command = f"docker run --name='carla-{os.getlogin()}' -d --gpus --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:0.9.13 {carla_cmd}"
    # run_command(command)
    subprocess.Popen(command, shell=True)
    time.sleep(5)
    print(f"Started CARLA Docker container {docker_name}")


def save_files():
    """Modified savefile.sh functionality to store all files and directories in ../data/output/."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = f"../data/save/{timestamp}/"

    output_dir = "../data/output/"

    # Check if output_dir exists
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist. Skipping save operation.")
        return

    # Create the save directory
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over all items (files and directories) in output_dir
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        target_path = os.path.join(save_dir, item)

        if os.path.isdir(item_path):
            # Use copytree for directories to handle nested structures
            if not os.listdir(item_path):  # Check if directory is empty
                print(f"Directory {item_path} is empty. Skipping...")
                continue

            shutil.copytree(item_path, target_path)
            print(f"Copied directory {item_path} to {target_path}")

        elif os.path.isfile(item_path):
            # Handle files directly in the output_dir root
            shutil.copy(item_path, save_dir)
            print(f"Copied file {item_path} to {save_dir}")

        else:
            print(f"{item_path} is neither a file nor a directory. Skipping...")

    # If save_dir is empty after copying, remove it
    if not os.listdir(save_dir):
        shutil.rmtree(save_dir)
        print(f"No files were saved. Removed empty save directory {save_dir}")


def stop_carla():
    """Equivalent to stop_carla.sh functionality."""
    docker_name = f"carla-{os.getlogin()}"
    run_command(f"docker rm -f {docker_name}")
    print(f"Stopped and removed Docker container {docker_name}")


def stop_autoware():
    """Equivalent to stop_autoware.sh functionality."""
    docker_name = f"autoware-{os.getlogin()}"
    run_command(f"docker rm -f {docker_name}")
    print(f"Stopped and removed Docker container {docker_name}")


def close_processes():
    """Equivalent to close.sh functionality."""
    process_grep = "/usr/bin/python2 /opt/ros/melodic/bin/rostopic echo /decision_maker/state"
    pids_output, _ = run_command(f"ps -u {os.getlogin()} -o pid,command | grep '{process_grep}' | awk '{{print $1}}'")
    pids = pids_output.split()

    for pid in pids:
        run_command(f"kill {pid}")
        print(f"Killed process {pid}")


def run_test(sim_port, target, density, town, duration):
    """Directly call the main function from fuzzer.py to run the simulation test."""

    # Get default argument values from argparse
    argument_parser = fuzzer.set_args()
    default_args = vars(argument_parser.parse_args([]))  # Get all default values as a dictionary

    # Update the default arguments with the specific values we want to pass
    custom_args = {
        "sim_port": sim_port,
        "target": target,
        "density": density,
        "town": town,
        "timeout": duration
    }

    # Merge default arguments with custom arguments
    default_args.update(custom_args)

    # Convert to SimpleNamespace for compatibility with the fuzzer's main function
    args = SimpleNamespace(**default_args)

    start_time = time.time()

    while True:
        # Initialize environment
        init_environment()

        current_time = time.time()
        total_duration = current_time - start_time
        if total_duration >= duration:
            print(f"Total duration exceeded {duration} seconds. Exiting...")
            break

        # Save the current directory and switch to the parent directory
        current_dir = os.getcwd()
        os.chdir('..')  # Switch to parent directory

        try:
            # Directly call the main function from fuzzer.py
            fuzzer.main(args)
        except (SystemExit, TimeoutError) as e:
            print(f"Exception caught: {e}. Continuing program execution.")
            # Perform additional actions or logging as needed
        except KeyboardInterrupt as e:
            # Capture any other general exceptions
            print(f"Unexpected exception caught: {e}")
            return

        os.chdir(current_dir)
        # Check if the Docker container is still running
        docker_name = f"carla-{os.getlogin()}"
        docker_status, _ = run_command(f"docker inspect -f '{{{{.State.Status}}}}' {docker_name}")
        if "running" not in docker_status:
            print(f"{docker_name} is not in 'running' state. Restarting...")

        time.sleep(1)


if __name__ == "__main__":
    # Example parameters for the test
    sim_port = 4000
    density = "0.4"
    town = "3"
    target = "autoware"
    duration = 86400  # Duration in seconds

    run_test(sim_port, target, density, town, duration)
