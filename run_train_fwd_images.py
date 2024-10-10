import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import shlex
import pathlib

# Define your parameter grids
vision_model = ["vae", "resnet_autoencoder"]
nz = [8, 12, 16]
lr = [1e-4, 5e-4]
z_predict = [1e-1, 1e-2, 1e-3]
z_reg = [1e-4, 1e-5, 1e-6]

# Generate all possible command combinations
cmds = []

for v in vision_model:
    for n in nz:
        for l in lr:
            for z in z_predict:
                for _z_reg in z_reg:
                    cmd = [
                        f"python train_fwd_images.py",
                        f"vision_model={v}",
                        f"nz={n}",
                        f"training.lr={l}",
                        f"training.z_predict={z}",
                        f"training.z_reg={_z_reg}",
                        # f"training.train_num_steps={train_num_steps}",
                    ]
                    cmd = " ".join(cmd)
                    cmds.append(cmd)

num_parallel = 10  # Number of parallel subprocesses


def run_command(cmd):
    """
    Executes a shell command and logs its output to a file.

    Args:
        cmd (str): The command to execute.

    Returns:
        tuple: (cmd, returncode, log_file)
    """
    import os

    # Create a safe filename by replacing spaces and special characters
    safe_cmd = cmd.replace(" ", "_").replace("/", "_").replace("=", "_")
    log_file = f"logs_cmd/{safe_cmd}.log"
    pathlib.Path(log_file).parent.mkdir(parents=True, exist_ok=True)


    print(f"Starting: {cmd} | Logging to: {log_file}")
    with open(log_file, "w") as f:
        # Execute the command
        process = subprocess.Popen(
            cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, text=True
        )
        process.wait()

    if process.returncode == 0:
        print(f"Completed: {cmd} | See {log_file} for details.\n")
    else:
        print(f"Failed: {cmd} | Check {log_file} for error messages.\n")

    return (cmd, process.returncode, log_file)


def main():
    # Use ProcessPoolExecutor to manage parallel execution
    with ProcessPoolExecutor(max_workers=num_parallel) as executor:
        # Submit all commands to the executor
        future_to_cmd = {executor.submit(run_command, cmd): cmd for cmd in cmds}

        # Handle results as they complete
        for future in as_completed(future_to_cmd):
            cmd = future_to_cmd[future]
            try:
                cmd, returncode, log_file = future.result()
                if returncode != 0:
                    print(f"Command failed: {cmd}. Check {log_file} for details.")
            except Exception as exc:
                print(f"Command generated an exception: {cmd} | Exception: {exc}")

    print("All commands have been processed.")


if __name__ == "__main__":
    main()
