import subprocess
import sys
import os


def run_script_in_succession(script_path):
    """
    Runs a single Python script and streams its output directly to the console.
    """
    if not os.path.exists(script_path):
        print(f"ERROR: Script not found at path: {script_path}")
        sys.exit(1)

    print(f"--- Starting execution of {script_path} ---")

    # add "-u" to force unbuffered output (prints appear immediately)
    command = [sys.executable, "-u", script_path]

    try:
        # By default, subprocess inherits stdout/stderr, streaming it to your console.
        subprocess.run(
            command,
            check=True  # Raises CalledProcessError if the script fails
        )

        print(f"--- Successfully finished {script_path} ---")

    except subprocess.CalledProcessError:
        pass
    except Exception as e:
        print(f"\nAn unexpected error occurred while running {script_path}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    files_to_run = [
        "lstm_global.py",
        "tft_global.py"
    ]

    for file in files_to_run:
        run_script_in_succession(file)

    print("\nMASTER RUNNER: All files executed successfully!")