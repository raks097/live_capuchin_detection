import os
def find_latest_checkpoint_with_run_id(base_directory):
    """
    Find the latest checkpoint file within a run directory in the given base directory.

    Args:
    - base_directory (str): The base directory containing run directories.

    Returns:
    - str: The path to the latest checkpoint file, or None if no checkpoint is found.
    """
    # Find all run directories
    run_directories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

    # Sort the directories by name in descending order
    run_directories.sort(reverse=True)

    for run_directory in run_directories:
        run_path = os.path.join(base_directory, run_directory)
        checkpoint_files = [f for f in os.listdir(run_path) if f.endswith('.pth')]

        if checkpoint_files:
            # Assuming there's only one checkpoint file per run directory
            return os.path.join(run_path, checkpoint_files[0])

    return None