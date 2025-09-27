import argparse
import glob
import logging
import os
import random
import shutil

# Configure logging for clear terminal output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_data_subset(source_dir, dest_dir, num_cases):
    """
    Randomly samples a specified number of patient folders from a source directory
    and copies them to a destination directory.

    Args:
        source_dir (str): The directory containing the original patient folders (e.g., 'data_brats_raw_all/train').
        dest_dir (str): The directory where the sampled patient folders will be copied (e.g., 'data_brats_raw/train').
        num_cases (int): The number of patient folders to sample.
    """
    # Ensure source directory exists
    if not os.path.isdir(source_dir):
        logging.error(f"Source directory not found: {source_dir}")
        return

    # Find all patient folders in the source directory
    all_patient_folders = [d for d in glob.glob(os.path.join(source_dir, 'BraTS2021_*')) if os.path.isdir(d)]

    if not all_patient_folders:
        logging.warning(f"No patient folders found in {source_dir}")
        return

    # Ensure the requested number of cases is not more than available
    if num_cases > len(all_patient_folders):
        logging.warning(
            f"Requested {num_cases} cases, but only {len(all_patient_folders)} are available in {source_dir}. "
            f"Copying all available cases."
        )
        num_cases = len(all_patient_folders)

    # Randomly select the patient folders
    selected_folders = random.sample(all_patient_folders, num_cases)
    logging.info(f"Randomly selected {len(selected_folders)} patient folders from {source_dir}.")

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Copy each selected folder to the destination
    for folder_path in selected_folders:
        patient_id = os.path.basename(folder_path)
        dest_path = os.path.join(dest_dir, patient_id)
        logging.info(f"  -> Copying {patient_id} to {dest_dir}")
        try:
            shutil.copytree(folder_path, dest_path)
        except FileExistsError:
            logging.warning(f"  -> Destination folder {dest_path} already exists. Skipping.")
        except Exception as e:
            logging.error(f"  -> Failed to copy {folder_path}: {e}")

    logging.info(f"Successfully copied {len(selected_folders)} patient folders to {dest_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create random subsets of the BraTS dataset for training and validation.")

    parser.add_argument("--source_root", type=str, default="data_brats_raw_all",
                        help="Root directory containing the full 'train' and 'val' sets.")
    parser.add_argument("--dest_root", type=str, default="data_brats_raw",
                        help="Root directory where the new 'train' and 'val' subsets will be created.")

    parser.add_argument("--train_num", type=int, default=20,
                        help="Number of patients to select for the training set.")
    parser.add_argument("--val_num", type=int, default=4,
                        help="Number of patients to select for the validation set.")

    parser.add_argument("--clean_dest", action='store_true',
                        help="If specified, deletes the destination 'train' and 'val' folders before creating new ones.")

    args = parser.parse_args()

    # Define source and destination paths based on root directories
    source_train_dir = os.path.join(args.source_root, 'train')
    source_val_dir = os.path.join(args.source_root, 'val')
    dest_train_dir = os.path.join(args.dest_root, 'train')
    dest_val_dir = os.path.join(args.dest_root, 'val')

    # Clean destination directories if requested
    if args.clean_dest:
        if os.path.exists(dest_train_dir):
            logging.info(f"Cleaning destination directory: {dest_train_dir}")
            shutil.rmtree(dest_train_dir)
        if os.path.exists(dest_val_dir):
            logging.info(f"Cleaning destination directory: {dest_val_dir}")
            shutil.rmtree(dest_val_dir)
        logging.info("Destination directories cleaned.\n")

    # Create the training subset
    create_data_subset(
        source_dir=source_train_dir,
        dest_dir=dest_train_dir,
        num_cases=args.train_num
    )

    # Create the validation subset
    create_data_subset(
        source_dir=source_val_dir,
        dest_dir=dest_val_dir,
        num_cases=args.val_num
    )

    logging.info("Dataset subset creation process finished.")
