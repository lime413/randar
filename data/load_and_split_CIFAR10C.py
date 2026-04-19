import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np


ZENODO_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
work_dir = Path("data")
archive_path = work_dir / "CIFAR-10-C.tar"
extract_dir = work_dir / "cifar10c"
output_dir = work_dir / "cifar10c-by-severity"

samples_per_severity = 10000
num_severities = 5


def download_file(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"Archive already exists: {dst}")
        return
    print(f"Downloading {url} -> {dst}")
    urllib.request.urlretrieve(url, dst)
    print("Download finished.")


def extract_tar(tar_path: Path, dst_dir: Path):
    if dst_dir.exists() and any(dst_dir.iterdir()):
        print(f"Extracted data already exists: {dst_dir}")
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {tar_path} -> {dst_dir}")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=dst_dir)
    print("Extraction finished.")


def find_dataset_root(base_dir: Path) -> Path:
    # After extraction, files may be directly in base_dir or in a nested folder.
    npy_files = list(base_dir.glob("*.npy"))
    if npy_files:
        return base_dir

    for subdir in base_dir.iterdir():
        if subdir.is_dir() and list(subdir.glob("*.npy")):
            return subdir

    raise FileNotFoundError(f"Could not find .npy files inside {base_dir}")


def split_by_severity(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = input_dir / "labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.npy in {input_dir}")

    labels = np.load(labels_path)
    np.save(output_dir / "labels.npy", labels)

    corruption_files = sorted(
        [p for p in input_dir.glob("*.npy") if p.name != "labels.npy"]
    )

    if not corruption_files:
        raise FileNotFoundError(f"No corruption .npy files found in {input_dir}")

    for severity in range(1, num_severities + 1):
        severity_dir = output_dir / f"severity_{severity}"
        severity_dir.mkdir(parents=True, exist_ok=True)

        start = (severity - 1) * samples_per_severity
        end = severity * samples_per_severity

        for corruption_path in corruption_files:
            print(f"Processing {corruption_path.name} for severity {severity}")
            arr = np.load(corruption_path, mmap_mode="r")
            severity_arr = arr[start:end]
            np.save(severity_dir / corruption_path.name, severity_arr)

    print(f"Done. Saved split dataset to: {output_dir}")


def main():
    download_file(ZENODO_URL, archive_path)
    extract_tar(archive_path, extract_dir)
    dataset_root = find_dataset_root(extract_dir)
    print(f"Dataset root found at: {dataset_root}")
    split_by_severity(dataset_root, output_dir)


if __name__ == "__main__":
    main()