#!/usr/bin/env python3
"""Download real publicly available medical imaging data for validation.

Uses MedMNIST (CC BY 4.0) which provides real medical images from published
clinical datasets at 28x28 resolution (~10-50 MB per dataset). Images are
upscaled to 224x224 on save to match model input requirements.

Datasets downloaded:
    - ChestMNIST: NIH ChestX-ray14 (14 pathology labels, multi-label)
    - PneumoniaMNIST: Guangzhou Women & Children's Medical Center (binary)
    - BreastMNIST: Breast ultrasound dataset (3-class)
    - OrganAMNIST: Abdominal CT organ classification (11-class)

Also extracts a sample DICOM from pydicom's built-in test data.

Total download: ~50-100 MB (MedMNIST 28x28 default splits).
Images saved at 224x224 in: data/sample_images/
Metadata stored in: data/sample_images/metadata.json

Author: Adam Jones
Date: 2026-02-27
License: Apache 2.0 (code), CC BY 4.0 (MedMNIST data)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = BASE_DIR / "data" / "sample_images"
CACHE_DIR = SAMPLE_DIR / ".medmnist_cache"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Number of sample images to save per dataset
NUM_CHEST = 10
NUM_PNEUMONIA = 10
NUM_BREAST = 5
NUM_ORGAN = 5

# Output image resolution (matches CXR model input)
OUTPUT_SIZE = 224

# MedMNIST downloads at 28x28 (default, small file size)
# We upscale to 224x224 using bicubic interpolation on save
DOWNLOAD_SIZE = 28

# MedMNIST ChestMNIST label names (14 pathology labels, multi-label)
CHEST_LABEL_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural Thickening",
    "Hernia",
]

# PneumoniaMNIST label names (binary)
PNEUMONIA_LABEL_NAMES = ["Normal", "Pneumonia"]

# BreastMNIST label names (3-class)
BREAST_LABEL_NAMES = ["Normal", "Benign", "Malignant"]

# OrganAMNIST label names (11-class)
ORGAN_LABEL_NAMES = [
    "Bladder",
    "Femur-L",
    "Femur-R",
    "Heart",
    "Kidney-L",
    "Kidney-R",
    "Liver",
    "Lung-L",
    "Lung-R",
    "Spleen",
    "Stomach",
]


def save_upscaled(img, filepath: Path) -> None:
    """Save a PIL image upscaled to OUTPUT_SIZE x OUTPUT_SIZE."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))
    img = img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BICUBIC)
    img.save(str(filepath))


def download_chest_xray() -> dict:
    """Download ChestMNIST (NIH ChestX-ray14) samples."""
    from medmnist import ChestMNIST

    print(f"\n{'='*60}")
    print("Downloading ChestMNIST (NIH ChestX-ray14)")
    print(f"{'='*60}")

    dataset = ChestMNIST(
        split="test", download=True, size=DOWNLOAD_SIZE,
        root=str(CACHE_DIR),
    )
    print(f"  Dataset size: {len(dataset)} test images")
    print(f"  Download size: {DOWNLOAD_SIZE}x{DOWNLOAD_SIZE}, output: {OUTPUT_SIZE}x{OUTPUT_SIZE}")
    print(f"  Labels: {len(CHEST_LABEL_NAMES)} pathologies (multi-label)")

    files = []

    # Select diverse samples: try to find images with different pathologies
    label_indices = {}
    for i in range(min(500, len(dataset))):
        _, label = dataset[i]
        label_array = np.array(label).flatten()
        for j, val in enumerate(label_array):
            if val == 1 and j not in label_indices:
                label_indices[j] = i

    # Build sample indices: prioritize diversity, then fill remaining
    selected = set()
    for label_idx in sorted(label_indices.keys()):
        if len(selected) >= NUM_CHEST:
            break
        selected.add(label_indices[label_idx])

    # Fill remaining with sequential indices
    idx = 0
    while len(selected) < NUM_CHEST and idx < len(dataset):
        selected.add(idx)
        idx += 1

    selected = sorted(selected)[:NUM_CHEST]

    for seq, i in enumerate(selected):
        img, label = dataset[i]
        label_array = np.array(label).flatten().tolist()

        filename = f"chest_xray_{seq:03d}.png"
        filepath = SAMPLE_DIR / filename
        save_upscaled(img, filepath)

        active_labels = [
            CHEST_LABEL_NAMES[j]
            for j in range(len(label_array))
            if j < len(CHEST_LABEL_NAMES) and label_array[j] == 1
        ]

        files.append({
            "filename": filename,
            "dataset_index": int(i),
            "label_vector": [int(x) for x in label_array],
            "label_names": active_labels if active_labels else ["No Finding"],
        })

        print(f"  Saved {filename} -> {active_labels if active_labels else ['No Finding']}")

    return {
        "source": "NIH ChestX-ray14 via MedMNIST v3",
        "license": "CC BY 4.0",
        "citation": "Wang et al., ChestX-ray8 (CVPR 2017); Yang et al., MedMNIST v2 (Sci Data 2023)",
        "num_samples": len(files),
        "total_dataset_size": len(dataset),
        "resolution": f"{OUTPUT_SIZE}x{OUTPUT_SIZE} (upscaled from {DOWNLOAD_SIZE}x{DOWNLOAD_SIZE})",
        "task": "multi-label classification (14 pathologies)",
        "label_names": CHEST_LABEL_NAMES,
        "files": files,
    }


def download_pneumonia() -> dict:
    """Download PneumoniaMNIST (pediatric CXR) samples."""
    from medmnist import PneumoniaMNIST

    print(f"\n{'='*60}")
    print("Downloading PneumoniaMNIST (Pediatric CXR)")
    print(f"{'='*60}")

    dataset = PneumoniaMNIST(
        split="test", download=True, size=DOWNLOAD_SIZE,
        root=str(CACHE_DIR),
    )
    print(f"  Dataset size: {len(dataset)} test images")

    files = []

    # Get balanced samples: some normal, some pneumonia
    normal_indices = []
    pneumonia_indices = []

    for i in range(min(300, len(dataset))):
        _, label = dataset[i]
        label_val = int(np.array(label).flatten()[0])
        if label_val == 0 and len(normal_indices) < NUM_PNEUMONIA // 2 + 1:
            normal_indices.append(i)
        elif label_val == 1 and len(pneumonia_indices) < NUM_PNEUMONIA // 2 + 1:
            pneumonia_indices.append(i)
        if len(normal_indices) + len(pneumonia_indices) >= NUM_PNEUMONIA:
            break

    selected = sorted(normal_indices + pneumonia_indices)[:NUM_PNEUMONIA]

    for seq, i in enumerate(selected):
        img, label = dataset[i]
        label_val = int(np.array(label).flatten()[0])

        filename = f"pneumonia_{seq:03d}.png"
        filepath = SAMPLE_DIR / filename
        save_upscaled(img, filepath)

        label_name = PNEUMONIA_LABEL_NAMES[label_val]
        files.append({
            "filename": filename,
            "dataset_index": int(i),
            "label": label_val,
            "label_name": label_name,
        })

        print(f"  Saved {filename} -> {label_name}")

    return {
        "source": "Guangzhou Women and Children's Medical Center via MedMNIST v3",
        "license": "CC BY 4.0",
        "citation": "Kermany et al., Cell 2018; Yang et al., MedMNIST v2 (Sci Data 2023)",
        "num_samples": len(files),
        "total_dataset_size": len(dataset),
        "resolution": f"{OUTPUT_SIZE}x{OUTPUT_SIZE} (upscaled from {DOWNLOAD_SIZE}x{DOWNLOAD_SIZE})",
        "task": "binary classification (normal vs pneumonia)",
        "label_names": PNEUMONIA_LABEL_NAMES,
        "files": files,
    }


def download_breast() -> dict:
    """Download BreastMNIST (breast ultrasound) samples."""
    from medmnist import BreastMNIST

    print(f"\n{'='*60}")
    print("Downloading BreastMNIST (Breast Ultrasound)")
    print(f"{'='*60}")

    dataset = BreastMNIST(
        split="test", download=True, size=DOWNLOAD_SIZE,
        root=str(CACHE_DIR),
    )
    print(f"  Dataset size: {len(dataset)} test images")

    files = []

    # Try to get representative samples from each class
    class_indices = {0: [], 1: [], 2: []}
    for i in range(len(dataset)):
        _, label = dataset[i]
        label_val = int(np.array(label).flatten()[0])
        if label_val in class_indices and len(class_indices[label_val]) < 2:
            class_indices[label_val].append(i)
        if sum(len(v) for v in class_indices.values()) >= NUM_BREAST:
            break

    # If we did not find enough classes, just take sequential samples
    selected = sorted(
        [idx for indices in class_indices.values() for idx in indices]
    )
    if len(selected) < NUM_BREAST:
        for i in range(len(dataset)):
            if i not in selected:
                selected.append(i)
            if len(selected) >= NUM_BREAST:
                break
        selected = sorted(selected)
    selected = selected[:NUM_BREAST]

    for seq, i in enumerate(selected):
        img, label = dataset[i]
        label_val = int(np.array(label).flatten()[0])

        filename = f"breast_us_{seq:03d}.png"
        filepath = SAMPLE_DIR / filename
        save_upscaled(img, filepath)

        label_name = BREAST_LABEL_NAMES[label_val] if label_val < len(BREAST_LABEL_NAMES) else f"Class_{label_val}"
        files.append({
            "filename": filename,
            "dataset_index": int(i),
            "label": label_val,
            "label_name": label_name,
        })

        print(f"  Saved {filename} -> {label_name}")

    return {
        "source": "Breast Ultrasound Images Dataset (BUSI) via MedMNIST v3",
        "license": "CC BY 4.0",
        "citation": "Al-Dhabyani et al., Data in Brief 2020; Yang et al., MedMNIST v2 (Sci Data 2023)",
        "num_samples": len(files),
        "total_dataset_size": len(dataset),
        "resolution": f"{OUTPUT_SIZE}x{OUTPUT_SIZE} (upscaled from {DOWNLOAD_SIZE}x{DOWNLOAD_SIZE})",
        "task": "3-class classification (normal, benign, malignant)",
        "label_names": BREAST_LABEL_NAMES,
        "files": files,
    }


def download_organ() -> dict:
    """Download OrganAMNIST (abdominal CT axial slices) samples."""
    from medmnist import OrganAMNIST

    print(f"\n{'='*60}")
    print("Downloading OrganAMNIST (Abdominal CT)")
    print(f"{'='*60}")

    dataset = OrganAMNIST(
        split="test", download=True, size=DOWNLOAD_SIZE,
        root=str(CACHE_DIR),
    )
    print(f"  Dataset size: {len(dataset)} test images")

    files = []

    # Try to get diverse organ classes
    class_indices = {}
    for i in range(min(500, len(dataset))):
        _, label = dataset[i]
        label_val = int(np.array(label).flatten()[0])
        if label_val not in class_indices:
            class_indices[label_val] = i
        if len(class_indices) >= NUM_ORGAN:
            break

    selected = sorted(class_indices.values())[:NUM_ORGAN]

    for seq, i in enumerate(selected):
        img, label = dataset[i]
        label_val = int(np.array(label).flatten()[0])

        filename = f"organ_ct_{seq:03d}.png"
        filepath = SAMPLE_DIR / filename
        save_upscaled(img, filepath)

        label_name = ORGAN_LABEL_NAMES[label_val] if label_val < len(ORGAN_LABEL_NAMES) else f"Organ_{label_val}"
        files.append({
            "filename": filename,
            "dataset_index": int(i),
            "label": label_val,
            "label_name": label_name,
        })

        print(f"  Saved {filename} -> {label_name}")

    return {
        "source": "Liver Tumor Segmentation Benchmark (LiTS) via MedMNIST v3",
        "license": "CC BY 4.0",
        "citation": "Bilic et al., Medical Image Analysis 2023; Yang et al., MedMNIST v2 (Sci Data 2023)",
        "num_samples": len(files),
        "total_dataset_size": len(dataset),
        "resolution": f"{OUTPUT_SIZE}x{OUTPUT_SIZE} (upscaled from {DOWNLOAD_SIZE}x{DOWNLOAD_SIZE})",
        "task": "11-class classification (organ identification from axial CT slices)",
        "label_names": ORGAN_LABEL_NAMES,
        "files": files,
    }


def download_dicom_sample() -> dict:
    """Extract a sample DICOM from pydicom's built-in test files."""
    print(f"\n{'='*60}")
    print("Extracting DICOM sample from pydicom test data")
    print(f"{'='*60}")

    try:
        import pydicom
        from pydicom.data import get_testdata_file

        # pydicom ships with several test DICOM files
        test_file = get_testdata_file("CT_small.dcm")
        ds = pydicom.dcmread(test_file)

        filename = "dicom_ct_sample.dcm"
        filepath = SAMPLE_DIR / filename
        ds.save_as(str(filepath))

        # Also save as PNG for quick viewing
        pixel_array = ds.pixel_array.astype(np.float32)
        if pixel_array.max() > 0:
            normalized = ((pixel_array - pixel_array.min()) /
                          (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(pixel_array, dtype=np.uint8)

        png_filename = "dicom_ct_sample_preview.png"
        png_filepath = SAMPLE_DIR / png_filename
        pil_img = Image.fromarray(normalized)
        pil_img = pil_img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BICUBIC)
        pil_img.save(str(png_filepath))

        info = {
            "filename": filename,
            "preview_png": png_filename,
            "modality": str(getattr(ds, "Modality", "Unknown")),
            "patient_id": str(getattr(ds, "PatientID", "Anonymous")),
            "rows": int(getattr(ds, "Rows", 0)),
            "columns": int(getattr(ds, "Columns", 0)),
            "bits_stored": int(getattr(ds, "BitsStored", 0)),
            "pixel_spacing": [float(x) for x in ds.PixelSpacing] if hasattr(ds, "PixelSpacing") else [],
            "slice_thickness": float(ds.SliceThickness) if hasattr(ds, "SliceThickness") else None,
        }

        print(f"  Saved {filename} ({info['modality']}, {info['rows']}x{info['columns']})")
        print(f"  Saved {png_filename} (preview)")

        return {
            "source": "pydicom test data (public domain CT slice)",
            "license": "MIT (pydicom), public domain (image data)",
            "files": [info],
        }

    except Exception as e:
        print(f"  Warning: Could not extract DICOM sample: {e}")
        return {"source": "pydicom test data", "error": str(e), "files": []}


def main():
    """Download all datasets and write metadata JSON."""
    print("=" * 60)
    print("MedMNIST Real Medical Imaging Data Download")
    print("=" * 60)
    print(f"Output directory: {SAMPLE_DIR}")
    print(f"Download resolution: {DOWNLOAD_SIZE}x{DOWNLOAD_SIZE}")
    print(f"Output resolution: {OUTPUT_SIZE}x{OUTPUT_SIZE}")
    print()

    metadata = {}

    # Download each dataset
    metadata["chest_xray"] = download_chest_xray()
    metadata["pneumonia"] = download_pneumonia()
    metadata["breast_ultrasound"] = download_breast()
    metadata["organ_ct"] = download_organ()
    metadata["dicom_sample"] = download_dicom_sample()

    # Write metadata
    metadata_path = SAMPLE_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written to: {metadata_path}")

    # Summary
    total_images = sum(
        d.get("num_samples", len(d.get("files", [])))
        for d in metadata.values()
    )
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"  Total images saved: {total_images}")
    print(f"  Datasets: {len(metadata)}")
    for name, data in metadata.items():
        n = data.get("num_samples", len(data.get("files", [])))
        src = data.get("source", "unknown")
        print(f"    {name}: {n} images ({src})")
    print(f"\n  All files in: {SAMPLE_DIR}")
    print(f"  Metadata: {metadata_path}")

    # Disk usage
    total_bytes = 0
    for pattern in ("*.png", "*.dcm"):
        for f in SAMPLE_DIR.glob(pattern):
            total_bytes += f.stat().st_size
    print(f"  Total image size on disk: {total_bytes / 1024 / 1024:.1f} MB")

    # Cache size
    cache_bytes = sum(
        f.stat().st_size
        for f in CACHE_DIR.rglob("*")
        if f.is_file()
    )
    print(f"  MedMNIST cache size: {cache_bytes / 1024 / 1024:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
