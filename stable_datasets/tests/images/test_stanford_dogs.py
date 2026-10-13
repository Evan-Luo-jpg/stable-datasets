import io

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import StanfordDogs
from stable_datasets.images.stanford_dogs import _to_rgb_bytes


NUM_CLASSES = 120


def _image_bytes(mode, fmt="JPEG"):
    buf = io.BytesIO()
    Image.new(mode, (8, 8)).save(buf, format=fmt)
    return buf.getvalue()


def test_to_rgb_bytes_keeps_rgb_unchanged():
    rgb = _image_bytes("RGB")
    assert _to_rgb_bytes(rgb) is rgb


# RGBA/PNG is the real case: the archive's one non-RGB file is an RGBA PNG named .jpg.
@pytest.mark.parametrize(("mode", "fmt"), [("L", "JPEG"), ("CMYK", "JPEG"), ("RGBA", "PNG")])
def test_to_rgb_bytes_converts_non_rgb(mode, fmt):
    converted = Image.open(io.BytesIO(_to_rgb_bytes(_image_bytes(mode, fmt))))
    assert converted.mode == "RGB", f"Expected RGB after conversion from {mode} {fmt}, got {converted.mode}."


@pytest.mark.large
def test_stanford_dogs_dataset():
    # StanfordDogs(split="train") automatically downloads and loads the dataset
    dogs_train = StanfordDogs(split="train")

    # Test 1: Check that the dataset has the expected number of samples
    expected_num_train_samples = 12000
    assert len(dogs_train) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(dogs_train)}."
    )

    # Test 2: Check that each sample has the keys "image" and "label"
    sample = dogs_train[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type, channels and dtype
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."
    image_np = np.array(image)
    assert image_np.ndim == 3, f"Image should have 3 dimensions (H, W, C), got shape {image_np.shape}"
    assert image_np.shape[2] == 3, f"Image should have 3 channels (RGB), got {image_np.shape[2]} channels"
    assert image_np.dtype == np.uint8, f"Image should have dtype uint8, got {image_np.dtype}"

    # Test 4: Validate label type and range
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert 0 <= label < NUM_CLASSES, f"Label should be between 0 and {NUM_CLASSES - 1}, got {label}."

    # Test 5: The official training split has exactly 100 images per class
    train_counts = np.bincount(dogs_train.table.column("label").to_numpy(), minlength=NUM_CLASSES)
    assert (train_counts == 100).all(), f"Expected 100 training images per class, got {train_counts}."

    # Test 6: Validate class names
    names = dogs_train.info.features["label"].names
    assert len(names) == NUM_CLASSES, f"Expected {NUM_CLASSES} class names, got {len(names)}."
    assert names[0] == "Chihuahua" and names[-1] == "African_hunting_dog", (
        f"Unexpected class name order: {names[0]}, {names[-1]}"
    )

    # Test 7: Check the test split
    dogs_test = StanfordDogs(split="test")
    expected_num_test_samples = 8580
    assert len(dogs_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(dogs_test)}."
    )

    # Test 8: Every image is decoded as 3-channel RGB (some source JPEGs are not RGB)
    for ds in (dogs_train, dogs_test):
        for i in range(len(ds)):
            assert ds[i]["image"].mode == "RGB", f"Image {i} has mode {ds[i]['image'].mode}, expected RGB."
