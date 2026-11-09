from PIL import Image

from src.dataset.data_module import DeepfakeDataModule
from src.dataset.dataset import DeepfakeDataset
from src.utils.dataset_paths import (
    dataset_length,
    get_dataset_from_path,
    get_source_from_path,
    normalize_source,
    resolve_input_paths,
)
from src.utils.files import CrossDataset, SubFFPP
from src.utils.model_paths import CLIP_L14_MODEL_ID, LOCAL_CLIP_L14_PATH, resolve_clip_model_path


def test_sub_ffpp_mapping_collects_split_images_without_writing_source(tmp_path, monkeypatch):
    root = tmp_path / "sub_FFPP"
    for split in ("train", "test"):
        for source in ("Deepfakes", "original"):
            frame = root / split / source / "001" / "000.png"
            frame.parent.mkdir(parents=True)
            Image.new("RGB", (2, 2)).save(frame)

    monkeypatch.setattr(SubFFPP, "root", root)

    assert SubFFPP.get_split("train") == [str(root / "train/Deepfakes/001/000.png"), str(root / "train/original/001/000.png")]
    assert SubFFPP.get_split("test") == [str(root / "test/Deepfakes/001/000.png"), str(root / "test/original/001/000.png")]


def test_original_source_is_normalized_to_real():
    path = "/data/hjr/deepfake/sub_FFPP/train/original/001/000.png"

    assert normalize_source(path.split("/")[-3]) == "real"


def test_empty_validation_dataset_has_zero_length():
    assert dataset_length(None) == 0


def test_empty_validation_loader_returns_no_loaders():
    data_module = object.__new__(DeepfakeDataModule)
    data_module.val_dataset = None

    assert data_module.val_dataloader() == []


def test_cross_dataset_mapping_includes_standard_and_wdf_layouts(tmp_path, monkeypatch):
    standard_frame = tmp_path / "CDF" / "Celeb-real" / "0001" / "000.png"
    wdf_frame = tmp_path / "WDF" / "fake_test" / "93" / "fake" / "96" / "000.png"
    standard_frame.parent.mkdir(parents=True)
    wdf_frame.parent.mkdir(parents=True)
    standard_frame.touch()
    wdf_frame.touch()

    monkeypatch.setattr(CrossDataset, "root", tmp_path)

    test_files = CrossDataset.get_test_files()

    assert test_files == {
        "CDF": [str(standard_frame)],
        "WDF": [str(wdf_frame)],
    }
    assert get_dataset_from_path(str(wdf_frame)) == "WDF"
    assert get_source_from_path(str(wdf_frame)) == "fake"


def test_cross_dataset_names_match_local_testset_root():
    assert CrossDataset.datasets == ("CDF", "DFD", "DFDC", "DFDCP", "DFR", "FFIW", "WDF")


def test_clip_l14_resolves_to_downloaded_local_snapshot():
    assert (LOCAL_CLIP_L14_PATH / "model.safetensors").is_file()
    assert resolve_clip_model_path(CLIP_L14_MODEL_ID) == str(LOCAL_CLIP_L14_PATH)


def test_dataset_accepts_direct_images_and_manifest_files(tmp_path):
    image_path = tmp_path / "real" / "001" / "000.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(image_path)

    manifest_path = tmp_path / "images.txt"
    manifest_path.write_text(f"{image_path}\n")

    paths = resolve_input_paths([str(image_path), str(manifest_path)])

    assert paths == [str(image_path), str(image_path)]


def test_dataset_converts_grayscale_frames_to_rgb_before_preprocess(tmp_path):
    image_path = tmp_path / "dataset" / "real" / "001" / "000.png"
    image_path.parent.mkdir(parents=True)
    Image.new("L", (2, 2), color=128).save(image_path)

    dataset = DeepfakeDataset(
        [str(image_path)],
        preprocess=lambda image: image.mode,
        binary=True,
    )

    assert dataset[0]["image"] == "RGB"
