"""Test protocol mirroring dfd-hr (ZiyingHuang/dfd-hr) for GenD.

Protocols:
  - dfdhr_test_files(): CDF-v2, FSh, DFD, DFDC, DFDCP from DeepfakeBench JSON (/data/hjr/dfbench) + WDF folder.
  - df40_test_files(): DF40 cross-method (Protocol-2) subsets uniface_ff ... simswap_ff from DeepfakeBench-style
    JSON under /data/hjr/df40/test (dfd-hr training/config/test_config_df40.yaml).

JSON side follows dfd-hr training/dataset/abstract_dataset.py (mode='test', image-level):
  - manifest: <json_root>/<dataset_name>.json
  - info[dataset_name][label]['test'] (+ ['c23'] for DeepFakeDetection / FaceShifter)
  - labels from label_dict, videos with no frames skipped
  - frames sorted by integer stem, then sample_frame_paths(frames, 32)
  - frame paths resolved inside the RGB root (relative -> root / path)

GenD derives labels from the path (parent of the video folder must contain "real") and groups videos by the
parent folder, so JSON frames are exposed through a symlink tree:
  <LINK_ROOT>/<dataset>/<real|fake>/<label>_<video>/<frame>.png -> original frame
GenD never resolves symlinks, so its loader and metrics run unchanged.
"""

import json
import os
from pathlib import Path, PureWindowsPath

DFBENCH_RGB_ROOT = Path(os.environ.get("DFBENCH_RGB_ROOT", "/data/hjr/dfbench/rgb"))
DFBENCH_JSON_ROOT = Path(os.environ.get("DFBENCH_JSON_ROOT", "/data/hjr/dfbench/dataset_json"))
LINK_ROOT = Path(os.environ.get("GEND_DFBENCH_LINK_ROOT", "datasets_links/dfbench"))
DF40_RGB_ROOT = Path(os.environ.get("DF40_RGB_ROOT", "/data/hjr/df40/test"))
DF40_JSON_ROOT = Path(os.environ.get("DF40_JSON_ROOT", "/data/hjr/df40/test/dataset_json"))
DF40_LINK_ROOT = Path(os.environ.get("GEND_DF40_LINK_ROOT", "datasets_links/df40"))
COMPRESSION = "c23"
FRAME_NUM = 32

# short name used in GenD results -> DeepfakeBench manifest name
JSON_DATASETS = {
    "CDF-v2": "Celeb-DF-v2",
    "FSh": "FaceShifter",
    "DFD": "DeepFakeDetection",
    "DFDC": "DFDC",
    "DFDCP": "DFDCP",
}
FOLDER_DATASETS = ("WDF",)

# DF40 cross-method evaluation (DFD-HR paper Tab. 1, Protocol-2): result name -> manifest name
DF40_DATASETS = {
    "UniFace": "uniface_ff",
    "BlendFace": "blendface_ff",
    "MobileSwap": "mobileswap_ff",
    "e4s": "e4s_ff",
    "FaceDancer": "facedancer_ff",
    "FSGAN": "fsgan_ff",
    "InSwapper": "inswap_ff",
    "SimSwap": "simswap_ff",
}

# subset of dfd-hr training/config/test_config.yaml label_dict
LABEL_DICT = {
    "CelebDFv2_real": 0,
    "CelebDFv2_fake": 1,
    "FF-real": 0,
    "FF-FH": 1,
    "DFD_real": 0,
    "DFD_fake": 1,
    "DFDC_Real": 0,
    "DFDC_Fake": 1,
    "DFDCP_Real": 0,
    "DFDCP_FakeA": 1,
    "DFDCP_FakeB": 1,
    # DF40 (dfd-hr training/config/test_config_df40.yaml)
    **{f"{m}_{s}": int(s == "Fake") for m in (
        "uniface", "blendface", "mobileswap", "e4s", "facedancer", "fsgan", "inswap", "simswap",
    ) for s in ("Real", "Fake")},
}
COMPRESSED_DATASETS = {"FF-DF", "FF-F2F", "FF-FS", "FF-NT", "FaceForensics++", "DeepFakeDetection", "FaceShifter"}


def sample_frame_paths(frame_paths: list[str], frame_num: int) -> list[str]:
    """Same as dfd-hr training/dataset/path_utils.py::sample_frame_paths."""
    if len(frame_paths) <= frame_num:
        return list(frame_paths)
    step = max(1, len(frame_paths) // frame_num)
    return [frame_paths[i] for i in range(0, len(frame_paths), step)][:frame_num]


def resolve_rgb_path(file_path: str, rgb_root: Path) -> Path:
    """Same rule as dfd-hr training/dataset/path_utils.py::resolve_rgb_path."""
    normalized = file_path.replace("\\", "/")
    root = rgb_root.expanduser().resolve()
    path = Path(normalized)
    if PureWindowsPath(normalized).is_absolute() and not path.is_absolute():
        raise ValueError(f"Resolved dataset path is outside RGB root: {file_path}")
    resolved = (path if path.is_absolute() else root / path).resolve()
    resolved.relative_to(root)  # raises if outside the RGB root
    return resolved


def _frame_key(path: str) -> int:
    return int(path.replace("\\", "/").rsplit("/", 1)[-1].split(".")[0])


def load_json_videos(manifest_name: str, json_root: Path = DFBENCH_JSON_ROOT) -> list[tuple[str, int, list[str]]]:
    """Returns [(unique_video_name, label, sampled_frame_paths)] for the test split."""
    with open(json_root / f"{manifest_name}.json") as f:
        info = json.load(f)[manifest_name]

    videos = []
    for label_name in info:
        split = info[label_name]["test"]
        if manifest_name in COMPRESSED_DATASETS:
            split = split[COMPRESSION]
        for video_name, video_info in split.items():
            if video_info["label"] not in LABEL_DICT:
                raise ValueError(f"Label {video_info['label']} is not in LABEL_DICT ({manifest_name})")
            frames = video_info["frames"]
            if len(frames) == 0:
                continue
            frames = sample_frame_paths(sorted(frames, key=_frame_key), FRAME_NUM)
            videos.append((f"{video_info['label']}_{video_name}", LABEL_DICT[video_info["label"]], frames))
    return videos


def _safe(name: str) -> str:
    return name.replace("\\", "_").replace("/", "_")


def build_json_dataset(
    short_name: str,
    manifest_name: str,
    rgb_root: Path = DFBENCH_RGB_ROOT,
    json_root: Path = DFBENCH_JSON_ROOT,
    link_root: Path = LINK_ROOT,
) -> list[str]:
    """Creates (or reuses) the symlink tree for one manifest and returns the link paths."""
    links, missing = [], []
    for video, label, frames in load_json_videos(manifest_name, json_root):
        video_dir = link_root / short_name / ("real" if label == 0 else "fake") / _safe(video)
        video_dir.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            src = resolve_rgb_path(frame, rgb_root)
            if not src.is_file():
                missing.append(str(src))
                continue
            dst = video_dir / src.name
            if dst.is_symlink() and Path(os.readlink(dst)) != src:
                dst.unlink()
            if not dst.is_symlink():
                dst.symlink_to(src)
            links.append(str(dst.absolute()))
    if missing:
        raise FileNotFoundError(f"{short_name}: {len(missing)} manifest frames missing, e.g. {missing[:3]}")
    return links


def dfdhr_test_files() -> dict[str, list[str]]:
    from .files import CrossDataset, cap_frames_per_video

    tst = {short: build_json_dataset(short, manifest) for short, manifest in JSON_DATASETS.items()}
    folder = CrossDataset.get_test_files()
    for name in FOLDER_DATASETS:
        if name not in folder:
            raise FileNotFoundError(f"{name} not found under {CrossDataset.root}")
        tst[name] = cap_frames_per_video(folder[name], FRAME_NUM)
    return tst


def df40_test_files() -> dict[str, list[str]]:
    """DF40 cross-method test sets, same manifests / sampling as dfd-hr test_config_df40.yaml."""
    return {
        short: build_json_dataset(short, manifest, DF40_RGB_ROOT, DF40_JSON_ROOT, DF40_LINK_ROOT)
        for short, manifest in DF40_DATASETS.items()
    }
