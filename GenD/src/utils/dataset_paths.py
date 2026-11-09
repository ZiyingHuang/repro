from collections.abc import Sized
from pathlib import Path

SOURCE_ALIASES = {
    "original": "real",
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def dataset_length(dataset: Sized | None) -> int:
    return 0 if dataset is None else len(dataset)


def _wdf_index(parts: tuple[str, ...]) -> int | None:
    try:
        return parts.index("WDF")
    except ValueError:
        return None


def normalize_source(source: str) -> str:
    return SOURCE_ALIASES.get(source, source)


def get_dataset_from_path(file_path: str) -> str:
    parts = Path(file_path).parts
    wdf_index = _wdf_index(parts)
    if wdf_index is not None:
        return "WDF"
    return parts[-4]


def get_source_from_path(file_path: str) -> str:
    parts = Path(file_path).parts
    wdf_index = _wdf_index(parts)
    if wdf_index is not None:
        source = parts[wdf_index + 1].removesuffix("_test")
    else:
        source = parts[-3]
    return normalize_source(source)


def get_video_from_path(file_path: str) -> str:
    parts = Path(file_path).parts
    wdf_index = _wdf_index(parts)
    if wdf_index is not None:
        return f"{parts[wdf_index + 2]}/{parts[-2]}"
    return parts[-2]


def resolve_input_paths(inputs: list[str]) -> list[str]:
    paths = []

    for input_path in inputs:
        input_file = Path(input_path)
        if input_file.suffix.lower() in IMAGE_SUFFIXES and input_file.is_file():
            paths.append(str(input_file))
            continue

        with input_file.open("r") as file:
            root = input_file.parent
            for line in file:
                path = Path(line.strip())
                if not path.is_absolute() and not path.exists():
                    path = root / path
                if not path.exists():
                    raise FileNotFoundError(f"File not found: {path}")
                paths.append(str(path))

    return paths
