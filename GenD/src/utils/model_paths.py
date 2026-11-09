from pathlib import Path

CLIP_L14_MODEL_ID = "openai/clip-vit-large-patch14"
LOCAL_CLIP_L14_PATH = Path("/home/huangjunrong/aid/main/CLIP-main/checkpoint/clip-vit-large-patch14")


def resolve_clip_model_path(model_name: str) -> str:
    required_files = ("config.json", "preprocessor_config.json", "model.safetensors")
    if model_name == CLIP_L14_MODEL_ID and all((LOCAL_CLIP_L14_PATH / name).is_file() for name in required_files):
        return str(LOCAL_CLIP_L14_PATH)
    return model_name
