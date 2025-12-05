import gdown
from pathlib import Path

# https://drive.google.com/file/d/1MYCo-EWakY9tE4PmaCXvoLhqn-A-kzps/view
FILE_ID = "1MYCo-EWakY9tE4PmaCXvoLhqn-A-kzps"

MODEL_PATH = Path("models/ensemble.pkl")

def download_model_if_needed() -> Path:
    """
    Downloads ensemble.pkl from Google Drive if it doesn't exist locally.
    Returns the path to the model file.
    """
    if MODEL_PATH.exists():
        print(f"[INFO] Model already exists at: {MODEL_PATH}")
        return MODEL_PATH

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print(f"[INFO] Downloading model from: {url}")
    gdown.download(url, str(MODEL_PATH), quiet=False)
    print(f"[INFO] Download complete: {MODEL_PATH}")

    return MODEL_PATH


if __name__ == "__main__":
    download_model_if_needed()
