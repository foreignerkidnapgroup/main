import json
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent  # 현재 폴더

def get_secret(
    key: str,
    default_value: Optional[str] = None,
    json_path: str = str(BASE_DIR / "secrets.json"),  # secrets.json
):
    with open(json_path) as f:
        secrets = json.loads(f.read())
    try:
        return secrets[key]
    except KeyError:
        if default_value:
            return default_value
        raise EnvironmentError(f"Set the {key} environment variable.")
    

CLOVA_SPEECH_URL = get_secret("CLOVA_SPEECH_URL")
CLOVA_SPEECH_KEY = get_secret("CLOVA_SPEECH_KEY")


# if __name__ == "__main__":
#     print(get_secret("CLOVA_SPEECH_URL"))
    