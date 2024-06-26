import os
from pathlib import Path


DATA_PATH = Path(os.environ["DATA_PATH"])
DATA_PATH.mkdir(exist_ok=True)