import sys
from pathlib import Path

# Add parent directory to path when running from inside the package
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from finance_forecast_research import config

print("Available Models in Config:")
for model_name in config.MODELS:
    print(f"- {model_name}")