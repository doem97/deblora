from .logger import CustomLogger
from .dota_dataset import LABEL_MAPPING

# Prevent unused import warnings by using __all__
__all__ = ["CustomLogger", "LABEL_MAPPING"]

# This ensures that these symbols are exported when using "from utils import *"
# It also signals to linters that these imports are intentional
