import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ai.train_model import AITrainer

trainer = AITrainer()

trainer.train("data/processed/combined_training.csv")