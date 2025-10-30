from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'data' / 'creditcard.csv'
MODELS_DIR = ROOT / 'models'

TARGET = "Class"
SCALE_COLS = ['Amount', 'Time']
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42