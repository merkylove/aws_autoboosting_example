import os

import numpy as np

RANDOM_SEED = 42


np.random.seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
