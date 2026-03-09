import torch
import numpy as np

cuda_available = torch.cuda.is_available()
USE_ONLY_CPU = False
if USE_ONLY_CPU:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if cuda_available else "cpu")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

EPOCHS = 10
WARMUP_EPOCHS = 10
BATCH_SIZE = 128
N_CLASSES = 10
N_WORKERS = 0
LR = 5e-4
OUTPUT_PATH = './outputs'

DATASET = 'fmnist'
IMAGE_SIZE = 28
PATCH_SIZE = 4
N_CHANNELS = 1
DATA_PATH = './data/'

EMBED_DIM = 64
N_ATTENTION_HEADS = 4
FORWARD_MUL = 2
N_LAYERS = 6
DROPOUT = 0.1
MODEL_PATH = './models'
LOAD_MODEL = False

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}