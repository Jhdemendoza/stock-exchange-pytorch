import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_STATE_SPACE = 40
BATCH_SIZE = 32
PCT_TRAIN = 0.8
NUM_DISCRETE_RETURNS = 10
MAX_POSSIBLE_VALUE = 0.03
GRU_OUTPUT_SIZE = 256

