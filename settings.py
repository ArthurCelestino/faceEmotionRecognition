import os
from datetime import date


EPOCHS = 150
BATCH_SIZE = 32

## defining the input size
WIDTH = 48
HEIGHT = 48
DEPTH= 1

INPUT_SHAPE = (HEIGHT, WIDTH, DEPTH)
## definitions of the optimizers
OPT_INIT_LR = 1e-3
OPT_DECAY = OPT_INIT_LR / EPOCHS


