import numpy as np
xp = np
gpu = False


# ===== CPU or GPU ===== #
# _set_cpu()

# ===== Network ===== #
# mag (only power) :1, complex: 2
AUDIO_CHANNELS = 2
VIS_CHANNNEL = 1792
AUDIO_LEN = 301
# Size of Fully Connected layer
FC_ROW = 1


# ===== Training ===== #
BATCH_SIZE = 6
ITERATION = 5000000
TRAIN = 1500000
EVALUATION = 10


def print_settings():

    print('==========================================')
    print('Audio channels:', AUDIO_CHANNELS)
    print('Iteration:', ITERATION)
    print('Batch size:', BATCH_SIZE)
    print('Train dataset Size:', TRAIN)
    print('Epoch:', (ITERATION * BATCH_SIZE) / TRAIN)
    print('Evaluation dataset Size:', EVALUATION)
    print('==========================================')