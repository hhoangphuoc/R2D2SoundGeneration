# THIS FILE INCLUDES ALL PRE-DEFINED PARAMETERS FOR 
# THE PROCESS OF R2D2 SOUND GENERATION

SAMPLE_RATE = 16000

# R2D2 SOUND GENERATION PARAMETERS
BASE_FREQ = 1000
FREQ_OFFSET = 100
CHIRP_RANGE = (5, 20) #number of chirps
CHIRP_SPEED = (0.1, 0.8) #chirp duration
BABBLE_RANGE = (10, 50) #babble frequency
BABBLE_AMOUNT = (5, 30) #babble amplitude

AMPLITUDE_SCALE = 0.5
DEF_CUTOFF_FREQUENCY = 4000

#CONFIGs
CARRIER_FREQ = 1000
MODULATOR_RATIO = 32
MODULATOR_INDEX = 2
FM_AMP = 0.5


MIN_DURATION = 15 #45  #minimum duration of each r2d2 audio
MAX_DURATION = 60 #180  #maximum duration of each r2d2 audio

NUM_SOUNDS = 100 # number of audios to generate
OUTPUT_DIR = "../data/r2d2"

