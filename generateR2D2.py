import os
import random
import numpy as np
import scipy.io.wavfile as wavfile
import numpy as np
import scipy.signal as signal
from sounddesign import generate_beep, random_line_slide, fm_synthesis, apply_low_pass_filter, apply_distortion, apply_delay, apply_reverb

import argparse
from tqdm import tqdm
#------------------------------------------------------------

# MANUAL IMPLEMENTATION 
# The implementation based on the guideline of creating R2D2 sound in Design Sound book by Andy Farnell
# The implementation is based on the following steps:
# 1. Random Line Slider with Noise
# 2. FM Synthesis
# 3. Apply Effects (if needed) - Low Pass Filter, Reverb, Distortion, etc.
# 3. Complete Patch
# 6. Write to WAV file

def apply_effects(audio, effects_list):
    for effect in effects_list:
        if effect == "low_pass_filter":
            audio = apply_low_pass_filter(audio)
        elif effect == "reverb":
            audio = apply_reverb(audio)
        elif effect == "distortion":
            audio = apply_distortion(audio)
        elif effect == "delay":
            audio = apply_delay(audio)
    return audio

        

def generate_r2d2_sound(
    length=10, 
    sample_rate=16000, 
    amplitude_scale=0.5, 
    beep_freq_range=(1000, 5000), #default range for beep frequency
    beep_dur_range=(0.05, 0.5), #default range for beep duration
    applied_effects=False,
    effects_list=[] #"low_pass_filter","reverb","distortion","delay"
    ):
    """Generates an R2-D2-like sound using Python.

    Args:
        length (float, optional): Length of the audio in seconds. Defaults to 5.

    Returns:
        numpy.ndarray: The generated audio signal.
    """
    t = np.linspace(0, length, int(length * sample_rate), False)

    # Generate beep parameters
    lower_num= int(length * 10)
    upper_num = int(length * 20)
    
    num_beeps = random.randint(lower_num, upper_num)  # Adjust number of beeps as needed
    beep_frequencies = np.random.uniform(beep_freq_range[0], beep_freq_range[1], num_beeps)
    beep_durations = np.random.uniform(beep_dur_range[0], beep_dur_range[1], num_beeps)
    beep_start_times = np.random.uniform(0, length - max(beep_durations), num_beeps)

    # Apply Random Line Slider
    carrier_freq = random_line_slide(length=length, sample_rate=sample_rate)  # adjusted to influence the overall pitch of the sound
    modulation_index = random_line_slide(length=length, sample_rate=sample_rate)
    modulator_freq = random_line_slide(length=length, sample_rate=sample_rate)

    # add random line slide to patch amplitude
    amplitude = random_line_slide(length, sample_rate) * amplitude_scale  # Scale amplitude
    
    # Apply FM synthesis
    audio = fm_synthesis(amplitude, carrier_freq, modulation_index, modulator_freq, length, sample_rate)


    # Add beeps
    for i in range(num_beeps):
        beep = generate_beep(beep_durations[i], sample_rate, beep_frequencies[i])
        start_index = int(beep_start_times[i] * sample_rate)
        end_index = start_index + len(beep)
        audio[start_index:end_index] += beep

    # Apply effects
    if applied_effects:
        audio = apply_effects(audio, effects_list)
    
    # Normalize and convert to integer
    audio /= np.max(np.abs(audio))
    audio = np.int16(audio * 32767)

    return audio


# # Generate single R2-D2 sound
# audio = generate_r2d2_sound()
# wavfile.write("r2d2_sound.wav", 16000, audio)

# create a function that generate multiple R2D2 sound in different length and save it to a folder
def generate_r2d2_dataset(
    num_audios=300, 
    output_path="../datasets/r2d2",
    min_audio_length=15,
    max_audio_length=45,
    sample_rate=16000, 
    amplitude_scale=0.5, 
    customised_beep=False,
    beep_freq_range=(1000, 5000),
    beep_dur_range=(0.05, 0.5),
    applied_effects=False,
    effects_list=[] #"low_pass_filter","reverb","distortion","delay"
    ):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in tqdm(range(num_audios), desc="Generating R2D2 dataset..."):
        length = np.random.uniform(min_audio_length, max_audio_length)
        audio = generate_r2d2_sound(
            length=length, 
            sample_rate=sample_rate, 
            amplitude_scale=amplitude_scale, 
            beep_freq_range=beep_freq_range,
            beep_dur_range=beep_dur_range,
            applied_effects=applied_effects,
            effects_list=effects_list
        )

        wavfile.write(f"{output_path}/r2d2_{i}.wav", sample_rate, audio)
    print("R2D2 dataset created!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_audios', type=int, default=400, help='Number of audios to generate')
    parser.add_argument('--output_path', type=str, default="../datasets/r2d2", help='Output path to save the generated audio')
    parser.add_argument('--min_audio_length', type=float, default=15, help='Minimum audio length')
    parser.add_argument('--max_audio_length', type=float, default=45, help='Maximum audio length')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate')
    parser.add_argument('--amp_scale', type=float, default=0.5, help='Emphasise of the beep to the overall sound')
    
    # Customised beep
    parser.add_argument('--customised_beep', type=bool, default=False, help='Customised beep')
    parser.add_argument('--beep_freq_range', type=tuple, default=(1000, 5000), help='Beep frequency range')
    parser.add_argument('--beep_dur_range', type=tuple, default=(0.05, 0.5), help='Beep duration range')

    # Effects
    parser.add_argument('--applied_effects', type=bool, default=False, help='Determine if the effects are applied')
    parser.add_argument('--effects_list', type=list, default=[], help='Effects list for tuning the sound')

    
    args = parser.parse_args()

    beep_freq_range = (1000, 5000)
    beep_dur_range = (0.05, 0.5)
    effects_list = []

    if args.customised_beep:
        beep_freq_range = args.beep_freq_range
        beep_dur_range = args.beep_dur_range
    elif args.applied_effects:
        effects_list = args.effects_list
    else:
        continue
        
    generate_r2d2_dataset(
        num_audios=args.num_audios,
        output_path=args.output_path,
        min_audio_length=args.min_audio_length,
        max_audio_length=args.max_audio_length,
        sample_rate=args.sr,
        amplitude_scale=args.amp_scale,
        customised_beep=args.customised_beep,
        beep_freq_range=beep_freq_range,
        beep_dur_range=beep_dur_range,
        applied_effects=args.applied_effects,
        effects_list=effects_list
    )