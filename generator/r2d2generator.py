import numpy as np
from pyo import *
from tqdm import tqdm
import params
import os
import argparse



class R2D2SoundGenerator:
    def __init__(
        self, 
        duration=10, 
        sample_rate=params.SAMPLE_RATE
        ):
        self.s = Server(sr=sample_rate, nchnls=1, buffersize=512, duplex=0).boot()
        self.s.start()
        self.duration = duration

        # Constants for R2D2 sound
        self.base_freq = params.BASE_FREQ
        self.freq_offset = params.FREQ_OFFSET
        self.chirp_range = params.CHIRP_RANGE 
        self.chirp_speed = params.CHIRP_SPEED #chirp duration
        self.babble_range = params.BABBLE_RANGE #babble frequency
        self.babble_amount = params.BABBLE_AMOUNT #babble amplitude

        # Random generators
        self.rand_gen1 = Randi(min=0, max=3, freq=Randi(1, 5))
        self.rand_gen2 = Randi(min=0, max=1000, freq=Randi(1, 3))
        self.rand_gen3 = Randi(min=0, max=9000, freq=Randi(0.1, 1))

        # FM Synthesis
        self.carrier_freq = Sig(params.CARRIER_FREQ)
        self.modulator_ratio = Sig(params.MODULATOR_RATIO)
        self.modulator_index = Sig(params.MODULATOR_INDEX)
        self.fm_amp = Sig(params.FM_AMP)

        self.modulator = Sine(freq=self.carrier_freq * self.modulator_ratio, mul=self.modulator_index)
        self.carrier = Sine(freq=self.carrier_freq + self.modulator + self.freq_offset, mul=self.fm_amp)

        # Phasor and cosine for waveform shaping
        self.phasor = Phasor(freq=self.rand_gen2, mul=0.6)
        self.shaped_wave = Cos(self.phasor + self.carrier)

        # Chirp effect (cute high-pitched sounds)
        self.chirp = Sine(freq=Randi(*self.chirp_range), mul=Randi(*self.chirp_speed), add=1)

        # Babble effect (pleasant randomness)
        self.babble = Sine(freq=Randi(*self.babble_range), mul=Randi(*self.babble_amount), add=0.6)

        # Combine effects
        self.combined_effect = self.shaped_wave * self.chirp * self.babble

        # Apply filters
        self.lowpass1 = Biquad(self.combined_effect, freq=10000, q=0.7, type=0)
        self.lowpass2 = Biquad(self.lowpass1, freq=10000, q=0.7, type=0)
        self.highpass1 = Biquad(self.lowpass2, freq=100, q=0.7, type=1)
        self.highpass2 = Biquad(self.highpass1, freq=100, q=0.7, type=1)

        # Final output with softer volume
        self.output = self.highpass2 * 0.02
        self.output.out()

    def change_parameters(self):
        # Simulate random parameter changes
        new_freq = self.rand_gen3.get() + self.base_freq
        self.carrier_freq.value = new_freq
        self.modulator_index.value = self.rand_gen1.get() * 0.8  # Soften sound by reducing modulation

    def generate_sound(self, filename):
        pat = Pattern(self.change_parameters, time=0.1).play()
        
        # Record the output
        recording = Record(self.output, filename=filename, chnls=1, fileformat=0, sampletype=0)
        
        # Start the server and record for the specified duration
        time.sleep(self.duration)
        recording.stop()
        self.s.stop()
        print(f"Sound saved to {filename}")

def generate_r2d2_dataset(
    num_sounds, 
    output_dir="../datasets/r2d2", 
    base_filename="r2d2", 
    min_duration=params.MIN_DURATION,
    max_duration=params.MAX_DURATION,
    sample_rate=params.SAMPLE_RATE
    ):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in tqdm(range(num_sounds), desc="Generating R2D2 sounds..."):
        duration = np.random.randint(min_duration, max_duration)
        try:
            generator = R2D2SoundGenerator(duration=duration, sample_rate=sample_rate)
            filename = f"{base_filename}_{i+1}.wav"
            generator.generate_sound(filename)
        except Exception as e:
            print(f"Error generating sound {i+1}: {e}")
            continue

    print("R2D2 sound dataset generated successfully!")
# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sounds", type=int, default=params.NUM_SOUNDS, help="Number of sounds to generate")
    parser.add_argument("--output_dir", type=str, default=params.OUTPUT_DIR, help="Output directory")
    parser.add_argument("--base_filename", type=str, default="r2d2", help="Sound filename")
    parser.add_argument("--min_duration", type=int, default=params.MIN_DURATION, help="Minimum duration of a sound")
    parser.add_argument("--max_duration", type=int, default=params.MAX_DURATION, help="Maximum duration of a sound")
    parser.add_argument("--sample_rate", type=int, default=params.SAMPLE_RATE, help="Sample rate")

    args = parser.parse_args()

    generate_r2d2_dataset(
        args.num_sounds, 
        args.output_dir, 
        args.base_filename, 
        args.min_duration, 
        args.max_duration, 
        args.sample_rate
    )