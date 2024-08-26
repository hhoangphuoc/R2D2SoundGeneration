import numpy as np
import librosa
import soundfile as sf
import random
from tqdm import tqdm
import os
import argparse

def augment_r2d2_sounds(
    samples_audio_path,
    # r2d2_path, 
    # chatter_path, 
    output_dir, 
    num_samples=120,
    sr=44100,
    min_length=45,
    max_length=120,
    ):
    """
    Augment R2D2 and robot chatter sounds to create a dataset of varied audio samples.
    
    :param r2d2_path: Path to the 15-minute R2D2 sound file
    :param chatter_path: Path to the robot chatter sound file
    :param output_dir: Directory to save the augmented audio files
    :param num_samples: Number of augmented samples to generate (default: 34)
    """
    # Load audio files
    audio_paths = [os.path.join(samples_audio_path, f) for f in os.listdir(samples_audio_path) if f.endswith('.wav')]
    audios = []
    for audio_path in audio_paths:
        audio, audio_sr = librosa.load(audio_path, sr=sr)
        if audio_sr != 44100:
            audio = librosa.resample(audio, audio_sr, 44100)
        audios.append(audio)
    #resample if necessary
    # if sr != 44100:
    #     r2d2_audio = librosa.resample(r2d2_audio, sr, 44100)
    #     chatter_audio = librosa.resample(chatter_audio, sr, 44100)
    
    for i in tqdm(range(num_samples), desc="Generating augmented r2d2 audio..."):
        # Randomly choose the length of the output audio (45-120 seconds)
        output_length = random.randint(min_length, max_length)
        
        # Initialize an empty array for the output audio
        output_audio = np.array([])
        
        while len(output_audio) < output_length * sr:
            # Randomly choose between several samples
            # source_audio = r2d2_audio if random.random() < 0.7 else chatter_audio
            source_audio = random.choice(audios) #randomly choose a sample audio
            
            # Random crop from the source audio
            start = random.randint(0, len(source_audio) - sr)
            end = min(start + random.randint(1, 5) * sr, len(source_audio))
            segment = source_audio[start:end]
            
            # Apply random augmentations
            # 1. Pitch shift
            segment = librosa.effects.pitch_shift(segment, sr=sr, n_steps=random.uniform(-2, 2))
            
            # 2. Time stretch
            segment = librosa.effects.time_stretch(segment, rate=random.uniform(0.8, 1.2))
            
            # 4. Random volume adjustment
            volume_factor = random.uniform(0.8, 1.2)
            segment = segment * volume_factor
            
            # Append the augmented segment to the output audio
            output_audio = np.concatenate((output_audio, segment))
        
        # Trim the output audio to the desired length
        output_audio = output_audio[:output_length * sr]
        
        # Normalize the audio
        output_audio = librosa.util.normalize(output_audio)
        
        # Save the augmented audio
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = f"{output_dir}/augmented_r2d2_{i+1}.wav"
        sf.write(output_path, output_audio, sr)
        
    print(f"Generated {num_samples} augmented r2d2 audio files!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_audio_path", type=str, required=True, help="Source directory foraudio samples")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for augmented audio")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of augmented samples to generate")
    parser.add_argument("--sr", type=int, help="Sampling rate")
    parser.add_argument("--min_length", type=int, help="Min length of each augmented sample")
    parser.add_argument("--max_length", type=int, help="Max length of each augmented sample")
    args = parser.parse_args()

    augment_r2d2_sounds(
        args.samples_audio_path, 
        args.output_dir, 
        args.num_samples,
        args.sr,
        args.min_length,
        args.max_length
    )

    # Usage example:
# augment_r2d2_sounds("samples/sample_r2d2.wav", "samples/robochatter.wav", "./aug_r2d2/", num_samples=34)


