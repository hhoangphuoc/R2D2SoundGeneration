# R2D2SoundGeneration

This is a sub-project that aims at generation a R2D2-alike sound dataset

## Requirements

The following packages is required to run the code:

```
numpy scipy pyo tqdm
```

## Generating Process

The data set is generated using the `r2d2augmentation.py` script. By running the following command:

```bash
python r2d2augmentation.py
--samples_audio_path <path_to_samples>
--output_dir <path_to_output>
--num_samples <num_samples>
--sr <sample_rate>
--min_length <min_length>
--max_length <max_length>
```

You can specify the parameters for the file as you want to
create your own size of the dataset. The following parameters can be specified:

- `samples_audio_path`: The path to the source audio samples.
- `output_dir`: Specify your output direction
- `num_samples`: The number of augmented samples to generate R2D2 Sound
- `sr`: The sample rate.
- `min_length`: The minimum length of each augmented sample.
- `max_length`: The maximum length of each augmented sample.

## Considerations
