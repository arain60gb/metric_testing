# from huggingface_hub import hf_hub_download
# import numpy as np
# import librosa
# import torch
# import torchaudio
# from scipy.signal import hilbert
# from audiocraft.metrics import CLAPTextConsistencyMetric
# import bittensor as bt


# class MetricEvaluator:
#     @staticmethod
#     def calculate_snr(file_path, silence_threshold=1e-4, constant_signal_threshold=1e-2):
#         audio_signal, _ = librosa.load(file_path, sr=None)
#         if np.max(np.abs(audio_signal)) < silence_threshold or np.var(audio_signal) < constant_signal_threshold:
#             return 0
#         signal_power = np.mean(audio_signal**2)
#         noise_signal = librosa.effects.preemphasis(audio_signal)
#         noise_power = np.mean(noise_signal**2)
#         if noise_power < 1e-10:
#             return 0
#         snr = 10 * np.log10(signal_power / noise_power)
#         return snr

#     @staticmethod
#     def calculate_hnr(file_path):
#         """
#         Harmonic to noise ratio is a measure of the relations between tone and noise.
#         A high value means less noise, a low value means more noise.
#         """
#         y, _ = librosa.load(file_path, sr=None)
#         if np.max(np.abs(y)) < 1e-4 or np.var(y) < 1e-2:
#             return 0
#         harmonic, percussive = librosa.effects.hpss(y)
#         harmonic_power = np.mean(harmonic**2)
#         noise_power = np.mean(percussive**2)
#         hnr = 10 * np.log10(harmonic_power / max(noise_power, 1e-10))
#         return hnr

#     @staticmethod
#     def calculate_consistency(file_path, text):
#         try:
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             pt_file = hf_hub_download(repo_id="lukewys/laion_clap", filename="music_audioset_epoch_15_esc_90.14.pt")
#             clap_metric = CLAPTextConsistencyMetric(pt_file, model_arch='HTSAT-base').to(device)
#             def convert_audio(audio, from_rate, to_rate, to_channels):
#                 resampler = torchaudio.transforms.Resample(orig_freq=from_rate, new_freq=to_rate)
#                 audio = resampler(audio)
#                 if to_channels == 1:
#                     audio = audio.mean(dim=0, keepdim=True)
#                 return audio

#             audio, sr = torchaudio.load(file_path)
#             audio = convert_audio(audio, from_rate=sr, to_rate=sr, to_channels=1)

#             clap_metric.update(audio.unsqueeze(0), [text], torch.tensor([audio.shape[1]]), torch.tensor([sr]))
#             consistency_score = clap_metric.compute()
#             return consistency_score
#         except Exception as e:
#             print(f"An error occurred while calculating music consistency score: {e}")
#             return None

# class Normalizer:
#     @staticmethod
#     def normalize_quality(quality_metric):
#         # Normalize quality to be within 0 to 1, with good values above 20 dB considered as high quality
#         return 1 / (1 + np.exp(-((quality_metric - 20) / 10)))

#     @staticmethod
#     def normalize_consistency(score):
#         if score is not None:
#             if score > 0:
#                 normalized_consistency = (score + 1) / 2
#             else:
#                 normalized_consistency = 0
#         else:
#             normalized_consistency = 0
#         return normalized_consistency

# class Aggregator:
#     @staticmethod
#     def geometric_mean(scores):
#         """Calculate the geometric mean of the scores, avoiding any non-positive values."""
#         scores = [max(score, 0.0001) for score in scores.values()]  # Replace non-positive values to avoid math errors
#         product = np.prod(scores)
#         return product ** (1.0 / len(scores))

# class MusicQualityEvaluator:
#     def __init__(self):
#         self.metric_evaluator = MetricEvaluator()
#         self.normalizer = Normalizer()
#         self.aggregator = Aggregator()

#     def evaluate_music_quality(self, file_path, text=None):
#         try:
#             snr_score = self.metric_evaluator.calculate_snr(file_path)
#             bt.logging.info(f'.......SNR......: {snr_score} dB')
#         except:
#             pass
#             bt.logging.error(f"Failed to calculate SNR")

#         try:
#             hnr_score = self.metric_evaluator.calculate_hnr(file_path)
#             bt.logging.info(f'.......HNR......: {hnr_score} dB')
#         except:
#             pass
#             bt.logging.error(f"Failed to calculate SNR")

#         try:
#             consistency_score = self.metric_evaluator.calculate_consistency(file_path, text)
#             bt.logging.info(f'....... Consistency Score ......: {consistency_score}')
#         except:
#             pass
#             bt.logging.error(f"Failed to calculate Consistency score")

#         # Normalize scores and calculate aggregate score
#         normalized_snr = self.normalizer.normalize_quality(snr_score)
#         normalized_hnr = self.normalizer.normalize_quality(hnr_score)
#         normalized_consistency = self.normalizer.normalize_consistency(consistency_score)

#         bt.logging.info(f'Normalized Metrics: SNR = {normalized_snr}dB, Normalized Metrics: HNR = {normalized_hnr}dB, Consistency = {normalized_consistency}')
#         aggregate_quality = self.aggregator.geometric_mean({'snr': normalized_snr, 'hnr': normalized_hnr})
#         aggregate_score = self.aggregator.geometric_mean({'quality': aggregate_quality, 'normalized_consistency': normalized_consistency}) if consistency_score > 0.2 else 0
#         bt.logging.info(f'....... Aggregate Score ......: {aggregate_score}')
#         return aggregate_score


import os
import torch
import torchaudio
import numpy as np
import bittensor as bt
from huggingface_hub import hf_hub_download
from audiocraft.metrics import CLAPTextConsistencyMetric
from audioldm_eval.datasets.load_mel import WaveDataset
from audioldm_eval.metrics.kl import calculate_kl
from audioldm_eval.metrics.fad import FrechetAudioDistance
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader

class MetricEvaluator:
    @staticmethod
    def calculate_kld(generated_audio_dir, target_audio_dir):
      # Sampling rate of your audio data
      orig_sampling_rate = 32000
      print(f"Calculating KLD score between ............ {generated_audio_dir} and {target_audio_dir}...")

      # Initialize datasets
      generated_dataset = WaveDataset("/tmp/music", orig_sampling_rate)
      target_dataset = WaveDataset("/root/metric_testing/audio_files", orig_sampling_rate)

      # Use DataLoader to handle batching
      generated_loader = DataLoader(generated_dataset, batch_size=1, shuffle=False)
      target_loader = DataLoader(target_dataset, batch_size=1, shuffle=False)

      # Load pre-trained Wav2Vec2 model and processor (MERT or any suitable model)
      model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
      processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

      # Define resampler to match the model's expected sampling rate (24kHz)
      resampler = torchaudio.transforms.Resample(orig_freq=orig_sampling_rate, new_freq=24000)

      # Function to extract features from the audio
      def extract_features(loader, model, processor, resampler):
          features_dict = {"hidden_states": [], "file_path_": []}  # Initialize with keys for the feature type (hidden_states)
          for audio, filename in loader:
              # Resample the audio to match the model's required sampling rate (24 kHz)
              audio = resampler(audio)
              audio = audio.squeeze(0)  # Remove batch dimension
              audio = audio.numpy()  # Convert to NumPy array

              # Process and extract features
              inputs = processor(audio, sampling_rate=24000, return_tensors="pt")

              # Pass the processed inputs to the model
              outputs = model(**inputs, output_hidden_states=True)

              # Use the last hidden state for comparison
              hidden_states = outputs.hidden_states[-1].mean(dim=1)  # Average over the time dimension

              # Store features (hidden states) and filenames
              features_dict["hidden_states"].append(hidden_states.squeeze(0))  # Assuming batch size of 1

              # Unpack the filename tuple and store as string
              features_dict["file_path_"].append(filename[0])  # Store the filename as a string

          return features_dict

      # Extract features for both generated and target audio
      generated_features = extract_features(generated_loader, model, processor, resampler)
      target_features = extract_features(target_loader, model, processor, resampler)

      # Calculate KLD using the feature layer name as "hidden_states"
      kl_metrics, _, _ = calculate_kl(generated_features, target_features, feat_layer_name="hidden_states", same_name=True)
      kld = max(0, kl_metrics["kullback_leibler_divergence_sigmoid"])
      return kld

    @staticmethod
    def calculate_fad(generated_audio_dir, target_audio_dir):
      # Initialize the Frechet Audio Distance calculator
      fad_calculator = FrechetAudioDistance()
      print(f"Calculating FAD score between ............ {generated_audio_dir} and {target_audio_dir}...")

      # Calculate the FAD score between the two directories
      fad_score = fad_calculator.score(
          background_dir='"/tmp/music"',  # Generated audio directory
          eval_dir="/root/metric_testing/audio_files",           # Target audio directory
          store_embds=False,                   # Set to True if you want to store embeddings for later reuse
          limit_num=None,                      # Limit the number of files to process, None means no limit
          recalculate=True                     # Set to True if you want to recalculate embeddings
      )

      # Extract the FAD score from the dictionary
      fad_value = fad_score['frechet_audio_distance']

      # Clamp the value to 0 if it's negative
      fad = max(0, fad_value)
      return fad

    @staticmethod
    def calculate_consistency(generated_audio_dir, text):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pt_file = hf_hub_download(repo_id="lukewys/laion_clap", filename="music_audioset_epoch_15_esc_90.14.pt")
            clap_metric = CLAPTextConsistencyMetric(pt_file, model_arch='HTSAT-base').to(device)

            def convert_audio(audio, from_rate, to_rate, to_channels):
                resampler = torchaudio.transforms.Resample(orig_freq=from_rate, new_freq=to_rate)
                audio = resampler(audio)
                if to_channels == 1:
                    audio = audio.mean(dim=0, keepdim=True)
                return audio

            # Check if generated_audio_dir is a file or directory
            if os.path.isfile(generated_audio_dir):
                file_path = generated_audio_dir
            else:
                # Get the single audio file path in the directory
                file_name = next((f for f in os.listdir(generated_audio_dir) if os.path.isfile(os.path.join(generated_audio_dir, f))), None)
                if file_name is None:
                    raise FileNotFoundError("No audio file found in the directory.")
                file_path = os.path.join(generated_audio_dir, file_name)

            # Load and process the audio
            audio, sr = torchaudio.load(file_path)
            audio = convert_audio(audio, from_rate=sr, to_rate=sr, to_channels=1)

            # Calculate consistency score
            clap_metric.update(audio.unsqueeze(0), [text], torch.tensor([audio.shape[1]]), torch.tensor([sr]))
            consistency_score = clap_metric.compute()

            return consistency_score
        except Exception as e:
            print(f"An error occurred while calculating music consistency score: {e}")
            return None

class Normalizer:
    @staticmethod
    def normalize_kld(kld_score, max_kld=2):
        if kld_score is not None:
            if kld_score > max_kld:
                normalized_kld = 0
            else:
                normalized_kld = 1 - (kld_score / max_kld)
        else:
            normalized_kld = 0
        return normalized_kld

    @staticmethod
    def normalize_fad(fad_score, min_score=0, max_score=10):
        if fad_score is not None:
            if fad_score > max_score:
                normalized_fad = 0
            else:
                normalized_fad = (fad_score - min_score) / (max_score - min_score)
        else:
            normalized_fad = 0
        return normalized_fad

    @staticmethod
    def normalize_consistency(score):
        if score is not None:
            if score > 0:
                normalized_consistency = (score + 1) / 2
            else:
                normalized_consistency = 0
        else:
            normalized_consistency = 0
        return normalized_consistency

class Aggregator:
    @staticmethod
    def geometric_mean(scores):
        """Calculate the geometric mean of the scores, avoiding any non-positive values."""
        scores = [max(score, 0.0001) for score in scores.values()]  # Replace non-positive values to avoid math errors
        product = np.prod(scores)
        return product ** (1.0 / len(scores))

class MusicQualityEvaluator:
    def __init__(self):
        self.metric_evaluator = MetricEvaluator()
        self.normalizer = Normalizer()
        self.aggregator = Aggregator()

    def evaluate_music_quality(self, generated_audio_dir, target_audio_dir, text=None):
        kld_score = None
        fad_score = None
        consistency_score = None

        try:
            kld_score = self.metric_evaluator.calculate_kld(generated_audio_dir, target_audio_dir)
            bt.logging.info(f'.......KLD......: {kld_score}')
            print(f'.......KLD......: {kld_score}')
        except Exception as e:
            bt.logging.error(f"Failed to calculate KLD: {e}")

        try:
            fad_score = self.metric_evaluator.calculate_fad(generated_audio_dir, target_audio_dir)
            bt.logging.info(f'.......FAD......: {fad_score}')
            print(f'.......FAD......: {fad_score}')
        except Exception as e:
            bt.logging.error(f"Failed to calculate FAD: {e}")

        try:
            consistency_score = self.metric_evaluator.calculate_consistency(generated_audio_dir, text)
            bt.logging.info(f'....... Consistency Score ......: {consistency_score}')
            print(f'....... Consistency Score ......: {consistency_score}')
        except Exception as e:
            bt.logging.error(f"Failed to calculate Consistency score: {e}")

        # Normalize scores and calculate aggregate score
        normalized_kld = self.normalizer.normalize_kld(kld_score)
        normalized_fad = self.normalizer.normalize_fad(fad_score)
        normalized_consistency = self.normalizer.normalize_consistency(consistency_score)

        bt.logging.info(f'Normalized Metrics: KLD = {normalized_kld}, Normalized Metrics: FAD = {normalized_fad}, Consistency = {normalized_consistency}')
        print(f'Normalized Metrics: KLD = {normalized_kld}, Normalized Metrics: FAD = {normalized_fad}, Consistency = {normalized_consistency}')
        aggregate_quality = self.aggregator.geometric_mean({'KLD': normalized_kld, 'FAD': normalized_fad})
        aggregate_score = self.aggregator.geometric_mean({'quality': aggregate_quality, 'normalized_consistency': normalized_consistency}) if consistency_score and consistency_score > 0.2 else 0
        bt.logging.info(f'....... Aggregate Score ......: {aggregate_score}')
        print(f'....... Aggregate Score ......: {aggregate_score}')
        return aggregate_score