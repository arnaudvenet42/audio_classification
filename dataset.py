from pathlib import Path
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


# Dataset
class UrbanSoundDataset(Dataset):
    #
    def __init__(self, audio_dir, target_sample_rate, num_samples, transformation=None, device='cpu'):
        self.audio_dir = audio_dir
        self.annotations = pd.read_csv(audio_dir / "UrbanSound8K.csv")
        self.annotations = self.annotations.loc[self.annotations.fold == 1]
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.device = device
        if transformation is not None:
            self.transformation = transformation.to(self.device)
    #
    def __len__(self):
        return len(self.annotations)
    #
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(str(audio_sample_path))
        # device
        signal = signal.to(self.device)
        # same sample rate
        signal = self._resample_if_necessary(signal, sr)
        # mono channel
        signal = self._mix_down_if_necessary(signal)
        # same duration
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        # mel spectrogram
        if self.transformation is not None:
            signal = self.transformation(signal)
        return signal, label
    #
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    #
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            signal = torch.concat([signal, torch.zeros(1, self.num_samples-length_signal)], axis=1)
        return signal
    #
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    #
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        return signal
    #
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index].fold}"
        filename = self.annotations.iloc[index].slice_file_name
        return self.audio_dir / fold / filename
    #
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index].classID



if __name__ == "__main__":
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    DATASET_DIR = Path(r"./data")
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050  # 1s duration

    transformation = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )


    dataset = UrbanSoundDataset(DATASET_DIR, SAMPLE_RATE, NUM_SAMPLES, transformation=transformation, device=device)
    signal, label = dataset[45]
    print(signal.shape)

    import matplotlib.pyplot as plt
    plt.imshow(signal[0])
    plt.show()
