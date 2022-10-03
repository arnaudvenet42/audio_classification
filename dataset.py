from pathlib import Path
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms import Resize
import numpy as np
import librosa


# Dataset
class UrbanSoundDataset(Dataset):
    #
    def __init__(self, audio_dir, target_sample_rate, num_samples, transformation=None, device='cpu', folder=1, data_augment=False):
        self.audio_dir = audio_dir
        self.annotations = pd.read_csv(audio_dir / "UrbanSound8K.csv")
        self.annotations = self.annotations.loc[self.annotations.fold == folder]
        self.target_sample_rate = target_sample_rate
        self.resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
        self.num_samples = num_samples
        self.device = device
        self.data_augment = data_augment
        if transformation is not None:
            self.transformation = transformation  # .to(self.device)
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
        # DA
        if self.data_augment:
            pitch_shift = np.random.randint(-2, 3)  # shifting pitch from -2 to 2 semi tone
            time_stretch = np.random.random() * (1.2 - 0.9) + 0.9  # speed up if > 1, slow down if < 1
            signal = torch.tensor(librosa.effects.time_stretch(librosa.effects.pitch_shift(signal.numpy(), sr=sr, n_steps=pitch_shift), rate=time_stretch))
        # same duration
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        # mel spectrogram
        if self.transformation is not None:
            signal = self.transformation(signal)
            # signal = signal / signal.max()
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
            signal = self.resampler(signal)
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


class MelSpectrogram3Channels:
    def __init__(self, sample_rate, n_ffts, hop_lengths, n_mels):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_ffts = n_ffts  # win_lengths
        self.hop_lengths = hop_lengths
        self.resized = Resize((128, 250))
        self.melspec_transforms = []
        for channel in range(3):
            self.melspec_transforms.append(torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_ffts[channel],
                hop_length=self.hop_lengths[channel],
                n_mels=self.n_mels, 
                center=False
            ))
    #
    def __call__(self, signal):
        melspecs = [self.melspec_transforms[channel](signal) for channel in range(3)]
        melspec_3channels = torch.vstack([self.resized(melspecs[0]/torch.max(melspecs[0])), self.resized(melspecs[1]/torch.max(melspecs[1])), self.resized(melspecs[2]/torch.max(melspecs[2]))])
        return melspec_3channels
    #
    def __repr__(self) -> str:
        return f"SAMPLE RATE: {self.sample_rate}\nN_FFTS: {self.n_ffts}\nHOP_LENGTHS: {self.hop_lengths}\nN_MELS: {self.n_mels}\nOUTPUT SIZE: (3, 128, 250)"



if __name__ == "__main__":
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    DATASET_DIR = Path(r"./data/audio")
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050*2  # 2s duration
    N_MELS = 128
    # Three channels MelSpec
    HOP_LENGTHS = [256, 512, 1024]  # to match 10ms, 25ms, 50ms with SAMPLE_RATE 22050
    N_FFTS = [512, 1024, 2048]  # to match 25ms, 50ms, 100ms with SAMPLE_RATE 22050
    # Three channels Mel Spec
    transformation = MelSpectrogram3Channels(
        sample_rate=SAMPLE_RATE,
        n_ffts=N_FFTS,
        hop_lengths=HOP_LENGTHS,
        n_mels=N_MELS
    )

    dataset = UrbanSoundDataset(DATASET_DIR, SAMPLE_RATE, NUM_SAMPLES, transformation=transformation, device=device)
    index = np.random.randint(len(dataset))
    signal, label = dataset[index]
    print(dataset.annotations.iloc[index])

    import matplotlib.pyplot as plt
    plt.imshow(signal.permute(1, 2, 0))
    plt.show()
