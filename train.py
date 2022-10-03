from pathlib import Path
import pandas as pd
import torch
import torchaudio
import torch.nn as nn
from dataset import UrbanSoundDataset, MelSpectrogram3Channels
from model import CNN, DenseNet

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
BATCH_SIZE = 24
LEARNING_RATE = 1e-4

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

if __name__ == "__main__":
    # dataset
    dataset = UrbanSoundDataset(DATASET_DIR, SAMPLE_RATE, NUM_SAMPLES, transformation=transformation, device=device, folder=1, data_augment=True)
    labels, counts = torch.tensor([dataset._get_audio_sample_label(k) for k in range(len(dataset))]).unique(return_counts=True)
    weights = counts/counts.sum()

    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=2)
    # iter(dataloader)
    # model
    # model = CNN().to(device)
    model = DenseNet().to(device)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train
    running_losses = []
    model.train()
    for epoch in range(10):  # loop over the dataset multiple times
        print(epoch)
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):  # signal, labels = next(iter(dataloader))
            # print(epoch, i)
            # get the signal; data is a list of [signal, labels]
            signal, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            # classic model
            # outputs = model(signal)
            # loss = (criterion(outputs, labels) * weights[labels]).mean()
            # inception model
            outputs = model(signal)
            loss = (criterion(outputs, labels) * weights[labels]/(weights[labels].sum())*len(labels)).mean()
            # loss = criterion(outputs, labels).mean()
            # optimization
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
        running_losses.append(running_loss)
        torch.save(model.state_dict(), "model/model_dense_DA_fold2.pth")
        print(running_loss)


    print('Finished Training')
    import matplotlib.pyplot as plt
    plt.plot(running_losses)
    plt.show()
    # save model
