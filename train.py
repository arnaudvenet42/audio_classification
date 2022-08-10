from pathlib import Path
import pandas as pd
import torch
import torchaudio
import torch.nn as nn
from dataset import UrbanSoundDataset
from model import CNN


DATASET_DIR = Path(r"./data")
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050  # 1s duration
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# transform
transformation = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

if __name__ == "__main__":
    # dataset
    dataset = UrbanSoundDataset(DATASET_DIR, SAMPLE_RATE, NUM_SAMPLES, transformation=transformation, device=device)
    labels, counts = torch.tensor([dataset._get_audio_sample_label(k) for k in range(len(dataset))]).unique(return_counts=True)
    weights = counts/counts.sum()

    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=2)
    # model
    model = CNN().to(device)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    # train
    running_losses = []
    model.train()
    for epoch in range(50):  # loop over the dataset multiple times
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
            # optimization
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
        running_losses.append(running_loss)
        print(running_loss)

    print('Finished Training')
    import matplotlib.pyplot as plt
    plt.plot(running_losses)
    plt.show()

    # save model
    torch.save(model.state_dict(), "model/model.pth")