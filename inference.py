from pathlib import Path
import torch
import torchaudio
import torch.nn as nn
from dataset import UrbanSoundDataset, MelSpectrogram3Channels
from model import CNN, DenseNet
from train import DATASET_DIR, SAMPLE_RATE, NUM_SAMPLES, transformation, device

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

model_path = Path("./model/model_dense_DA.pth")
model = DenseNet().to(device)
model.load_state_dict(torch.load(model_path))

dataset = UrbanSoundDataset(DATASET_DIR, SAMPLE_RATE, NUM_SAMPLES, transformation=transformation, device=device, folder=2, data_augment=False)


# signal, target = dataset[0]
# signal = signal.unsqueeze(0)

preds = []
targets = []
for signal, target in dataset:
    predicted_index, target = predict(model, signal, target, class_mapping)
    preds.append(predicted_index)
    targets.append(target)


from torchmetrics import ConfusionMatrix
confmat = ConfusionMatrix(num_classes=len(class_mapping))
mat = confmat(torch.tensor(preds), torch.tensor(targets))  # lines = targets, columns = preds
mat = (mat.T / mat.sum(axis=1)).T
mat.diag().sum()/mat.sum()  
# CUSTUM MODEL best i had 64.39% -> on training set .., model2 -> 60.60%
# DENSENET MODEL without DA : 10epochs -> 99.7% on training set (1), 56.84 -> on test set (2) (no DA)
# DENSENET MODEL with DA : +10epochs -> ?% on training set (1), 59.19 -> on test set (2) 
# DENSENET MODEL with DA : +10epochs -> 100% on training set (1), 65.25 -> on test set (2) 
# .. DA is shit ?
# DENSENET MODEL with DA on fold2 : 10epochs -> 98.6% on training set (2), 65.25 -> on test set (1) 
import matplotlib.pyplot as plt
import matplotlib
fig, ax = plt.subplots()
ax.imshow(mat)
plt.xlabel("predictions")
plt.ylabel("targets")
plt.xticks(ticks=range(10), labels=class_mapping, rotation=20)
plt.yticks(ticks=range(10), labels=class_mapping)
plt.show(block=False)




def predict(model, signal, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(signal.unsqueeze(0))
        predicted_index = predictions[0].argmax()
        # predicted = class_mapping[predicted_index]
        # expected = class_mapping[target]
    return predicted_index, target





