from pathlib import Path
from model import CNN
from dataset import UrbanSoundDataset
from train import transformation, device
from train import DATASET_DIR, SAMPLE_RATE, NUM_SAMPLES
import torch

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

model_path = Path("./model/model.pth")

model = CNN().to(device)
model.load_state_dict(torch.load(model_path))

dataset = UrbanSoundDataset(DATASET_DIR, SAMPLE_RATE, NUM_SAMPLES, transformation=transformation, device=device)


signal, target = dataset[0]
signal = signal.unsqueeze(0)

preds = []
targets = []
for signal, target in dataset:
    predicted_index, target = predict(model, signal, target, class_mapping)
    preds.append(predicted_index)
    targets.append(target)

from torchmetrics import ConfusionMatrix
confmat = ConfusionMatrix(num_classes=len(class_mapping))
mat = confmat(torch.tensor(preds), torch.tensor(targets))
mat = mat / mat.sum(axis=1)
mat.diag().sum()/mat.sum()
import matplotlib.pyplot as plt
plt.imshow(mat)
plt.show()




def predict(model, signal, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(signal.unsqueeze(0))
        predicted_index = predictions[0].argmax()
        # predicted = class_mapping[predicted_index]
        # expected = class_mapping[target]
    return predicted_index, target





