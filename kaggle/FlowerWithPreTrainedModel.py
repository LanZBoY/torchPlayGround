import torch
from torchvision.models import resnet50, ResNet50_Weights
from kaggle.dataset.dataUtils import FlowerDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from tqdm import tqdm

def train_step(model, loss_fn, optimizer, dataloader):
    model.train()
    for epoch in range(EPOCH):
        print(f'Current Epoch is {epoch + 1}')
        for y, x in tqdm(dataloader):
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        print(f'Current loss = {loss.item()}')



def test_step(model, dataloader):
    correct = 0.
    model.eval()
    for y, x in dataloader:
        pred = model(x)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    print(f'Total data {len(dataloader.dataset)}')
    print(f'Correct = {correct}')
    acc = correct/len(dataloader.dataset)
    print(f'The acc is {acc*100.}%')


DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
EPOCH = 20
BATCH_SIZE = 32
ROOT_PATH = './dataset/flowers_/'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(DEVICE)

# print(model)

train_dataset = FlowerDataset(
    root_path=ROOT_PATH, 
    split='train', 
    transform=Compose([
        Resize(size=(150, 150)),
        ToTensor(),
        Normalize(mean, std)
        ]),
        device=DEVICE
    )
test_dataset = FlowerDataset(
        root_path=ROOT_PATH, 
        split='test',
        transform=Compose([
        Resize(size=(150, 150)),
        ToTensor(),
        Normalize(mean, std)
        ]),
        device=DEVICE)

trainDataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
testDataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

## Test shape is fit or not
for y, x in trainDataloader:
    pred = model(x)
    print(pred.shape)
    break

outputLayer = torch.nn.Linear(1000, 5)

model_s = torch.nn.Sequential(model, outputLayer).to(DEVICE)

for y, x in trainDataloader:
    pred = model_s(x)
    print(pred.shape)
    break

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_s.parameters(), lr=1e-3)

train_step(model_s, loss_fn, optimizer, trainDataloader)
test_step(model_s, trainDataloader)
test_step(model_s, testDataloader)