import os 
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision.transforms import ToTensor,Resize, Normalize,Compose, ToPILImage
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCH = 20
BATCH_SIZE = 32
class FlowerDataset(Dataset):

    def __init__(self, root_path, split = 'train', transform = None, device = 'cpu') -> None:
        super(FlowerDataset, self).__init__()
        self.transform = transform
        self.root_path = os.path.join(root_path, split)
        self.class_list = os.listdir(self.root_path)
        self.datas = []
        self.device = device
        for label, class_name in enumerate(self.class_list):
            classPath = os.path.join(self.root_path, class_name)
            img_list = os.listdir(classPath)
            for imgName in img_list:
                imgPath = os.path.join(classPath, imgName)
                self.datas.append((label, imgPath))
    
    def __len__(self):
        print("test")
        return len(self.datas)
    
    def __getitem__(self, index):
        label, imgPath = self.datas[index]
        label = torch.tensor(data=label)
        img = Image.open(imgPath)
        if self.transform is not None:
            img = self.transform(img)
        return label.to(self.device), img.to(self.device)
root_path = './kaggle/dataset/flowers_'
show_dataset = FlowerDataset(
        root_path=root_path, 
    split='train', 
    transform=Compose(
        [
        Resize(size=(150, 150)),
        ToTensor(),
        ]
        ),
        device=DEVICE
    )

dataloader = DataLoader(dataset=show_dataset, batch_size = BATCH_SIZE, shuffle=True)
