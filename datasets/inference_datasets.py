import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class Infer_Orid5k(Dataset):
    def __init__(self, images_dir, csv_path):
        self.images_dir = images_dir
        self.datas = pd.read_excel(csv_path)   

        self.transform = transforms.Compose([
            transforms.Resize(256),                         # 先把短边缩放到 256
            transforms.CenterCrop(224),                     # 再裁出中心 224×224
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        
    def __getitem__(self, index):
        row = self.datas.iloc[[index]]
        left_filename = row['Left-Fundus'].values[0]
        right_filename = row['Right-Fundus'].values[0]
        left_imgpath = os.path.join(self.images_dir, left_filename)
        right_imgpath = os.path.join(self.images_dir, right_filename)

        left_img = Image.open(left_imgpath).convert('RGB')
        right_img = Image.open(right_imgpath).convert('RGB')
    
        left_img = self.transform(left_img)
        right_img = self.transform(right_img)

        data = {
            'left_image':left_img, 
            'right_image':right_img, 
            'idx': left_filename.replace("_left.jpg", ""),
            'left_filename': left_filename,     # 左眼文件名
            'right_filename': right_filename,   # 右眼文件名    
        }
        return data

    def __len__(self):
        return len(self.datas)