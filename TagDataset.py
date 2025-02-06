from torch.utils.data import Dataset
import pandas as pd
from preprocessing import preprocess_image

class TagDataset(Dataset):
    def __init__(self, root_dir, images_dir):
        self.root_dir = root_dir
        self.df = pd.read_csv(self.root_dir)
        self.df['image_path'] = images_dir + self.df['img_name']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image_path'][idx]
        text = self.df['text'].astype(int).astype(str)[idx]
        img = preprocess_image(file_name)
        # Преобразование изображения в формат, подходящий для модели
        
        return img, text