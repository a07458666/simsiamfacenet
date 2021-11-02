import os
import PIL.Image as Image
from torch.utils.data import Dataset
import glob

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, data_list):
    images = []
    for img_name, idx, labels in data_list:
        item = (img_name, int(idx))
        images.append(item)
    return images


class FaceImages(Dataset):
    
    def __init__(self, img_dir, transform, specific = '**'):
        self.img_dir = img_dir
        self.img_path_list = glob.glob(os.path.join(img_dir, specific + '/*.jpg'))
        self.transform = transform
        
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = FaceImages.read_image(img_path)
        target = int(img_path.split('/')[5])
        return self.transform(img), self.transform(img), target
    
    @staticmethod
    def read_image(img_path):
        #return cv2.imread(img_path)
        return Image.open(img_path, mode='r').convert('RGB')