import os
import PIL.Image as Image
from torch.utils.data import Dataset
import glob
import numpy as np

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
        target = int(img_path.split('/')[-2])
        return self.transform(img), self.transform(img), target
    
    @staticmethod
    def read_image(img_path):
        #return cv2.imread(img_path)
        return Image.open(img_path, mode='r').convert('RGB')

class TripletImageLoader(Dataset):
    def __init__(self, root, choic_count = 2, transform=None, target_transform=None):
        classes, class_to_idx, idx_to_class = self.find_classes(root)
        imgs, targets, label_to_indices = self.make_dataset(root, class_to_idx)
        self.targets = targets
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.choic_count = choic_count
        self.label_to_indices = label_to_indices #{label: np.where(np.array(targets) == int(self.class_to_idx[label]))[0] for label in self.classes}
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        imgs = []
        targets = []
        path, target = self.imgs[index]
        random_choice = np.random.choice(list(self.label_to_indices[str(target)]), size=self.choic_count)
        for choice_index in random_choice:
            path, target = self.imgs[choice_index]
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            imgs.append(img)
            targets.append(target)

        return imgs, targets

    def __len__(self):
        return len(self.imgs)
    
    def find_classes(self, dir):
        classes = os.listdir(dir)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        idx_to_class = {}
        for key, value in class_to_idx.items():
            idx_to_class[value] = key
        return classes, class_to_idx, idx_to_class

    def make_dataset(self, dir, class_to_idx):
        images = []
        targets = []
        label_to_indices = {}
        for target in os.listdir(dir):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            idx = class_to_idx[target]
            label_to_indices[str(idx)] = []
            for filename in os.listdir(d):
                if is_image_file(filename):
                    label_to_indices[str(idx)].append(len(images))
                    path = '{0}/{1}'.format(target, filename)
                    item = (path, idx)
                    images.append(item)
                    targets.append(idx)

        return images, targets, label_to_indices