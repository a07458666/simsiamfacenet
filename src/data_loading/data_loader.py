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

def find_classes(dir):
    classes = []
    for c in os.listdir(dir):
        d = os.path.join(dir, c)
        if not os.path.isdir(d):
            continue
        if(c == ".ipynb_checkpoints"):
            continue
        classes.append(c)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = {}
    for key, value in class_to_idx.items():
        idx_to_class[value] = key
    return classes, class_to_idx, idx_to_class

def make_dataset(dir, class_to_idx):
    images = []
    targets = []
    label_to_indices = {}
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        if(target == ".ipynb_checkpoints"):
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

class FaceImages(Dataset):
    
    def __init__(self, root, transform, specific = '**'):
        self.root = root
        self.img_path_list = glob.glob(os.path.join(root, specific + '/*.jpg')) + glob.glob(os.path.join(root, specific + '/*.png'))
        self.transform = transform
        classes, class_to_idx, idx_to_class = find_classes(root)
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = FaceImages.read_image(img_path)
        target = self.class_to_idx[img_path.split('/')[-2]]
        return self.transform(img), self.transform(img), target
    
    @staticmethod
    def read_image(img_path):
        #return cv2.imread(img_path)
        return Image.open(img_path, mode='r').convert('RGB')

class TripletImageLoader(Dataset):
    def __init__(self, root, choic_count = 1, transform=None, target_transform=None):
        classes, class_to_idx, idx_to_class = find_classes(root)
        imgs, targets, label_to_indices = make_dataset(root, class_to_idx)
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
        random_choice = np.append(random_choice, index)
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
    
class TripletSSLImageLoader(Dataset):
    def __init__(self, root, choic_count = 1, transform=None, target_transform=None):
        classes, class_to_idx, idx_to_class = find_classes(root)
        imgs, targets, label_to_indices = make_dataset(root, class_to_idx)
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
        imgs_1 = []
        imgs_2 = []
        targets = []
        path, target = self.imgs[index]
#         random_choice = np.random.choice(list(self.label_to_indices[str(target)]), size=self.choic_count)
#         random_choice.appned(index)
#         for choice_index in random_choice:
#             path, target = self.imgs[choice_index]
#             img = Image.open(os.path.join(self.root, path)).convert('RGB')
#             if self.transform is not None:
#                 img_1 = self.transform(img)
#                 img_2 = self.transform(img)
#             if self.target_transform is not None:
#                 target = self.target_transform(target)
#             imgs_1.append(img_1)
#             imgs_2.append(img_2)
#             targets.append(target)
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img_1 = self.transform(img)
            img_2 = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        imgs_1.append(img_1)
        imgs_2.append(img_2)
        targets.append(target)
        return imgs_1, imgs_2, targets

    def __len__(self):
        return len(self.imgs)