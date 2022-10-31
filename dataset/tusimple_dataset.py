import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class Tusimple_dataset(Dataset):
    # Cifar-10 Dataset
    def __init__(self, root, label_dir, img_dir, transforms):
        self.root = root
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.transforms = transforms
        anno_path = os.path.join(root, label_dir, 'train.txt')

        with open(anno_path, 'r') as f:
            self.img_dir = [os.path.join(root, img_dir, l[:l.find(' ')]+'.jpg') for l in f.readlines()]

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir[idx]).convert('RGB')
        img_name = '/'.join(self.img_dir[idx].split('/')[2:])
        return self.transforms(img), img_name
    
def collate_fn(batch):
    imgs, names = [list(e) for e in zip(*batch)]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in imgs])
        storage = imgs[0].storage()._new_shared(numel)
        out = imgs[0].new(storage) 
    return torch.stack(imgs, 0, out=out), names
    
    