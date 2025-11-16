import os
from PIL import Image
import numpy as np

from torch.utils import data

from datasets.data_utils import DataAugmentation



IMG_NORMAL_FOLDER_NAME = "/home/datasets/SMDD/normal"
IMG_INSPECTION_FOLDER_NAME = '/home/datasets/SMDD/inspection'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"

#IGNORE = 255

label_suffix='.jpg' # replace according to the dataset image type

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


def load_image_label_list_from_npy(npy_path, img_name_list):
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_inspection_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_INSPECTION_FOLDER_NAME, img_name)


def get_img_normal_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_NORMAL_FOLDER_NAME, img_name)


def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace('.jpg', label_suffix))


class ImageDataset(data.Dataset):
    """
    Base Dataset Class
    root_dir: folder path of the dataset
    """
    def __init__(self, root_dir, split='train', img_size=256, is_train=True, to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  #train | train_aug | val
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        self.img_name_list = load_img_name_list(self.list_path)

        self.normal_size = len(self.img_name_list)  # get the size of dataset
        self.to_tensor = to_tensor
        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
                random_color_tf=True
            )
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                is_evaluation=True
            )

    def __getitem__(self, index):
        name = self.img_name_list[index]
        normal_path = get_img_normal_path(self.root_dir, self.img_name_list[index % self.normal_size])
        inspection_path = get_img_inspection_path(self.root_dir, self.img_name_list[index % self.normal_size])

        img_normal = np.asarray(Image.open(normal_path).convert('RGB'))
        img_inspection = np.asarray(Image.open(inspection_path).convert('RGB'))

        [img_normal, img_inspection], _ = self.augm.transform([img_normal, img_inspection],[], to_tensor=self.to_tensor)

        return {'name': name, 'N': img_normal, 'I': img_inspection}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.normal_size


class SMDDDataset(ImageDataset):
    """
    SMDD dataset class
    root_dir: folder path of the dataset
    img_size: spatial size of the images (input to the model)
    """

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(Dataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform

    def __getitem__(self, index):
        name = self.img_name_list[index]
        normal_path = get_img_normal_path(self.root_dir, self.img_name_list[index % self.normal_size])
        inspection_path = get_img_inspection_path(self.root_dir, self.img_name_list[index % self.normal_size])
        img_normal = np.asarray(Image.open(normal_path).convert('RGB'))
        img_inspection = np.asarray(Image.open(inspection_path).convert('RGB'))
        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.normal_size])

        label = np.array(Image.open(L_path), dtype=np.uint8)
        # if you are getting error because of dim mismatch ad [:,:,0] at the end
        # Note: label should be grayscale (single channel image)
        if self.label_transform == 'norm':
            label = label // 255
        
        [img_normal, img_inspection], [label] = self.augm.transform([img_normal, img_inspection], [label], to_tensor=self.to_tensor)
        
        return {'name': name, 'N': img_normal, 'I': img_inspection, 'L': label}

