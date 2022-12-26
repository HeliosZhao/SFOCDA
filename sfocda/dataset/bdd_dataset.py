import os
import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image

class BDDDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, name)
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC) ## PIL image resize (width height)

        image = np.asarray(image, np.float32)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        
        return image.copy(), np.array(size), name, name

class BDDDataSet_Aug(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val', augmentation=None):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.augmentation = augmentation
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, name)
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC) ## PIL image resize (width height)

        image_aug = self.augmentation(image)
            
        image = np.asarray(image, np.float32)
        image_aug = np.asarray(image_aug, np.float32)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        image_aug = image_aug[:, :, ::-1]  # change to BGR
        image_aug -= self.mean
        image_aug = image_aug.transpose((2, 0, 1))
        
        return image.copy(), image_aug.copy(), np.array(size), name
    

class BDDTestDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val', test_classes=19):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, name)
            label_file = img_file[:-4] + '_train_id.png'
            self.files.append({
                "img": img_file,
                "name": name,
                "label": label_file
            })

        # if self.n_classes == 19:
        #     self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]
        #     self.class_names = ["unlabelled","road","sidewalk","building","wall",
        #         "fence","pole","traffic_light","traffic_sign","vegetation",
        #         "terrain","sky","person","rider","car",
        #         "truck","bus","train","motorcycle","bicycle",
        #     ]
        #     self.to19 = dict(zip(range(19), range(19)))
        # elif self.n_classes == 16:
        #     self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 23, 24, 25, 26, 28, 32, 33,]
        #     self.class_names = ["unlabelled","road","sidewalk","building","wall",
        #         "fence","pole","traffic_light","traffic_sign","vegetation",
        #         "sky","person","rider","car","bus",
        #         "motorcycle","bicycle",
        #     ] # terrain truck train
        #     self.to19 = dict(zip(range(16), [0,1,2,3,4,5,6,7,8,10,11,12,13,15,17,18]))
        # elif self.n_classes == 13:
        #     self.valid_classes = [7, 8, 11, 19, 20, 21, 23, 24, 25, 26, 28, 32, 33,]
        #     self.class_names = ["unlabelled","road","sidewalk","building","traffic_light",
        #         "traffic_sign","vegetation","sky","person","rider",
        #         "car","bus","motorcycle","bicycle",
        #     ]
        #     self.to19 = dict(zip(range(13), [0,1,2,6,7,8,10,11,12,13,15,17,18]))

        # self.ignore_index = 250
        # self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))   #zip: return tuples
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]
        label = Image.open(datafiles["label"])
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize((1280,720), Image.NEAREST)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.uint8)
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        
        return image.copy(), label.copy(), np.array(size), name
