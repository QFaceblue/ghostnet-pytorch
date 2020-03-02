import torch
import random
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np



class DATASET(data.Dataset):
    def __init__(self, root, set_name='train', 
            img_scale_min=0.8, normalize=True, augmentation=None):
        self.root = root
        self.set_name = set_name
        self.size = 224 # TODO:
        self.img_scale_min = img_scale_min
        self.normalize = normalize
        self.augmentation = augmentation
        self.normalizer = transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))

        # TODO:
        self.dic = []
        self.images_path = []
        self.labels = []
        for dir_name in os.listdir(self.root):
            path = os.path.join(self.root, dir_name)
            if not os.path.isfile(path): # is dir
                self.dic.append(dir_name)
                for image_name in os.listdir(path):
                    if image_name.endswith('.jpg') or \
                        image_name.endswith('.png') or \
                            image_name.endswith('.bmp'):
                        self.images_path.append(os.path.join(path, image_name))
                        self.labels.append(dir_name)
        self.dic = sorted(list(set(self.dic)))
        for i in range(len(self.labels)):
            self.labels[i] = int(self.dic.index(self.labels[i]))
        ###########


    def __len__(self):
        # int

        # TODO:
        return len(self.labels)
        ###########


    def __getitem__(self, index):
        # img: FloatTensor(c, h, w)
        # target: LongTensor(1)

        # TODO:
        image_path, label = self.images_path[index], self.labels[index]
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        target = torch.LongTensor([label])
        ###########

        if self.set_name == 'train':
            if self.augmentation is not None:
                img = self.augmentation(img)
            img = random_resize_fix(img, self.size, self.img_scale_min)
        else:
            img = center_fix(img, self.size)
        img = transforms.ToTensor()(img)
        if self.normalize:
            img = self.normalizer(img)
        return img, target 


    def collate_fn(self, data):
        # imgs: FloatTensor(b, c, h, w)
        # labels: LongTensor(b)
        imgs, labels = zip(*data)
        imgs = torch.stack(imgs)
        labels = torch.cat(labels, dim=0)
        return imgs, labels

        
    def show(self):
        for i in range(15):
            plt.subplot(3, 5, i+1)
            idx = random.randint(0, self.__len__()-1)
            img, label = self.__getitem__(idx)
            plt.axis('off')
            plt.imshow(transforms.ToPILImage()(img))
            plt.title(self.dic[label])
        plt.show()



def center_fix(img, size):
    w, h = img.size
    size_min = min(w,h)
    size_max = max(w,h)
    sw = sh = float(size) / size_max
    ow = int(w * sw + 0.5)
    oh = int(h * sh + 0.5)
    ofst_w = round((size - ow) / 2.0)
    ofst_h = round((size - oh) / 2.0)
    img = img.resize((ow,oh), Image.BILINEAR)
    img = img.crop((-ofst_w, -ofst_h, size-ofst_w, size-ofst_h))
    return img



def random_resize_fix(img, size, img_scale_min):
    w, h = img.size
    size_min = min(w,h)
    size_max = max(w,h)
    scale_rate = float(size) / size_max
    scale_rate *= random.uniform(img_scale_min, 1.0)
    ow, oh = int(w * scale_rate + 0.5), int(h * scale_rate + 0.5)
    img = img.resize((ow,oh), Image.BILINEAR)
    max_ofst_h = size - oh
    max_ofst_w = size - ow
    ofst_h = random.randint(0, max_ofst_h)
    ofst_w = random.randint(0, max_ofst_w)
    img = img.crop((-ofst_w, -ofst_h, size-ofst_w, size-ofst_h))
    return img



if __name__ == '__main__':

    set_name = 'train' # train, eval
    img_scale_min = 0.5

    def augment(img):
        if random.random()<0.5:
            img = transforms.RandomRotation(20)(img)
        # img = transforms.RandomCrop(28, padding=5)(img)
        return img

    dataset = DATASET('C:\\Users\\ASUS\\Desktop\\testimg', set_name=set_name,
                img_scale_min=img_scale_min, normalize=False, augmentation=augment)
    print('len:', len(dataset))
    print('dic:', dataset.dic)
    dataset.show()
