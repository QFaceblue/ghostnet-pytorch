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
        self.size = 32
        self.img_scale_min = img_scale_min
        self.normalize = normalize
        self.augmentation = augmentation
        self.normalizer = transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        self.dic = ['0','1','2','3','4','5','6','7','8' ,'9']

        if self.set_name == 'train':
            f1 = open(os.path.join(self.root, 'train-images.idx3-ubyte'), 'rb')
            l1 = open(os.path.join(self.root, 'train-labels.idx1-ubyte'), 'rb')
            raw_data = f1.read()
            raw_label = l1.read()
            self.labels = []
            self.data = []
            for i in range(60000):
                data_idx = 28*28*i+16
                label_idx = i+8
                labels_np=np.array(list(raw_label[label_idx:label_idx+1]),dtype='int64')
                data_np=np.array(list(raw_data[data_idx:data_idx+28*28]),dtype='uint8')
                self.labels.append(labels_np)
                self.data.append(data_np)
            self.labels = np.concatenate(self.labels)
            self.data = np.concatenate(self.data)
            self.data = self.data.reshape((60000, 28, 28))
            f1.close()
            l1.close()
        else:
            f1 = open(os.path.join(root, 't10k-images.idx3-ubyte'), 'rb')
            l1 = open(os.path.join(root, 't10k-labels.idx1-ubyte'), 'rb')
            raw_data = f1.read()
            raw_label = l1.read()
            self.labels = []
            self.data = []
            for i in range(10000):
                data_idx = 28*28*i+16
                label_idx = 8+i
                labels_np=np.array(list(raw_label[label_idx:label_idx+1]),dtype='int64')
                data_np=np.array(list(raw_data[data_idx:data_idx+28*28]),dtype='uint8')
                self.labels.append(labels_np)
                self.data.append(data_np)
            self.labels = np.concatenate(self.labels)
            self.data = np.concatenate(self.data)
            self.data = self.data.reshape((10000, 28, 28))
            f1.close()
            l1.close()



    def __len__(self):
        # int
        if self.set_name == 'train':
            return 60000
        else:
            return 10000


    def __getitem__(self, index):
        # img: FloatTensor(c, h, w)
        # target: LongTensor(1)
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        target = torch.LongTensor([int(target)])

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
        img = transforms.RandomCrop(28, padding=5)(img)
        return img

    dataset = DATASET('D:\\dataset\\mnist', set_name=set_name,
                img_scale_min=img_scale_min, normalize=False, augmentation=augment)
    print('len:', len(dataset))
    dataset.show()
