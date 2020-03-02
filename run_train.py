import torch
import json
import time
import api
import random
import math
import numpy as np
import torchvision.transforms as transforms


# Read config.json and set current GPU
with open('config.json', 'r') as load_f:
    cfg = json.load(load_f)
torch.cuda.set_device(cfg['device'][0])


# TODO: Define augment
def aug_func(img):
    if random.random() < cfg['random_xflip']['probability']:
        img = transforms.RandomHorizontalFlip()(img)
    if random.random() < cfg['random_color']['probability']:
        img = transforms.ColorJitter(
            brightness=cfg['random_color']['brightness'], 
            contrast=cfg['random_color']['contrast'], 
            saturation=cfg['random_color']['saturation'], 
            hue=cfg['random_color']['hue'])(img)
    if random.random() < cfg['random_rotation']['probability']:
        img = transforms.RandomRotation(cfg['random_rotation']['degrees'])(img)
    if random.random() < cfg['random_crop']['probability']:
        img = transforms.RandomCrop(size=cfg['random_crop']['size'], 
            padding=cfg['random_crop']['padding'])(img)
    return img
if cfg['augmentation']:
    augmentation = aug_func
else:
    augmentation = None


# Get train/eval dataset and dataloader
loader = __import__('dataset.' + cfg['dataset'], fromlist=(cfg['dataset'],))
dataset_train = loader.DATASET(
    cfg['root_train'], set_name='train', 
    img_scale_min=cfg['img_scale_min'], 
    normalize=False, augmentation=augmentation)
dataset_eval = loader.DATASET(
    cfg['root_eval'], set_name='eval', 
    img_scale_min=cfg['img_scale_min'], 
    normalize=False, augmentation=None)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg['nbatch_train'], 
                    shuffle=True, num_workers=cfg['num_workers'], collate_fn=dataset_train.collate_fn)
loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=cfg['nbatch_eval'], 
                    shuffle=False, num_workers=0, collate_fn=dataset_eval.collate_fn)


# Prepare the network and read log
model = __import__('models.' + cfg['model'], fromlist=(cfg['model'],))
net = model.Model(classes=len(dataset_train.dic))
log = []
device_out = 'cuda:%d' % (cfg['device'][0])
if cfg['load']:
    net.load_state_dict(torch.load('net.pkl', map_location=device_out))
    log = list(np.load('log.npy'))
net = torch.nn.DataParallel(net, device_ids=cfg['device'])
net = net.cuda(cfg['device'][0])
net.train()


# Prepare optimizer
lr_base = cfg['lr_base']
lr_gamma = cfg['lr_gamma']
lr_schedule = cfg['lr_schedule']
if cfg['optimizer'] == 'adam':
    opt = torch.optim.Adam(net.parameters(), lr=lr_base, 
        weight_decay=cfg['weight_decay'])
else:
    opt = torch.optim.SGD(net.parameters(), lr=lr_base, 
        momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])


# Prepare lr_func
WARM_UP_ITERS = cfg['warmup_iter']
WARM_UP_FACTOR = cfg['warmup_factor'] # 1.0 / 3.0
def lr_func(step):
    lr = lr_base
    if step < WARM_UP_ITERS:
        alpha = float(step) / WARM_UP_ITERS
        warmup_factor = WARM_UP_FACTOR * (1.0 - alpha) + alpha
        lr = lr*warmup_factor 
    else:
        for i in range(len(lr_schedule)):
            if step < lr_schedule[i]:
                break
            lr *= lr_gamma
    return float(lr)


# Prepare API structure
trainer = api.Trainer(net, dataset_train, loader_train, cfg['device'], opt, cfg['grad_clip'], lr_func)
evaluator = api.Evaluator(net, dataset_eval, loader_eval, cfg['device'])
if cfg['load']:
    trainer.epoch = log[-1][1]
    trainer.step = trainer.epoch * int(math.ceil(len(dataset_train) / float(cfg['nbatch_train'])))


# Run epoch
while True:
    if trainer.epoch >= cfg['epoches']:
        break
    trainer.step_epoch()
    acc = evaluator.step_epoch()
    log.append([acc, trainer.epoch])
    if cfg['save']:
        torch.save(net.module.state_dict(),'net.pkl')
        np.save('log.npy', log)
print('Schedule finished!')
