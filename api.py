import torch
import time
import json
import torchvision.transforms as transforms



class Trainer(object):
    def __init__(self, net, dataset, loader, device, 
                    opt, grad_clip=3, lr_func=None):
        '''
        external initialization structure: 
            net(DataParallel), dataset(Dataset), loader(DataLoader), device(List), opt(Optimizer)
        grad_clip: limit the gradient size of each iteration
        lr_func: lr_func(step) -> float
        self.step, self.epoch for outside use
        '''
        self.net = net
        self.dataset = dataset
        self.loader = loader
        self.device = device
        self.opt = opt
        self.grad_clip = grad_clip
        self.lr_func = lr_func
        self.step = 0
        self.epoch = 0

    def step_epoch(self):
        '''
        train one epoch
        '''
        lr = -1
        for i, (img, labels) in enumerate(self.loader):
            if self.lr_func is not None:
                lr = self.lr_func(self.step)
                for param_group in self.opt.param_groups:
                    param_group['lr'] = lr
            if i == 0:
                g_batch_size = int(img.shape[0])
            time_start = time.time()
            self.opt.zero_grad()
            loss = self.net(img, labels)
            loss = loss.mean()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
            self.opt.step()
            maxmem = int(torch.cuda.max_memory_allocated(device=self.device[0]) / 1024 / 1024)
            time_end = time.time()
            totaltime = int((time_end - time_start) * 1000)
            print('total_step:%d: epoch:%d, step:%d/%d, loss:%f, maxMem:%dMB, time:%dms, lr:%f' % \
                (self.step, self.epoch, i*g_batch_size, len(self.dataset), loss, maxmem, totaltime, lr))
            self.step += 1
        self.epoch += 1



class Evaluator(object):
    def __init__(self, net, dataset, loader, device):
        '''
        external initialization structure: 
            net(DataParallel), dataset(Dataset), loader(DataLoader), device(List)
        '''
        self.net = net
        self.dataset = dataset
        self.loader = loader
        self.device = device

    def step_epoch(self):
        '''
        return acc
        note: this function will set self.net.train() at last
        '''
        with torch.no_grad():
            self.net.eval()
            num = 0.0
            acc_total = torch.zeros(1)
            for i, (img, labels) in enumerate(self.loader):
                if i == 0:
                    g_batch_size = int(img.shape[0])
                pred = self.net(img) # (batch_size)
                acc = (pred.cpu() == labels).float().mean().view(1)
                acc_total += acc
                num += 1.0
                print('  Eval: {}/{}, acc:{}'.format(i*g_batch_size, len(self.dataset), float(acc)), end='\r')
            acc_total = float(acc_total / num)
            print('acc_total:', acc_total)
            self.net.train()
            return acc_total
