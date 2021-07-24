import logging
import utils.gpu as gpu
from model.yolov3 import Yolov3
from model.loss.yolo_loss import YoloV3Loss
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import utils.datasets as data
import time
import random
import argparse
from eval.evaluator import *
from utils.tools import *
from tensorboardX import SummaryWriter
import config.yolov3_config_voc as cfg
from utils import cosine_lr_scheduler
from tqdm import tqdm


# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='2'


class Trainer(object):
    def __init__(self,  weight_path, resume, gpu_id):
        init_seeds(0)
        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.
        self.epochs = cfg.TRAIN["EPOCHS"] # 在cfg中这里的值是50作为传入
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        # Dataset是将图片转换为Tensor形式
        self.train_dataset = data.VocDataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        # Dataloader是将Dataset的数据加载到内存中
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.TRAIN["BATCH_SIZE"],
                                           num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                           shuffle=True)
        self.yolov3 = Yolov3().to(self.device)
        # self.yolov3.apply(tools.weights_init_normal)

        self.optimizer = optim.SGD(self.yolov3.parameters(), lr=cfg.TRAIN["LR_INIT"],
                                   momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])
        #self.optimizer = optim.Adam(self.yolov3.parameters(), lr = lr_init, weight_decay=0.9995)

        self.criterion = YoloV3Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])

        self.__load_model_weights(weight_path, resume) # 加载预训练权重

        # 使用余弦退火函数进行lr
        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.optimizer,
                                                          T_max=self.epochs*len(self.train_dataloader),
                                                          lr_init=cfg.TRAIN["LR_INIT"],
                                                          lr_min=cfg.TRAIN["LR_END"],
                                                          warmup=cfg.TRAIN["WARMUP_EPOCHS"]*len(self.train_dataloader))


    def __load_model_weights(self, weight_path, resume):
        if resume:
            pretrain_model = 'best_84'
            if pretrain_model == 'last':
                last_weight = os.path.join(os.path.split(weight_path)[0], "last.pt")
                print("===> resume path:",last_weight)
                chkpt = torch.load(last_weight, map_location=self.device)
                self.yolov3.load_state_dict(chkpt['model'])
                #self.yolov3.load_state_dict(chkpt,strict=True)

                self.start_epoch = chkpt['epoch'] + 1
                if chkpt['optimizer'] is not None:
                    self.optimizer.load_state_dict(chkpt['optimizer'])
                    self.best_mAP = chkpt['best_mAP']
                del chkpt
            elif pretrain_model == 'best_84':
                last_weight = os.path.join(os.path.split(weight_path)[0], "best_84.pt")
                print("===> resume path:", last_weight)
                chkpt = torch.load(last_weight, map_location=self.device)
                self.yolov3.load_state_dict(chkpt['model'])
                # self.yolov3.load_state_dict(chkpt,strict=True)

                self.start_epoch = chkpt['epoch'] + 1
                if chkpt['optimizer'] is not None:
                    self.optimizer.load_state_dict(chkpt['optimizer'])
                    self.best_mAP = chkpt['best_mAP']
                del chkpt


        else:
            self.yolov3.load_darknet_weights(weight_path)


    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(os.path.split(self.weight_path)[0], "best.pt")
        last_weight = os.path.join(os.path.split(self.weight_path)[0], "last.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.yolov3.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight)

        if epoch > 0 and epoch % 10 == 0:
            torch.save(chkpt, os.path.join(os.path.split(self.weight_path)[0], 'backup_epoch%g.pt'%epoch))
        del chkpt


    def train(self):
        print(self.yolov3)
        print("Train datasets number is : {}".format(len(self.train_dataset)))

        for epoch in range(self.start_epoch, self.epochs):
            self.yolov3.train()

            mloss = torch.zeros(4)
            # sbbox = small bounding box, mbbox = middle bounding box, lbbox = large bounding box
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)  in tqdm(enumerate(self.train_dataloader)): # 从train_dataloader进行加载

                self.scheduler.step(len(self.train_dataloader)*epoch + i)

                imgs = imgs.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)

                p, p_d = self.yolov3(imgs)

                # 经过前向传播得到三个loss giou是框的loss conf是框内是否有物体的loss cls是label的loss
                # p, p_d是head输出的结果
                loss, loss_giou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)

                self.optimizer.zero_grad()
                loss.backward() #将总loss进行backward,总loss是giou,conf,cls loss的总和
                self.optimizer.step()

                # Update running mean of tracked metrics
                loss_items = torch.tensor([loss_giou, loss_conf, loss_cls, loss])
                mloss = (mloss * i + loss_items) / (i + 1) # 累计loss求平均

                # Print batch results 每10次print一次
                if i%10==0:
                    s = ('Epoch:[ %d | %d ]    Batch:[ %d | %d ]    loss_giou: %.4f    loss_conf: %.4f    loss_cls: %.4f    loss: %.4f    '
                         'lr: %g') % (epoch, self.epochs - 1, i, len(self.train_dataloader) - 1, mloss[0],mloss[1], mloss[2], mloss[3],
                                      self.optimizer.param_groups[0]['lr'])
                    print(s)

                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i+1)%10 == 0:
                    self.train_dataset.img_size = random.choice(range(10,20)) * 32
                    print("multi_scale_img_size : {}".format(self.train_dataset.img_size))

            mAP = 0
            # 大于20才进行计算mAP因为网络刚开始效果不好
            if epoch >= 20:
                print('*'*20+"Validate"+'*'*20)
                with torch.no_grad():
                    APs = Evaluator(self.yolov3).APs_voc()
                    for i in APs:
                        print("{} --> mAP : {}".format(i, APs[i]))
                        mAP += APs[i]
                    mAP = mAP / self.train_dataset.num_classes
                    print('mAP:%g'%(mAP))

            self.__save_model_weights(epoch, mAP) #保证最佳mAP的时候数据进行保存
            print('best mAP : %g' % (self.best_mAP))

'''
if __name__ == '__main__' 就相当于是 Python 模拟的程序入口。Python 本身并没有规定这么写，这只是一种编码习惯。由于模块之间相互引用，
不同模块可能都有这样的定义，而入口程序只能有一个。到底哪个入口程序被选中，这取决于 __name__ 的值
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 添加可选参数
    parser.add_argument('--weight_path', type=str, default='weight/darknet53_448.weights', help='weight file path') # 权重参数的路径
    parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag') # 是否进行fine-tuned
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id') # 使用哪张GPU
    opt = parser.parse_args()

    Trainer(weight_path=opt.weight_path,
            resume=opt.resume,
            gpu_id=opt.gpu_id).train()