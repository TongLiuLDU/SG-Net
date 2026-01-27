import os
import numpy as np
import argparse
from datetime import datetime
import logging
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model.SG_Net import SG_Net
from utils.dataloader import get_loader, test_dataset, PolypDataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter

from ptflops import get_model_complexity_info
import warnings
warnings.filterwarnings("ignore")

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

l = [0, 1, 2, 3]
ss = [x for x in powerset(l)]
#ss = [[0],[1],[2],[3]]
print(ss)

import random
def set_seed(seed=42):  
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, path, dataset,is_use_deepvision=False):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, args.img_size)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res = model(image)  # forward

        if is_use_deepvision:
            res = res[0]+res[1]+res[2]+res[3]
        
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)  # additive aggregation and upsampling
        res = res.sigmoid().data.cpu().numpy().squeeze()  # apply sigmoid activation for binary segmentation
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # eval Dice
        input = res >= 0.5
        target = np.array(gt >= 0.5)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1, num1


def train(train_loader, model, optimizer, epoch, test_path, model_name='Transnext',is_use_deepvision=False):
    model.train()
    global best
    global total_train_time
    global max_memory_usage  # 添加一个全局变量来存储最大显存占用
    max_memory_usage = 0  # 初始化最大显存占用
    time_before_epoch_start = time.time()
    size_rates = [1]
    loss_record = AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()

            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            # ---- rescale ----
            trainsize = int(round(args.img_size * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            # ---- forward ----
            outputs = model(images)
            
            if is_use_deepvision:
                loss = 0.0
                for s in ss:
                    iout = 0.0
                    if(s==[]):
                        continue
                    for idx in range(len(s)):
                        iout += outputs[s[idx]]
                    loss += structure_loss(iout, gts)
            else:
                loss = structure_loss(outputs, gts)

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()

            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, args.batchsize)

            # ---- monitoring maximum memory usage ----
            current_memory = torch.cuda.max_memory_allocated()
            max_memory_usage = max(max_memory_usage, current_memory)

        # ---- train visualization ----
        if i % 50 == 0 or i == total_step:
            inter_info = '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:0.4f}, max memory usage: {:.2f} MB'.format(
                datetime.now(), epoch, args.epoch, i, total_step, loss_record.show(), max_memory_usage / (1024 ** 2)
            )
            print(inter_info)
            logging.info(inter_info)

    time_after_epoch_end = time.time()
    total_train_time += (time_after_epoch_end - time_before_epoch_start)
    print('total train time till current epoch: ' + str(total_train_time))
    logging.info('total train time till current epoch: ' + str(total_train_time))

    # save model
    save_path = (args.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + '' + model_name + '-last.pth')

    # choose the best model
    global dict_plot

    if (epoch + 1) % 1 == 0:
        total_dice = 0
        total_images = 0
        for dataset in ['test']:
            dataset_dice, n_images = test(model, test_path, dataset, is_use_deepvision=is_use_deepvision)  
            total_dice += (n_images * dataset_dice)
            total_images += n_images
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dict_plot[dataset].append(dataset_dice)

        dataset_test_dice = total_dice / total_images
        meandice = dataset_test_dice
        valid_meandice = total_dice / total_images
        dict_plot['valid'].append(valid_meandice)
        dict_plot['test'].append(dataset_test_dice)
        print('Test dice score: {}'.format(dataset_test_dice))
        logging.info('Test dice score: {}'.format(dataset_test_dice))

        if meandice > best:
            print('##################### Dice score improved from {} to {}'.format(best, meandice))
            logging.info('##################### Dice score improved from {} to {}'.format(best, meandice))
            best = meandice
            torch.save(model.state_dict(), save_path + '' + model_name + '-best.pth')
            
            
            
            
    print("最好的dice：",best)

    # 在每个epoch结束时，重置最大显存占用
    torch.cuda.reset_peak_memory_stats()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    dict_plot = {'valid': [], 'test': []}
    name = ['valid', 'test']
    ##################model_name#############################


    set_seed(43)
    
    # model_name = "SG-Net"       # add all

    # current_time = time.strftime("%H%M%S")
    # print("The current time is", current_time)
    # model_name = model_name +'_t'+current_time

    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='SG-Net', help='name for the model for saving')
    parser.add_argument('--encoder', type=str, default='Transnext', help='Name of encoder: PVT or MERIT')
    parser.add_argument("--num_classes", type=int, default=1, help="output channel of network")
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--augmentation', default=True, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
    parser.add_argument('--img_size', type=int, default=256, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str, default='./datasets/ISIC2016/train',
                        help='path to train dataset')
    parser.add_argument('--test_path', type=str, default='./datasets/ISIC2016',
                        help='path to testing Kvasir dataset')
    parser.add_argument('--train_save', type=str, default='./model_out/ISIC2016/')

    args = parser.parse_args()

    args.train_save = args.train_save + args.model_name + '/'

    if not os.path.exists(args.train_save):
        os.makedirs(args.train_save)

    logging.basicConfig(filename=args.train_save + 'train.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    # model = model(num_classes=args.num_classes).cuda(0)
    


    model = SG_Net(deep_supervision=True,drop_path_rate=0.4).cuda(0)

    # logging.info(model)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    macs, params = get_model_complexity_info(model, (3, args.img_size, args.img_size), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    best = 0

    params = model.parameters()

    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, args.lr, weight_decay=3e-5)
    else:
        optimizer = torch.optim.SGD(params, args.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(args.train_path)
    gt_root = '{}/masks/'.format(args.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=args.batchsize, trainsize=args.img_size, shuffle=True,num_workers=8,
                              augmentation=args.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)
    total_train_time = 0

    for epoch in range(1, args.epoch):
        adjust_lr(optimizer, args.lr, epoch, args.decay_rate, args.decay_epoch)
        # cos_lr(optimizer, args.lr, epoch, args.epoch)
        train(train_loader, model, optimizer, epoch, args.test_path, model_name=args.model_name,is_use_deepvision=True)
    print('avg train time: ' + str(total_train_time / (args.epoch - 1)))
    logging.info('avg train time: ' + str(total_train_time / (args.epoch - 1)))
    
    dataset_dice, n_images = test(model, args.test_path, 'test',is_use_deepvision=True)
    print('Test dice score: {}'.format(dataset_dice))
    logging.info('Test dice score: {}'.format(dataset_dice))
