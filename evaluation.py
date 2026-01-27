import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import imageio
import cv2
from skimage import img_as_ubyte
import pandas as pd
import sys
import time
from thop import profile

from model.SG_Net import SG_Net
from utils.dataloader import test_dataset


def calculate_metrics(Y_test, yp):
    jacard = 0
    dice = 0
    smooth = 1e-15
    for i in range(len(Y_test)):
        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()
        
        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection
        
        if (np.sum(y2)==0) and (np.sum(yp_2)==0):
            jacard += 1.0
            dice += 1.0
        elif(np.sum(intersection)==0):
            jacard += 0.0
            dice += 0.0
        else:
            jacard += ((np.sum(intersection) + smooth)/(np.sum(union) + smooth))
            dice += (2. * np.sum(intersection) + smooth) / (np.sum(yp_2) + np.sum(y2) + smooth)

    jacard /= len(Y_test)
    dice /= len(Y_test)
    
    return jacard, dice

def confusion_matrix_scorer(Y, Y_pred):
    Y = Y.astype(np.int8)
    Y_pred = Y_pred.astype(np.int8)
    P = len(np.where(Y  == 1)[0])
    N = len(np.where(Y == 0)[0])
    
    FP = len(np.where(Y - Y_pred  == -1)[0])
    FN = len(np.where(Y - Y_pred == 1)[0])
    TP = len(np.where(Y + Y_pred ==2)[0])
    TN = len(np.where(Y + Y_pred == 0)[0])
    
    return P, N, TN, FP, FN, TP

def get_metrics(Y, pred):
    Y = np.reshape(Y, pred.shape)
    smooth = 1e-15
    P, N, TN, FP, FN, TP = 0, 0, 0, 0, 0, 0
    sensitivity, specificity, accuracy, precision, F1, MCC = 0, 0, 0, 0, 0, 0
    
    for i in range(len(Y)):
        _p, _n, _tn, _fp, _fn, _tp = confusion_matrix_scorer(Y[i], pred[i])
        P += _p; N += _n; TN += _tn; FP += _fp; FN += _fn; TP += _tp
        
        if (_tp + _fn) > 0:
            sensitivity += (_tp / (_tp + _fn))
        if (_tn + _fp) > 0:
            specificity += (_tn / (_tn + _fp))
        if (_tp + _tn + _fp + _fn) > 0:
            accuracy += ((_tp + _tn)/(_tp + _fn + _fp + _tn))
        if (_tp + _fp) > 0:
            precision += (_tp / (_tp + _fp))
        if (2 * _tp + _fp + _fn) > 0:
            F1 += (2 * _tp) / (2 * _tp + _fp + _fn)
        
        mcc_numerator = (_tp * _tn) - (_fp * _fn)
        mcc_denominator = np.sqrt(float((_tp + _fp) * (_tp + _fn) * (_tn + _fp) * (_tn + _fn)))
        if mcc_denominator > 0:
            MCC += mcc_numerator / mcc_denominator
            
    num_samples = len(Y)
    return (sensitivity/num_samples, specificity/num_samples, accuracy/num_samples, 
            precision/num_samples, F1/num_samples, MCC/num_samples)

def get_metrics_and_print(Y, yp, method="CustomEncoder_MSDUNet", testset='test', write=False):
    yp = (yp >= 0.5).astype(np.uint8)
    Y = (Y >= 0.5).astype(np.uint8)
    
    sensitivity, specificity, accuracy, precision, f1, mcc_cal = get_metrics(Y, yp)
    jacard, dice = calculate_metrics(Y, yp)

    print(f"Metrics for {method} on {testset}")
    print(f"Dice: {dice:.4f}")
    print(f"Jacard (mIoU): {jacard:.4f}")
    print(f"Sensitivity/Recall: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"MCC: {mcc_cal:.4f}")
        
    if write:
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_file = os.path.join(results_dir, f'results_{testset}.csv')
        file_exists = os.path.isfile(results_file)
        
        results = pd.DataFrame([[method, jacard, dice, sensitivity, specificity, accuracy, precision, f1, mcc_cal]], 
                               columns=['Method', 'mIoU/Jacard', 'DICE', 'Sensitivity/Recall', 'Specificity', 
                                        'Accuracy', 'Precision', 'F-score', 'MCC'])
        
        results.to_csv(results_file, mode='a', index=False, header=not file_exists)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=256, help='testing size')
    parser.add_argument('--pth_path', type=str, default=r'None')
    parser.add_argument('--test_path', type=str, default=r'None')
    parser.add_argument('--save_path', type=str, default='./results/SG-Net/')

    opt = parser.parse_args()

    model = SG_Net(drop_path_rate=0.4)
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(opt.save_path, exist_ok=True)

    image_root = '{}/images/'.format(opt.test_path)
    gt_root = '{}/masks/'.format(opt.test_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    all_preds = []
    all_gts = []
    time_bank = []
    
    num_images = len(os.listdir(gt_root))

    for i in range(num_images):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        
        if i == 0:
            flops, params = profile(model, inputs=(image,))
            print(f'FLOPs: {flops / 1e9:.2f} G')
            print(f'Parameters: {params / 1e6:.2f} M')
        
        torch.cuda.synchronize()
        start = time.time()
        res = model(image)
        torch.cuda.synchronize()
        end = time.time()
        time_bank.append(end-start)
        
        res = F.upsample(res[0]+res[1]+res[2]+res[3], size=(opt.testsize, opt.testsize), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        all_preds.append(res)
        all_gts.append(cv2.resize(gt, (opt.testsize, opt.testsize)))
        
        imageio.imsave(opt.save_path + name, img_as_ubyte(res))

    print(f"Inference took {sum(time_bank):.2f} seconds for {num_images} images.")
    print(f'Mean time per image: {np.mean(time_bank):.4f} seconds')

    all_gts = np.array(all_gts)
    all_preds = np.array(all_preds)
    
    get_metrics_and_print(all_gts, all_preds, method='CGDFMNet', testset='ISIC2016_test', write=True) 