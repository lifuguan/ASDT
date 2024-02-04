#-*-coding:utf-8-*-
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import imageio
import importlib
from tool import imutils
import argparse
import cv2
import os.path
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def _crf_with_alpha(pred_prob, ori_img):
    bgcam_score = pred_prob.cpu().data.numpy()
    crf_score = imutils.crf_inference_inf(ori_img, bgcam_score, labels=21)

    return crf_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default='./output/model_weights/exp1/RRM_150_3_18000.pth', type=str)
    parser.add_argument("--network", default="network.RRM", type=str)
    parser.add_argument("--out_cam_pred", default='./output/result/no_crf', type=str)
    parser.add_argument("--out_la_crf", default='./output/result/crf/exp1/rrm_18000_max_train', type=str)
    parser.add_argument("--LISTpath", default="./voc12/train_aug(id).txt", type=str)
    parser.add_argument("--IMpath", default="/data/zwy/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages", type=str)

    args = parser.parse_args()


    model = getattr(importlib.import_module(args.network), 'SegNet')()
    model.load_state_dict(torch.load(args.weights,map_location='cuda:0'))

    model.eval()
    model.cuda()
    im_path = args.IMpath
    img_list = open(args.LISTpath).readlines()
    pred_softmax = torch.nn.Softmax(dim=0)
    
    if not os.path.isdir(args.out_la_crf):
        os.makedirs(args.out_la_crf)
            
    for i in img_list:
        if args.LISTpath == "./voc12/test.txt":
            tmp = i.split("/")[2][:-5]
        if i.endswith("\n"):
            i = i.strip()
        #img_temp = cv2.imread(os.path.join(im_path, i[:-1] + '.jpg'))
        img_temp = cv2.imread(os.path.join(im_path, i + '.jpg'))
        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB).astype(np.float)
        img_original = img_temp.astype(np.uint8)
        img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
        img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
        img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225

        input = torch.from_numpy(img_temp[np.newaxis, :].transpose(0, 3, 1, 2)).float().cuda()

        output1, output2 = model(input,require_mcam=False,require_seg=True)

        output1 = F.interpolate(output1, (img_temp.shape[0], img_temp.shape[1]),mode='bilinear',align_corners=False)
        output2 = F.interpolate(output2, (img_temp.shape[0], img_temp.shape[1]), mode='bilinear', align_corners=False)

        output1 = torch.squeeze(output1)
        
        output2 = torch.squeeze(output2)
        pred_prob1 = pred_softmax(output1)
        pred_prob2 = pred_softmax(output2)

        pred_prob = torch.where(pred_prob1 > pred_prob2, pred_prob1, pred_prob2)

        output = torch.argmax(output, dim=0).cpu().numpy()

        print(i)

        if args.out_la_crf is not None:
            crf_la = _crf_with_alpha(pred_prob, img_original)

            crf_img = np.argmax(crf_la, 0)

            imageio.imsave(os.path.join(args.out_la_crf, i + '.png'), crf_img.astype(np.uint8))

