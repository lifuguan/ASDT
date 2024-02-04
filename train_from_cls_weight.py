import os
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from tool import pyutils, torchutils
import argparse
import importlib
import torch.nn.functional as F
from DenseEnergyLoss import DenseEnergyLoss
import random
import tool.myTool as mytool
from tool.myTool import compute_joint_loss, compute_seg_label, compute_cam_up
from tool import imutils

def _crf_with_alpha(pred_prob, ori_img):
    bgcam_score = pred_prob.cpu().data.numpy()
    crf_score = imutils.crf_inference_inf(ori_img, bgcam_score, labels=21)

    return crf_score

def seed_torch(seed=42):

    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False


if __name__ == '__main__':

    seed_torch()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU_id')

    parser.add_argument("--LISTpath", default="voc12/train_aug(id).txt", type=str)
    parser.add_argument("--IMpath", default="data/JPEGImages", type=str)
    parser.add_argument("--SAVEpath", default="./output/model_weights/exp1", type=str)

    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_step", default=20000, type=int)
    parser.add_argument("--network", default="network.RRM", type=str)
    parser.add_argument("--lr", default=0.0007, type=float)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--wt_dec", default=1e-5, type=float)
    parser.add_argument("--weights",default='./netWeights/res38_cls.pth', type=str)

    parser.add_argument("--session_name", default="RRM_", type=str)
    parser.add_argument("--crop_size", default=321, type=int)
    parser.add_argument("--class_numbers", default=20, type=int)

    parser.add_argument('--crf_la_value', type=int, default=4)
    parser.add_argument('--crf_ha_value', type=int, default=32)

    parser.add_argument('--densecrfloss', type=float, default=1e-7,
                        metavar='M', help='densecrf loss (default: 0)')
    parser.add_argument('--rloss-scale', type=float, default=0.5,
                        help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    parser.add_argument('--sigma-rgb', type=float, default=15.0,
                        help='DenseCRF sigma_rgb')
    parser.add_argument('--sigma-xy', type=float, default=100,
                        help='DenseCRF sigma_xy')

    args = parser.parse_args()

    gpu_id = args.gpu_id

    if not os.path.isdir(args.SAVEpath):
        os.makedirs(args.SAVEpath)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    save_path = os.path.join(args.SAVEpath,args.session_name)
    print("dloss weight", args.densecrfloss)
    critersion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='elementwise_mean').cuda()
    DenseEnergyLosslayer = DenseEnergyLoss(weight=args.densecrfloss, sigma_rgb=args.sigma_rgb,
                                     sigma_xy=args.sigma_xy, scale_factor=args.rloss_scale)

    model = getattr(importlib.import_module(args.network), 'SegNet')()

    pceloss = torch.nn.CrossEntropyLoss(weight=None, reduction='elementwise_mean').cuda()

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))

    max_step = args.max_step

    batch_size = args.batch_size
    img_list = mytool.read_file(args.LISTpath)

    data_list = []
    for i in range(int(max_step//100)):
        np.random.shuffle(img_list)
        data_list.extend(img_list)

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer_cls([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    weights_dict = torch.load(args.weights,map_location=torch.device('cpu'))

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss')

    timer = pyutils.Timer("Session started: ")

    data_gen = mytool.chunker(data_list, batch_size)

    # T and N
    T = 150

    N = 6

    every_part = int(T/N)

    save_path = save_path + str(T) + "_" + str(N) + "_"

    for iter in range(max_step + 1):

        chunk = data_gen.__next__()
        img_list = chunk
        images, ori_images, label, croppings, img_original = mytool.get_data_from_chunk_v2(chunk,args)
        b, _, w, h = ori_images.shape
        c = args.class_numbers
        label = label.cuda(non_blocking=True)

        x_f, cam, seg, seg2 = model(images, require_seg = True, require_mcam = True)

        # prepare pgt label for seg2

        seg_feat = F.interpolate(seg[0], (w, h), mode='bilinear', align_corners=False)

        seg_feat2 = F.interpolate(seg2[0], (w, h), mode='bilinear', align_corners=False)

        seg_prob = F.softmax(seg_feat, dim=1)

        cam_up = compute_cam_up(cam, label, w, h, b)

        seg_label1 = np.zeros((b,w,h))

        seg_label2 = np.zeros((b,w,h))

        for i in range(b):

            cam_up_single = cam_up[i]
            cam_label = label[i].cpu().numpy()
            ori_img = ori_images[i].transpose(1,2,0).astype(np.uint8)
            norm_cam = cam_up_single/(np.max(cam_up_single, (1, 2), keepdims=True) + 1e-5)

            seg_label1[i] = compute_seg_label(ori_img, cam_label, norm_cam)

            if iter > 1000 and int(iter / every_part) % N == 0:

                seg_prob_tmp = seg_prob[i]

                img_original_tmp = img_original[i]

                crf_la = _crf_with_alpha(seg_prob_tmp, img_original_tmp)

                seg_label2[i] = np.argmax(crf_la, 0)

        closs = F.multilabel_soft_margin_loss(x_f, label)

        celoss, dloss = compute_joint_loss(ori_images, seg[0], seg_label1, croppings, critersion, DenseEnergyLosslayer)

        label_pgt = torch.from_numpy(seg_label2).long().cuda()

        if iter > 1000:

            w_pgt = 0.1

            celoss2, dloss2 = compute_joint_loss(ori_images, seg2[0], seg_label1, croppings, critersion,
                                                 DenseEnergyLosslayer)

            # celoss2 = pceloss(seg_feat2, label_pgt.detach()) * w_pgt

            loss = closs + (celoss + dloss)*0.5 + (celoss2 + dloss2) * 0.5

        else:

            celoss2, dloss2 = compute_joint_loss(ori_images, seg2[0], seg_label1, croppings, critersion,
                                                 DenseEnergyLosslayer)

            loss = closs + (celoss + dloss) * 0.5 + (celoss2 + dloss2) * 0.5

        print('closs: %.4f'% closs.item(),'celoss: %.4f'%celoss.item(), 'dloss: %.4f'%dloss.item(), 'ce_loss2: %.4f'%celoss2.item(),
              'dloss2: %.4f' % dloss2.item())

        avg_meter.add({'loss': loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (optimizer.global_step - 1) % 50 == 0:
            timer.update_progress(optimizer.global_step / max_step)

            print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                  'Loss:%.4f' % (avg_meter.pop('loss')),
                  'imps:%.1f' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                  'Fin:%s' % (timer.str_est_finish()),
                  'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

            if (optimizer.global_step - 1) % 2000 == 0 and optimizer.global_step > 10000:
                torch.save(model.module.state_dict(), save_path + '%d.pth' % (optimizer.global_step - 1))

    #torch.save(model.module.state_dict(), args.session_name + 'final.pth')

