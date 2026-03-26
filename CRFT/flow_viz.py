# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 1
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    '''

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)


UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


# from https://github.com/gengshan-y/VCN
def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def save_vis_flow_tofile(flow, output_path):
    vis_flow = flow_to_image(flow)
    from PIL import Image
    img = Image.fromarray(vis_flow)
    img.save(output_path)


def flow_tensor_to_image(flow):
    """Used for tensorboard visualization"""
    flow = flow.permute(1, 2, 0)  # [H, W, 2]
    flow = flow.detach().cpu().numpy()
    flow = flow_to_image(flow)  # [H, W, 3]
    flow = np.transpose(flow, (2, 0, 1))  # [3, H, W]

    return flow

"""szx"""
def get_warp_flow(img, flow, start=0): #tensor[16,1,360,640],tensor[16,2,320,576],tensor[16,2,1,1]

    batch_size, _, patch_size_h, patch_size_w = flow.shape
    # start = start.to('cuda:1')
    # flow = flow.to('cuda:1')
    grid_warp = get_grid(batch_size, patch_size_h, patch_size_w, start)[:, :2, :, :] - flow
    img_warp = transformer(img, grid_warp)

    return img_warp

def get_grid(batch_size, H, W, start=0):

    if torch.cuda.is_available():
        xx = torch.arange(0, W).cuda()
        yy = torch.arange(0, H).cuda()
    else:
        xx = torch.arange(0, W)
        yy = torch.arange(0, H)
    xx = xx.view(1, -1).repeat(H, 1)
    yy = yy.view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    ones = torch.ones_like(xx).cuda() if torch.cuda.is_available() else torch.ones_like(xx)
    grid = torch.cat((xx, yy, ones), 1).float()

    # grid = grid.to('cuda:1')
    grid[:, :2, :, :] = grid[:, :2, :, :] + start  # add the coordinate of left top
    return grid

def transformer(I, vgrid, train=True):
    # I: Img, shape: batch_size, 1, full_h, full_w
    # vgrid: vgrid, target->source, shape: batch_size, 2, patch_h, patch_w
    # outsize: (patch_h, patch_w)

    def _interpolate(im, x, y, out_size):
        # x: x_grid_flat
        # y: y_grid_flat
        # im = im.to('cuda:1')
        # x = x.to('cuda:1')
        # y = y.to('cuda:1')

        num_batch, num_channels, height, width = im.size()

        out_height, out_width = out_size[0], out_size[1]
        zero = 0
        max_y = height - 1
        max_x = width - 1

        # do sampling
        x0 = torch.floor(x).int() 
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)  # same as np.clip 截断操作
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)

        dim1 = width * height
        dim2 = width

        if torch.cuda.is_available():
            base = torch.arange(0, num_batch).int().cuda() #tensor[16]
        else:
            base = torch.arange(0, num_batch).int()

        base = base * dim1
        base = base.repeat_interleave(out_height * out_width, axis=0) #2949120
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im = im.permute(0, 2, 3, 1)
        im_flat = im.reshape([-1, num_channels]).float()

        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(out_height * out_width * num_batch, num_channels)
        Ia = torch.gather(im_flat, 0, idx_a)

        idx_b = idx_b.unsqueeze(-1).long()
        idx_b = idx_b.expand(out_height * out_width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)

        idx_c = idx_c.unsqueeze(-1).long()
        idx_c = idx_c.expand(out_height * out_width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)

        idx_d = idx_d.unsqueeze(-1).long()
        idx_d = idx_d.expand(out_height * out_width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return output

    def _transform(I, vgrid): #tensor[16,1,360,640] tensor[16,2,320,576]

        C_img = I.shape[1]
        B, C, H, W = vgrid.size()

        x_s_flat = vgrid[:, 0, ...].reshape([-1]) #tensor[2949120]
        y_s_flat = vgrid[:, 1, ...].reshape([-1])
        out_size = vgrid.shape[2:] #size2[320,576]
        input_transformed = _interpolate(I, x_s_flat, y_s_flat, out_size)

        output = input_transformed.reshape([B, H, W, C_img])
        return output

    output = _transform(I, vgrid)
    if train:
        output = output.permute(0, 3, 1, 2)
    return output


def save_images(img1_show, img2_show, flow_gt_show, img1_warp, img2_warp, flow_show, label):
    output_dir = os.path.join('/root/autodl-tmp/result_flow_OS_CRFT/', label)
    
    os.makedirs(output_dir, exist_ok=True)

    filenames = [
         label + 'opt.tif',
         label + 'sar.tif',
         label + 'flogt.tif',
         label + 'optwarp.tif',
         label + 'sarwarp.tif',
         label + 'floprep.tif',
    ]
    savePath = [os.path.join(output_dir, fname) for fname in filenames]
    for i, img_array in enumerate([img1_show, img2_show, flow_gt_show, img1_warp, img2_warp, flow_show]):
        img_array = img_array.astype(np.uint8)

        if img_array.ndim == 2:
            img_pil = Image.fromarray(img_array, mode='L')
        elif img_array.ndim == 3:
            img_pil = Image.fromarray(img_array, mode='RGB')
        else:
            raise ValueError(f'Unsupported Image Formate for image{i + 1}')

        img_pil.save(savePath[i])


def visualizeResults(flow_pr, flow_gt, image1, image2, label):
    flow_pr_for_show = flow_pr[0] if flow_pr.dim() == 4 else flow_pr
    flow_gt_for_show = flow_gt[0] if flow_gt.dim() == 4 else flow_gt

    # Add check for channel dimension before permuting
    if flow_pr_for_show.shape[0] == 2:
        flow_pr_for_show = flow_pr_for_show.permute(1, 2, 0)
    if flow_gt_for_show.shape[0] == 2:
        flow_gt_for_show = flow_gt_for_show.permute(1, 2, 0)

    flow_pr_show = flow_pr_for_show.cpu().numpy()
    flow_gt_show = flow_gt_for_show.cpu().numpy()
    flow_show = flow_to_image(flow_pr_show)
    flow_gt_show = flow_to_image(flow_gt_show)

    x1_full = image1[:, 0, :, :].unsqueeze(0)
    if torch.cuda.is_available():
        x1_full = x1_full.cuda()
    x2_full = image2[:, 0, :, :].unsqueeze(0)
    if torch.cuda.is_available():
        x2_full = x2_full.cuda()
    # H_flow_gt = flow_gt.unsqueeze(0).cuda()
    # img1_warp = flow_viz.get_warp_flow(x1_full, H_flow_gt)
    # img2_warp = flow_viz.get_warp_flow(x2_full, -H_flow_gt)
    img1_warp = get_warp_flow(x1_full, flow_pr)
    img2_warp = get_warp_flow(x2_full, -flow_pr)
    img1_warp = img1_warp.cpu().numpy().squeeze(0).squeeze(0)
    img2_warp = img2_warp.cpu().numpy().squeeze(0).squeeze(0)
    img1_show = x1_full.cpu().numpy().squeeze(0).squeeze(0)
    img2_show = x2_full.cpu().numpy().squeeze(0).squeeze(0)
    plt.gray()
    plt.subplot(231)
    plt.axis('off')
    plt.title('optical')
    plt.imshow(img1_show)
    plt.subplot(232)
    plt.axis('off')
    plt.title('SAR warped')
    plt.imshow(img2_show)
    plt.subplot(233)
    plt.axis('off')
    plt.title('flow truth')
    plt.imshow(flow_gt_show / 255.0)
    plt.subplot(234)
    plt.axis('off')
    plt.title('SAR warped back')
    plt.imshow(img2_warp)
    plt.subplot(235)
    plt.axis('off')
    plt.title('optical warped forward')
    plt.imshow(img1_warp)
    plt.subplot(236)
    plt.axis('off')
    plt.title('flow predict')
    plt.imshow(flow_show / 255.0)
    
    #plt.show()

    save_images(img1_show, img2_show, flow_gt_show, img1_warp, img2_warp, flow_show, label)
