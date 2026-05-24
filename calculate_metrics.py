import os
import numpy as np
import torch
from hdf5storage import loadmat
from torch import nn
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        Itrue = im_true.clamp(0., 1.)*data_range
        Ifake = im_fake.clamp(0., 1.)*data_range
        err=Itrue-Ifake
        err=torch.pow(err,2)
        err = torch.mean(err,dim=0)
        err = torch.mean(err,dim=0)

        psnr = 10. * torch.log10((data_range ** 2) / err)
        psnr=torch.mean(psnr)
        return psnr


class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs.clamp(0., 1.)*255- label.clamp(0., 1.)*255
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse

class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()
        self.eps=2.2204e-16
    def forward(self,im1, im2):
        assert im1.shape == im2.shape
        H,W,C=im1.shape
        im1 = np.reshape(im1,( H*W,C))
        im2 = np.reshape(im2,(H*W,C))
        core=np.multiply(im1, im2)
        mole = np.sum(core, axis=1)
        im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
        im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
        deno = np.multiply(im1_norm, im2_norm)
        sam = np.rad2deg(np.arccos(((mole+self.eps)/(deno+self.eps)).clip(-1,1)))
        return np.mean(sam)



class Loss_SAM_1(nn.Module):
    def __init__(self):
        super(Loss_SAM_1, self).__init__()
        # 防止除0错误
        self.eps = 2.2204e-16
    def forward(self,im1, im2):
        assert im1.shape == im2.shape
        H, W, C = im1.shape
        # im1 = np.reshape(im1, (H*W, C))
        im1 = im1.reshape(H * W, C)
        # im2 = np.reshape(im2, (H*W, C))
        im2 = im2.reshape(H * W, C)
        # 计算对应元素乘积(元素级乘法)
        core = im1 * im2
        mole = torch.sum(core, dim=1)
        # im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
        im1_norm = torch.sqrt(torch.sum(torch.square(im1), dim=1))
        # im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
        im2_norm = torch.sqrt(torch.sum(torch.square(im2), dim=1))
        deno = im1_norm * im2_norm
        # clip(-1, 1)将值限制在-1到1之间
        # np.rad2deg()将弧度转化为度数
        # sam = np.rad2deg(np.arccos(((mole+self.eps)/(deno+self.eps)).clip(-1, 1)))
        sam = torch.rad2deg(torch.acos(torch.clamp((mole + self.eps)/(deno + self.eps), -1, 1)))
        return torch.mean(sam)
    

class Loss_SSIM(nn.Module):
    def __init__(self):
        super(Loss_SSIM, self).__init__()
        pass

    def forward(self, img1, img2, window_size=11, size_average=True):
        (_, channel, _, _) = img1.size()
        window = create_window(window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

# 输入四维张量
class Loss_ERGAS(nn.Module):
    def __init__(self):
        super(Loss_ERGAS, self).__init__()


    def forward(self, img_tgt, img_fus):
        scale = 8
        img_tgt = img_tgt.squeeze(0).data.cpu().numpy()
        img_fus = img_fus.squeeze(0).data.cpu().numpy()
        img_tgt = np.squeeze(img_tgt)
        img_fus = np.squeeze(img_fus)
        img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
        img_fus = img_fus.reshape(img_fus.shape[0], -1)

        rmse = np.mean((img_tgt - img_fus) ** 2, axis=1)
        rmse = rmse ** 0.5
        mean = np.mean(img_tgt, axis=1)

        ergas = np.mean((rmse / mean) ** 2)
        ergas = 100 / scale * ergas ** 0.5

        return ergas
    
class Loss_ERGAS_1(nn.Module):
    def __init__(self):
        super(Loss_ERGAS_1, self).__init__()


    def forward(self, img_tgt, img_fus):
        scale = 8
        img_tgt = img_tgt.squeeze(0)
        img_fus = img_fus.squeeze(0)
        img_tgt = img_tgt.squeeze()
        img_fus = img_fus.squeeze()
        img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
        img_fus = img_fus.reshape(img_fus.shape[0], -1)

        rmse = torch.mean((img_tgt - img_fus) ** 2, dim=1)
        rmse = rmse ** 0.5
        mean = torch.mean(img_tgt, dim=1)

        ergas = torch.mean((rmse / mean) ** 2)
        ergas = 100 / scale * ergas ** 0.5

        return ergas
    
def qnr_index(fused, msi, hsi, alpha=1, beta=1):
    """
    计算 QNR 指标
    :param fused: (H, W, B) 融合结果 (HRHSI)
    :param msi:   (H, W, C) 原始 HRMSI
    :param hsi:   (H, W, B) 原始 LRHSI 上采样到 HR 尺度
    :param alpha: 光谱权重
    :param beta:  空间权重
    :return: QNR 值, D_lambda, D_s
    """
    fused = np.clip(fused, 0, 1).astype(np.float64)
    msi   = np.clip(msi, 0, 1).astype(np.float64)
    hsi   = np.clip(hsi, 0, 1).astype(np.float64)

    H, W, B = fused.shape
    _, _, C = msi.shape

    # --- 保证尺寸一致 ---
    if hsi.shape[:2] != (H, W):
        hsi = resize(hsi, (H, W, B), order=1, preserve_range=True, anti_aliasing=True)
    if msi.shape[:2] != (H, W):
        msi = resize(msi, (H, W, C), order=1, preserve_range=True, anti_aliasing=True)

    # --- 光谱失真 Dλ ---
    D_lambda = 0
    for i in range(B):
        for j in range(i + 1, B):
            corr_f = np.corrcoef(fused[:, :, i].ravel(), fused[:, :, j].ravel())[0, 1]
            corr_h = np.corrcoef(hsi[:, :, i].ravel(), hsi[:, :, j].ravel())[0, 1]
            D_lambda += abs(corr_f - corr_h)
    D_lambda /= (B * (B - 1) / 2)

    # --- 空间失真 Ds ---
    D_s = 0
    for c in range(C):
        psnr_f = psnr(msi[:, :, c], fused[:, :, c], data_range=1.0)
        psnr_h = psnr(msi[:, :, c], hsi[:, :, c], data_range=1.0)
        D_s += abs(psnr_f - psnr_h) / (abs(psnr_h) + 1e-6)
    D_s /= C

    # --- QNR ---
    QNR = (1 - D_lambda) ** alpha * (1 - D_s) ** beta
    return QNR

if __name__ == '__main__':
    SAM=Loss_SAM()
    RMSE=Loss_RMSE()
    PSNR=Loss_PSNR()
    psnr_list=[]
    sam_list=[]
    sam=AverageMeter()
    rmse=AverageMeter()
    psnr=AverageMeter()
    path1 = r'result/Cave/Rea/'
    path2 = r'result/Cave/Fak/'
    imglist = os.listdir(path1)
    for i in range(0, len(imglist)):
        img1 = loadmat(path2 + imglist[i])
        img2 = loadmat(path1 + imglist[i])
        # print(img2)
        lable = img1["fak"]
        recon = img2["rea"]
        sam_temp=SAM(lable,recon)
        psnr_temp=PSNR(torch.Tensor(lable), torch.Tensor(recon))
        sam.update(sam_temp)
        rmse.update(RMSE(torch.Tensor(lable),torch.Tensor(recon)))
        psnr.update(psnr_temp)
        psnr_list.append(psnr_temp)
        sam_list.append(sam_temp)
    print(sam.avg)
    print(rmse.avg)
    print(psnr.avg.cpu().detach().numpy())
    print(psnr_list)
    print(sam_list)
