import time
import hdf5storage as h5
import numpy as np
import torch
from scipy.io import loadmat
import os
from torch import nn
from calculate_metrics import Loss_SAM, Loss_RMSE, Loss_PSNR, Loss_SSIM, Loss_ERGAS
from utils import create_F, Gaussian_downsample, fspecial, AverageMeter
from Model.ASSR import  ASSR
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
dataset = 'CAVE'
path = './Dataset/Cave/Test/'
imglist = os.listdir(path)

R = create_F()
R_inv = np.linalg.pinv(R)
R_inv = torch.Tensor(R_inv)
SRF = torch.Tensor(R).cuda().float()

model_path = r'PKL/Cave/Train_ASSR/4/cave_pkl/End.pkl'
net2 = ASSR(31,3,64).cuda()
checkpoint = torch.load(model_path)
net2.load_state_dict(checkpoint)

RealSavePath = './result/ASSR_Net/Cave/Rea_2/'
FakSavePath = './result/ASSR_Net/Cave/Fak_2/'

RMSE = []
training_size = 64
stride = 32
PSF = fspecial('gaussian', 8, 3)
downsample_factor = 8
loss_func = nn.L1Loss().cuda()


# print(net2)
def reconstruction(net2, R, HSI_LR, MSI, HRHSI, downsample_factor, training_size, stride, val_loss):
    index_matrix = torch.zeros((HSI_LR.shape[1], MSI.shape[2], MSI.shape[3])).cuda()
    abundance_t = torch.zeros((HSI_LR.shape[1], MSI.shape[2], MSI.shape[3])).cuda()
    a = []
    for j in range(0, MSI.shape[2] - training_size + 1, stride):
        a.append(j)
    a.append(MSI.shape[2] - training_size)
    b = []
    for j in range(0, MSI.shape[3] - training_size + 1, stride):
        b.append(j)
    b.append(MSI.shape[3] - training_size)
    for j in a:
        for k in b:
            temp_hrms = MSI[:, :, j:j + training_size, k:k + training_size]
            temp_lrhs = HSI_LR[:, :, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                        int(k / downsample_factor):int((k + training_size) / downsample_factor)]
            temp_hrhs = HRHSI[:, :, j:j + training_size, k:k + training_size]
            with torch.no_grad():
                out = net2(temp_lrhs, temp_hrms)
                # _,_,out = net2(temp_hrms,temp_lrhs)
                loss_temp = loss_func(out, temp_hrhs.cuda())
                val_loss.update(loss_temp)
                HSI = out.squeeze()
                # 去掉维数为一的维度
                HSI = torch.clamp(HSI, 0, 1)
                abundance_t[:, j:j + training_size, k:k + training_size] = abundance_t[:, j:j + training_size,
                                                                           k:k + training_size] + HSI
                index_matrix[:, j:j + training_size, k:k + training_size] = 1 + index_matrix[:, j:j + training_size,
                                                                                k:k + training_size]

    HSI_recon = abundance_t / index_matrix
    assert torch.isnan(HSI_recon).sum() == 0
    return HSI_recon, val_loss


val_loss = AverageMeter()
SAM = Loss_SAM()
RMSE = Loss_RMSE()
PSNR = Loss_PSNR()
SSIM = Loss_SSIM().cuda()
ERGAS = Loss_ERGAS().cuda()
sam = AverageMeter()
rmse = AverageMeter()
psnr = AverageMeter()
ssim = AverageMeter()
ergas = AverageMeter()
for i in range(0, len(imglist)):
    net2.eval()
    img = h5.loadmat(path + imglist[i])
    if dataset == 'CAVE':
        img1 = img["b"]
        img1=img1/img1.max() # 归一化
    elif dataset == 'Harvard':
        img1 = img["ref"]
        img1 = img1 / img1.max()  # 归一化
    # print("real_hyper's shape =",img1.shape)
    print('org:', img1.min(), img1.max())
    HRHSI = torch.Tensor(np.transpose(img1, (2, 0, 1)))
    MSI = torch.tensordot(torch.Tensor(R), HRHSI, dims=([1], [0]))
    HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
    MSI_1 = torch.unsqueeze(MSI, 0)
    HSI_LR1 = torch.unsqueeze(torch.Tensor(HSI_LR), 0)  # 加维度 (b,c,h,w)
    time1 = time.time()
    to_fet_loss_hr_hsi = torch.unsqueeze(torch.Tensor(HRHSI), 0)
    with torch.no_grad():
        prediction, val_loss = reconstruction(net2, R, HSI_LR1.cuda(), MSI_1.cuda(), to_fet_loss_hr_hsi,
                                              downsample_factor, training_size, stride, val_loss)
        Fuse = prediction.cpu().detach().numpy()
    time2 = time.time()
    print(time2 - time1)

    print('pre:', prediction.min(), prediction.max())
    sam.update(SAM(np.transpose(HRHSI.cpu().detach().numpy(), (1, 2, 0)),
                   np.transpose(prediction.squeeze().cpu().detach().numpy(), (1, 2, 0))))
    rmse.update(RMSE(HRHSI.cpu().permute(1, 2, 0), prediction.squeeze().cpu().permute(1, 2, 0)))
    psnr.update(PSNR(HRHSI.cpu().permute(1, 2, 0), prediction.squeeze().cpu().permute(1, 2, 0)))
    ssim.update(SSIM(HRHSI.cuda().unsqueeze(0), prediction.unsqueeze(0)))
    ergas.update(ERGAS(HRHSI.cuda().unsqueeze(0), prediction.unsqueeze(0)))
    faker_hyper = np.transpose(Fuse, (1, 2, 0))
    print(i, ':', imglist[i], faker_hyper.shape)
    print(PSNR(HRHSI.cpu().permute(1, 2, 0), prediction.squeeze().cpu().permute(1, 2, 0)))
    rea_data_path = os.path.join(RealSavePath + imglist[i])
    fak_data_path = os.path.join(FakSavePath + imglist[i])
    h5.savemat(fak_data_path, {'fak': faker_hyper}, format='5')
    h5.savemat(rea_data_path, {'rea': img1}, format='5')
print("val loss:", val_loss.avg.cpu().detach().numpy())
print("val  PSNR:", psnr.avg.cpu().detach().numpy(), "  RMSE:", rmse.avg.cpu().detach().numpy(), "  SAM:", sam.avg, "val loss:", val_loss.avg.cpu().detach().numpy(), "ssim:", ssim.avg.cpu().detach().numpy(), "ergas:", ergas.avg)
