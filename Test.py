import sys
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import imageio
from EGANet import EGANetModel
from MGCC import MGCC
from MobileUViT import MobileUViT
from PFDNet import PFD_Net
from PraNet_Res2Net import PraNet
from dcnet_Comparion.DC_Net import DC_Net
from lib.networks import MERIT_MSGAA_Cascaded
from model import SHAT
from CFMANet import CfaNet
from UNet import UNet
sys.path.append('model')
from MSFBNet import MSFBNet

from dataloader import test_dataset
from collections import OrderedDict
from skimage.util import img_as_ubyte
import time

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--pth_path', type=str, default='')
parser.add_argument('--test_path', type=str, default='')


# for _data_name in ['test', 'STU', 'CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
for _data_name in ['test']:
    opt = parser.parse_args()
    targetdir = os.listdir(opt.test_path)
    if (_data_name not in targetdir):
        continue

    data_path = os.path.join(opt.test_path, _data_name)
    data_path = data_path.replace('\\', '/')
    print("#" * 20)
    print("Now test dir is: ", data_path)
    print("#" * 20)
    time.sleep(10)
    save_path = '' + '/{}/{}/'.format(opt.test_path.split('/')[-2], _data_name)
    model = MSFBNet(1)

    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weights = torch.load(pth_path, map_location=device)

    # weights = torch.load(pth_path)

    new_state_dict = OrderedDict()

    for k, v in weights.items():
        if 'total_ops' not in k and 'total_params' not in k:
            name = k
            new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    total_infer_time = 0.0
    total_images = test_loader.size

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        # ------------------ Start timing ------------------
        start_time = time.time()
        res3,res2, res1, res0 = model(image)
        # res3 = model(image)
        res = res3
        res = F.interpolate(res, size=(gt.shape), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        end_time = time.time()
        # ------------------ End timing --------------------

        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        total_infer_time += (end_time - start_time)
        imageio.imsave(save_path + name, img_as_ubyte(res))

    avg_time_per_image = total_infer_time / total_images
    fps = 1.0 / avg_time_per_image

    print("=" * 40)
    print(f"Inference Time per image: {avg_time_per_image * 1000:.2f} ms")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    print("=" * 40)
