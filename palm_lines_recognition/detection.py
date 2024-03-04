import numpy as np
from PIL import Image
import torch
from palm_lines_recognition.model import *


MODEL_PATH = 'palm_lines_recognition/weights/checkpoint_aug_epoch70.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device for UNet model: ', device)
net = UNet(n_channels=3, n_classes=1)
net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))


def detect(jpeg_dir, output_dir, resize_value, device=torch.device('cpu')):
    pil_img = Image.open(jpeg_dir)
    img = np.asarray(pil_img.resize((resize_value, resize_value), resample=Image.NEAREST)) / 255
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).permute(0,3,1,2).to(device)
    pred = net(img).squeeze(0)
    pred = torch.Tensor(np.apply_along_axis(lambda x: [1,1,1] if x > 0.03 else [0,0,0], 0, pred.cpu().detach()))
    Image.fromarray((pred.permute((1,2,0)).numpy() * 255).astype(np.uint8)).save(output_dir)