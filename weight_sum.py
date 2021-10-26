from Server.server_model import Yolov4_server, Yolov4_local
from models import Yolov4
from torch import nn
import torch
import os
from tool.darknet2pytorch import Darknet

# model_local = Darknet_Local('/home/sihun/yolov4_split_learning/Local/cfg/yolov4_21nia.cfg')
# model_server = Darknet_Server('/home/sihun/yolov4_split_learning/Local/cfg/yolov4_21nia.cfg')

model_server = Yolov4_server(n_classes=10)
model_local = Yolov4_local()

model = Yolov4(n_classes=10, inference=True)
# model=Darknet('/home/sihun/yolov4_split_learning/Local/cfg/yolov4_21nia.cfg', inference=True)
# model = nn.Sequential(model.down1, model.down2, model.down3, model.down4, model.down5, model.neek)

# Local Side
pretrained_dict = torch.load('C:/Users/hyeli/Anaconda3/envs/hlnam/Yolo_inference/weight/Yolov4_cli_epoch123.pth')
# model_dict = model.state_dict()
# 1. filter out unnecessary keys
# pretrained_dict1 = {k1: v for (k, v), k1 in zip(pretrained_dict1.items(), model_dict)}
# 2. overwrite entries in the existing state dict
# model_dict.update(pretrained_dict1)
# model.load_state_dict(pretrained_dict1)


# Server Side
pretrained_dict2 = torch.load('C:/Users/hyeli/Anaconda3/envs/hlnam/Yolo_inference/weight/Yolov4_ser_epoch123.pth')
model_dict = model.state_dict()
# 1. filter out unnecessary keys
# pretrained_dict2 = {k1: v for (k, v), k1 in zip(pretrained_dict2.items(), model_dict)}

# cnt = 105
# temp='0'
# pretrained_dict2_={}
# for i, (k,v) in enumerate(pretrained_dict2.items()):
#
#     if k.split('.')[1]!=temp:
#         cnt=105+int(k.split('.')[1])
#         temp = k.split('.')[1]
#     x=k.split('.')
#     x[1]=str(cnt)
#     x='.'.join(x)
#     pretrained_dict2_[x]=v


# 2. overwrite entries in the existing state dict
# model_dict.update(pretrained_dict2_)
pretrained_dict.update(pretrained_dict2)
model.load_state_dict(pretrained_dict)
os.makedirs('./weightsum', exist_ok=True)
save_path = os.path.join('./weightsum', 'model.pth')
torch.save(model.state_dict(), save_path)