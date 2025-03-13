import torch
import torch.nn as nn
import torch.optim as optim

import time
import subprocess
import shutil
import os
import random
from PIL import Image
import numpy as np
import pickle






#### Feature discriminator #########
class FeatureDiscriminator(nn.Module):
    def __init__(self):
        super(FeatureDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)

discriminate_loss = nn.BCEWithLogitsLoss()
###################################3



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)



import lora_models.backbones as backbones
from lora_models.backbones.mylora import Linear as LoraLinear
from lora_models.backbones.mylora import DVLinear as DVLinear
from util.visualizer import save_images
from util import html




############# import cyclegan module ###########################

from options.train_options import TrainOptions
from models import create_model
from data import create_dataset
opt = TrainOptions().parse()  # get test options
opt.model = 'cycle_gan'
opt.epoch = 'latest'

opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   

dataset_cyclegan = create_dataset(opt)
cyclegan = create_model(opt)      # create a model given opt.model and other options
cyclegan.setup(opt)               # regular setup: load and print networks; create schedulers
cyclegan.eval()

# precopy

dataset_cyclegan_dataset_A_paths = dataset_cyclegan.dataset.A_paths.copy()
dataset_cyclegan_dataset_B_paths = dataset_cyclegan.dataset.B_paths.copy()

min_length = min(len(dataset_cyclegan_dataset_A_paths), len(dataset_cyclegan_dataset_B_paths))
dataset_cyclegan.dataset.A_paths = dataset_cyclegan_dataset_A_paths[:min_length]
dataset_cyclegan.dataset.B_paths = dataset_cyclegan_dataset_B_paths[:min_length]

dataset_cyclegan_dataset_A_paths = dataset_cyclegan.dataset.A_paths.copy()
dataset_cyclegan_dataset_B_paths = dataset_cyclegan.dataset.B_paths.copy()

print('len A path is ', len(dataset_cyclegan_dataset_A_paths))
print('len B path is ', len(dataset_cyclegan_dataset_B_paths))




gt_pose_44 = np.loadtxt(opt.pose_gt_path, delimiter=',')
if gt_pose_44.shape[-1] == 17:
    gt_pose_44 = gt_pose_44[:,1:]


############################################################


def mark_only_part_as_trainable(model: nn.Module, bias: str = 'none', warm_up: bool = True) -> None:
    for n, p in model.named_parameters():
        if warm_up:
            if 'lora_A' not in n and 'lora_B' not in n and 'residual_' not in n and 'conv_depth_' not in n :
                p.requires_grad = False
        else:
            if 'lora_U' not in n and 'lora_V' not in n and 'residual_' not in n and 'conv_depth_' not in n :
                p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, backbones.galora.LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


############################### retrieval module ############################


import argparse
import os
from functools import partial
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform):
        self.root, self.transform = root, transform
        self.imgs = sorted(glob(os.path.join(root, '*.jpg')) + glob(os.path.join(root, '*.png')))
        
    def __getitem__(self, idx):
        return self.transform(Image.open(self.imgs[idx]).convert("RGB"))

    def __len__(self):
        return len(self.imgs)


class ImageFileDataset(Dataset):
    def __init__(self, file_list, transform):
        self.file_list, self.transform = file_list, transform

    def __getitem__(self):
        return self.transform(Image.open(self.file_list).convert("RGB"))

    def __len__(self):
        return 1


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)
        self.single = True

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x) # (1,1200,384)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1) # (1,1202,384)

        x = x + self.pos_embed
        x = self.pos_drop(x) # (1,1202,384)

        for i, blk in enumerate(self.blocks):
            x = blk(x) # (1,1202,384)

        x = self.norm(x) # (1,1202,384)

        return x, x[:, 0], x[:, 1]

    def forward(self, x):
        full_features, x, x_dist = self.forward_features(x)
        
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        return full_features, F.normalize( (x + x_dist) / 2, p=2, dim=1)


args = opt

model=DistilledVisionTransformer(
        img_size=args.img_resize, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=256)
checkpoint = torch.load(args.checkpoint, map_location='cpu')
state_dict = {k.replace('module.backbone.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('module.backbone')}
model.load_state_dict(state_dict)

device = torch.device("cuda:7")
model.to(device)
model=model.train()


lora_type = 'lora'
for t_layer_i, blk in enumerate(model.blocks):
    mlp_in_features = blk.mlp.fc1.in_features
    mlp_hidden_features = blk.mlp.fc1.out_features
    mlp_out_features = blk.mlp.fc2.out_features


    if lora_type == "dvlora":
        blk.mlp.fc1 = DVLinear(mlp_in_features, mlp_hidden_features, r=args.r, lora_alpha=args.r)
        blk.mlp.fc2 = DVLinear(mlp_hidden_features, mlp_out_features, r=args.r, lora_alpha=args.r)
        blk.mlp.fc1 = blk.mlp.fc1.to(device)
        blk.mlp.fc2 = blk.mlp.fc2.to(device)
    elif lora_type == "lora":
        blk.mlp.fc1 = LoraLinear(mlp_in_features, mlp_hidden_features, r=args.r)
        blk.mlp.fc2 = LoraLinear(mlp_hidden_features, mlp_out_features, r=args.r)
        blk.mlp.fc1 = blk.mlp.fc1.to(device)
        blk.mlp.fc2 = blk.mlp.fc2.to(device)

mark_only_part_as_trainable(model)



base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Resize(args.img_resize,antialias=False)
])


image_path = args.scene_path


dataset = ImageFolderDataset(image_path, transform=base_transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                        shuffle=False, num_workers=4, pin_memory=True)





feature_discriminator = FeatureDiscriminator()
feature_discriminator.to(device)
feature_discriminator.train()


optimizer_feature_discriminator = torch.optim.Adam(
    filter(lambda p: p.requires_grad, feature_discriminator.parameters()),
    lr=1e-5
)







database_features = np.zeros((len(dataset), 256), dtype="float32")


error_list_prev = []
error_list_after = []
error_list_np = []

index_lists = []


feature_sum = torch.zeros(256, device=device)
feature_squared_sum = torch.zeros(256, device=device)
feature_count = 0

all_features = []

with torch.inference_mode():
    for i, A_and_B in enumerate(tqdm(dataset_cyclegan)):


        if i < 1:
            continue
        if i > len(dataset_cyclegan_dataset_A_paths) -10:
            break

        virtual_gt = A_and_B['B']
        x = F.interpolate(virtual_gt, size=(480, 640), mode='bilinear', align_corners=False)

        full_features, x= model(x.to(device))
        database_features[i*args.batch_size:(i+1)*args.batch_size] = x.cpu().numpy()

        all_features.append(full_features.cpu().numpy())



database_features_np = database_features
database_features = torch.tensor(database_features).to(device)






confidence_buffer = []
time_duration_list = []
for i, A_and_B in enumerate(tqdm(dataset_cyclegan)):
    start_time = time.time()
    real_image = A_and_B['A']
    y = cyclegan.netG_A(real_image.to(device))
    y = F.interpolate(y, size=(480, 640), mode='bilinear', align_corners=False)



    if i < 1:
        continue
    if i > len(dataset_cyclegan_dataset_A_paths)-10:
        break


    # Ensure the model is in training mode
    model.train()
    optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=2e-5
    )

    
    _, query_feature = model(y.to(device))
    query_feature = query_feature[0]
    query_feature_np = query_feature.detach().cpu().numpy()



    similarities_np = np.dot(database_features_np, query_feature_np) / (np.linalg.norm(database_features_np, axis=1) * np.linalg.norm(query_feature_np) + 1e-8)
    max_sim = np.max(similarities_np)


    #################### discriminate loss #####################
    # warm up
    if i < int(len(dataset_cyclegan.dataset) * 0.05):
        # for inter_ in range(3):
        _, virtual_image_features = model(y.to(device))
        virtual_image_features = virtual_image_features # (256,)
        # print('virtual_image_features shape is ', virtual_image_features.shape)

        with torch.no_grad():
            gt_random_index = random.randint(0, len(dataset_cyclegan.dataset)-1)
            gt_virtual_image = dataset_cyclegan.dataset[gt_random_index]['B']
            gt_virtual_image = gt_virtual_image.unsqueeze(0)
            gt_virtual_image = F.interpolate(gt_virtual_image, size=(480, 640), mode='bilinear', align_corners=False)
            _, gt_virtual_features = model(gt_virtual_image.to(device))
            gt_virtual_features = gt_virtual_features # (256,)



        for param in feature_discriminator.parameters():
            param.requires_grad = False


        optimizer.zero_grad()  # set G_A and G_B's gradients to zero
        dis_result = feature_discriminator.forward(virtual_image_features)

        loss = torch.nn.functional.cross_entropy(dis_result, torch.tensor(1).unsqueeze(0).to(device))
        
        
        loss.backward()
        optimizer.step()

        # D_A and D_B
        for param in feature_discriminator.parameters():
            param.requires_grad = True


        optimizer_feature_discriminator.zero_grad()   # set D_A and D_B's gradients to zero
        dis_result = feature_discriminator.forward(gt_virtual_features)
        loss = torch.nn.functional.cross_entropy(dis_result, torch.tensor(0).unsqueeze(0).to(device))
        loss.backward()
        optimizer_feature_discriminator.step()  # update D_A and D_B's weights


    #########################################################################




    if len(confidence_buffer) < 50:
        confidence_buffer.append(max_sim)

        max_index_np = np.argmax(similarities_np)
        index_lists.append(max_index_np)

        gt_pose_trans = gt_pose_44[i].reshape(4, 4)[:3, 3]
        out_pose = gt_pose_44[max_index_np].reshape(4, 4)
        t_err = np.linalg.norm(gt_pose_trans - out_pose[:3, 3])
        error_list_np.append(t_err)



    elif max_sim > np.mean(confidence_buffer) - 2.5 * np.std(confidence_buffer):
        confidence_buffer.pop(0)
        confidence_buffer.append(max_sim)

        max_index_np = np.argmax(similarities_np)
        index_lists.append(max_index_np)

        gt_pose_trans = gt_pose_44[i].reshape(4, 4)[:3, 3]
        out_pose = gt_pose_44[max_index_np].reshape(4, 4)
        t_err = np.linalg.norm(gt_pose_trans - out_pose[:3, 3])
        error_list_np.append(t_err)
        print('t_err is ', t_err)


    else:
        print('Not OK and continue')

        if i > int(len(dataset_cyclegan.dataset) * 0.05):

            max_index_np = np.argmax(similarities_np)
            index_lists.append(max_index_np)
            top_10_index = np.argsort(similarities_np)[::-1][0:15]

            gt_pose_trans = gt_pose_44[i].reshape(4, 4)[:3, 3]
            out_pose = gt_pose_44[max_index_np].reshape(4, 4)
            t_err = np.linalg.norm(gt_pose_trans - out_pose[:3, 3])
            error_list_np.append(t_err)
            print('t_err is ', t_err)

            start = max(1, i-7)
            end = min(len(dataset_cyclegan_dataset_A_paths)-1, i+7)
            A_index_list = np.arange(start, end)
            B_index_list = top_10_index

            dataset_cyclegan.dataset.A_paths = [dataset_cyclegan_dataset_A_paths[i] for i in A_index_list]
            dataset_cyclegan.dataset.B_paths = [dataset_cyclegan_dataset_B_paths[i] for i in B_index_list]

            GAN_train_length = min(len(A_index_list), len(B_index_list))
            for iter_ in range(1):
                for j, data in enumerate(dataset_cyclegan):
                    if j >= GAN_train_length-4:
                        break

                    cyclegan.set_input(data)  # unpack data from data loader
                    cyclegan.optimize_parameters()  

            confidence_buffer.pop(0)
            confidence_buffer.append(max_sim)
            dataset_cyclegan.dataset.A_paths = [dataset_cyclegan_dataset_A_paths[i] for i in range(len(dataset_cyclegan_dataset_A_paths))]
            dataset_cyclegan.dataset.B_paths = [dataset_cyclegan_dataset_B_paths[i] for i in range(len(dataset_cyclegan_dataset_B_paths))]



    time_duration_list.append(time.time()-start_time)

print('all time is ', np.sum(time_duration_list))

np.save('error_list_np.npy', error_list_np)
