import torch
import copy
from clip import clip
from PIL import Image, ImageOps
from tqdm import tqdm
import json
from collections import defaultdict


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
for _, param in model.named_parameters():
    param.requires_grad = False

# Json and COCO dataset dir path
json_path = 'xxx/instances_train2017_seen_2_proposal.json'
file_dir = "xxx/coco/train2017/"
save_path = "xxx/coco/zero-shot/clip_feat.pkl"

with open(json_path, "r") as f:
    data = json.load(f)

img2ann_gt = defaultdict(list)
for temp in data['annotations']:
    img2ann_gt[temp['image_id']].append(temp)

dic = {}
for image_id in tqdm(img2ann_gt.keys()):
    file_name = file_dir + f"{image_id}".zfill(12) + ".jpg"
    image = Image.open(file_name).convert("RGB")
    
    for value in img2ann_gt[image_id]:
        ind = value['id']
        bbox = copy.deepcopy(value['bbox'])
        if (bbox[1] < 16) or (bbox[2] < 16):
            continue
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        roi = preprocess(image.crop(bbox)).to(device).unsqueeze(0)
        roi_features = model.encode_image(roi)
        
        category_id = value['category_id']

        if category_id in dic.keys():
            dic[category_id].append(roi_features)
        else:
            dic[category_id] = [roi_features]


for key in dic.keys():
    dic[key] = torch.cat(dic[key], 0)

torch.save(dic, save_path)
