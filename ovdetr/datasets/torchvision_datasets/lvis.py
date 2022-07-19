import os
import os.path
from io import BytesIO

import tqdm
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class LvisDetection(VisionDataset):
    """`LVIS Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root,
        annFile,
        transform=None,
        target_transform=None,
        transforms=None,
        cache_mode=False,
        local_rank=0,
        local_size=1,
    ):
        super(LvisDetection, self).__init__(root, transforms, transform, target_transform)
        from lvis import LVIS

        self.lvis = LVIS(annFile)
        self.ids = list(sorted(self.lvis.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.lvis.load_imgs(img_id)[0]["file_name"]
            with open(os.path.join(self.root, path), "rb") as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), "rb") as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert("RGB")
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def __getitem__(self, index):
        lvis = self.lvis
        img_id = self.ids[index]
        ann_ids = lvis.get_ann_ids(img_ids=[img_id])
        target = lvis.load_anns(ann_ids)

        split_folder, file_name = lvis.load_imgs([img_id])[0]["coco_url"].split("/")[-2:]
        path = os.path.join(split_folder, file_name)

        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
