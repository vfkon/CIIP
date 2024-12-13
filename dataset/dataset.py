import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import random
import numpy as np
import tifffile as tiff
import pandas as pd

import os
import glob
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch.utils.data as data

# from .util.mask import (bbox2mask, brush_stroke_mask,
#                         get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

# mapping from igbp to dfc2020 classes
DFC2020_CLASSES = [
    0,  # class 0 unused in both schemes
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    3,  # --> will be masked if no_savanna == True
    3,  # --> will be masked if no_savanna == True
    4,
    5,
    6,  # 12 --> 6
    7,  # 13 --> 7
    6,  # 14 --> 6
    8,
    9,
    10
    ]

# indices of sentinel-2 high-/medium-/low-resolution bands
S2_BANDS_HR = [2, 3, 4, 8]
S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
S2_BANDS_LR = [1, 10, 11]


# util function for reading s2 data
def load_s2(path, use_hr, use_mr, use_lr):
    bands_selected = []
    if use_hr:
        bands_selected = bands_selected + S2_BANDS_HR
    if use_mr:
        bands_selected = bands_selected + S2_BANDS_MR
    if use_lr:
        bands_selected = bands_selected + S2_BANDS_LR
    bands_selected = sorted(bands_selected)
    with rasterio.open(path) as data:
        s2 = data.read(bands_selected)
    s2 = s2.astype(np.float32)
    s2 = np.clip(s2, 0, 10000)
    s2 /= 10000
    s2 = s2.astype(np.float32)
    s2 = torch.tensor(s2)
    mean = torch.as_tensor([0.5]*len(bands_selected),
                           dtype=s2.dtype, device=s2.device)
    std = torch.as_tensor([0.5]*len(bands_selected),
                          dtype=s2.dtype, device=s2.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    s2.sub_(mean).div_(std)
    return s2


# util function for reading s1 data
def load_s1(path):
    with rasterio.open(path) as data:
        s1 = data.read()
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    s1 /= 25
    s1 += 1
    s1 = s1.astype(np.float32)
    s1 = torch.tensor(s1, dtype = torch.float32)
    mean = torch.as_tensor([0.5, 0.5],
                           dtype=s1.dtype, device=s1.device)
    std = torch.as_tensor([0.5, 0.5],
                          dtype=s1.dtype, device=s1.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    s1.sub_(mean).div_(std)
    return s1


# util function for reading lc data
def load_lc(path, no_savanna=False, igbp=True):

    # load labels
    with rasterio.open(path) as data:
        lc = data.read(1)

    # convert IGBP to dfc2020 classes
    if igbp:
        lc = np.take(DFC2020_CLASSES, lc)
    else:
        lc = lc.astype(np.int64)

    # adjust class scheme to ignore class savanna
    if no_savanna:
        lc[lc == 3] = 0
        lc[lc > 3] -= 1

    # convert to zero-based labels and set ignore mask
    lc -= 1
    lc[lc == -1] = 255
    return lc


# util function for reading data from single sample
def load_sample(sample, use_s1, use_s2hr, use_s2mr, use_s2lr, use_s2_cr,
                no_savanna=False, igbp=True, unlabeled=False):

    use_s2 = use_s2hr or use_s2mr or use_s2lr
    #return_dict = {}
    # load s2 data
    if use_s2:
        #return_dict['image'] = load_s2(sample["s2"], use_s2hr, use_s2mr, use_s2lr)
        image = load_s2(sample["s2"], use_s2hr, use_s2mr, use_s2lr)
    else:
        image = None
    if use_s2_cr:
        #return_dict['image_cr'] = load_s2(sample["s2cr"], use_s2hr, use_s2mr, use_s2lr)
        image_cr = load_s2(sample["s2cr"], use_s2hr, use_s2mr, use_s2lr)
    else:
        image_cr = None
    # load s1 data
    if use_s1:
        #return_dict['image_s1'] = load_s1(sample["s1"])
        image_sar = load_s1(sample["s1"])
    else:
        image_sar = None
    # load label
    #return_dict['id'] = sample["id"]
    if not unlabeled:
        label = load_lc(sample["label"], no_savanna=no_savanna, igbp=igbp)
    else:
        label = None
    return image_cr, image_sar, image, label



# calculate number of input channels
def get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr):
    n_inputs = 0
    if use_s2hr:
        n_inputs += len(S2_BANDS_HR)
    if use_s2mr:
        n_inputs += len(S2_BANDS_MR)
    if use_s2lr:
        n_inputs += len(S2_BANDS_LR)
    if use_s1:
        n_inputs += 2
    return n_inputs


# select channels for preview images
def get_display_channels(use_s2hr, use_s2mr, use_s2lr):
    if use_s2hr and use_s2lr:
        display_channels = [3, 2, 1]
        brightness_factor = 3
    elif use_s2hr:
        display_channels = [0, 1, 2]
        brightness_factor = 3
    elif not (use_s2hr or use_s2mr or use_s2lr):
        display_channels = 0
        brightness_factor = 1
    else:
        display_channels = 0
        brightness_factor = 3
    return (display_channels, brightness_factor)



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(
            dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')


# calculate number of input channels
def get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr):
    n_inputs = 0
    if use_s2hr:
        n_inputs += len(S2_BANDS_HR)
    if use_s2mr:
        n_inputs += len(S2_BANDS_MR)
    if use_s2lr:
        n_inputs += len(S2_BANDS_LR)
    if use_s1:
        n_inputs += 2
    return n_inputs


# select channels for preview images
def get_display_channels(use_s2hr, use_s2mr, use_s2lr):
    if use_s2hr and use_s2lr:
        display_channels = [3, 2, 1]
        brightness_factor = 3
    elif use_s2hr:
        display_channels = [2, 1, 0]
        brightness_factor = 3
    elif not (use_s2hr or use_s2mr or use_s2lr):
        display_channels = 0
        brightness_factor = 1
    else:
        display_channels = 0
        brightness_factor = 3
    return (display_channels, brightness_factor)


class SEN12MS(data.Dataset):
    """PyTorch dataset class for the SEN12MS dataset"""
    # expects dataset dir as:
    #       - SEN12MS_holdOutScenes.txt
    #       - ROIsxxxx_y
    #           - lc_n
    #           - s1_n
    #           - s2_n
    #
    # SEN12SEN12MS_holdOutScenes.txt contains the subdirs for the official
    # train/val split and can be obtained from:
    #   https://github.com/MSchmitt1984/SEN12MS/blob/master/splits

    def __init__(self, path, mode="train", no_savanna=False, use_s2hr=True,
                 use_s2mr=False, use_s2lr=False, use_s2cr=True, use_s1=True):
        """Initialize the dataset"""

        # inizialize
        super(SEN12MS, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s2cr = use_s2cr
        self.use_s1 = use_s1
        self.no_savanna = no_savanna
        assert mode in ["train", "val"]

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr)

        # provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = get_display_channels(
                                                            use_s2hr,
                                                            use_s2mr,
                                                            use_s2lr)

        # provide number of classes
        if no_savanna:
            self.n_classes = max(DFC2020_CLASSES) - 1
            self.no_savanna = True
        else:
            self.n_classes = max(DFC2020_CLASSES)
            self.no_savanna = False

        # make sure parent dir exists
        assert os.path.exists(path)

        # find and index samples
        self.samples = []
        if mode == "train":
            pbar = tqdm(total=162556)   # we expect 541,986 / 3 * 0.9 samples
        else:
            pbar = tqdm(total=18106)   # we expect 541,986 / 3 * 0.1 samples
        pbar.set_description("[Load]")

        val_list = list(pd.read_csv(os.path.join(path,
                                                 "SEN12MS_holdOutScenes.txt"),
                                    header=None)[0])
        val_list = [x.replace("s1_", "s2_") for x in val_list]
        val_list = [x.replace('_s1', '_s2') for x in val_list]
        # compile a list of paths to all samples
        if mode == "train":
            train_list = []
            for seasonfolder in ['ROIs1970_fall_s2', 'ROIs1158_spring_s2',
                                 'ROIs2017_winter_s2', 'ROIs1868_summer_s2']:
                train_list += [os.path.join(seasonfolder, x) for x in
                               os.listdir(os.path.join(path, seasonfolder))]
            train_list = [x for x in train_list if "s2_" in x]
            train_list = [x for x in train_list if x not in val_list]
            sample_dirs = train_list
        elif mode == "val":
            sample_dirs = val_list

        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"),
                                     recursive=True)

            # INFO there is one "broken" file in the sen12ms dataset with nan
            #      values in the s1 data. we simply ignore this specific sample
            #      at this point. id: ROIs1868_summer_xx_146_p202
            if folder == "ROIs1868_summer/s2_146":
                broken_file = os.path.join(path, "ROIs1868_summer",
                                           "s2_146",
                                           "ROIs1868_summer_s2_146_p202.tif")
                s2_locations.remove(broken_file)
                pbar.write("ignored one sample because of nan values in "
                           + "the s1 data")

            for s2_loc in s2_locations:
                s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_").replace('_s2',
                                                                                                      '_s1')
                s2cr_loc = s1_loc.replace('_s1', '_s2_cloudy').replace("_s1_", "_s2_cloudy_").replace("s1_", "s2_cloudy_")
                lc_loc = s2_loc.replace('_s2', '_lc').replace("_s2_", "_lc_").replace("s2_", "lc_")

                pbar.update()
                self.samples.append({"label": lc_loc, "s1": s1_loc, "s2": s2_loc, "s2cr": s2cr_loc,
                                     "id": os.path.basename(s2_loc)})

        pbar.close()

        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])

        print("loaded", len(self.samples),
              "samples from the sen12ms subset", mode)

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        image_cloud, image_sar, image_clear, image_label = load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                           self.use_s2lr, self.use_s2cr, no_savanna=self.no_savanna,
                           igbp=True, unlabeled=True)
        ret = {}
        ret['gt_image'] = image_clear[[2,1,0],:,:]
        #ret['cond_image_sar'] = image_sar
        ret['cond_image'] = torch.cat([image_cloud[[2,1,0],:,:], image_sar], dim = 0)
        ret['path'] = sample['id']
        return ret


    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)
