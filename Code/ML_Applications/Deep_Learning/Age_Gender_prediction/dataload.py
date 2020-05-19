import os
import cv2
import re
import glob
import math
import torch
import numpy as np

from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset

from config import parser

age_cls_unit = int(parser['RacNet']['age_cls_unit'])

# distribution of IMDB-WIKi dataset I: IMDB-Wiki
imdb_distr = [25, 63, 145, 54, 46, 113, 168, 232, 455, 556,
                752, 1089, 1285, 1654, 1819, 1844, 2334, 2828,
                3346, 4493, 6279, 7414, 7706, 9300, 9512, 11489,
                10481, 12483, 11280, 13096, 12766, 14346, 13296,
                12525, 12568, 12278, 12694, 11115, 12089, 11994,
                9960, 9599, 9609, 8967, 7940, 8267, 7733, 6292,
                6235, 5596, 5129, 4864, 4466, 4278, 3515, 3644,
                3032, 2841, 2917, 2755, 2853, 2380, 2169, 2084,
                1763, 1671, 1556, 1343, 1320, 1121, 1196, 949,
                912, 710, 633, 581, 678, 532, 491, 428, 367,
                339, 298, 203, 258, 161, 136, 134, 121, 63, 63,
                82, 40, 37, 24, 16, 18, 11, 4, 9]
imdb_distr[age_cls_unit - 1] = sum(imdb_distr[age_cls_unit - 1:])
imdb_distr = imdb_distr[:age_cls_unit]
imdb_distr = np.array(imdb_distr, dtype='float')

# distribution of test dataset: FG-NET
fg_distr = [10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            9, 8, 8, 9, 9, 5, 7, 6, 6, 7, 6, 9, 5, 4, 6, 5, 7, 6, 3, 3, 5, 5, 4, 4, 2,
            3, 5, 2, 2, 2, 3, 2, 3, 3, 2, 2, 2, 0, 0, 1, 0, 1, 3, 1, 1, 0, 0, 0, 1, 0, 0]
fg_distr[age_cls_unit - 1] = sum(fg_distr[age_cls_unit - 1:])
fg_distr = fg_distr[:age_cls_unit]
fg_distr = np.array(fg_distr, dtype='float') + 1

# step 1: correct different distribution between datasets
loss_weight = fg_distr / imdb_distr

# step 2: normalize the weight so that the expected weight for a random sample
#         from training dataset equals to 1, i.e. sum(weight * 1/imdb_distr ) = 1
loss_weight = loss_weight / sum(loss_weight / imdb_distr)

# >>> (loss_weight * 100).astype('int')
# array([1398,  554,  241,  647,  760,  309,  208,  150,   76,   57,   46,
#          32,   27,   21,   19,   18,   14,   12,   10,    7,    4,    3,
#           4,    3,    2,    2,    2,    1,    2,    1,    2,    1,    1,
#           1,    1,    2,    1,    1,    1,    1,    1,    1,    1,    1,
#           1,    2,    1,    1,    1,    2,    1,    2,    2,    2,    2,
#           2,    1,    1,    2,    0])


loss_weight = torch.from_numpy(np.array(loss_weight, dtype='float'))
loss_weight = loss_weight.type(torch.FloatTensor)


class FaceDataset(Dataset):
  """ read images from disk dynamically """

  def __init__(self, datapath, transformer):
    """
    init function
    :param datapath: datapath to aligned folder  
    :param transformer: image transformer
    """
    if datapath[-1] != '/':
      print("[WARNING] PARAM: datapath SHOULD END WITH '/'")
      datapath += '/'
    self.datapath     = datapath
    self.pics         = [f[len(datapath) : ] for f in
                         glob.glob(datapath + "*.jpg")]
    self.transformer  = transformer
    self.age_divde = float(parser['DATA']['age_divide'])
    self.age_cls_unit = int(parser['RacNet']['age_cls_unit'])

    self.age_cls = {x: self.GaussianProb(x) for x in range(1, self.age_cls_unit + 1)}
    self.age_cls_zeroone = {x: self.ZeroOneProb(x) for x in range(1, self.age_cls_unit + 1)}

  def __len__(self):
    return len(self.pics)

  def GaussianProb(self, true, var = 2.5):
    x = np.array(range(1, self.age_cls_unit + 1), dtype='float')
    probs = np.exp(-np.square(x - true) / (2 * var ** 2)) / (var * (2 * np.pi ** .5))
    return probs / probs.max()

  def ZeroOneProb(self, true):
    x = np.zeros(shape=(self.age_cls_unit, ))
    x[true - 1] = 1
    return x


  def __getitem__(self, idx):
    """
    get images and labels
    :param idx: image index 
    :return: image: transformed image, gender: torch.LongTensor, age: torch.FloatTensor
    """
    # read image and labels
    img_name = self.datapath + self.pics[idx]
    img = io.imread(img_name)
    if len(img.shape) == 2: # gray image
      img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    (age, gender) = re.findall(r"([^_]*)_([^_]*)_[^_]*.jpg", self.pics[idx])[0]
    age = max(1., min(float(age), float(self.age_cls_unit)))

    # preprcess images
    if self.transformer:
      img = transforms.ToPILImage()(img)
      image = self.transformer(img)
    else:
      image = torch.from_numpy(img)

    # preprocess labels
    gender = float(gender)
    gender = torch.from_numpy(np.array([gender], dtype='float'))
    gender = gender.type(torch.LongTensor)

    age_rgs_label = torch.from_numpy(np.array([age / self.age_divde], dtype='float'))
    age_rgs_label = age_rgs_label.type(torch.FloatTensor)

    age_cls_label = self.age_cls[int(age)]
    # age_cls_label = self.age_cls_zeroone[int(age)]

    age_cls_label = torch.from_numpy(np.array([age_cls_label], dtype='float'))
    age_cls_label = age_cls_label.type(torch.FloatTensor)

    # image of shape [256, 256]
    # gender of shape [,1] and value in {0, 1}
    # age of shape [,1] and value in [0 ~ 10)
    return image, gender, age_rgs_label, age_cls_label



