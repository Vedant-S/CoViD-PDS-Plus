import os
import cv2
import re
import time
import random
import shutil
import glob
import scipy.io

import numpy as np

from shutil import copyfile
from datetime import datetime, timedelta
from multiprocessing import Pool

from config import config, parser
from align_faces import FaceAligner

parser = parser['DATA']

def parse_matlab_date(x):
  """
  :param x: date string in matlab format
  :return: int, year
  """
  x, date = int(x), -1
  try:
    date = (datetime.fromordinal(int(x))
          + timedelta(days=x % 1)
          - timedelta(days=366)).year
  except:
    print("[convertMatlabDate] Failed to parse string {}".format(x))
  return date


def clear_dir(path):
  """
  remove all files in this directionary
  :param path: path to dir
  :return: 
  """
  if os.path.exists(path):
    shutil.rmtree(path)
    os.mkdir(path)
  return


def addlabels(data = 'wiki', clean = False):
  """
  move pictures to labled dir and rename to [Age]_[Gender]_[Name].jpg format
  :param data: 'wiki' or 'imdb'
  :return: 
  """
  # 1, clean previous
  origin_dir = config.wiki_raw if data == 'wiki' else config.imdb_raw
  if clean: clear_dir(config.labeled)

  # 2, read meta data
  mat = scipy.io.loadmat(origin_dir + data + '.mat')[data][0][0]

  # records
  no_face_image = 0
  multiple_face_image = 0
  wrong_age = 0
  wrong_gender = 0
  successful = 0
  for dob, dop, path, gender, name, face_score, face_score2 \
          in zip(mat[0][0], mat[1][0], mat[2][0], mat[3][0], mat[4][0], mat[6][0], mat[7][0]):
    if face_score < 0 or not np.isnan(face_score2):
      if face_score < 0: no_face_image += 1
      if not np.isnan(face_score2): multiple_face_image += 1
      continue

    age = dop - parse_matlab_date(dob)
    if age < int(parser['age_lower']) or age > int(parser['age_upper']):
      wrong_age += 1
      continue

    if gender not in [1.0, 0.0]:
      wrong_gender += 1
      continue

    newName = "{}_{}_{}.jpg".format(age,
                                    int(gender),
                                    name[0]
                                    .replace(' ', '')
                                    .replace('/', '')
                                    .replace(':', ''))
    # 2.1 check duplicate
    # 2.1 if duplicate exist, append a random number to it name
    newNameNoDupli = newName
    while os.path.exists(config.labeled + newNameNoDupli):
      newNameNoDupli = "{}{}{}".format(newName[:-4], random.randint(1, 9999), newName[-4:])
    # 2.2 save as a new file
    copyfile(origin_dir + path[0], config.labeled + newNameNoDupli)
    successful += 1
  print("{} Successful, {} no_face_image, {} multiple_face_image, {} wrong_age, {} wrong_gender"
        .format(successful, no_face_image, multiple_face_image, wrong_age, wrong_gender))
  return


# sort photos by their names
def sort_out_by_name(clean = False):
  pwd = os.getcwd()
  if clean:
    clear_dir(config.named)
  os.chdir(config.aligned)
  for img in glob.glob("*.jpg"):
    name = re.findall(r'[^_]*_[^_]*_([\D]*)[0-9]*.jpg', img)
    if not len(name): continue
    name = name[0].lower()
    if not os.path.exists(config.named + name + '/'):
      os.mkdir(config.named + name + '/')
    copyfile(img, config.named + name + '/' + img)
  os.chdir(pwd)



# TODO: any other ways to get around this public variable?
FL = FaceAligner()
def sub_align_face(picname):
  """
  sub thread function to get and store aligned faces
  :param picname: pic names
  :return: 
  """
  aligned = FL.getAligns(picname)
  if len(aligned) == 0:
    return
    # copyfile(picname, config.aligned + picname)
  cv2.imwrite(config.aligned + picname, aligned[0])


def creat_fgnet_val(clean = False):
  if clean:
    clear_dir(config.val)
  pwd = os.getcwd()
  os.chdir(config.fgnet_raw)
  for pic in glob.glob("*"):
    name, age = re.findall(r'(\d)*A(\d*).*', pic)[0]
    newName = "{}_1_{}.jpg".format(age,
                                    name[0]
                                    .replace(' ', '')
                                    .replace('/', '')
                                    .replace(':', ''))
    # 2.1 check duplicate
    # 2.1 if duplicate exist, append a random number to it name
    newNameNoDupli = newName
    while os.path.exists(config.labeled + newNameNoDupli):
      newNameNoDupli = "{}{}{}".format(newName[:-4], random.randint(1, 9999), newName[-4:])
    # 2.2 save as a new file
    copyfile(config.fgnet_raw + pic, config.val + newNameNoDupli)
  os.chdir(pwd)


def align_faces(clean = False):
  """
  get aligned faces from labeled folder and store it in aligned folder for training
  :param data: 'wiki' or 'imdb'
  :param clean: if set, clean aligned folder, else append or rewrite to it
  :return: 
  """
  if clean: clear_dir(config.aligned)
  os.chdir(config.labeled)
  jobs = glob.glob("*.jpg")

  # un-parallel
  # for picname in jobs:
  #   aligned = FL.getAligns(picname)
  #   if len(aligned) != 1: return
  #   cv2.imwrite(config.aligned + picname, aligned[0])

  # parallel
  with Pool() as pool:
    try:
      pool.map(sub_align_face, jobs)
    finally:
      pool.close()
  return



def sub_divideTrainVal(img):
  """
  distribute images randomly to train or test foled by 95% train prob
  :param img: image path
  :return: 
  """
  if np.random.rand() < float(parser['train_test_div']):
    copyfile(config.aligned + img, config.train + img)
  else:
    copyfile(config.aligned + img, config.val + img)
  return

def divideTrainVal():
  """
  distribute images randomly to train or test foled by 95% train prob
  :return: 
  """
  pwt = os.getcwd()
  os.chdir(config.aligned)

  # clean
  clear_dir(config.train)
  clear_dir(config.val)

  # read into mem
  # train, val = [], []

  # parallel
  with Pool() as pool:
    try:
      pool.map(sub_divideTrainVal, glob.glob("*.jpg"))
    finally:
      pool.close()

  # for img in glob.glob("*.jpg"):
  #   if np.random.rand() < float(parser['train_test_div']):
  #     cv2.imwrite(config.train + img, cv2.imread(img))
      # train.append([cv2.imread(img), img])
    # else:
    #   cv2.imwrite(config.val + img, cv2.imread(img))
      # val.append([cv2.imread(img), img])

  # dump out of mem
  # for img, name in train:
  #   cv2.imwrite(config.train + img, name)
  # for img, name in val:
  #   cv2.imwrite(config.val + img, name)
  os.chdir(pwt)
  return


if __name__ == "__main__":
  print("labeling..")
  addlabels(data='imdb', clean=True)
  
  print("aligning..")
  align_faces(clean = True)
  
  print("dividing..")
  divideTrainVal()

#   creat_fgnet_val(clean=True)
  pass












