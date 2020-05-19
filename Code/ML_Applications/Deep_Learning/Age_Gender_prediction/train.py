import os
import re
import cv2
import time
import copy
import math
import glob
import datetime
import numpy as np
import pandas as pd

from multiprocessing import cpu_count

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import Dataset
from torch.autograd import Variable

from config import config, parser
from align_faces import FaceAligner
from dataload import FaceDataset, loss_weight
from agegenpredmodel import AgeGenPredModel, image_transformer

from collections import OrderedDict


class AgePredModel:
  """ train/test class for age/gender prediction """

  def __init__(self,
               load_best=True,
               model_name='res18_cls70',
               eval_use_only=False,
               new_last_layer=False,
               new_training_process=True):
    """
    :param load_best: if set, load the best weight, else load the latest weights,
                      usually load the best when doing evaluation, and latest when
                      doing training.
    :param model_name: name used for saving model weight and training info.
    :param eval_use_only: if set, model will not load training/testing data, and 
                          change the bahavior of some layers(dropout, batch norm).
    :param new_last_layer: if the model only changed last fully connected layers,
                            if set, only train last fully connected layers at first 
                            2 epochs.
    :param new_training_process: if set, create a new model and start training.
    """
    # init params
    self.model = AgeGenPredModel()
    self.model_name = model_name
    self.use_gpu = torch.cuda.is_available()
    self.transformer = image_transformer()
    self.load_best = load_best
    self.new_train = new_training_process
    self.new_last_layer = new_last_layer
    self.checkpoint_best = config.model + "{}_best.nn".format(model_name.lower())
    self.checkpoint_last = config.model + "{}_last.nn".format(model_name.lower())
    self.csv_path = config.model + self.model_name + ".csv"

    # training details
    self.batch_size = int(parser['TRAIN']['batch_size'])
    self.num_epochs = int(parser['TRAIN']['num_epochs'])
    self.loading_jobs = int(parser['TRAIN']['jobs_to_load_data'])
    self.max_no_reduce = int(parser['TRAIN']['max_no_reduce'])
    self.age_cls_unit = int(parser['RacNet']['age_cls_unit'])
    self.weight_decay = float(parser['TRAIN']['weight_decay'])
    self.age_divide = float(parser['DATA']['age_divide'])
    self.min_lr_rate = float(parser['TRAIN']['min_lr_rate'])
    self.lr_reduce_by = float(parser['TRAIN']['lr_reduce_by'])
    self.lr_rate = float(parser['TRAIN']['init_lr_rate'])

    # reduce loss on gender so the model focus on age pred
    self.reduce_gen_loss = float(parser['TRAIN']['reduce_gen_loss'])
    self.reduce_age_mae = float(parser['TRAIN']['reduce_age_mae'])

    self.weight_loaded = False
    self.age_cls_criterion = nn.BCELoss(weight=loss_weight)
    self.age_rgs_criterion = nn.L1Loss()
    self.gender_criterion = nn.CrossEntropyLoss()
    self.aligner = FaceAligner()

    if self.use_gpu:
      self.model = self.model.cuda()
      self.age_cls_criterion = self.age_cls_criterion.cuda()
      self.age_rgs_criterion = self.age_rgs_criterion.cuda()
      self.gender_criterion = self.gender_criterion.cuda()

    # csv checkpoint details
    columns = ['Timstamp', 'Epoch', 'Phase', 'AGE ACC', 'AGE MAE', 'GEN ACC',
               'BEST AGE ACC', 'BEST AGE MAE', 'BEST GEN ACC', 'Lr_rate']
    self.csv_checkpoint = pd.DataFrame(data=[], columns=columns)
    if not self.new_train and os.path.exists(self.csv_path):
      self.csv_checkpoint = pd.read_csv(self.csv_path)

    # load no training data when evaluation,
    if not eval_use_only:
      self.load_data()

  def load_data(self):
    """
    initiate dataloader processes
    :return: 
    """
    print("[AgePredModel] load_data: start loading...")
    image_datasets = {x: FaceDataset(config.pics + x + '/', self.transformer[x])
                      for x in ['train', 'val']}
    self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       num_workers=self.loading_jobs)
                        for x in ['train', 'val']}
    self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print("[AgePredModel] load_data: Done! Get {} for train and {} for test!"
          .format(self.dataset_sizes['train'],
                  self.dataset_sizes['val']))
    print("[AgePredModel] load_data: loading finished !")

  @staticmethod
  def rand_init_layer(m):
    """
    initialization method
    :param m: torch.module
    :return: 
    """
    if isinstance(m, nn.Conv2d):
      n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
      m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
      m.weight.data.fill_(1)
      m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
      size = m.weight.size()
      fan_out = size[0]  # number of rows
      fan_in = size[1]  # number of columns
      variance = np.sqrt(2.0 / (fan_in + fan_out))
      m.weight.data.normal_(0.0, variance)

  def soft_load_statedic(self, state_dict):
    """
    WARNING: Always run model = nn.DataParallel after this!
    load network parameters in a soft way, the original load_statedic
     func from torch is prone to raise exceptions when mismatch. this 
     function skip all incapatible weights and print the info intead of
     raising a exception.
    :param state_dict: saved dict 
    :return: 
    """
    # remove `module.` prefix when using nn.DataParallel
    new_state_dict = OrderedDict()
    for name, weight in state_dict.items():
      if len(name) >= 7 and name[:7].lower() == 'module.':
        name = name[7:]
      new_state_dict[name] = weight
    state_dict = new_state_dict

    # start loading
    own_state = self.model.state_dict()
    error_layers = []
    for name, param in state_dict.items():
      if name in own_state:
        if isinstance(param, nn.Parameter):
          param = param.data
        try:
          own_state[name].copy_(param)
        except Exception:
          print('[soft_load_statedic] WARNING: incapatible dim found for {} = {} != {}.'
                .format(name, own_state[name].size(), param.size()))
          error_layers.append(name)
      else:
        print('[soft_load_statedic] Unexpected key "{}" in saved state_dict'.format(name))
    missing = set.union(set(own_state.keys()) - set(state_dict.keys()), set(error_layers))
    if len(missing) > 0:
      print('[soft_load_statedic] keys in state_dict: "{}" not loaded!'.format(missing))
    return

  def train_model(self):
    print("[AgePredModel] train_model: Start training...")

    # 1.0.0.0 define Vars
    best_gen_acc = 0.
    best_age_acc = 0.
    best_age_mae = 99.
    not_reduce_rounds = 0

    # 2.0.0.0 init optimizer
    self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                lr=self.lr_rate,
                                weight_decay=self.weight_decay)

    # 3.0.0.0 load weights if possible
    checkpoint_path = self.checkpoint_best if self.load_best else self.checkpoint_last
    if self.new_train:
      print("[new_training_process] NO WEIGHT LOADED!")
    elif os.path.exists(checkpoint_path):
      checkpoint = torch.load(checkpoint_path, map_location=None if self.use_gpu else 'cpu')
      self.soft_load_statedic(checkpoint['state_dic'])
      print("[train_model] Params Loading Finished!")
      self.weight_loaded = True
      try:
        best_gen_acc = checkpoint['best_gen_acc']
        best_age_acc = checkpoint['best_age_acc']
        best_age_mae = checkpoint['best_age_mae']
        # self.lr_rate = checkpoint['lr_rate']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in self.optimizer.param_groups:
          param_group['lr'] = self.lr_rate
        print("[train_model] Load Optimizer Successful!")
      except:
        print("[train_model] ERROR: Loading Params/Optimizer Error!")
    else:
      print("[train_model] Checkpoint Not Found, Train From Scratch!")

    # report model params
    all_params = sum([np.prod(p.size()) for p in self.model.parameters()])
    trainable_params = sum([np.prod(p.size()) for p in
                            filter(lambda p: p.requires_grad, self.model.parameters())])
    print("[AgePredModel] Model has {}k out of {}k trainable params "
          .format(trainable_params // 1000, all_params // 1000))

    # use when having multiple GPUs available
    if torch.cuda.device_count() > 1:
      self.model = nn.DataParallel(self.model)

    # 4.0.0.0 start each epoch
    layer_to_freeze = 0
    for epoch in range(self.num_epochs):
      print('\nStart Epoch {}/{} ...'.format(epoch + 1, self.num_epochs))
      print('-' * 16)

      # automatically freeze some layers on first 2 epochs
      if epoch == 0:
        new_layer_to_freeze = 8  # resnet-18 has 8 modules in pytorch
      elif epoch == 1:
        new_layer_to_freeze = 6
      else:
        new_layer_to_freeze = 0
      if (self.new_last_layer or self.new_train) \
              and layer_to_freeze != new_layer_to_freeze:
        layer_to_freeze = new_layer_to_freeze
        # free some layers
        model = self.model
        if torch.cuda.device_count() > 1:
          model = self.model.module
        for i, child in enumerate(model.resNet.children()):
          requires_grad = i >= int(layer_to_freeze)
          for param in child.parameters():
            param.requires_grad = requires_grad
        # re-define the optimizer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=self.lr_rate,
                                    weight_decay=self.weight_decay)

      # 4.1.0.0 loop over training and validation phase
      for phase in ['train', 'val']:
        # 4.1.1.0 shift train/eval model
        self.model.train(phase == 'train')
        torch.cuda.empty_cache()

        epoch_age_tp = 0.
        epoch_age_mae = 0.
        epoch_gender_tp = 0.
        processed_data = 0

        # 4.1.2.0 iterate over each batch.
        epoch_start_time = time.time()
        for data in self.dataloaders[phase]:
          # 4.1.2.1 get the inputs and labels
          inputs, gender_true, age_rgs_true, age_cls_true = data
          processed_data += inputs.size(0)

          # 4.1.2.2 wrap inputs&oputpus into Variable
          #         NOTE: set voloatile = True when
          #         doing evaluation helps reduce
          #         gpu mem usage.
          volatile = phase == 'val'
          if self.use_gpu:
            inputs = Variable(inputs.cuda(), volatile=volatile)
            gender_true = Variable(gender_true.cuda(), volatile=volatile)
            # age_rgs_true  = Variable(age_rgs_true.cuda(), volatile=volatile)
            age_cls_true = Variable(age_cls_true.cuda(), volatile=volatile)
          else:
            inputs = Variable(inputs, volatile=volatile)
            gender_true = Variable(gender_true, volatile=volatile)
            # age_rgs_true  = Variable(age_rgs_true, volatile=volatile)
            age_cls_true = Variable(age_cls_true, volatile=volatile)

          # 4.1.2.3 zero gradients
          self.optimizer.zero_grad()

          # 4.1.2.4 forward and get outputs
          gender_out, age_out = self.model(inputs)
          _, gender_pred = torch.max(gender_out, 1)
          _, max_cls_pred_age = torch.max(age_out, 1)
          gender_true = gender_true.view(-1)
          age_cls_true = age_cls_true.view(-1, self.age_cls_unit)

          # 4.1.2.5 get the loss
          gender_loss = self.gender_criterion(gender_out, gender_true)
          age_cls_loss = self.age_cls_criterion(age_out, age_cls_true)
          # age_rgs_loss  = self.age_rgs_criterion(age_out, age_rgs_true)

          # *Note: reduce some age loss and gender loss
          #         enforce the model to focuse on reducing
          #         age classification loss
          gender_loss *= self.reduce_gen_loss
          # age_rgs_loss *= self.reduce_age_mae

          # loss = gender_loss + age_rgs_loss + age_cls_loss
          # loss = age_rgs_loss
          loss = age_cls_loss
          loss = gender_loss + age_cls_loss

          gender_loss_perc = 100 * (gender_loss / loss).cpu().data.numpy()[0]
          age_cls_loss_perc = 100 * (age_cls_loss / loss).cpu().data.numpy()[0]
          # age_rgs_loss_perc = 100 * (age_rgs_loss / loss).cpu().data.numpy()[0]

          age_rgs_loss_perc = 0
          # age_cls_loss_perc = 0
          # gender_loss_perc = 0

          # convert cls result to rgs result by weigted sum
          weigh = np.linspace(1, self.age_cls_unit, self.age_cls_unit)
          age_cls_raw = age_out.cpu().data.numpy()
          age_cls_raw = np.sum(age_cls_raw * weigh, axis=1)
          age_rgs_true = age_rgs_true.view(-1)
          age_rgs_true = age_rgs_true.cpu().numpy() * self.age_divide
          age_rgs_loss = np.mean(np.abs(age_cls_raw - age_rgs_true))

          # 4.1.2.6 backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            self.optimizer.step()

          # 4.1.2.7 statistics
          gender_pred = gender_pred.cpu().data.numpy()
          gender_true = gender_true.cpu().data.numpy()
          batch_gender_tp = np.sum(gender_pred == gender_true)

          max_cls_pred_age = max_cls_pred_age.cpu().data.numpy()
          age_cls_true = age_rgs_true
          batch_age_tp = np.sum(np.abs(age_cls_true - max_cls_pred_age) <= 2)  # if true, MAE < 5

          epoch_age_mae += age_rgs_loss * inputs.size(0)
          epoch_age_tp += batch_age_tp
          epoch_gender_tp += batch_gender_tp

          # 4.1.2.8 print info for each bach done
          print("|| {:.2f}% {}/{} || LOSS = {:.2f} || DISTR% {:.0f} : {:.0f} : {:.0f} "
                "|| AMAE/AACC±2/GACC = {:.2f} / {:.2f}% / {:.2f}% "
                "|| LR {} || ETA {:.0f}s || BEST {:.2f} / {:.2f}% / {:.2f}% ||"
                .format(100 * processed_data / self.dataset_sizes[phase],
                        processed_data,
                        self.dataset_sizes[phase],
                        loss.cpu().data.numpy()[0],
                        age_rgs_loss_perc,
                        age_cls_loss_perc,
                        gender_loss_perc,
                        age_rgs_loss,
                        100 * batch_age_tp / inputs.size(0),
                        100 * batch_gender_tp / inputs.size(0),
                        self.lr_rate,
                        (self.dataset_sizes[phase] - processed_data) * (
                        time.time() - epoch_start_time) / processed_data,
                        best_age_mae,
                        100 * best_age_acc,
                        100 * best_gen_acc),
                end='\r')

          # 4.1.2.9 unlink cuda variables and free up mem
          del inputs, gender_true, age_rgs_true, age_cls_true
          del age_rgs_loss, loss  # , gen_loss, age_cls_loss
          del gender_loss_perc, age_cls_loss_perc, age_rgs_loss_perc

        # 4.1.3.0 epoch done
        epoch_gender_acc = epoch_gender_tp / self.dataset_sizes[phase]
        epoch_age_acc = epoch_age_tp / self.dataset_sizes[phase]
        epoch_age_mae /= self.dataset_sizes[phase]

        # 4.1.4.0 print info after each epoch done
        print('\n--{} {}/{} Done! '
              '|| AMAE/AACC±2/GACC = {:.2f} / {:.2f}% / {:.2f}%  '
              '|| COST {:.0f}s'
              .format(phase.upper(),
                      epoch,
                      self.num_epochs,
                      epoch_age_mae,
                      100 * epoch_age_acc,
                      100 * epoch_gender_acc,
                      time.time() - epoch_start_time))

        # 4.1.5.0, save model weights
        if phase == 'val' and epoch_age_mae < best_age_mae:
          best_gen_acc = epoch_gender_acc
          best_age_acc = epoch_age_acc
          best_age_mae = epoch_age_mae
          best_model_wts = copy.deepcopy(self.model.state_dict())
          torch.save({'epoch': epoch,
                      'state_dic': best_model_wts,
                      "best_gen_acc": best_gen_acc,
                      "best_age_acc": best_age_acc,
                      "best_age_mae": best_age_mae,
                      "lr_rate": self.lr_rate,
                      "optimizer": self.optimizer.state_dict()
                      }, self.checkpoint_best)
          not_reduce_rounds = 0
          print("--New BEST FOUND!! || "
                " AMAE/AACC/AACC±2/GACC = {:.2f} / {:.2f}% / {:.2f}%"
                .format(best_age_mae,
                        100 * best_age_acc,
                        100 * best_gen_acc))
        elif phase == 'val':
          not_reduce_rounds += 1
          torch.save({'epoch': epoch,
                      'state_dic': self.model.state_dict(),
                      "best_gen_acc": best_gen_acc,
                      "best_age_acc": best_age_acc,
                      "best_age_mae": best_age_mae,
                      "lr_rate": self.lr_rate,
                      "optimizer": self.optimizer.state_dict()
                      }, self.checkpoint_last)

        # 4.1.6.0 save csv logging file
        try:
          self.csv_checkpoint.loc[len(self.csv_checkpoint)] = [str(datetime.datetime.now()),
                                                               epoch,
                                                               phase,
                                                               epoch_age_acc,
                                                               epoch_age_mae,
                                                               epoch_gender_acc,
                                                               best_age_acc,
                                                               best_age_mae,
                                                               best_gen_acc,
                                                               self.lr_rate]
          self.csv_checkpoint.to_csv(self.csv_path, index=False)
        except:
          print("Error when saving csv files! [tip]: Please check csv column names.")
          print(self.csv_checkpoint.columns)

        # 4.1.7.0 reduce learning rate if nessessary
        if phase == "val" \
                and not_reduce_rounds >= self.max_no_reduce \
                and self.lr_rate > self.min_lr_rate:
          self.lr_rate = max(self.min_lr_rate, self.lr_rate / self.lr_reduce_by)
          print("[reduce_lr_rate] Reduce Learning Rate From {} --> {}"
                .format(self.lr_rate * self.lr_reduce_by, self.lr_rate))
          for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_rate
          not_reduce_rounds = 0

          # 4.2.0.0 train/val loop ends
    # 5.0.0.0 Trainning Completes!
    return self.model

# """
#
#   # evaluate function is just a pruned version of train function
#
#   def evaluate(self):
#     checkpoint_path = self.checkpoint_best if self.load_best else self.checkpoint_last
#     checkpoint = torch.load(checkpoint_path, map_location=None if self.use_gpu else 'cpu')
#     self.soft_load_statedic(checkpoint['state_dic'])
#     self.model.train(mode=False)
#
#     epoch_age_tp = 0.
#     epoch_age_mae = 0.
#     epoch_gender_tp = 0.
#     processed_data = 0
#
#     # 4.1.2.0 Iterate over data.
#     epoch_start_time = time.time()
#     phase = 'val'
#     for data in self.dataloaders[phase]:
#       # 4.1.2.1 get the inputs and labels
#       inputs, gender_true, age_rgs_true, age_cls_true = data
#       processed_data += self.batch_size
#
#       # 4.1.2.2 wrap inputs&oputpus into Variable
#       #         NOTE: set voloatile = True when
#       #         doing evaluation helps reduce
#       #         gpu mem usage.
#       volatile = phase == 'val'
#       if self.use_gpu:
#         inputs = Variable(inputs.cuda(), volatile=volatile)
#         gender_true = Variable(gender_true.cuda(), volatile=volatile)
#         age_cls_true = Variable(age_cls_true.cuda(), volatile=volatile)
#       else:
#         inputs = Variable(inputs, volatile=volatile)
#         gender_true = Variable(gender_true, volatile=volatile)
#         age_cls_true = Variable(age_cls_true, volatile=volatile)
#
#       # 4.1.2.4 forward and get outputs
#       gender_out, age_cls_out = self.model(inputs)
#       _, gender_pred = torch.max(gender_out, 1)
#       _, age_cls_pred = torch.max(age_cls_out, 1)
#       gender_true = gender_true.view(-1)
#       age_cls_true = age_cls_true.view(-1, 99)
#
#       # 4.1.2.5 get loss
#       # print(age_cls_out.size(), age_cls_true.size(), loss_weight.size())
#       gender_loss = self.gender_criterion(gender_out, gender_true)
#       age_cls_loss = self.age_cls_criterion(age_cls_out, age_cls_true)
#       # age_rgs_loss  = self.age_rgs_criterion(age_rgs_pred, age_rgs_true)
#
#       # *Note: reduce some age loss and gender loss
#       #         enforce the model to focuse on reducing
#       #         age classification loss
#       gender_loss *= self.reduce_gen_loss
#       # age_rgs_loss *= self.reduce_age_mae
#
#       # loss = gender_loss + age_rgs_loss + age_cls_loss
#       # loss = age_cls_loss
#       loss = gender_loss + age_cls_loss
#
#       gender_loss_perc = 100 * (gender_loss / loss).cpu().data.numpy()[0]
#       age_cls_loss_perc = 100 * (age_cls_loss / loss).cpu().data.numpy()[0]
#       # age_rgs_loss_perc = 100 * (age_rgs_loss / loss).cpu().data.numpy()[0]
#
#       age_rgs_loss_perc = 0
#       weigh = np.linspace(1, 99, 99)
#       age_cls_raw = age_cls_out.cpu().data.numpy()
#       age_cls_raw = np.sum(age_cls_raw * weigh, axis=1)
#       age_rgs_true = age_rgs_true.view(-1)
#       age_rgs_true = age_rgs_true.cpu().numpy() * self.age_divide
#       age_rgs_loss = np.mean(np.abs(age_cls_raw - age_rgs_true))
#
#       # 4.1.2.7 statistics
#       gender_pred = gender_pred.cpu().data.numpy()
#       gender_true = gender_true.cpu().data.numpy()
#       batch_gender_tp = np.sum(gender_pred == gender_true)
#
#       age_cls_pred = age_cls_pred.cpu().data.numpy()
#       age_cls_true = age_rgs_true
#       batch_age_tp = np.sum(np.abs(age_cls_true - age_cls_pred) <= 2)  # if true, MAE < 5
#
#       epoch_age_mae += age_rgs_loss * inputs.size(0)
#       epoch_age_tp += batch_age_tp
#       epoch_gender_tp += batch_gender_tp
#
#       # 4.1.2.8 print info for each bach done
#       print("|| {:.2f}% {}/{} || LOSS = {:.2f} || DISTR% {:.0f} : {:.0f} : {:.0f} "
#             "|| AMAE/AACC±2/GACC = {:.2f} / {:.2f}% / {:.2f}% "
#             "|| LR {} || ETA {:.0f}s "
#             .format(100 * processed_data / self.dataset_sizes[phase],
#                     processed_data,
#                     self.dataset_sizes[phase],
#                     loss.cpu().data.numpy()[0],
#                     age_rgs_loss_perc,
#                     age_cls_loss_perc,
#                     gender_loss_perc,
#                     age_rgs_loss,
#                     # self.age_divide * age_rgs_loss.cpu().data.numpy()[0],
#                     100 * batch_age_tp / inputs.size(0),
#                     100 * batch_gender_tp / inputs.size(0),
#                     self.lr_rate,
#                     (self.dataset_sizes[phase] - processed_data) * (time.time() - epoch_start_time) / processed_data,
#             end='\r'))
#
#       # 4.1.2.9 unlink cuda variables and free up mem
#       del inputs, gender_true, age_rgs_true, age_cls_true
#       del age_cls_loss, age_rgs_loss, loss  # , gen_loss
#       del gender_loss_perc, age_cls_loss_perc, age_rgs_loss_perc
#
#     # 4.1.3.0 epoch done
#     epoch_gender_acc = epoch_gender_tp / self.dataset_sizes[phase]
#     epoch_age_acc = epoch_age_tp / self.dataset_sizes[phase]
#     epoch_age_mae /= self.dataset_sizes[phase]
#
#     # 4.1.4.0 print info after each epoch done
#     print('\n--{} Done! '
#           '|| AMAE/AACC±2/GACC = {:.2f} / {:.2f}% / {:.2f}%  '
#           '|| COST {:.0f}s'
#           .format(phase.upper(),
#                   epoch_age_mae,
#                   # self.age_divide * epoch_age_mae,
#                   100 * epoch_age_acc,
#                   100 * epoch_gender_acc,
#                   time.time() - epoch_start_time))
# """

  def getAgeGender(self,
                   img,
                   transformed=False,
                   return_all_faces=True,
                   return_info=False):
    """
    evaluation/test funtion
    :param img: str or numpy array represent the image
    :param transformed: if the image is transformed into standarlized pytorch image.
            applicable when using this in train loop
    :param return_all_faces: if set, return prediction results of all faces detected.
            set to False if it's known that all images comtain only 1 face
    :param return_info: if set, return a list of rects (x, y, w, h) represents loc of faces
    :return: a list of [gender_pred, age_pred]
    """
    # load model params
    if not self.weight_loaded:
      path = self.checkpoint_best if self.load_best else self.checkpoint_last
      checkpoint = torch.load(path, map_location='gpu' if self.use_gpu else 'cpu')
      self.soft_load_statedic(checkpoint['state_dic'])
      # self.model.load_state_dict(checkpoint['state_dic'])
      self.model.train(False)
      self.weight_loaded = True

    # load images if not provided
    if type(img) == str:
      img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)

    # get faces and rects
    aligned = self.aligner.getAligns(img, return_info=return_info)
    if return_info:
      aligned, rects, scores = aligned
    if not len(aligned):  # no face detected
      scores = [1]
      rects = [(0, 0, img.shape[0], img.shape[1])]
      faces = [img]
    else:
      faces = aligned
    if not return_all_faces:
      faces = faces[0]
    faces = [transforms.ToPILImage()(fc) for fc in faces]
    if not transformed:
      faces = [self.transformer['val'](fc) for fc in faces]

    # get predictions of each face
    preds = self.model.evaluate(faces)

    if return_info:
      return preds, rects, scores
    return preds


if __name__ == "__main__":
  a = AgePredModel(model_name='res18_cls70',
                   new_training_process=False,
                   new_last_layer=True)
  a.train_model()
  # a.evaluate()
  # print(a.getAgeGender(config.val + "6_0_MurderofElisaIzquierdo.jpg"))
  # a.divideTrainVal()
  # a.img2matrix()
  # face_dataset = FaceDataset()
  # print(face_dataset[1])





