import os
import time
import configparser

def time_count(fn):
  """
  Funtion wrapper used to memsure time consumption
  :param fn: function pointer
  :return: wrapper
  """
  def _wrapper(*args, **kwargs):
    start = time.clock()
    result = fn(*args, **kwargs)
    print(">>[Time Count]: Funtion '%s' Costs %fs" % (fn.__name__, time.clock() - start))
    return result
  return _wrapper


def CreatPathIfNotExists(fn):
  """
  function wrapper to check if returned path/dir exists,
  if not, create the dir
  :param fn: function pointer
  :return: wrapper
  """
  def _wrapper(*args, **kwargs):
    result = fn(*args, **kwargs)
    if not os.path.exists(result):
      os.makedirs(result)
    return result
  return _wrapper


class Config:

  def __init__(self):
    self.root = os.getcwd() + '/'
    self.parser  = configparser.ConfigParser()
    self.parser.read(self.root + 'config.ini')

  @property
  @CreatPathIfNotExists
  def model(self):
    return self.root + parser['PATH']['model']

  @property
  @CreatPathIfNotExists
  def pics(self):
    return self.root + parser['PATH']['pics']

  @property
  @CreatPathIfNotExists
  def wiki_raw(self):
    return self.root + parser['PATH']['wiki_raw']

  @property
  @CreatPathIfNotExists
  def fgnet_raw(self):
    return self.root + parser['PATH']['fgnet_raw']

  @property
  @CreatPathIfNotExists
  def labeled(self):
    return self.root + parser['PATH']['labeled']

  @property
  @CreatPathIfNotExists
  def named(self):
    return self.root + parser['PATH']['named']

  @property
  @CreatPathIfNotExists
  def imdb_raw(self):
    return self.root + parser['PATH']['imdb_raw']

  @property
  @CreatPathIfNotExists
  def aligned(self):
    return self.root + parser['PATH']['aligned']

  @property
  @CreatPathIfNotExists
  def train(self):
    return self.root + parser['PATH']['train_folder']

  @property
  @CreatPathIfNotExists
  def val(self):
    return self.root + parser['PATH']['val_folder']

  @property
  def NetworkParamsParser(self):
    return self.parser['RacNet']

  @property
  def TrainingParamsParser(self):
    return self.parser['TRAIN']

  @property
  def DataProcessParamsParser(self):
    return self.parser['DATA']

  @property
  def ConfigParser(self):
    return self.parser['CONFIG']


config = Config()
parser = config.parser

















