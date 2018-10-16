#!-*-coding:utf-8-*-
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np

def get_rmse_log(pred, label):
  clipped_pred = np.clip(pred, 1, float('inf'))
  rmse = np.sqrt(np.mean(np.square(np.log(clipped_pred) - np.log(label))))
  
  return rmse
