
import json
import glob
import cv2
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer
import numpy as np
from PIL import Image
import random
import time
import csv
import traceback

class BaseDataset(Dataset):
    def _try_getitem(self, idx):
        raise NotImplementedError
    def __getitem__(self, idx):
        wait = 0.1
        while True:
            try:
                ret = self._try_getitem(idx)
                return ret
            except KeyboardInterrupt:
                break
            except (Exception, BaseException) as e: 
                exstr = traceback.format_exc()
                print(exstr)
                print('read error, waiting:', wait)
                time.sleep(wait)
                wait = min(wait*2, 1000)

class TranslationDataset(BaseDataset):
    def __init__(self, data_file, input_l, output_l, sos_id=1, eos_id=2, pad_id=0):
        with open(data_file, 'r') as fp:
            reader = csv.reader(fp)
            self.samples = [row for row in reader]
            self.input_l = input_l
            self.output_l = output_l
            self.sos_id = sos_id
            self.pad_id = pad_id
            self.eos_id = eos_id
    def __len__(self):
        return len(self.samples)
    def _try_getitem(self, idx):
        source = [int(x) for x in self.samples[idx][1].split()]
        if len(source)<self.input_l:
            source.extend([self.pad_id] * (self.input_l-len(source)))
        if len(self.samples[idx])<3:
            return np.array(source)[:self.input_l]
        target = [self.sos_id] + [int(x) for x in self.samples[idx][2].split()] + [self.eos_id]
        if len(target)<self.output_l:
            target.extend([self.pad_id] * (self.output_l-len(target)))
        return np.array(source)[:self.input_l], np.array(target)[:self.output_l]
