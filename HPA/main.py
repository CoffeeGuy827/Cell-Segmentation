import os 
import time 
import json 
import torch 
import random 
import warnings
import torchvision
import numpy as np 
import pandas as pd 

import sys
from utils import *
from data import HumanDataset
from tqdm import tqdm 
from config import config
from datetime import datetime
from models.model import*
from torch import nn,optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import f1_score

path_to_data = sys.argv[1]

config.train_data = path_to_data+"train/"
config.best_models = sys.argv[2]



# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')




if not os.path.exists("./logs/"):
    os.mkdir("./logs/")

log = Logger()
log.open("logs/%s_log_train.txt"%config.model_name,mode="a")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |------------ Train -------------|----------- Valid -------------|----------Best Results---------|------------|\n')
log.write('mode     iter     epoch    |         loss   f1_macro        |         loss   f1_macro       |         loss   f1_macro       | time       |\n')
log.write('-------------------------------------------------------------------------------------------------------------------------------\n')


# 2. evaluate fu
def main():
    fold = 0
    # 4.1 mkdirs
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)
    if not os.path.exists(config.weights + config.model_name + os.sep +str(fold)):
        os.makedirs(config.weights + config.model_name + os.sep +str(fold))
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")


    resume = False
    all_files = pd.read_csv(path_to_data+"train.csv")
    #print(all_files)
    train_data_list,val_data_list = train_test_split(all_files,test_size = 0.13,random_state = 2050)

    # load dataset
    train_gen = HumanDataset(train_data_list,config.train_data,mode="train")
    train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=4)

    val_gen = HumanDataset(val_data_list,config.train_data,augument=False,mode="train")
    val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=True,num_workers=4)


        
if __name__ == "__main__":
    main()
