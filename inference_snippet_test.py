
import numpy as np

import requests
import os
#import torchvision.models as models
#from model.model_b3_p import Reab3p16
import model_adjusted.model_b3_p as m
#import model.model_b3_p as m
import torch 
import os
import tensorflow as tf 
import argparse
import zipfile
from data.load_data import load_data,Dataset
from data import preprocess 
import config
from torch.autograd import Variable
import torch.optim as optim
from collections import OrderedDict

import config

# Adjusted from original load_data from github repository.
def load_data_for_inference(args, data_dir, data_split, filename):
    data_files = []
    # data_dir = '../process_data/reason_data/reason_data/RAVEN-10000/'
    # data_dir = "C:/Users/Hertz/Documents/SJSU Coursework/MS Project_big files/RAVEN 1000/center_single"
    # filename = "RAVEN_0_train.npz"
    # for subdir in os.listdir(data_dir):
    #     for filename in os.listdir(data_dir + subdir):
    #         if "npz" in filename:
    #             data_files.append(data_dir + subdir + "/" + filename)
    data_files.append(data_dir + "/" + filename)


    df = [data_file for data_file in data_files if data_split in data_file and "npz" in data_file][:]


    print("df", df)
    #data_files = [data_file for data_file in data_files if data_split in data_file]
    print("Nums of "+data_split+" : ", len(df))
    # train_loader = torch.utils.data.DataLoader(Dataset(train_files), batch_size=args.batch_size, shuffle=True,num_workers=args.numwork)#
    loader = torch.utils.data.DataLoader(Dataset(args,df), batch_size=args.batch_size, num_workers=args.numwork)
    return loader

# original load_data from original github repository. 
def load_data(args, data_split, data_dir):
    data_files = []
    # data_dir = '../process_data/reason_data/reason_data/RAVEN-10000/'
    data_dir = "C:/Users/Hertz/Documents/SJSU Coursework/MS Project_big files/RAVEN 1000/center_single"
    # filename = "RAVEN_0_train.npz"
    # for subdir in os.listdir(data_dir):
    #     for filename in os.listdir(data_dir + subdir):
    #         if "npz" in filename:
    #             data_files.append(data_dir + subdir + "/" + filename)
	
    for filename in os.listdir(data_dir):
        if "npz" in filename:
            data_files.append(data_dir + "/" + filename)
    data_files.append(data_dir + "/" + filename)


    df = [data_file for data_file in data_files if data_split in data_file and "npz" in data_file][:]


    print("df", df)
    #data_files = [data_file for data_file in data_files if data_split in data_file]
    print("Nums of "+data_split+" : ", len(df))
    # train_loader = torch.utils.data.DataLoader(Dataset(train_files), batch_size=args.batch_size, shuffle=True,num_workers=args.numwork)#
    loader = torch.utils.data.DataLoader(Dataset(args,df), batch_size=args.batch_size, num_workers=args.numwork)
    return loader

def run_inference(data_dir, data_split, filename):
    torch.multiprocessing.freeze_support()
    device = torch.device('cpu')
    
    weights_path = "C:\\Users\\Hertz\\Documents\\SJSU Coursework\\MS Project_big files\\git\\distracting_feature\\distracting_feature\\epochs\\epoch860"
    
    weights_path = "C:\\Users\\Hertz\\Documents\\SJSU Coursework\\MS Project_big files\\git\\distracting_feature\\distracting_feature\\epochs\\epoch860" # got ~20 percent here. 
    
    weights_path = "C:\\Users\\Hertz\\Documents\\SJSU Coursework\\MS Project_big files\\git\\distracting_feature\\distracting_feature\\epochs\\epochbest(200K_79.2)"
    
    #model_path = "C:\\Users\\Hertz\\Documents\\SJSU Coursework\\MS Project_big files\\git\\distracting_feature\\distracting_feature\\epochs\\epoch860" # got ~70 percent here. 

    #image_path = "C:\\Users\\sonam\\Desktop\\MS_Project\\test_model\\RAVEN_1368_test\\image.npy"
    # image_path = os.path.join(RAVEN_folder, file_folder)
    # image_path = str(RAVEN_folder)+"/"+ str(file)
    # image_path = "C:/Users/Hertz/Documents/SJSU Coursework/MS Project_big files/RAVEN-10000-release/RAVEN-10000/center_single/RAVEN_10_train.npz"
    image_path = "C:/Users/Hertz/Documents/SJSU Coursework/MS Project_big files/RAVEN-10000-release/RAVEN-10000/"
    ap = argparse.ArgumentParser()
    #ap.add_argument("type_loss", type=bool)
    #ap.add_argument("image_path", type=str)
    ap.add_argument('--type_loss', type=bool, default=True)
    ap.add_argument('--image_path', type=str,default= image_path)
    ap.add_argument('--regime', type=str, default='all')
    ap.add_argument('--image_type', type=str, default='image')
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--numwork', dest='numwork', type=int, default=1)
    args = ap.parse_args()

    # args = config.get_args()
    # args = args[0]
    # weights_path = args.path_weight+"/"+args.load_weight

    pretrained_dict = torch.load(weights_path, map_location = device) 
    # print("pretrained_dict:", pretrained_dict)
    r_model = m.Reab3p16(args)
    model_dict = r_model.state_dict()                           ## https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.htmlA state_dict is an integral entity 
    pretrained_dict1 = {}                                      ##..if you are interested in saving or loading models from PyTorch
    for k, v in pretrained_dict.items():                      ##filter out unnecessary keys k
        if k[:7]=="module.":
            k = k[7:]
        if k in model_dict:                                   ##only when keys match(like conv2D..and so forth)
            pretrained_dict1[k] = v                   
    model_dict.update(pretrained_dict1)                      ##overwrite entries in the existing state dict 
    r_model.load_state_dict(model_dict) 

    # print("pretrained_dict1:", pretrained_dict1)

    with torch.no_grad():
        r_model.eval()
        accuracy_all = []
        #data_dir = "C:/Users/Hertz/Documents/SJSU Coursework/MS Project_big files/RAVEN-10000-release/RAVEN-10000/"
        #data_dir = "C:/Users/Hertz/Documents/SJSU Coursework/MS Project_big files/RAVEN 1000/"
        loader_try = load_data_for_inference(args, data_dir, data_split, filename)

        # print(loader_try)

        

        # data_split = "train"
        # data_files = [image_path]
        # print("datafiles: ", data_files)

        # df = [data_file for data_file in data_files][:]
        # print("df: ", df)
        loader = torch.utils.data.DataLoader(Dataset(args,loader_try), batch_size=args.batch_size, num_workers=args.numwork)

        # checkpoint = torch.load(model_path, map_location=device)
        count = 0
        for x, y, style,me in loader_try:
            count = count +1
            #print("count", count)
            # print("style:", style)
            x, y = Variable(x), Variable(y)
            pred= r_model(x)
            # print("pred:", pred)
            pred = pred[0].data.max(1)[1]
            
            # print("pred:", pred)
            # print("y",y)
            correct = pred.eq(y.data).cpu().numpy()
            accuracy = correct.sum() * 100.0 / len(y)
            print("accuracy", accuracy)
            accuracy_all.append(accuracy) 
        accuracy_all = sum(accuracy_all) / len(accuracy_all)
        print(accuracy_all)
        print("pred:",pred.data)
        print("y:",y.data)
        print("count:", count)
        return pred
        

def run():
    torch.multiprocessing.freeze_support()
    device = torch.device('cpu')
    
    weights_path = "C:\\Users\\Hertz\\Documents\\SJSU Coursework\\MS Project_big files\\git\\distracting_feature\\distracting_feature\\epochs\\epoch860"
    
    weights_path = "C:\\Users\\Hertz\\Documents\\SJSU Coursework\\MS Project_big files\\git\\distracting_feature\\distracting_feature\\epochs\\epoch860" # got ~20 percent here. 
    
    weights_path = "C:\\Users\\Hertz\\Documents\\SJSU Coursework\\MS Project_big files\\git\\distracting_feature\\distracting_feature\\epochs\\epochbest(73.6)"
    
    #model_path = "C:\\Users\\Hertz\\Documents\\SJSU Coursework\\MS Project_big files\\git\\distracting_feature\\distracting_feature\\epochs\\epoch860" # got ~70 percent here. 

    #image_path = "C:\\Users\\sonam\\Desktop\\MS_Project\\test_model\\RAVEN_1368_test\\image.npy"
    # image_path = os.path.join(RAVEN_folder, file_folder)
    # image_path = str(RAVEN_folder)+"/"+ str(file)
    # image_path = "C:/Users/Hertz/Documents/SJSU Coursework/MS Project_big files/RAVEN-10000-release/RAVEN-10000/center_single/RAVEN_10_train.npz"
    image_path = "C:/Users/Hertz/Documents/SJSU Coursework/MS Project_big files/RAVEN-10000-release/RAVEN-10000/"
    ap = argparse.ArgumentParser()
    #ap.add_argument("type_loss", type=bool)
    #ap.add_argument("image_path", type=str)
    ap.add_argument('--type_loss', type=bool, default=True)
    ap.add_argument('--image_path', type=str,default= image_path)
    ap.add_argument('--regime', type=str, default='all')
    ap.add_argument('--image_type', type=str, default='image')
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--numwork', dest='numwork', type=int, default=1)
    args = ap.parse_args()

    # args = config.get_args()
    # args = args[0]
    # weights_path = args.path_weight+"/"+args.load_weight

    pretrained_dict = torch.load(weights_path, map_location = device) 
    # print("pretrained_dict:", pretrained_dict)
    r_model = m.Reab3p16(args)
    model_dict = r_model.state_dict()                           ## https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.htmlA state_dict is an integral entity 
    pretrained_dict1 = {}                                      ##..if you are interested in saving or loading models from PyTorch
    for k, v in pretrained_dict.items():                      ##filter out unnecessary keys k
        if k[:7]=="module.":
            k = k[7:]
        if k in model_dict:                                   ##only when keys match(like conv2D..and so forth)
            pretrained_dict1[k] = v
            print(k)                   
    model_dict.update(pretrained_dict1)                      ##overwrite entries in the existing state dict 
    r_model.load_state_dict(model_dict) 

    # print("pretrained_dict1:", pretrained_dict1)

    with torch.no_grad():
        r_model.eval()
        accuracy_all = []
        #data_dir = "C:/Users/Hertz/Documents/SJSU Coursework/MS Project_big files/RAVEN-10000-release/RAVEN-10000/"
        data_dir = "C:/Users/Hertz/Documents/SJSU Coursework/MS Project_big files/RAVEN 1000/"
        loader_try = load_data(args,"train", data_dir)

        # print(loader_try)

        

        # data_split = "train"
        # data_files = [image_path]
        # print("datafiles: ", data_files)

        # df = [data_file for data_file in data_files][:]
        # print("df: ", df)
        loader = torch.utils.data.DataLoader(Dataset(args,loader_try), batch_size=args.batch_size, num_workers=args.numwork)

        # checkpoint = torch.load(model_path, map_location=device)
        count = 0
        for x, y, style,me in loader_try:
            count = count +1
            #print("count", count)
            # print("style:", style)
            x, y = Variable(x), Variable(y)
            pred= r_model(x)
            # print("pred:", pred)
            pred = pred[0].data.max(1)[1]
            # print("pred:", pred)
            # print("y",y)
            correct = pred.eq(y.data).cpu().numpy()
            accuracy = correct.sum() * 100.0 / len(y)
            print("accuracy", accuracy)
            accuracy_all.append(accuracy) 
        accuracy_all = sum(accuracy_all) / len(accuracy_all)
        print(accuracy_all)

if __name__ == '__main__':
    run()