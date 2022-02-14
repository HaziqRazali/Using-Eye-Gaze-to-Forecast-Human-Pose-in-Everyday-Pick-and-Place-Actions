import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.expanduser("~"),"tmp")
import cv2
import sys
import time
import json
import torch
import numpy as np
import argparse
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathlib import Path
from collections import OrderedDict
from misc.misc import *

from torch.nn.functional import cross_entropy, mse_loss, l1_loss

torch.manual_seed(1337)

# Import for args
##################################################### 
parser = argparse.ArgumentParser()
parser.add_argument('--args', required=True, type=str)
args, unknown = parser.parse_known_args()
args_import = "from misc.{} import *".format(args.args)
exec(args_import)
args = argparser()

# Imports for Architecture and Data Loader 
##################################################### 
architecture_import = "from models.{} import *".format(args.architecture)
exec(architecture_import)
data_loader_import = "from dataloaders.{} import *".format(args.data_loader)
exec(data_loader_import)
            
# Prepare Data Loaders
##################################################### 
long_dtype, float_dtype = get_dtypes(args)
# load data
va_data = dataloader(args, args.test_set)
# data loader
va_loader = torch.utils.data.DataLoader(va_data, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True, pin_memory=torch.cuda.is_available())

# Prepare Dump File
#####################################################
dump = [None]*int((len(va_data) - len(va_data)%args.batch_size))

# Prepare Network and Optimizers
##################################################### 
net = model(args)
print("Total # of parameters: ", count_parameters(net))
net.type(float_dtype)

# Load weights and initialize checkpoints
##################################################### 
print("Attempting to load from: " + os.path.join(args.weight_root,args.model_name))
if os.path.join(args.weight_root,args.model_name) is not None and os.path.isdir(os.path.join(args.weight_root,args.model_name)):
    
    # load the best epoch for each task
    for epoch_name,layer_names in zip(args.epoch_names,args.layer_names):
    
        # load checkpoint dictionary
        checkpoint = torch.load(os.path.join(args.weight_root,args.model_name,epoch_name))
                
        # load weights
        model_state = checkpoint["model_state"]
        layer_dict = {k:v for k,v in model_state.items() for layer_name in layer_names if layer_name in k}
        print("=======================")
        print("epoch_name", epoch_name)
        print("layer_dict.keys()")
        for key in layer_dict.keys():
            print(key)
        print()
        net.load_state_dict(layer_dict,strict=args.strict)
    
    print("Model Loaded")
    
else:
    print(os.path.join(args.weight_root,args.model_name) + " not found")
    sys.exit() 
    
# Main Loop
####################################################  
def loop(net, inp_data, optimizer, counter, args, mode):        
    # {'human_joints_t0':human_joints_t0, 'human_joints_t1':human_joints_t1, 'object_data':object_data, "key_object":key_object, "frame":frame, "key_frame":key_frame}
    
    # move to gpu
    for k,v in inp_data.items():
        inp_data[k] = inp_data[k].cuda() if torch.cuda.device_count() > 0 and type(v) != type([]) else inp_data[k]
        
    # Forward pass
    t1 = time.time()
    out_data = net(inp_data, mode="te")
    t2 = time.time()
    print("Foward Pass Time: ", 1/(t2-t1), flush=True) 

    # move all to cpu numpy
    #losses = iterdict(losses)
    inp_data = iterdict(inp_data)
    out_data = iterdict(out_data)
    
    return {"out_data":out_data}
    
# save results 
####################################################  
def save(out, inp, args):
        
    # handle conflicting names
    keys = set(inp.keys()) | set(out.keys())
    for key in keys:
        if key in inp and key in out:
            inp["_".join(["true",key])] = inp[key]
            out["_".join(["pred",key])] = out[key]
            del inp[key]
            del out[key]
    
    # merge dict
    data = {**inp, **out}
        
    # remove items i do not want
    data = {k:v for k,v in data.items() if "posterior" not in k and "prior" not in k}
    
    # json can only save list
    for k,v in data.items():
        data[k] = data[k].tolist() if isinstance(v, type(np.array([0]))) else data[k]
        
    # save each frame
    for i in range(len(data["sequence"])):
    
        # create folder
        model = args.result_name    
        foldername = os.path.join(args.result_root,model,data["sequence"][i]) 
        path = Path(foldername)
        path.mkdir(parents=True,exist_ok=True)
        
        # save filename
        filename = os.path.join(args.result_root,model,data["sequence"][i],str(int(data["inp_frame"][i])).zfill(10)+".json") 
                
        # create json for each frame 
        data_i = {k:v[i] if type(v) == type([]) else v for k,v in data.items()}
        
        # unpad 
        if args.unpad is not None:
            for unpad in args.unpad:            
                try:
                    data_i[unpad] = data_i[unpad][:data_i["unpadded_length"]]
                except:
                    sys.exit()
                
        # write to json
        with open(filename, 'w') as f:
            json.dump(data_i, f)
                  
# validation
####################################################  
with torch.no_grad():        
    net.eval()
    va_losses = {}
    for batch_idx, va_data in enumerate(va_loader):
    
        print("Validation batch ", batch_idx, " of ", len(va_loader))
        
        # forward pass
        va_output = loop(net=net,inp_data=va_data,optimizer=None,counter=None,args=args,mode="va")
                
        # save results
        save(va_output["out_data"], va_data, args)