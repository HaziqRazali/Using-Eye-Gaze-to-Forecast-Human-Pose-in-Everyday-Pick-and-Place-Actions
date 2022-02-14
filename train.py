import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.expanduser("~"),"tmp")
import cv2
import sys
import time
import torch
import socket
import argparse
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.nn.functional import cross_entropy, mse_loss, binary_cross_entropy

from collections import OrderedDict
from tensorboardX import SummaryWriter
from misc.misc import *
from misc.losses import *
from pathlib import Path

torch.manual_seed(1337)

# pid 
print("MACHINE", socket.gethostname())
print("PID", os.getpid(), flush=True)

# Import for args
##################################################### 
parser = argparse.ArgumentParser()
parser.add_argument('--args', required=True, type=str)
args, unknown = parser.parse_known_args()
args_import = "from misc.{} import *".format(args.args)
print(args_import)
exec(args_import)
args = argparser()
for k,v in vars(args).items():
    print(k,v)

# Imports for Architecture, Data Loader 
##################################################### 
architecture_import = "from models.{} import *".format(args.architecture)
exec(architecture_import)
data_loader_import = "from dataloaders.{} import *".format(args.data_loader)
exec(data_loader_import)
    
# Prepare Logging
##################################################### 
datetime = time.strftime("%c")  
writer = SummaryWriter(os.path.join(args.log_root,args.model_name,datetime))

# default checkpoints 
checkpoint = {
    'model_summary': None,  
    'model_state': None,
    'optim_state': None,
    'epoch': 0,
    'tr_counter': 0,
    'va_counter': 0}
# additional checkpoints
for task_name in args.task_names:
    checkpoint["_".join([task_name,"loss"])] = np.inf
    checkpoint["_".join([task_name,"epoch"])] = np.inf
    
tr_counter = 0 
va_counter = 0
epoch = 0
path = Path(os.path.join(args.weight_root,args.model_name))
path.mkdir(parents=True, exist_ok=True)
  
# Prepare Data Loaders
##################################################### 
long_dtype, float_dtype = get_dtypes(args)
# load data
tr_data = dataloader(args, "train")
va_data = dataloader(args, "val")
# data loader
tr_loader = torch.utils.data.DataLoader(tr_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=torch.cuda.is_available())
va_loader = tr_loader #torch.utils.data.DataLoader(va_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=torch.cuda.is_available())

# Prepare Network and Optimizers
##################################################### 
net = model(args)
print(net)
# must set model type before initializing the optimizer 
# https://discuss.pytorch.org/t/code-that-loads-sgd-fails-to-load-adam-state-to-gpu/61783/2 
print("Total # of parameters: ", count_parameters(net))
net.type(float_dtype)
#print(torch.cuda.current_device())
#sys.exit()
#net = net.to('cuda:'+str(torch.cuda.current_device()))
optimizer = optim.Adam(net.parameters(), lr=args.lr)
compute_loss = compute_loss(args)
 
# Maybe load weights and initialize checkpoints
##################################################### 
if args.restore_from_checkpoint == 1:
    
    # if i am loading elsewhere
    if os.path.join(args.load_weight_root,args.load_model_name) is not None and os.path.isdir(os.path.join(args.load_weight_root,args.load_model_name)):
        load_weight_root = args.load_weight_root
        load_model_name  = args.load_model_name
    elif os.path.join(args.weight_root,args.model_name) is not None and os.path.isdir(os.path.join(args.weight_root,args.model_name)):
        load_weight_root = args.weight_root
        load_model_name  = args.model_name        
    else:
        sys.exit("Model load weights not found")
        
    # load the best epoch for each task
    for epoch_name,layer_names in zip(args.epoch_names,args.layer_names):
    
        # load checkpoint dictionary
        checkpoint = torch.load(os.path.join(load_weight_root,load_model_name,epoch_name))
                
        # load weights
        model_state = checkpoint["model_state"]
        layer_dict = {k:v for k,v in model_state.items() for layer_name in layer_names if layer_name in k}
        print("epoch_name", epoch_name)
        print("layer_dict.keys()")
        print(layer_dict.keys())
        print()
        net.load_state_dict(layer_dict,strict=False)
        
# Main Loop
####################################################  
def loop(net, inp_data, optimizer, counter, epoch, args, mode):        
           
    # move to gpu
    for k,v in inp_data.items():
        inp_data[k] = inp_data[k].cuda() if torch.cuda.device_count() > 0 and not isinstance(v, list) else inp_data[k]
        
    # maybe freeze some layers
    if eval(args.freeze) is not None:
        net = eval(args.freeze)(net, args.stages[epoch])
        
    # Forward pass
    out_data = net(inp_data, mode=mode)
        
    # compute unscaled losses        
    losses = {}
    for loss_name, loss_function in zip(args.loss_names, args.loss_functions):        
        losses[loss_name] = compute_loss(inp_data, out_data, loss_name, loss_function)
        
    # write unscaled losses to log file
    for k,v in losses.items():
        if v != 0:
            writer.add_scalar(os.path.join(k,mode), v.item(), counter)
    
    # scale the losses since save_checkpoint checks the dictionary losses
    # also backprop the scaled losses
    total_loss = 0
    for loss_name, loss_weight in zip(args.loss_names, args.loss_weights):
        losses[loss_name] *= loss_weight[epoch]
        total_loss += losses[loss_name]
        
    if mode == "tr" and optimizer is not None:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    # move all to cpu numpy
    losses = iterdict(losses)
    inp_data = iterdict(inp_data)
    out_data = iterdict(out_data)
    
    return {"out_data":out_data, "losses":losses}

tr_counter = checkpoint['tr_counter']
va_counter = checkpoint['va_counter']
epoch = checkpoint["epoch"]
print(epoch, "tr_counter:", tr_counter, "va_counter:", va_counter)

# Train
####################################################
for i in range(epoch, 100000): 
    
    # training ---------------------
    net.train()
    start = time.time()
    t1 = time.time()
    for batch_idx, tr_data in enumerate(tr_loader):
        t2 = time.time()
        
        if batch_idx%100 == 0:
            print("Epoch " + str(i).zfill(2) + " training batch ", batch_idx, " of ", len(tr_loader), "Time:", t2-t1, flush=True)
            
        tr_output = loop(net=net,inp_data=tr_data,optimizer=optimizer,counter=tr_counter,epoch=i,args=args,mode="tr")
        tr_counter= tr_counter+1
        if batch_idx!=0 and batch_idx%args.tr_step == 0:
            break
        
        t1 = time.time()
                                       
        #break
    end = time.time()
    tr_time = end - start
    # training ---------------------
    
    # validation ---------------------
    start = time.time()
    t1 = time.time()
    with torch.no_grad():        
        net.eval()
        va_losses = {}
        for batch_idx, va_data in enumerate(va_loader):
            t2 = time.time()
            
            if batch_idx%50 == 0:
                print("Epoch " + str(i).zfill(2) + " validation batch ", batch_idx, " of ", len(va_loader), "Time:", t2-t1, flush=True)
            
            va_output = loop(net=net,inp_data=va_data,optimizer=None,counter=va_counter,epoch=i,args=args,mode="va")
            va_counter= va_counter+1 
                       
            # accumulate loss                    
            for key in args.loss_names:
                va_losses = collect(va_losses, key=key, value=va_output["losses"][key]) if key in va_output["losses"] else va_losses
            
            if batch_idx!=0 and batch_idx%args.va_step == 0:
                break
                
            t1 = time.time()
            
    end = time.time()
    va_time = end - start     
    
    #sys.exit()        
    # average loss
    for k,v in va_losses.items():
        va_losses[k] = np.mean(va_losses[k])
        
    # checkpoint
    for task_name,task_component in zip(args.task_names,args.task_components):
        
        # compute task loss for current epoch
        task_loss = np.sum([va_losses[x] for x in task_component])
        
        # maybe reset best loss
        if args.reset_loss is not None and any([i == x for x in args.reset_loss]):
            print("Resetting loss at epoch", i)
            checkpoint["_".join([task_name,"loss"])] = np.inf
        
        # maybe save model
        checkpoint = save_model(checkpoint, net, args, optimizer, tr_counter, va_counter, task_name, current_epoch=i, current_loss=task_loss)
    
    # print
    print("Curr Loss:       ", va_losses, flush=True)
    print("Best Loss:       ", {k:v for k,v in checkpoint.items() if "loss" in k}, flush=True)
    print("Best Epoch:      ", {k:v for k,v in checkpoint.items() if "epoch" in k}, flush=True)
    print("Training time:   ", tr_time, flush=True)
    print("Validation time: ", va_time, flush=True)
