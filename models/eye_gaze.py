import torch.distributions as tdist
import torch.nn.functional as F
import torch
import torch.nn as nn

from torch.nn import GRU, LSTM
#import torch.nn.GRU as GRU
#import torch.nn.LSTM as LSTM
import time

from models.components import *

class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()
                 
        for key, value in args.__dict__.items():
            setattr(self, key, value)
        
        """
        Eye
        """
        self.gaze_encoder     = eval(self.gaze_encoder_type)(args)
        self.gaze_attention   = multi_attention(self.gaze_attention_units, self.gaze_attention_activations, self.object_padded_length)
        self.object_attention = attention(self.object_attention_units, self.object_attention_activations, self.object_padded_length)
                
        """
        Pose
        """
          
        self.inp_pose_encoder = make_mlp([self.pose_encoder_units[0]+3]+self.pose_encoder_units[1:],self.pose_encoder_activations)
        self.key_pose_encoder = make_mlp(self.pose_encoder_units,self.pose_encoder_activations)
        self.pose_mu          = make_mlp(self.pose_mu_var_units,self.pose_mu_var_activations)
        self.pose_log_var     = make_mlp(self.pose_mu_var_units,self.pose_mu_var_activations)
        self.key_pose_decoder = make_mlp(self.pose_decoder_units,self.pose_decoder_activations)
        
        self.norm = tdist.Normal(torch.tensor(0.0), torch.tensor(1.0))
                        
    def forward(self, data, mode):
    
        """
        for k,v in data.items():
            if type(v) == type(torch.Tensor(0)):
                print(k, v.shape)
        sys.exit()
        """
        """        
        inp_frame torch.Size([32])
        key_frame torch.Size([32])
        center torch.Size([32, 3])
        gaze_vector torch.Size([32, 5, 2, 3])
        inp_pose torch.Size([32, 21, 3])
        key_pose torch.Size([32, 21, 3])
        object_scores torch.Size([32])
        key_object torch.Size([32, 1, 3])
        time torch.Size([32])
        
        relative_gaze torch.Size([32, self.object_padded_length, 5, 3])
        key_relative_gaze torch.Size([32, self.object_padded_length, 3])
        objects torch.Size([32, self.object_padded_length, 3])
        object_padded_length torch.Size([32])
        """
                
        actions  = data["action"]
        gaze     = data["relative_gaze"]                      
        key_gaze = data["key_relative_gaze"]                  
        objects  = data["objects"]                           
        inp_pose = data["inp_pose"].view(self.batch_size,-1) 
        key_pose = data["key_pose"].view(self.batch_size,-1)
        key_object = data["key_object"].view(self.batch_size,-1)
        object_padded_length = data["object_padded_length"]

        """
        predict object
        """
        
        gaze_scores, object_scores = [], []             
        pred_key_gaze, pred_key_object = [], []
        for gaze_i,object_i,unpadded_length_i in zip(gaze,objects,object_padded_length): 
            
            # attend over gaze
            gaze_i = gaze_i.unsqueeze(0)                                                             # [1, object_padded_length, gaze_length, 3]
            gaze_features_i = self.gaze_encoder(gaze_i.view(-1,self.gaze_length,3))                  # [object_padded_length, gaze_length, 128]
            gaze_scores_i, gaze_features_i = self.gaze_attention(gaze_features_i, unpadded_length_i) # [1, object_padded_length, gaze_length, 1] [1, object_padded_length, 128]
            pred_key_gaze_i = torch.sum(gaze_i * gaze_scores_i, dim=2, keepdim=False)                # [1, object_padded_length, 3]
                        
            # attend over object
            object_i = object_i.unsqueeze(0)                                                # [1, object_padded_length, 3]
            object_scores_i, _ = self.object_attention(gaze_features_i, unpadded_length_i)  # [1, object_padded_length, 1]                                    
            pred_key_object_i  = torch.sum(object_i * object_scores_i, dim=1, keepdim=True) # [1, 1, 3]
                        
            # collect
            gaze_scores.append(gaze_scores_i)
            object_scores.append(object_scores_i)
            pred_key_gaze.append(pred_key_gaze_i)
            pred_key_object.append(pred_key_object_i)
        
        gaze_scores = torch.cat(gaze_scores)         # [32, self.object_padded_length, 5, 1]
        object_scores = torch.cat(object_scores)     # [32, self.object_padded_length, 1]
        pred_key_gaze = torch.cat(pred_key_gaze)     # [32, self.object_padded_length, 3]
        pred_key_object = torch.cat(pred_key_object) # [32, 1, 3]
                                
        """
        compute pose
        """
        
        # use pred key object as input
        if self.key_object == "pred":
            object_scores_argmax = torch.argmax(object_scores.squeeze(),dim=1)
            key_object = torch.stack([object[object_score_argmax] for object,object_score_argmax in zip(objects,object_scores_argmax)]) # [32, 3]
            key_objects = key_object.unsqueeze(0)
        
        # use true key object as input
        if self.key_object == "true":
            key_object = key_object # [32, 3]
            key_objects = key_object.unsqueeze(0)
            
        # if pick action  - compute the pose for every single object
        # if place action - compute the pose for the weighted sum of the objects
        if self.key_object == "all":
            
            key_objects = objects # [32, self.object_padded_length, 3]
            
            # to make life simple, do not do any truncation, only truncate when saving the json files
            key_objects = []
            for action, object, object_score in zip(actions, objects, object_scores):
                
                # compute the pose for every single object
                if action == "pick":
                    key_object = object # [self.object_padded_length, 3]
                
                # compute the pose for the weighted sum of the objects    
                if action == "place":
                    key_object = torch.sum(object * object_score, dim=0, keepdim=True) # [1, 3]
                    key_object = key_object.repeat(self.object_padded_length,1)               # [self.object_padded_length, 3]
                            
                key_objects.append(key_object)
                
            key_objects = torch.stack(key_objects)     # [32, self.object_padded_length, 3]
            key_objects = key_objects.permute(1, 0, 2) # [self.object_padded_length, 32, 3]
                    
        pred_key_pose_list,time_list = [],[]
        for key_object in key_objects:
        
            # feed x and y
            inp_pose_features = torch.cat((inp_pose, key_object), dim=1)
            inp_pose_features = self.inp_pose_encoder(inp_pose_features)
            key_pose_features = self.key_pose_encoder(key_pose)
            
            # get gaussian parameters
            pose_posterior = torch.cat((inp_pose_features,key_pose_features),dim=1)
            pose_posterior_mu = self.pose_mu(pose_posterior)
            pose_posterior_log_var = self.pose_log_var(pose_posterior)
            
            # sample
            pose_posterior_std = torch.exp(0.5*pose_posterior_log_var)
            pose_posterior_eps = self.norm.sample([self.batch_size, pose_posterior_mu.shape[1]]).cuda()
            pose_posterior_z   = pose_posterior_mu + pose_posterior_eps*pose_posterior_std
            
            z_p = pose_posterior_z if mode == "tr" else self.norm.sample([self.batch_size, self.pose_mu_var_units[-1]]).cuda()
            
            # forecast
            pred_key_pose = torch.cat((z_p,inp_pose_features),dim=1)
            pred_key_pose = self.key_pose_decoder(pred_key_pose)
            pred_key_pose_list.append(pred_key_pose)
        
        pred_key_pose_list = torch.stack(pred_key_pose_list)
                
        # process
        gaze_scores = gaze_scores.squeeze()
        object_scores = object_scores.squeeze()        
        pred_key_pose_list = pred_key_pose_list.permute(1,0,2)
        pred_key_pose_list = pred_key_pose_list.view(self.batch_size,-1,21,3)
        pred_key_pose_list = pred_key_pose_list.squeeze()
                
        return {"gaze_scores":gaze_scores, "key_relative_gaze":pred_key_gaze, 
                "object_scores":object_scores, 
                "key_pose":pred_key_pose_list,
                "pose_posterior":{"mu":pose_posterior_mu, "log_var":pose_posterior_log_var},
                "key_object":pred_key_object}
