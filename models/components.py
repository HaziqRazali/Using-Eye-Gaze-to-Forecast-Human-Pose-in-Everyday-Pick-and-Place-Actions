import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
 
def make_mlp(dim_list, activations, dropout=0):

    if len(dim_list) == 0 and len(activations) == 0:
        return nn.Identity()

    assert len(dim_list) == len(activations)+1
    
    layers = []
    for dim_in, dim_out, activation in zip(dim_list[:-1], dim_list[1:], activations):
                
        # append layer
        layers.append(nn.Linear(dim_in, dim_out))
        
        # # # # # # # # # # # # 
        # append activations  #
        # # # # # # # # # # # #
            
        activation_list = re.split('-', activation)
        for activation in activation_list:
                                        
            if 'leakyrelu' in activation:
                layers.append(nn.LeakyReLU(negative_slope=float(re.split('=', activation)[1]), inplace=True))
                
            elif activation == 'relu':
                layers.append(nn.ReLU())
                
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
                
            elif activation == "none":
                pass
                                
            elif activation == "batchnorm":
                layers.append(nn.BatchNorm1d(dim_out))    
                        
            else:
                print("unknown activation")
                sys.exit()
            
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
                   
    return nn.Sequential(*layers)

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        for key, value in args.__dict__.items():
            setattr(self, key, value)
        
        assert len(self.gaze_encoder_units) == len(self.gaze_encoder_activations)+1
        assert len(self.gaze_encoder_units) == len(self.gaze_encoder_kernels)
        assert len(self.gaze_encoder_kernels) == len(self.gaze_encoder_paddings)
        
        layers = []
        for dim_in, dim_out, kernel, pad, activation in zip(self.gaze_encoder_units[:-1], self.gaze_encoder_units[1:], self.gaze_encoder_kernels, self.gaze_encoder_paddings, self.gaze_encoder_activations):
            layers.append(nn.Conv1d(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel, padding=pad))
            if activation == "relu":
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)
        
    def forward(self, data):
    
        data = data.permute(0,2,1)
        data = self.model(data)
        data = data.permute(0,2,1)
        
        return data
            
class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        for key, value in args.__dict__.items():
            setattr(self, key, value)
        self.model = eval("nn."+self.gaze_encoder_type)(self.gaze_encoder_units[0],self.gaze_encoder_units[1],self.gaze_encoder_units[2],batch_first=False)
    
    def init_hidden(self):
        # [num_layers, batch, hidden_size]
        return (torch.zeros(self.gaze_encoder_units[2], self.padded_length*1, self.gaze_encoder_units[1]).cuda(), torch.zeros(self.gaze_encoder_units[2], self.padded_length*1, self.gaze_encoder_units[1]).cuda())
        #return (torch.zeros(self.gaze_encoder_units[2], self.batch_size*10, self.gaze_encoder_units[1]).cuda(), torch.zeros(self.gaze_encoder_units[2], self.batch_size*10, self.gaze_encoder_units[1]).cuda())
    
    def forward(self, data):
    
        # data = [1*self.padded_length, gaze_length, 3]
        data = data.permute(1,0,2) # [gaze_length, 1*self.padded_length, 3]
        
        hi = self.init_hidden()
        h = []
        for d in data:
            _, hi = self.model(torch.unsqueeze(d,0),hi)
            h.append(hi[0])
        h = torch.cat(h) # [gaze_length, batch*15, 128]
        
        return h.permute(1,0,2) # return [batch*15, gaze_length, 128]

# attention over the gaze relative to each object / grid
class multi_attention(nn.Module):
    def __init__(self, units, activations, padded_length):
        super(multi_attention, self).__init__()
        self.fc = make_mlp(units, activations)
        self.padded_length = padded_length
        
    def forward(self, data, unpadded_length):
                
        # truncate data
        data = data[:unpadded_length] # data = [num_objects/grid, gaze_length, hidden_size]
                
        # reshape
        data = data.view(-1, unpadded_length, data.shape[1], data.shape[2]) # [batch (1), num_objects/grid, gaze_length, hidden size]   
        
        # attention
        scores = self.fc(data)                       # [batch (1), num_objects/grid, gaze_length, 1]
        scores = F.softmax(scores,dim=-2)            # [batch (1), num_objects/grid, gaze_length, 1] torch.sum(scores[0,0]) = 1 
        data = data * scores                         # [batch (1), num_objects/grid, gaze_length, hidden size]
        data = torch.sum(data,dim=-2)                # [batch (1), num_objects/grid, hidden size]
         
        # pad it back
        scores_pad = torch.zeros([scores.shape[0], self.padded_length, scores.shape[2], scores.shape[3]], dtype=scores.dtype, device=scores.device) # [1, padded_length, gaze_length, 1]
        scores_pad[:scores.shape[0], :scores.shape[1], :scores.shape[2], :scores.shape[3]] = scores
        data_pad = torch.zeros([data.shape[0], self.padded_length, data.shape[2]], dtype=data.dtype, device=data.device) # [1, padded_length, hidden_size]
        data_pad[:data.shape[0], :data.shape[1], :data.shape[2]] = data
        
        return scores_pad, data_pad

# attention over each object
class attention(nn.Module):
    def __init__(self, units, activations, padded_length):
        super(attention, self).__init__()
        
        assert len(units) != 0
        assert len(activations) != 0
        
        self.fc = make_mlp(units, activations)
        self.padded_length = padded_length
        
    def forward(self, data, unpadded_length):
    
        # truncate data
        data = data[:,:unpadded_length] # data = [batch (1), num_objects/grid, hidden_size]
                     
        # attention   
        scores = self.fc(data)             # [batch (1), num_objects/grid, 1]
        scores = F.softmax(scores,dim=-2)  # [batch (1), num_objects/grid, 1]
        data = data * scores               # [batch (1), num_objects/grid, feature dim]
        data = torch.sum(data,dim=-2)      # [batch (1), feature dim]
        
        # pad it back
        scores_pad = torch.zeros([scores.shape[0], self.padded_length, scores.shape[2]], dtype=scores.dtype, device=scores.device) # [1, 100, 1]
        scores_pad[:scores.shape[0], :scores.shape[1], :scores.shape[2]] = scores
                        
        return scores_pad, data

# create GRU hidden state        
def init_hidden(batch_size, hidden_units, num_layers, bidirectional):
    # [num_layers, batch, hidden_size]
    num_directions = 2 if bidirectional == 1 else 1
    return torch.zeros(num_directions*num_layers, batch_size, hidden_units).cuda()
