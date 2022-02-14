import os
import argparse

def argparser():

    # Argument Parser
    ##################################################### 
    parser = argparse.ArgumentParser()
    
    # # # # # #
    # dataset #
    # # # # # #
    
    parser.add_argument('--dataset_root', required=True, type=str)
    parser.add_argument('--data_loader',  required=True, type=str)
    parser.add_argument('--batch_size',   default=8, type=int)
    parser.add_argument('--test_set', type=str)
    
    # actions
    parser.add_argument('--actions', nargs="*", default=["pick","place"], type=str)
    parser.add_argument('--use_instructions', default=0, type=int)    
    
    # objects
    parser.add_argument('--furniture_names', nargs="*", type=str)
    parser.add_argument('--grid_sizes', nargs="*", action="append", type=int)
    parser.add_argument('--object_padded_length', default=0, type=int)
    
    # gaze
    parser.add_argument('--inp_length', default=0, type=int)
    parser.add_argument('--out_length', default=0, type=int)    
    parser.add_argument('--time_step_size', default=0, type=int)    
    parser.add_argument('--gaze_length', default=0, type=int)
    parser.add_argument('--fixation_threshold', default=180, type=float)
    
    # variables to unpad at test time
    parser.add_argument('--unpad', nargs="*", type=str)
    
    # variables not to save at test time
    parser.add_argument('--remove', nargs="*", type=str)
    
    # # # # # # # # # # 
    # model settings  # 
    # # # # # # # # # #
    
    parser.add_argument('--architecture', required=True, type=str)
            
    # # # # # # # # # # # # # #
    # Gaze Attention Network  #
    # # # # # # # # # # # # # #
    
    # gaze encoder
    parser.add_argument('--gaze_encoder_type', type=str)
    parser.add_argument('--gaze_encoder_units', nargs="*", type=int)
    parser.add_argument('--gaze_encoder_activations', nargs="*", type=str)
    parser.add_argument('--gaze_encoder_kernels', nargs="*", type=int)
    parser.add_argument('--gaze_encoder_paddings', nargs="*", type=int)
    
    # gaze attention
    parser.add_argument('--gaze_attention_units', nargs="*", type=int)
    parser.add_argument('--gaze_attention_activations', nargs="*", type=str)
    
    # # # # # # # # # # # # # # # 
    # Object Attention Network  #
    # # # # # # # # # # # # # # #
    
    # object attention
    parser.add_argument('--object_attention_units', nargs="*", type=int)
    parser.add_argument('--object_attention_activations', nargs="*", type=str)
    
    # # # # # # # # #
    # Pose Network  #
    # # # # # # # # #
    
    # mlp encoder
    parser.add_argument('--key_object', type=str)
    parser.add_argument('--pose_encoder_units', nargs="*", type=int)
    parser.add_argument('--pose_encoder_activations', nargs="*", type=str)
    parser.add_argument('--pose_mu_var_units', nargs="*", type=int)
    parser.add_argument('--pose_mu_var_activations', nargs="*", type=str)
        
    # mlp decoder
    parser.add_argument('--pose_decoder_units', nargs="*", type=int)
    parser.add_argument('--pose_decoder_activations', nargs="*", type=str)
                        
    # # # # # # # # # # # # # # # # # # # # # # # # 
    # checkpointing                               #
    # # # # # # # # # # # # # # # # # # # # # # # # 
    
    parser.add_argument('--log_root', default="", type=str)
    parser.add_argument('--weight_root', default="", type=str)
    
    parser.add_argument('--model_name', default="", type=str)
    parser.add_argument('--epoch_names', nargs="*", type=str)                   # task specific epoch names
    parser.add_argument('--layer_names', nargs="*", action="append", type=str)  # task specific layer names
    parser.add_argument('--strict', default=1, type=int)
    
    parser.add_argument('--restore_from_checkpoint', default=0, type=int)
    
    parser.add_argument('--task_names', nargs="*", type=str)                       # task names
    parser.add_argument('--task_components', nargs="*", action="append", type=str) # task components e.g. key_pose = key_pose + pose_posterior
    parser.add_argument('--result_root', default="", type=str)
    parser.add_argument('--result_name', default="", type=str)

    # # # # # # # # # # # # #
    # general optimization  #
    # # # # # # # # # # # # #
    
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--tr_step', type=int)
    parser.add_argument('--va_step', type=int)
    parser.add_argument('--loss_names', nargs="*", type=str)
    parser.add_argument('--loss_functions', nargs="*", type=str)
    parser.add_argument('--loss_weights', nargs="*", type=str)
    
    parser.add_argument('--freeze', default="None", type=str)
    parser.add_argument('--debug', default=0, type=int)
    parser.add_argument('--reset_loss', nargs="*", type=int)
                
    # parse
    args, unknown = parser.parse_known_args()
    
    args.log_root = os.path.join("./logs/",args.log_root)
    args.weight_root = os.path.join("./weights/",args.weight_root)
    args.result_root = os.path.join("./results/",args.result_root)
        
    # the lists containing the loss schedules must have the same length
    if args.loss_weights is not None:
        args.loss_weights = [eval(x) for x in args.loss_weights]
        assert len(set(map(len,args.loss_weights))) == 1
    
    # the loss names, functions and weights must have the same length    
    if args.loss_names is not None or args.loss_functions is not None or args.loss_weights is not None:
        assert len(args.loss_names) == len(args.loss_functions)
        assert len(args.loss_functions) == len(args.loss_weights)
        
    return args
