#!/bin/bash

# read arguments
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2- -d=)   
    
    case "$KEY" in
            arch)       				arch=${VALUE} ;;
            root)       				root=${VALUE} ;;
            name)       				name=${VALUE} ;;
			
			# # # # # # # # # # # #
			# dataset parameters  #
            # # # # # # # # # # # #
			
			data_loader)        		data_loader=${VALUE} ;;
			batch_size)					batch_size=${VALUE} ;;
			
			use_instructions)			use_instructions=${VALUE} ;;
			grid_sizes1)				grid_sizes1=${VALUE} ;;
			grid_sizes2)				grid_sizes2=${VALUE} ;;
			grid_sizes3)				grid_sizes3=${VALUE} ;;
			object_padded_length)		object_padded_length=${VALUE} ;;
			
			inp_length)					inp_length=${VALUE} ;;
			out_length)					out_length=${VALUE} ;;
			
			gaze_length)				gaze_length=${VALUE} ;;
			time_step_size)				time_step_size=${VALUE} ;;
			fixation_threshold)			fixation_threshold=${VALUE} ;;
			
			# # # # # # # # # # # #
			# learning parameters #
            # # # # # # # # # # # #
			
			lr)							lr=${VALUE} ;;
			tr_step)					tr_step=${VALUE} ;;
			va_step)					va_step=${VALUE} ;;
			loss_names)					loss_names=${VALUE} ;;
			loss_functions)				loss_functions=${VALUE} ;;
			loss_weights)				loss_weights=${VALUE} ;;
			task_names)					task_names=${VALUE} ;;
			task_component1)			task_component1=${VALUE} ;;
			task_component2)			task_component2=${VALUE} ;;
					
			# # # # # # # # #
			# gaze network  #
            # # # # # # # # #
			
			gaze_encoder_type)				gaze_encoder_type=${VALUE} ;;
			gaze_encoder_units)				gaze_encoder_units=${VALUE} ;;
			gaze_encoder_activations)		gaze_encoder_activations=${VALUE} ;;
			gaze_encoder_kernels)			gaze_encoder_kernels=${VALUE} ;;
			gaze_encoder_paddings)			gaze_encoder_paddings=${VALUE} ;;
			
			gaze_attention_units)			gaze_attention_units=${VALUE} ;;
			gaze_attention_activations)		gaze_attention_activations=${VALUE} ;;
			
			object_attention_units)			object_attention_units=${VALUE} ;;
			object_attention_activations)	object_attention_activations=${VALUE} ;;
			
			# # # # # # # # #
			# pose network  #
            # # # # # # # # #
			
			pose_encoder_units)			pose_encoder_units=${VALUE} ;;
			pose_encoder_activations)	pose_encoder_activations=${VALUE} ;;
			pose_mu_var_units)			pose_mu_var_units=${VALUE} ;;
			pose_mu_var_activations)	pose_mu_var_activations=${VALUE} ;;
			pose_decoder_units)			pose_decoder_units=${VALUE} ;;
			pose_decoder_activations)	pose_decoder_activations=${VALUE} ;;
								
            *)   
    esac  
done

# args
args="args_mogaze"

# # # # # # # # #
# optimization  #
# # # # # # # # #

gpu_num=${!#}
lr=1e-3
tr_step=1000
va_step=500
loss_names=${loss_names:-}
loss_functions=${loss_functions:-}
loss_weights=${loss_weights:-}

# # # # # # # # # # # # #
# checkpoint task names #
# # # # # # # # # # # # #

task_names=${task_names:-}
task_component1=${task_component1:-}
task_component2=${task_component2:-}

# # # # # #
# dataset #
# # # # # #

dataset_root="./data"
data_loader="mogaze_eye"
batch_size=48

# actions
actions="pick place"
use_instructions=${use_instructions:-1}

# furnitures
furniture_names="table laiva_shelf vesken_shelf"
grid_sizes1=${grid_sizes1:-5 5} # "5 5" *1 = 25
grid_sizes2=${grid_sizes2:-4 2} # "4 2" *5 = 40
grid_sizes3=${grid_sizes3:-3 2} # "3 2" *4 = 24
object_padded_length=${object_padded_length:-89} # 25 + 40 + 24 = 89

# gaze data
inp_length=${inp_length:-0}
out_length=${out_length:-0}
gaze_length=${gaze_length:-9}
time_step_size=12
fixation_threshold=180

# # # # #
# model #
# # # # #

architecture="eye_gaze"

# # # # # # # # # # # # # #
# gaze attention network  #
# # # # # # # # # # # # # #
 
gaze_encoder_type="CNN"
gaze_encoder_units="3 64 128"
gaze_encoder_activations="relu relu"
gaze_encoder_kernels="3 3 3"
gaze_encoder_paddings="1 1 1"
gaze_attention_units="128 1"
gaze_attention_activations="none"

# # # # # # # # # # # # # # #
# object attention network  #
# # # # # # # # # # # # # # #

object_attention_units="128 1"
object_attention_activations="none"

# # # # # # # # #
# pose network  #
# # # # # # # # #

pose_encoder_units="63 256 128"
pose_encoder_activations="relu relu"
pose_mu_var_units="256 64 8"
pose_mu_var_activations="relu none"
pose_decoder_units="136 256 63"
pose_decoder_activations="relu none"

# use true key object for training
key_object="true"
key_pose="true" 

# # # # # # # # #
# checkpointing #
# # # # # # # # #

restore_from_checkpoint=0
log_root=$root
weight_root=$root
model_name=$name

cd ../..
CUDA_VISIBLE_DEVICES=${gpu_num} python train.py --args $args \
`# dataset` \
--dataset_root $dataset_root --data_loader $data_loader --batch_size $batch_size \
--actions $actions --use_instructions $use_instructions \
--furniture_names $furniture_names --grid_sizes $grid_sizes1 --grid_sizes $grid_sizes2 --grid_sizes $grid_sizes3 --object_padded_length $object_padded_length \
--inp_length $inp_length --out_length $out_length --gaze_length $gaze_length --time_step_size $time_step_size --fixation_threshold $fixation_threshold \
`# optimization` \
--lr $lr --tr_step $tr_step --va_step $va_step --loss_names $loss_names --loss_functions $loss_functions --loss_weights $loss_weights \
--task_names $task_names --task_components $task_component1 --task_components $task_component2 \
`# model` \
--architecture $architecture \
`# gaze network` \
--gaze_encoder_type $gaze_encoder_type --gaze_encoder_units $gaze_encoder_units --gaze_encoder_activations $gaze_encoder_activations --gaze_encoder_kernels $gaze_encoder_kernels --gaze_encoder_paddings $gaze_encoder_paddings \
--gaze_attention_units $gaze_attention_units --gaze_attention_activations $gaze_attention_activations \
--object_attention_units $object_attention_units --object_attention_activations $object_attention_activations \
`# pose network` \
--pose_encoder_units $pose_encoder_units --pose_encoder_activations $pose_encoder_activations \
--pose_mu_var_units $pose_mu_var_units --pose_mu_var_activations $pose_mu_var_activations \
--pose_decoder_units $pose_decoder_units --pose_decoder_activations $pose_decoder_activations \
--key_object $key_object --key_pose $key_pose \
`# checkpointing` \
--log_root $log_root --weight_root $weight_root --model_name $model_name \
--restore_from_checkpoint $restore_from_checkpoint
cd shell_scripts/ICRA2022