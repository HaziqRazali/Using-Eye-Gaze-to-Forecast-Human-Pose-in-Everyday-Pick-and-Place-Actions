./train.sh arch="eye_gaze" root="ICRA2022" name="vanilla" \
`# dataset parameters` \
data_loader="mogaze_eye" \
use_instructions=1 \
gaze_length=9 time_step_size=12 \
grid_sizes1="5 5" grid_sizes2="4 2" grid_sizes3="3 2" object_padded_length=89 \
`# learning parameters` \
loss_names="key_relative_gaze key_object key_pose pose_posterior" \
loss_functions="mse mse mse kl_loss" \
loss_weights="[0.1]*1000 [1.0]*1000 [1.0]*1000 [1.0]*1000" \
task_names="key_object key_pose" \
task_component1="key_object" task_component2="key_pose pose_posterior" \
0