./test.sh arch="eye_gaze" root="ICRA2022" name="vanilla" \
`# dataset parameters` \
data_loader="mogaze_eye" \
use_instructions=1 \
gaze_length=9 time_step_size=12 \
grid_sizes1="5 5" grid_sizes2="4 2" grid_sizes3="3 2" object_padded_length=89 \
`# checkpointing` \
epoch_names="key_object_epoch_0049_best_0049.pt key_pose_epoch_0051_best_0051.pt" \
strict=0 \
result_root="ICRA2022" result_name="vanilla" \
0