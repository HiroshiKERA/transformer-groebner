wandb_project_name=neurips-cr
gpu_id=1

task=shape
nvars=2
field=GF7
max_coefficient=100  # should be higher than coeff_bound
max_degree=20  # should be higher than max_degree_F * 2 + max_degree_G

# encoding_method=standard
encoding_method=hybrid

# positional_encoding=embedding
# positional_encoding=sinusoidal

epochs=8
batch_size=16

data_name=${task}_n=${nvars}_field=${field}
data_path=data/${task}_no_choose_degree/${data_name}/data_${field}_n=${nvars}
data_config_path=config/${data_name}.yaml

echo $data_path

model=bart

group=${encoding_method}_${model}_no_choose_degree
_save_path=${field}_n=${nvars}
save_path=results/${task}/${group}/${_save_path}
run_name=${task}_${_save_path}

mkdir -p $save_path
CUDA_VISIBLE_DEVICES=$gpu_id  nohup python3 src/main.py  --save_path $save_path \
                                            --data_path $data_path \
                                            --data_config_path $data_config_path \
                                            --task $task \
                                            --num_variables $nvars \
                                            --field $field \
                                            --max_coefficient $max_coefficient \
                                            --max_degree $max_degree \
                                            --epochs $epochs \
                                            --batch_size $batch_size \
                                            --test_batch_size $batch_size \
                                            --group $group \
                                            --exp_name $wandb_project_name \
                                            --exp_id $run_name \
                                            --regression_weights 0.01 \
                                            --encoding_method $encoding_method > ${save_path}/run.log &

                                            # --max_steps_per_epoch $max_steps_per_epoch \
                                            # --positional_encoding $positional_encoding \