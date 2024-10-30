#!/bin/bash
wandb_project_name=neurips-cr
gpu_id=1
task=shape
max_coefficient=100
max_degree=20
epochs=8
batch_size=16

# Define arrays for different configurations
fields=("QQ")
nvars=(4)
encoding_methods=("hybrid")
regression_weights=0.01
# cont_model=ffn2

# Define density values for each nvar
declare -A density_map
density_map[2]=1.0
density_map[3]=0.6
density_map[4]=0.3
density_map[5]=0.2

for field in "${fields[@]}"; do
    for nvar in "${nvars[@]}"; do
        for encoding_method in "${encoding_methods[@]}"; do
            # Skip standard encoding for RR field
            if [ "$field" = "RR" ] && [ "$encoding_method" = "standard" ]; then
                continue
            fi

            # Set model based on encoding method
            if [ "$encoding_method" = "standard" ]; then
                model="bart"
            else
                model="bart+"
            fi

            # Get density for current nvar
            density=${density_map[$nvar]}

            # Set data_name and paths based on density
            if (( $(echo "$density < 1.0" | bc -l) )); then
                data_name=${task}_n=${nvar}_field=${field}_density=${density}
                _save_path=${field}_n=${nvar}_density=${density}
            else
                data_name=${task}_n=${nvar}_field=${field}
                _save_path=${field}_n=${nvar}
            fi

            data_path=data/${task}/${data_name}/data_${field}_n=${nvar}
            data_config_path=config/${data_name}.yaml
            group=${encoding_method}_${model}  # _$cont_model
            save_path=results/${task}/${group}/${_save_path}
            run_name=${task}_${_save_path}

            mkdir -p $save_path
            
            echo "Starting experiment: Field=${field}, Nvars=${nvar}, Density=${density}, Encoding=${encoding_method}, Model=${model}"
            
            CUDA_VISIBLE_DEVICES=$gpu_id nohup python3 src/main.py \
                --save_path $save_path \
                --model $model \
                --data_path $data_path \
                --data_config_path $data_config_path \
                --task $task \
                --num_variables $nvar \
                --field $field \
                --max_coefficient $max_coefficient \
                --max_degree $max_degree \
                --epochs $epochs \
                --batch_size $batch_size \
                --test_batch_size $batch_size \
                --group $group \
                --exp_name $wandb_project_name \
                --exp_id $run_name \
                --regression_weights $regression_weights \
                --encoding_method $encoding_method \
                > ${save_path}/run.log &
                
                # --continuous_embedding_model $cont_model \

            sleep 1
        done
    done
done