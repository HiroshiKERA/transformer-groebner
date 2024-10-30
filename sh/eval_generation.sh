#!/bin/bash
experiment=generation
gpu_id=2

task=shape

batch_size=100
th_GF=0.5  # only used for finite fields
th_QR=0.31 # only used for RR/QQ field

# Define arrays for different configurations
fields=("QQ")
nvars=(2 3 4 5)
encoding_methods=("standard" "hybrid")

# Define density values for each nvar
declare -A density_map
density_map[2]=1.0
density_map[3]=0.6
density_map[4]=0.3
density_map[5]=0.2

for field in "${fields[@]}"; do
    for nvar in "${nvars[@]}"; do
        for encoding_method in "${encoding_methods[@]}"; do
            
            th=0.0

            # Skip standard encoding for RR field
            if [ "$field" = "RR" ] && [ "$encoding_method" = "standard" ]; then
                continue
            fi

            # Set model based on encoding method
            if [ "$encoding_method" = "standard" ]; then
                model="bart"
            else
                model="bart+"
                if [ "$field" = "RR" ] || [ "$field" = "QQ" ]; then
                    th=$th_QR
                else
                    th=$th_GF
                fi
            fi

            # Get density for current nvar
            density=${density_map[$nvar]}

            # Set data_name and paths based on density
            if (( $(echo "$density < 1.0" | bc -l) )); then
                data_name=${task}_n=${nvar}_field=${field}_density=${density}
                _model_path=${field}_n=${nvar}_density=${density}
            else
                data_name=${task}_n=${nvar}_field=${field}
                _model_path=${field}_n=${nvar}
            fi

            if [ "$field" = "QQ" ] && [ "$encoding_method" = "hybrid" ]; then
                data_path=data/${task}/${data_name}/data_${field}_n=${nvar}.test.lex.infix+
            else
                data_path=data/${task}/${data_name}/data_${field}_n=${nvar}.test.lex.infix
            fi

            data_config_path=config/${data_name}.yaml

            group=${encoding_method}_${model}
            model_path=results/${task}/${group}/${_model_path}

            save_path=results/${experiment}/${task}/${group}/${data_name}
            mkdir -p "$save_path"
            
            echo "Starting experiment: Field=${field}, Nvars=${nvar}, Density=${density}, Encoding=${encoding_method}, Model=${model}"
            # echo "data: $data_path"
            
            CUDA_VISIBLE_DEVICES=$gpu_id python3 src/experiments/generation.py \
                --save_path $save_path \
                --data_path $data_path \
                --model_path $model_path \
                --field $field \
                --disable_tqdm \
                --th $th \
                --batch_size $batch_size > ${save_path}/run.log

            echo
        done
    done
done