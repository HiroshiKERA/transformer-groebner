#!/bin/bash

task="shape"
num_variables_range=(2 3 4 5)
fields=("GF7" "GF31" "QQ" "RR")

# num_variablesとdensitiesの対応を関数化
get_density() {
    local num_var=$1
    case $num_var in
        3) echo "0.6" ;;
        4) echo "0.3" ;;
        5) echo "0.2" ;;
        *) echo "1.0" ;; # デフォルト値
    esac
}

for num_variables in "${num_variables_range[@]}"; do
    for field in "${fields[@]}"; do

        density=$(get_density $num_variables)

        if [[ $(echo "$density != 1.0" | bc -l) -eq 1 ]]; then
            config="${task}_n=${num_variables}_field=${field}_density=${density}"
        else
            config="${task}_n=${num_variables}_field=${field}"
        fi
        save_dir="data/${task}_no_choose_degree/${config}"
        mkdir -p "$save_dir"
        
        echo "Running configuration: $config with density $density"
        
        sage src/dataset/build_dataset.sage \
            --save_path "$save_dir" \
            --testset_only \
            --config_path "config/${config}.yaml" > "${save_dir}/run_${config}.log" # 2>&1
        
        echo "Completed: $config"
        echo "------------------------"
    done
done

echo "All datasets generated."