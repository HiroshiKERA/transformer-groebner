# task="shape"
task="cauchy"
num_variables_range=(2 3)
fields=("QQ")

for num_variables in "${num_variables_range[@]}"; do
    for field in "${fields[@]}"; do

        config="${task}_n=${num_variables}_field=${field}_uniform"
        save_dir="data/${task}/${config}"
        mkdir -p "$save_dir"
        
        echo "Running configuration: $config"
        
        sage src/dataset/build_dataset.sage \
            --save_path "$save_dir" \
            --config_path "config/${config}.yaml" > "${save_dir}/run_${config}.log" 2>&1
        
        echo "Completed: $config"
        echo "------------------------"
    done
done

echo "All datasets generated."