experiment=showcase

gpu_id=0

num_samples_to_show=12
print_style=latex  # latex or sage

task=shape
num_variables_range=(2 3 4 5)
fields=("QQ")

get_density() {
    local num_var=$1
    case $num_var in
        3) echo "0.6" ;;
        4) echo "0.3" ;;
        5) echo "0.2" ;;
        *) echo "1.0" ;; 
    esac
}

for nvars in "${num_variables_range[@]}"; do
    density=$(get_density $nvars)

    for field in "${fields[@]}"; do

        if [[ $(echo "$density != 1.0" | bc -l) -eq 1 ]]; then
            data_name=${task}_n=${nvars}_field=${field}_density=${density}
            _model_path=${field}_n=${nvars}_density=${density}
        else
            data_name=${task}_n=${nvars}_field=${field} 
            _model_path=${field}_n=${nvars}
        fi

        data_path=data/${task}/${data_name}/data_${field}_n=${nvars}.test.lex.infix
        model_path=results/${task}/standard_bart/${_model_path}
        generation_path=results/generation/${task}/standard_bart/${data_name}
        generation_results=${generation_path}/generation_results.yaml        
        # mkdir -p $save_path
        
        echo "Running experiment for n=$nvars, field=$field"
        
        CUDA_VISIBLE_DEVICES=$gpu_id sage src/experiments/showcase.sage --data_path $data_path \
                                            --model_path $model_path \
                                            --num_variables $nvars \
                                            --field $field \
                                            --generation_results $generation_results \
                                            --num_samples_to_show $num_samples_to_show \
                                            --print_style $print_style > $generation_path/showcase.txt
        
        
        echo "Finished experiment for n=$nvars, field=$field"
        echo "----------------------------------------"
    done
done

echo "All experiments completed."