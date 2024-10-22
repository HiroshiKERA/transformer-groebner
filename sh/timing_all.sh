experiment=timing
# timeout=5

model=bart

task=shape
num_variables_range=(2 3 4 5)
fields=("GF7" "GF31" "QQ")

# num_variables_range=(2)
# fields=("QQ")


get_density() {
    local num_var=$1
    case $num_var in
        3) echo "0.6" ;;
        4) echo "0.3" ;;
        5) echo "0.2" ;;
        *) echo "1.0" ;; # デフォルト値
    esac
}

for nvars in "${num_variables_range[@]}"; do
    density=$(get_density $nvars)

    for field in "${fields[@]}"; do

        if [[ $(echo "$density != 1.0" | bc -l) -eq 1 ]]; then
            data_name=${task}_n=${nvars}_field=${field}_density=${density}
        else
            data_name=${task}_n=${nvars}_field=${field} 
        fi

        data_path=data/${task}_no_choose_degree/${data_name}/data_${field}_n=${nvars}.test.lex.infix

        _model_path=${field}_n=${nvars}
        model_path=results/${task}/${model}/${_model_path}
        save_path=results/${experiment}/${task}_no_choose_degree/${data_name}
        
        mkdir -p $save_path
        
        echo "Running experiment for n=$nvars, field=$field"
        
        sage src/experiments/timing.sage --data_path $data_path \
                                         --model_path $model_path \
                                         --save_path $save_path \
                                         --num_variables $nvars \
                                         --field $field \
                                        #  --timeout $timeout
        
        echo "Finished experiment for n=$nvars, field=$field"
        echo "----------------------------------------"
    done
done

echo "All experiments completed."