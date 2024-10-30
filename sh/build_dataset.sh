task=shape
num_variables=2
field=QQ
density=1.0

config=${task}_n=${num_variables}_field=${field}

if [[ $(echo "$density != 1.0" | bc -l) -eq 1 ]]; then
    config=${task}_n=${num_variables}_field=${field}_density=${density}
fi

# save_dir=data/${task}/${config} #_reduced
save_dir=dump/${task}/${config} #_reduced

mkdir -p $save_dir
sage src/dataset/build_dataset.sage  --save_path $save_dir \
                                     --testset_only \
                                     --config_path config/${config}.yaml > ${save_dir}/run_${config}.log
                                        
                                        # --strictly_conditioned
