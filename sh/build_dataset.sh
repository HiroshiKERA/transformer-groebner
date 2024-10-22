task=shape
num_variables=2
field=GF7
density=1.0

config=${task}_n=${num_variables}_field=${field}

if [[ $(echo "$density != 1.0" | bc -l) -eq 1 ]]; then
    config=${task}_n=${num_variables}_field=${field}_density=${density}
fi

save_dir=data/${task}_no_choose_degree/${config} #_reduced
# save_dir=dump/${task}/${config} #_reduced

mkdir -p $save_dir
sage src/dataset/build_dataset.sage     --save_path $save_dir \
                                        --config_path config/${config}.yaml > ${save_dir}/run_${config}.log

                                        # --testset_only 
                                        # --strictly_conditioned
