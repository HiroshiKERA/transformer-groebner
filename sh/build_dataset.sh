task=shape
num_variables=2
field=GF7

config=${task}_n=${num_variables}_field=${field}_init
save_dir=data/${task}/${config}

mkdir -p $save_dir
sage src/dataset/build_dataset.sage     --save_path $save_dir \
                                        --strictly_conditioned \
                                        --config_path config/${config}.yaml # > ${save_dir}/run_${config}.log

                                        # --testset_only 
                                        # --strictly_conditioned
