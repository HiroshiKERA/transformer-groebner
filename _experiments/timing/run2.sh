
term_order=lex
encoding=prefix
num_prints=10
batch_size=200
fields=(
    "F7"
)

for n in {2..2} ; do
    for field in "${fields[@]}" ; do

        echo ${n}, ${field}

        # data_path=results2/timing/gb_dataset_n=${n}_field=${field}/data
        data_path=data_dev/gb_dataset_n=${n}_field=${field}/data
        load_path=results2/shape_gb_${term_order}/gb_dataset_n=${n}_field=${field}
        save_path=results2/timing/gb_dataset_n=${n}_field=${field}
        success_table_path=results2/timing/gb_dataset_n=${n}_field=${field}/success_table.pickle

        mkdir -p $save_path

        CUDA_VISIBLE_DEVICES=0 python3 experiments/prediction/prediction.py \
                            --data_path $data_path\
                            --load_path $load_path\
                            --save_path $save_path\
                            --data_encoding $encoding\
                            --term_order $term_order\
                            --field $field\
                            --batch_size $batch_size > $save_path/process_failure.log

        # sage experiments/prediction/showcase.sage \
        #                     --num_variables $n\
        #                     --field $field\
        #                     --data_path $data_path\
        #                     --load_path $load_path\
        #                     --save_path $save_path\
        #                     --data_encoding $encoding\
        #                     --term_order $term_order\
        #                     --num_prints $num_prints >> $save_path/process_failure.log

        # sage experiments/timing/timing_failure.sage --dryrun

        sage experiments/timing/timing_failure.sage \
                            --num_variables $n\
                            --field $field\
                            --data_path $data_path\
                            --load_path $load_path\
                            --save_path $save_path\
                            --data_encoding $encoding\
                            --term_order $term_order\
                            --success_table_path $success_table_path >> $save_path/process_failure.log
    done
done