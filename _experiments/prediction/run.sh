term_order=lex
field=F7
model_field=F7
encoding=prefix
batch_size=200
num_prints=10
gb_type=shape
main_dir=results-uniform

for n in {2..2} ; do
    # shape
    data_path=data_uniform/${gb_type}/gb_dataset_n=${n}_field=${field}/data    

    load_path=${main_dir}/${gb_type}/${term_order}/gb_dataset_n=${n}_field=${field}
    # load_path=${main_dir}/${gb_type}/${term_order}/gb_dataset_n=${n}_field=${model_field}-C-ffn-0.01-MSE
    # load_path=results2/shape_gb_${term_order}/support_learning/gb_dataset_n=${n}_field=${field}
    
    save_path=${main_dir}/${gb_type}/prediction_${term_order}_field=${field}/gb_dataset_n=${n}
    # save_path=${main_dir}/${gb_type}/prediction_${term_order}_field=${model_field}_on_${field}/gb_dataset_n=${n}-C-ffn-0.01-MSE
    # save_path=results2/prediction_${term_order}_field=${field}/support_learning/gb_dataset_n=${n}

    # cauchy
    # data_path=data_dev/${gb_type}/gb_dataset_n=${n}_field=${field}/data
    # load_path=results2/${gb_type}/shape_gb_${term_order}/gb_dataset_n=${n}_field=${field}
    # # load_path=results2/${gb_type}/shape_gb_${term_order}/regression_weights=0.01/gb_dataset_n=${n}_field=${field}-C

    # save_path=results2/${gb_type}/prediction_${term_order}_field=${field}/gb_dataset_n=${n}
    # # save_path=results2/${gb_type}/prediction_${term_order}_field=${field}/regression_weight=0.01/gb_dataset_n=${n}-C



    mkdir -p $save_path

    CUDA_VISIBLE_DEVICES=1 python3 experiments/prediction/prediction.py \
                        --data_path $data_path\
                        --load_path $load_path\
                        --save_path $save_path\
                        --data_encoding $encoding\
                        --term_order $term_order\
                        --field $field\
                        --batch_size $batch_size > $save_path/process.log

    sage experiments/prediction/showcase.sage \
                        --num_variables $n\
                        --field $field\
                        --data_path $data_path\
                        --load_path $load_path\
                        --save_path $save_path\
                        --data_encoding $encoding\
                        --term_order $term_order\
                        --num_prints $num_prints >> $save_path/process.log
done
