term_order=lex
field=F7
encoding=prefix
batch_size=250
num_prints=10

for n in {2..2} ; do
    data_path=data/gb_dataset_n=${n}_field=${field}_ex/data
    load_path=results/shape_gb_${term_order}/gb_dataset_n=${n}_field=${field}
    save_path=results/prediction_${term_order}_field=${field}/gb_dataset_n=${n}_ex
    mkdir -p $save_path

    CUDA_VISIBLE_DEVICES=0 python experiments/prediction/prediction.py \
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