# conda activate sage 

items=(
    # 'gb_dataset_n=2_field=QQ_ex'
    'gb_dataset_n=2_field=F7_ex'
    'gb_dataset_n=2_field=F31_ex'
)

for item in "${items[@]}" ; do
    save_dir=data/${item} 
    mkdir -p $save_dir
    echo =====================================
    echo $save_dir
    echo =====================================
    
    echo -- src/data/gbdataset.sage ----------
    sage src/data/gbdataset.sage generation $save_dir experiments/generalization/config/${item}.yaml > ${save_dir}/run.log
    echo ""
done