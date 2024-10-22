# nohup bash experiments/timing/run.sh &

field=QQ
load_path=results/timing
profile_base=dataset_profiles_N=1000.yaml
runtime_base=timing_results_N=1000.yaml
echo $field 
echo $load_path
echo $profile_base
python experiments/timing/table_generator.py --field $field \
                                             --profile_paths ${load_path}/gb_dataset_n=2_field=${field}_dense/${profile_base} \
                                                             ${load_path}/gb_dataset_n=3_field=${field}_dense/${profile_base} \
                                                             ${load_path}/gb_dataset_n=4_field=${field}_dense/${profile_base} \
                                                             ${load_path}/gb_dataset_n=5_field=${field}_dense/${profile_base} \
                                             --runtime_paths ${load_path}/gb_dataset_n=2_field=${field}_dense/${runtime_base} \
                                                             ${load_path}/gb_dataset_n=3_field=${field}_dense/${runtime_base} \
                                                             ${load_path}/gb_dataset_n=4_field=${field}_dense/${runtime_base} \
                                                             ${load_path}/gb_dataset_n=5_field=${field}_dense/${runtime_base} 

python experiments/timing/table_generator.py --field $field \
                                             --profile_paths ${load_path}/gb_dataset_n=2_field=${field}/${profile_base} \
                                                             ${load_path}/gb_dataset_n=3_field=${field}/${profile_base} \
                                                             ${load_path}/gb_dataset_n=4_field=${field}/${profile_base} \
                                                             ${load_path}/gb_dataset_n=5_field=${field}/${profile_base} \
                                             --runtime_paths ${load_path}/gb_dataset_n=2_field=${field}/${runtime_base} \
                                                             ${load_path}/gb_dataset_n=3_field=${field}/${runtime_base} \
                                                             ${load_path}/gb_dataset_n=4_field=${field}/${runtime_base} \
                                                             ${load_path}/gb_dataset_n=5_field=${field}/${runtime_base} 

# python experiments/timing/table_generator.py --field $field \
#                                              --profile_paths ${load_path}/gb_dataset_n=2_field=${field}_ex/${profile_base} \
#                                              --runtime_paths ${load_path}/gb_dataset_n=2_field=${field}_ex/${runtime_base}



