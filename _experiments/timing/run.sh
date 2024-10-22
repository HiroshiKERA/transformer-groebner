# nohup bash experiments/timing/run.sh &

field=RR
for n in {2..5} ; do
    base=gb_dataset_n=${n}_field=${field}
    config_path=config/${base}.yaml
    save_path=results2/timing/${base}
    mkdir -p $save_path

    # sage experiments/timing/timing.sage --dryrun
    # python3 experiments/timing/timing.sage.py --field $field --config_path $config_path --save_path $save_path > ${save_path}/run.log
    sage experiments/timing/timing.sage --field $field --config_path $config_path --save_path $save_path > ${save_path}/run.log
done

# field=F7
# for n in {2..5} ; do
#     base=gb_dataset_n=${n}_field=${field}
#     config_path=config/${base}.yaml
#     save_path=results/timing/${base}
#     mkdir -p $save_path

#     sage experiments/timing/timing.sage --dryrun
#     python experiments/timing/timing.sage.py --field $field --config_path $config_path --save_path $save_path > ${save_path}/run.log
# done

# field=F31
# for n in {2..5} ; do
#     base=gb_dataset_n=${n}_field=${field}
#     config_path=config/${base}.yaml
#     save_path=results/timing/${base}
#     mkdir -p $save_path

#     sage experiments/timing/timing.sage --dryrun
#     python experiments/timing/timing.sage.py --field $field --config_path $config_path --save_path $save_path > ${save_path}/run.log
# done


# field=QQ
# for n in {2..5} ; do
#     base=gb_dataset_n=${n}_field=${field}_dense
#     config_path=config/${base}.yaml
#     save_path=results/timing/${base}
#     mkdir -p $save_path

#     sage experiments/timing/timing.sage --dryrun
#     python experiments/timing/timing.sage.py --field $field --config_path $config_path --save_path $save_path > ${save_path}/run.log
# done

# field=F7
# for n in {2..5} ; do
#     base=gb_dataset_n=${n}_field=${field}_dense
#     config_path=config/${base}.yaml
#     save_path=results/timing/${base}
#     mkdir -p $save_path

#     sage experiments/timing/timing.sage --dryrun
#     python experiments/timing/timing.sage.py --field $field --config_path $config_path --save_path $save_path > ${save_path}/run.log
# done

# field=F31
# for n in {2..5} ; do
#     base=gb_dataset_n=${n}_field=${field}_dense
#     config_path=config/${base}.yaml
#     save_path=results/timing/${base}
#     mkdir -p $save_path

#     sage experiments/timing/timing.sage --dryrun
#     python experiments/timing/timing.sage.py --field $field --config_path $config_path --save_path $save_path > ${save_path}/run.log
# done


# field=QQ
# for n in {2..2} ; do
#     base=gb_dataset_n=${n}_field=${field}_ex
#     config_path=experiments/generalization/config/${base}.yaml
#     save_path=results/timing/${base}
#     mkdir -p $save_path

#     sage experiments/timing/timing.sage --dryrun
#     python experiments/timing/timing.sage.py --field $field --config_path $config_path --save_path $save_path > ${save_path}/run.log
# done

# field=F7
# for n in {2..2} ; do
#     base=gb_dataset_n=${n}_field=${field}_ex
#     config_path=experiments/generalization/config/${base}.yaml
#     save_path=results/timing/${base}
#     mkdir -p $save_path

#     sage experiments/timing/timing.sage --dryrun
#     python experiments/timing/timing.sage.py --field $field --config_path $config_path --save_path $save_path > ${save_path}/run.log
# done

# field=F31
# for n in {2..2} ; do
#     base=gb_dataset_n=${n}_field=${field}_ex
#     config_path=experiments/generalization/config/${base}.yaml
#     save_path=results/timing/${base}
#     mkdir -p $save_path

#     sage experiments/timing/timing.sage --dryrun
#     python experiments/timing/timing.sage.py --field $field --config_path $config_path --save_path $save_path > ${save_path}/run.log
# done
