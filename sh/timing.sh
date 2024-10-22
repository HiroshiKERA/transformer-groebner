experiment=timing
task=shape
nvars=2
field=GF7

model=bart

data_name=${task}_n=${nvars}_field=${field}
data_path=data/${task}/${data_name}/data_${field}_n=${nvars}.test.lex.infix

_model_path=${field}_n=${nvars}_ep=${epochs}_bs=${batch_size}
model_path=results/${task}/${model}/${_model_path}

save_path=dump/results/${experiment}/${task}/${data_name}

mkdir -p $save_path
sage src/experiments/timing.sage    --data_path $data_path \
                                    --model_path $model_path \
                                    --save_path $save_path \
                                    --num_variables $nvars \
                                    --field $field \
                                    --timeout 1