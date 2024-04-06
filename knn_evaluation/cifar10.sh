python TIL_linear_eval_alltasks.py \
    --dataset "cifar10" \
    --eval "accum" \
    --n-tasks 5 \
    --num-classes 10 \
    --data-dir "/home/sol-ex-inanis/Data/" \
    --ckpt-json "/home/sol-ex-inanis/Code/baselines_cassle/ckpt_jsons/barlow-cifar10.json"

python TIL_linear_eval_alltasks.py \
    --dataset "cifar10" \
    --eval "curr" \
    --n-tasks 5 \
    --num-classes 10 \
    --data-dir "/home/sol-ex-inanis/Data/" \
    --ckpt-json "/home/sol-ex-inanis/Code/baselines_cassle/ckpt_jsons/barlow-cifar10.json"

python TIL_linear_eval_alltasks.py \
--dataset "cifar10" \
--eval "all" \
--n-tasks 5 \
--num-classes 10 \
--data-dir "/home/sol-ex-inanis/Data/" \
--ckpt-json "/home/sol-ex-inanis/Code/baselines_cassle/ckpt_jsons/barlow-cifar10.json"