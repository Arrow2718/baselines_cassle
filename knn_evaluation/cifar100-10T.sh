python TIL_linear_eval_alltasks.py \
    --dataset "cifar100" \
    --eval "accum" \
    --n-tasks 10 \
    --num-classes 100 \
    --data-dir "/home/sol-ex-inanis/Data/" \
    --ckpt-json "/home/sol-ex-inanis/Code/baselines_cassle/ckpt_jsons/barlow-cifar100-10T.json"

python TIL_linear_eval_alltasks.py \
    --dataset "cifar100" \
    --eval "curr" \
    --n-tasks 10 \
    --num-classes 100 \
    --data-dir "/home/sol-ex-inanis/Data/" \
    --ckpt-json "/home/sol-ex-inanis/Code/baselines_cassle/ckpt_jsons/barlow-cifar100-10T.json"

python TIL_linear_eval_alltasks.py \
    --dataset "cifar100" \
    --eval "all" \
    --n-tasks 10 \
    --num-classes 100 \
    --data-dir "/home/sol-ex-inanis/Data/" \
    --ckpt-json "/home/sol-ex-inanis/Code/baselines_cassle/ckpt_jsons/barlow-cifar100-10T.json"