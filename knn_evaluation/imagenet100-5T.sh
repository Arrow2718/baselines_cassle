python TIL_linear_eval_alltasks.py \
    --dataset "imagenet100" \
    --eval "accum" \
    --n-tasks 5 \
    --num-classes 100 \
    --data-dir "/home/sol-ex-inanis/Data/" \
    --ckpt-json "/home/sol-ex-inanis/Code/baselines_cassle/ckpt_jsons/barlow-imagenet100-10T.json" \
    --train-dir tmp/tiny-imagenet-200/train  \
    --val-dir tmp/tiny-imagenet-200/val
python TIL_linear_eval_alltasks.py \
    --dataset "imagenet100" \
    --eval "curr" \
    --n-tasks 5 \
    --num-classes 100 \
    --data-dir "/home/sol-ex-inanis/Data/" \
    --ckpt-json "/home/sol-ex-inanis/Code/baselines_cassle/ckpt_jsons/barlow-imagenet100-10T.json" \
    --train-dir tmp/tiny-imagenet-200/train  \
    --val-dir tmp/tiny-imagenet-200/val
python TIL_linear_eval_alltasks.py \
    --dataset "imagenet100" \
    --eval "all" \
    --n-tasks 5 \
    --num-classes 100 \
    --data-dir "/home/sol-ex-inanis/Data/" \
    --ckpt-json "/home/sol-ex-inanis/Code/baselines_cassle/ckpt_jsons/barlow-imagenet100-10T.json" \
    --train-dir tmp/tiny-imagenet-200/train  \
    --val-dir tmp/tiny-imagenet-200/val