CUDA_VISIBLE_DEVICES=0 \
python job_launcher.py --script bash_files/continual/baseline_scripts/cifar10_barlow_first_task.sh

CUDA_VISIBLE_DEVICES=0 \
python job_launcher.py --script bash_files/continual/baseline_scripts/cifar10_barlow_rest.sh