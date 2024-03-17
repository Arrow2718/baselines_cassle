CUDA_VISIBLE_DEVICES=0 \
python job_launcher.py --script bash_files/continual/baseline_scripts/mnist_barlow_first_task.sh

CUDA_VISIBLE_DEVICES=0 \
python job_launcher.py --script bash_files/continual/baseline_scripts/mnist_barlow_rest.sh