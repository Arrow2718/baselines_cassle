import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter, knn_monitor
from datasets import get_dataset
from models.optimizers import get_optimizer, LR_Scheduler
from utils.loggers import *


def evaluate_single(model, dataset, test_loader, memory_loader, device, k, last=False) -> Tuple[list, list, list, list]:
    accs, accs_mask_classes = [], []
    knn_accs, knn_accs_mask_classes = [], []
    correct = correct_mask_classes = total = 0
    knn_acc, knn_acc_mask = knn_monitor(args,model.net.module.backbone, dataset, memory_loader, test_loader, device, args.cl_default, task_id=k, k=min(args.train.knn_k, len(dataset.memory_loaders[k].dataset))) 

    return knn_acc


def main(device, args):

    dataset = get_dataset(args)
    dataset_copy = get_dataset(args)
    train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args)
    model = get_model(args, device, len(train_loader), get_aug(train=False, train_classifier=False, **args.aug_kwargs))

    model_accuracies = np.zeros((args.n_tasks,args.n_tasks))
    max_task_acc = np.zeros(args.n_tasks)
    forgetting_rates = np.zeros((args.n_tasks,args.n_tasks))

    # model_accuracies = [[0.0 for col in range(dataset_copy.N_TASKS)] for row in range(dataset_copy.N_TASKS)]

    # max_task_acc = [0.0 for col in range(dataset_copy.N_TASKS)]
    # forgetting_rates = [0.0 for col in range(dataset_copy.N_TASKS)]
    #print(model_accuracies)

    train_loaders, memory_loaders, test_loaders = [], [], []
    for t in range(args.n_tasks):
      tr, me, te = dataset.get_data_loaders(args)
      print("tr sample size for",t,"   ",len(tr.dataset))
      print("te sample size for",t,"   ",len(te.dataset))
      print("me sample size for",t,"   ",len(me.dataset))

      train_loaders.append(tr)
      memory_loaders.append(me)
      test_loaders.append(te)

    for t in tqdm(range(0, args.n_tasks), desc='Evaluatinng'):
      if args.eval.type == 'all':
          eval_tids = [j for j in range(args.n_tasks)]
      elif args.eval.type == 'curr':
          eval_tids = [t]
      elif args.eval.type == 'accum':
          eval_tids = [j for j in range(t + 1)]
      else:
          sys.exit('Stopped!! Wrong eval-type.')
          
      #ckpt_dir ="/home/phd2201101011/Project_OCL/Woked_UCL-main/checkpoints/cifar100_10_T_si"
      #ckpt_dir ="/home/phd2201101011/Project_OCL/Woked_UCL-main/checkpoints/cifar100_10_T_der"
      #model_path  = f"/home/deepakprasad/OCL/UCL/checkpoints/Imagenet32-32_5T_alp_0.4_results/mixup_ucl_simsiam-tinyimg-experiment-resnet18_mixup_run_0_{t}.pth"
      #model_path  = f"/home/deepakprasad/OCL/UCL/checkpoints/4_OCT_Imagenet32-32_5T_alp_0.1_500_epochs_results/mixup_ucl_simsiam-tinyimg-experiment-resnet18_mixup_run_0_{t}.pth"
      
      model_path = os.path.join(f"{args.ckpt_dir}", f"{args.cl_model_name}_{args.name}_{t}.pth")
      print("model_path", model_path)

      #, f"{args.model.cl_model}_{args.name}_{t}.pth
      #model_path = os.path.join(ckpt_dir, "")
      save_dict = torch.load(model_path, map_location='cpu')
      std_list =  save_dict['state_dict'].items()
      #print("len(std_list)", len(std_list))
      for k, v in list(std_list)[:16] :
         if 'backbone.' not in k:
            print("save_dict['state_dict'].items()", k)

      #model.net.module.backbone.load_state_dict({k[16:]:v for k, v in save_dict['state_dict'].items() if 'backbone.' in k}, strict=True)
      model.net.module.backbone.load_state_dict({k[16:]:v for k, v in save_dict['state_dict'].items() if 'backbone.' in k}, strict=True)

      model = model.to(device)
      knn_acc_list = []
      
      cl_default = True
      for i in eval_tids:
          acc, acc_mask = knn_monitor(args,model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i],
                  device, cl_default = cl_default, task_id=i, k=min(args.train.knn_k, len(eval_tids)))
        
          print(acc)
          knn_acc_list.append(acc)
          print("i", i)
          model_accuracies[t][i] = acc

      for i in range(t):
        for j in range(t):
            if max_task_acc[i] < model_accuracies[j][i]:
                max_task_acc[i] = model_accuracies[j][i]
        forgetting_rates[t][i] = max_task_acc[i] - model_accuracies[t][i]
     


    model_wise_sum_acc = [ (model_accuracies[i][:].sum() / (i + 1)) for i in range(args.n_tasks) ]
    model_wise_fgt_acc = [ (forgetting_rates[i][:].sum() / (i + 1)) for i in range(args.n_tasks) ]

    avg_acc = np.mean(model_wise_sum_acc)

    avg_fgt = np.sum(model_wise_fgt_acc)/(args.n_tasks-1)


    # model_wise_sum_acc = [ (model_accuracies[i][:].sum() / (i + 1)) for i in range(dataset.N_TASKS) ]

    # avg_acc = np.mean(model_wise_sum_acc)

    # avg_fgt = np.mean(forgetting_rates)



    results = { "model_acc" : model_accuracies,
            "forgetting_rates": forgetting_rates,
            "avg_acc" : avg_acc,
            "avg_fgt" : avg_fgt  }




    print(results)
    chk_path, chk_dir_name = os.path.split(os.path.split(args.ckpt_dir)[0])


    
    result_path = os.path.join(f"{args.result_dir}",f"{chk_dir_name}")
    
    result_file_path = os.path.join(f"{args.result_dir}",f"{chk_dir_name}", f"{args.result_mod}.json")

    print("result_file_path", result_file_path)

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(result_file_path, "w") as f:
	    f.write( str(results) )


if __name__ == "__main__":
    args = get_args()
    device = "cuda"
    main(device=device, args=args)