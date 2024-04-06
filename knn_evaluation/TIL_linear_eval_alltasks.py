import os
import torch
from tqdm import tqdm
import argparse
import numpy as np
from knn_monitor import knn_monitor
import sys
from pytorch_lightning import  seed_everything
from torchvision.models import resnet18, resnet50
import torch.nn as nn
from ..cassle.utils.classification_dataloader import prepare_datasets
from ..cassle.utils.pretrain_dataloader import split_dataset
from torch.utils.data import DataLoader
import json


def main(device, args):
    
    
    with open(args.ckpt_json, "r") as f:
        ckpt_dict = json.load(f)

    seed_everything(5)

    tasks = torch.randperm(args.num_classes).chunk(args.n_tasks)
    print("tasks", tasks)
    
    model_accuracies = np.zeros((args.n_tasks,args.n_tasks))
    max_task_acc = np.zeros(args.n_tasks)
    forgetting_rates = np.zeros((args.n_tasks,args.n_tasks))

    model = resnet18()
    model.fc =nn.Identity()
    
    memory_dataset, test_dataset = prepare_datasets(args.dataset, T_train = nn.Module(), T_val = nn.Module(), data_dir= args.data_dir, train_dir= args.train_dir, val_dir= args.val_dir)

    for idx in range(args.n_tasks):
        task_memory_dataset, _ = split_dataset(dataset =memory_dataset, task_idx= idx, num_tasks= args.n_tasks, split_strategy= "class", tasks= tasks )
        task_test_dataset, = split_dataset(dataset =test_dataset, task_idx= idx, num_tasks= args.n_tasks, split_strategy= "class", tasks= tasks )

        memory = DataLoader(task_memory_dataset, batch_size= 512, num_workers= 6)
        test =  DataLoader(task_test_dataset, batch_size= 512, num_workers= 6)

        memory_loaders.append(memory)
        test_loaders.append(test)

    memory_loaders, test_loaders =  [], []

    for t in tqdm(range(0, args.n_tasks), desc='Evaluatinng'):
      if args.eval == 'all':
          eval_tids = [j for j in range(args.n_tasks)]
      elif args.eval == 'curr':
          eval_tids = [t]
      elif args.eval == 'accum':
          eval_tids = [j for j in range(t + 1)]
      else:
          sys.exit('Stopped!! Wrong eval-type.')
          
      
      model_path = ckpt_dict["ckpt"][t]
      print("model_path", model_path)

      save_dict = torch.load(model_path, map_location='cpu')
    
      model.load_state_dict(save_dict)

      model = model.to(device)
      knn_acc_list = []
      
      cl_default = True
      for i in eval_tids:
          acc, acc_mask = knn_monitor(args,model.net.module.backbone, memory_loaders[i], test_loaders[i],
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


   



    results = { "model_acc" : model_accuracies,
            "forgetting_rates": forgetting_rates,
            "avg_acc" : avg_acc,
            "avg_fgt" : avg_fgt  }




    print(results)
    
    result_file_path = os.path.join(args.result_dir, args.dataset + f"_TIL_{args.eval}.txt")
    print("result_file_path", result_file_path)

   
    os.makedirs(result_file_path, exist_ok=True)
    with open(result_file_path, "w") as f:
	    f.write( str(results) )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type = int)
    parser.add_argument("--num_classes", type = int)
    parser.add_argument("--data-dir", type = str)
    parser.add_argument("--train-dir", type = str)
    parser.add_argument("--val-dir", type = str)
    parser.add_argument("--ckpt-json", type = str)
    
    args = parser.parse_args()
    
    main(device="cuda", args=args)