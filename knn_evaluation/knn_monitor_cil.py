from tqdm import tqdm
import torch.nn.functional as F 
import torch
import numpy as np
from utils.metrics import mask_classes

# code copied from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=RI1Y8bSImD7N
# test using a knn monitor



def knn_monitor_cil(args,net, dataset, memory_data_loaders, test_data_loader, device, cl_default, task_id, k=200, t=0.1, hide_progress=False):
    net.eval()
    classes = args.n_classes_per_task * args.n_tasks
    #classes = 100
    total_top1 = total_top1_mask = total_top5 = total_num = 0.0
    #feature_bank = []
    features_dict = {}

    # Extracting labels
    labels_bank = []

    idx = 0

    for loader in tqdm(memory_data_loaders[ 0 : task_id + 1 ], desc = f" fetching training labels for {idx} loader", leave = False, position = 2):

        labels_bank.append( torch.tensor( loader.dataset.targets,  device = 'cuda') )
        idx += 1
        # print(labels_bank)
    feature_labels = torch.cat( labels_bank, dim = 0 )
    
    with torch.no_grad():
    ### Generating Feature Banks :
    
        for mask in tqdm(range(task_id + 1), desc = f"Generating features using model {task_id} ", leave = False, position = 3):
            
            # creating feature bank with each mask   
            mask_bank = [] 
            
            data_idx = 0
            for loader in tqdm(memory_data_loaders[ 0 : task_id + 1 ], desc = f"Generating features using mask {mask}", leave = False, position=4):

            # extracting features

                for data, target in tqdm(loader, desc = f"Loader {data_idx} using mask {mask}", leave=False, disable=False, position = 4):
                    if cl_default:
                        feature = net(data.cuda(non_blocking=True), return_features=True)
                    else:
                        feature = net(data.cuda(non_blocking=True))

                    mask_bank.append(feature)

                data_idx += 1

            features_dict[mask] = F.normalize(torch.cat( mask_bank, dim=0).contiguous(), dim = 1)
                
            
        test_labels = torch.tensor( test_data_loader.dataset.targets, device =  "cuda") 

        pred_scores = []
    
        for mask in tqdm(range(task_id + 1), desc = f"generating test features and pred_scores for model {task_id} ", leave = False, position = 3):

            pred_bank = []
           

            for data, target in tqdm(test_data_loader, desc = f"extracting pred scores using mask {mask}", leave = False, disable= False, position = 4):

                if cl_default:
                    feature = net(data.cuda(non_blocking=True), return_features=True)
                else:
                    feature = net(data.cuda(non_blocking=True))
                feature = F.normalize(feature, dim=1)


                batch_pred = knn_predict( feature, features_dict[mask], feature_labels, classes, k, t )

                

                pred_bank.append(batch_pred)
               
            
            pred_bank = torch.cat( pred_bank, dim = 0 )

            pred_bank = pred_bank[: , mask * args.n_classes_per_task  :  (mask + 1) * args.n_classes_per_task  ]
            # pred_bank = F.normalize(pred_bank)
            # print("pred_bank.size() after slice: ",pred_bank.size())
            pred_scores.append(pred_bank)
       
      
        pred_scores = torch.cat(pred_scores, dim = 1)


        total_num = len(test_data_loader.dataset)

        _, preds = torch.max(pred_scores.data, 1)
        # print("preds.size(): ",preds.size())
        # print("--------------------------------------------------------------------------------------------------------------")
        # print(test_labels[0])
        
        total_top1 = torch.sum(preds == test_labels).item()

        acc = total_top1 / total_num * 100  

        

    return acc




# # knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# # implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
# def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
#     # compute cos similarity between each feature vector and feature bank ---> [B, N]
#     sim_matrix = torch.mm(feature, feature_bank)
#     # [B, K]
#     sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
#     # [B, K]
#     sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
#     sim_weight = (sim_weight / knn_t).exp()

#     # counts for each class
#     one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
#     # [B*K, C]
#     one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
#     # weighted score ---> [B, C]
#     pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

#     return pred_scores

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    sim_matrix = torch.mm(feature, feature_bank.t())
   
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    return pred_scores
