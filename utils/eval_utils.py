from sklearn import metrics
import numpy as np
import torch
import torch.nn.functional as F
import os

def save_best_record(epoch, auc, file_path):
    fo = open(file_path, "w")
    fo.write("epoch: {}\n".format(epoch))
    fo.write(str(auc))
    fo.close()

def cal_auc(scores,labels):
    fpr, tpr, thresholds=metrics.roc_curve(labels,scores,pos_label=1)
    auc=metrics.auc(fpr,tpr)
    return auc

def eval(total_scores,total_labels,logger):
    total_scores = np.array(total_scores)
    total_labels = np.array(total_labels)
    
    auc = cal_auc(total_scores, total_labels)
    
    if logger == None:
        print('AUC \t{}'.format(auc))
    else:
        logger.info('===================')
        
        logger.info('AUC\t {}'.format(auc))
    return auc

    

