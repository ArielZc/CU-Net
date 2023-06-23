from asyncore import write
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from torch import nn
from torch.nn import init

def read_testing_txt(file_path):
    vids_dict={}
    with open(file_path) as f:
        lines=f.readlines()
        for line in lines:
            line_list=line.strip().split('\t')
            annotations=[]
            for i in range(2,6):
                if int(line_list[i])==-1:
                    break
                else:
                    annotations.append(int(line_list[i]))

            vids_dict[line_list[0]]=[line_list[1],annotations,int(line_list[-1])]

    return vids_dict


def get_anno(mask_txt, segment_len):
    test_dict = read_testing_txt(mask_txt)
    
    annotation_dict = {}
    for key in test_dict.keys():
        ano_type, anno, frames_num = test_dict[key]
        annotation = np.zeros(frames_num - frames_num % (segment_len), dtype=int)
        if len(anno) >= 2:
            front = anno[0]
            back = anno[1]
            if front < annotation.shape[0]:
                annotation[front:min(back, annotation.shape[0])] = 1
        if len(anno) == 4:
            front = anno[-2]
            back = anno[-1]
            if front < annotation.shape[0]:
                annotation[front:min(back, annotation.shape[0])] = 1
        annotation_dict[key] = annotation
    
  
    return annotation_dict

def UCF_test(mask_txt, segment_len,norm):
    annotation_dict = get_anno(mask_txt,segment_len)
    annos = []
    labels=[]
    names=[]
    output_feats=[]
    rgb_list_file = '../data/ucf-i3d-test.list'
    file_list = list(open(rgb_list_file))
    v_names = []
    for v_count in range(0,len(file_list)):
        file = file_list[v_count]
        p,f = os.path.split(file)
        f = f[0:-9]
        v_name = f
        key = v_name
        v_names.append(v_name)

        feat=np.load(file_list[v_count].strip('\n'), allow_pickle=True)
        feat = np.array(feat, dtype=np.float32)
        if norm==2:
            feat=feat/np.linalg.norm(feat,axis=-1,keepdims=True)
        
        if 'Normal' in key:
            labels.append('Normal')
        else:
            labels.append('Abnormal')
        anno = annotation_dict[key + '.mp4']
        if anno.shape[0]==0:
            print(key)
        if anno.shape[0] < feat.shape[0] * 16:
            num_dif = feat.shape[0] * 16 - anno.shape[0]
            anno = anno.tolist()
            last_anno = anno[-1]
            for j in range(0,num_dif):
                anno.append(0)
            anno = np.array(anno)
        output_feats.append(feat)
        annos.append(anno)
        names.append(key)
    return output_feats,labels,annos