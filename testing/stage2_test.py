
import sys
from torch import logit
sys.path.append('..')
from apex import amp
from models.stage2_net import *
from configs import test_options
from utils.eval_utils import *
from stage2training.UCF_dataset import *


def load_model(model, state_dict):
    new_dict={}
    for key,value in state_dict.items():
        new_dict[key]=value
    #print('new_dict:',new_dict)
    model.load_state_dict(new_dict)

def get_model_pretrain(args, model_path_pre):
   
    checkpoint = model_path_pre
    print('checkpoint:', checkpoint)
    model=Temporal_I3D_SGA_STD(args.dropout_rate).cuda()


    opt_level='O1'
    model=amp.initialize(model,opt_level=opt_level,keep_batchnorm_fp32=None)
    load_model(model, torch.load(checkpoint)['model'])

    return model

def eval_epoch(args, model, fea_size):
    model = model.eval()
    total_scores = []
    total_labels = []
    
    with torch.no_grad():
        model = model.eval()
        for test_feat, label, test_anno in zip(test_feats, test_labels, test_annos):
            temp_score = []
            for i in range(test_feat.shape[0]):
                feat = test_feat[i]
               
                win_size = args.win_size
                fea_list = []
                if i >= win_size:
                    for w_index in range(1, win_size+1):
                        fea_list.append(test_feat[i - w_index])
                elif i == 0:
                    fea_list.append(feat)
                else:
                    for interval_index in range(1, i +1):
                        fea_list.append(test_feat[i - interval_index])

                fea = np.array(fea_list)
                fea = np.mean(fea,axis=0)
                feat = np.concatenate((feat, fea),axis=1)
                
                feat = torch.from_numpy(np.array(feat)).float().cuda().view([-1,fea_size])
                feat = torch.reshape(feat,(1,10,fea_size))
                logits= model(feat)
                logits = logits[:,:,-1]
                score = torch.mean(logits,dim=1).item()
                temp_score.extend([score]*16)

            total_labels.extend(test_anno[:len(temp_score)].tolist())
            total_scores.extend(temp_score)

    return eval(total_scores,total_labels,logger=None)

if __name__=='__main__':
    args=test_options.parse_args()
    if args.dataset == 'ucf':
        model_path_pre = '../train_ckpts/stage_2/UCF_0.8622.pth'
        test_mask_txt_path = '../data/Temporal_Anomaly_Annotation_New.txt'
        model=get_model_pretrain(args, model_path_pre)
        fea_size = 4096
        test_feats,test_labels,test_annos=UCF_test(test_mask_txt_path, args.segment_len, args.norm)
        eval_epoch(args, model, fea_size)
        

    
    

    