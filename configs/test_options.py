import argparse
import os
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='ucf')
    parser.add_argument('--win_size',type=int,default=5) 
    parser.add_argument('--dropout_rate',type=float,default=0.8)
    parser.add_argument('--segment_len',type=int,default=16) 
    parser.add_argument('--norm',type=int,default=0) 

    args=parser.parse_args()
   

    return args
