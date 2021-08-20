import os
import sys
import argparse

argparser = argparse.ArgumentParser(sys.argv[0]) 

argparser.add_argument("--dataset",
                        type=str,
                        # default = 'MetaQA-1H',
                        default = 'MetaQA-2H',
                        # default = 'MetaQA-3H',
                        # default = "PQ-2H",
                        # default = "PQ-3H",
                        # default = "PQ-mix",
                        # default = "PQL-2H",
                        # default = "PQL-3H",
                        # default = "PQL-mix",
                        help="dataset for training")

argparser.add_argument("--KGE_model",
                        type=str,
                        # default='ComplEx',
                        # default='ConvE',
                        default='TuckER',
                        # default='DistMult',
                        help="KGE model")

argparser.add_argument('--seed', type=int, default=2021,
                    help='random seed')

argparser.add_argument('--partition', type=float, default= 0.1,
                    help='valid, test set partition ratio')

argparser.add_argument('--train', action='store_true',
                    help='train model')

argparser.add_argument('--eval', action='store_true',
                    help='eval model')


argparser.add_argument('--group_examples_by_query', type = bool, default= True, help='group examples by topic entity + query relation')

argparser.add_argument('--add_reversed_edges', action='store_true',
                    help='add reversed edges to extend training set')

# general parameters                    
argparser.add_argument("--max_hop",
                        type=int,
                        default=3,
                        help="max reasoning hop")

argparser.add_argument('--entity_dim', type=int, default=200,
                    help='entity embedding dimension')

argparser.add_argument('--relation_dim', type=int, default=200,
                    help='relation embedding dimension')

argparser.add_argument('--word_dim', type=int, default=300,
                    help='word embedding dimension')

argparser.add_argument('--word_dropout_rate', type=float, default=0.3,
                    help='word embedding dropout rate')

argparser.add_argument('--word_padding_idx', type=int, default=0,
                    help='word padding index')

argparser.add_argument('--is_train_word_emb', type=bool, default=True,
                    help='train word embedding or not')
                    
argparser.add_argument('--label_smoothing_epsilon', type=float, default=0.1,
                    help='epsilon used for label smoothing')

argparser.add_argument('--grad_norm', type=float, default=10,
                    help='norm threshold for gradient clipping')

argparser.add_argument('--emb_dropout_rate', type=float, default=0.3,
                    help='Knowledge graph embedding dropout rate')

# TransE parameters
argparser.add_argument('--margin', type=float, default=1,
                    help='margin used for base MAMES training')

# ConvE parameters
argparser.add_argument('--hidden_dropout_rate', type=float, default=0.3,
                    help='ConvE hidden layer dropout rate')
argparser.add_argument('--feat_dropout_rate', type=float, default=0.2,
                    help='ConvE feature dropout rate')
argparser.add_argument('--emb_2D_d1', type=int, default=10,
                    help='ConvE embedding 2D shape dimension 1 (default: 10)')
argparser.add_argument('--emb_2D_d2', type=int, default=20,
                    help='ConvE embedding 2D shape dimension 2 (default: 20)')
argparser.add_argument('--num_out_channels', type=int, default=32,
                    help='ConvE number of output channels of the convolution layer')
argparser.add_argument('--kernel_size', type=int, default=4,
                    help='ConvE kernel size')

argparser.add_argument("--min_freq", type=int, default=0, help="Minimum frequency for words")

# Transformer parameters
argparser.add_argument('--head_num', type=int, default=4,
                    help='Transformer head number')

argparser.add_argument('--hidden_dim', type=int, default=100,
                    help='Transformer hidden dimension')

argparser.add_argument('--encoder_layers', type=int, default=2,
                    help='Transformer encoder layers number')

argparser.add_argument('--encoder_dropout_rate', type=float, default=0.3,
                    help='Transformer encoder dropout rate')

argparser.add_argument('--kge_framework_freeze', type=str, default="no_freeze", help='KGE framework parameters freeze')

# Reinforce Learning
argparser.add_argument('--history_dim', type=int, default=200,
                    help='path encoder LSTM hidden dimension')

argparser.add_argument('--relation_only', type=bool, default=False,
                    help='search with relation information only, ignoring entity representation')

argparser.add_argument('--rl_dropout_rate', type=float, default=0.3,
                    help='reinforce learning dropout rate')

argparser.add_argument('--history_layers', type=int, default=2,
                    help='path encoder LSTM layers')

argparser.add_argument('--bucket_interval', type=int, default=10,
                    help='adjacency list bucket size')

argparser.add_argument('--strategy', type=str, 
                    default='sample',
                    # default='avg',
                    # default='top1',
                    help='baseline used by the policy gradient algorithm') 

argparser.add_argument('--gamma', type=float, default=0.98,
                    help='moving average weight') 

argparser.add_argument('--tau', type=float, default=1.00,
                    help='GAE tau')

argparser.add_argument('--entropy_coef', type=float, default=0.02,
                    help = "entropy loss coefficient") 

argparser.add_argument('--value_loss_coef', type=float, default=0.5,
                    help = "A3C value loss coefficient") 

argparser.add_argument('--use_keqa_vector', type=bool, default=True,
                    help='use vector predicted by KEQA or not')
                
argparser.add_argument('--use_reward_shaping', type=bool, default=True,
                    help = "use reward shaping or not")

# general
argparser.add_argument('--batch_size', type=int, default=32,
                    help='mini-batch size')

argparser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='learning rate')

argparser.add_argument('--valid_episodes', type=int, default=100,
                    help = "each valid episodes")

argparser.add_argument('--beam_size', type=int, default=3,
                    help='size of beam used in beam search inference')

argparser.add_argument('--num_workers', type=int, default=3,
                    help = "train workers")

argparser.add_argument("--num_epochs",
                        type=int,
                        default=40,
                        help="maximum training epochs")

argparser.add_argument("--early_stop_patience",
                        type=int,
                        default=25,
                        help="early stop epoch")

args = argparser.parse_args()