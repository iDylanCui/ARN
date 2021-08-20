import os
import pickle
import time
import math
import torch
import pickle
import random
import itertools
import pandas as pd
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
from shared_adam import SharedAdam
from parse_args import args
from util import build_er2id, read_vocab, flatten, load_all_triples_from_txt, get_adjacent, build_qa_vocab, process_text_file, get_all_entity_neighours, initialize_action_space, set_seed, index2word
from util import load_data_in_dict
from policy_network import Policy_Network
from enviroment import Environment
from train_worker import TrainWorker
from valid_worker import ValidWorker
from test_woker import TestWorker
import warnings
warnings.filterwarnings('ignore') 

START_RELATION = 'START_RELATION' 
NO_OP_RELATION = 'NO_OP_RELATION'
NO_OP_ENTITY = 'NO_OP_ENTITY'
DUMMY_RELATION = 'DUMMY_RELATION'
DUMMY_ENTITY = 'DUMMY_ENTITY'
PADDING_ENTITIES = [DUMMY_ENTITY, NO_OP_ENTITY]
PADDING_ENTITIES_ID = [0, 1]
PADDING_RELATIONS = [DUMMY_RELATION, START_RELATION, NO_OP_RELATION]
flag_words = ['<pad>', '<unk>']

def run_train(args, d_train_data, d_valid_data, d_word2id, d_entity2id, d_relation2id, keqa_checkpoint_path, d_entity_neighours, d_entity2bucketid, d_action_space_buckets, checkpoint_path, intermediate_result_path):
    d_entity_neighours = mp.Manager().dict(d_entity_neighours) 

    shared_model = Policy_Network(args, len(d_word2id), len(d_entity2id), len(d_relation2id), keqa_checkpoint_path)


    shared_model.share_memory()
    params = filter(lambda p: p.requires_grad, shared_model.parameters())
    optim = SharedAdam(params, lr = args.learning_rate) 
    optim.share_memory()
    
    set_seed(args.seed)

    done = mp.Value('i', False) 
    train_gstep = mp.Value('i', 0)
    valid_gstep = mp.Value('i', 0)

    l_intimediate_result = mp.Manager().list() 
    d_ckpt = mp.Manager().dict()

    train_env = Environment(args, d_train_data, len(d_entity2id), shuffle = True)
    valid_env = Environment(args, d_valid_data, len(d_entity2id), shuffle = False) 

    num_episodes = math.ceil(len(d_train_data) / args.batch_size) * args.num_epochs 

    procs = []
    p = ValidWorker(args, len(procs), done, shared_model, valid_env, train_gstep, valid_gstep, d_entity_neighours, d_entity2bucketid, d_action_space_buckets, l_intimediate_result, d_ckpt, checkpoint_path, len(d_word2id), len(d_entity2id), len(d_relation2id), keqa_checkpoint_path)

    p.start()
    procs.append(p)

    for _ in range(args.num_workers):
        p = TrainWorker(args, len(procs), done, shared_model, optim, train_env, train_gstep, valid_gstep, d_entity_neighours, d_entity2bucketid, d_action_space_buckets, num_episodes, len(d_word2id), len(d_entity2id), len(d_relation2id), keqa_checkpoint_path)
        p.start()
        procs.append(p)
    
    for p in procs:
        p.join()

    best_episode = d_ckpt['best_episode_num']
    best_hits1 = d_ckpt['best_hits@1']
    
    running_episodes = []
    intermediate_results = []
    for i in range(len(l_intimediate_result)):
        episode, i_result = l_intimediate_result[i]
        running_episodes.append(episode)
        intermediate_results.append(i_result)
    
    dataframe = pd.DataFrame({'episodes': running_episodes,'hits@1': intermediate_results})
    dataframe.to_csv(intermediate_result_path, index=False, sep=',') 
    return best_episode, best_hits1

def run_test(args, d_test_data, d_word2id, d_entity2id, d_relation2id, keqa_checkpoint_path, d_entity_neighours, d_entity2bucketid, d_action_space_buckets, reqa_checkpoint_path):
    d_id2word = index2word(d_word2id)
    d_entity_neighours = mp.Manager().dict(d_entity_neighours) 
    d_results = mp.Manager().dict()

    test_env = Environment(args, d_test_data, len(d_entity2id), shuffle = False)

    procs = []
    p = TestWorker(args, len(procs), test_env, d_entity_neighours, d_entity2bucketid, d_action_space_buckets, d_entity2id, d_relation2id, reqa_checkpoint_path, d_results, len(d_word2id), len(d_entity2id), len(d_relation2id), keqa_checkpoint_path)

    p.start()
    procs.append(p)
    
    for p in procs:
        p.join()
    
    test_result = round(d_results['hits@1'], 4)
    return test_result
 

def get_dataset_path(args):
    if args.dataset.startswith("MetaQA"):
        kb_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Dataset/MetaQA-Vanilla/kb.txt"
        entity2id_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Dataset/MetaQA-Vanilla/entity2id.txt"
        relation2id_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Dataset/MetaQA-Vanilla/relation2id.txt"

        if args.dataset.endswith("1H"):
            args.max_hop = 1
            dataset_file = "1-hop"
        
        elif args.dataset.endswith("2H"):
            args.max_hop = 2
            dataset_file = "2-hop"
        
        elif args.dataset.endswith("3H"):
            args.max_hop = 3
            dataset_file = "3-hop"
        
        train_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Dataset/MetaQA-Vanilla/{}/train_qa.txt".format(dataset_file)
        valid_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Dataset/MetaQA-Vanilla/{}/valid_qa.txt".format(dataset_file)
        test_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Dataset/MetaQA-Vanilla/{}/test_qa.txt".format(dataset_file)
        word2id_file = os.path.abspath(os.path.join(os.getcwd(), "../..") + "/Dataset/MetaQA-Vanilla/{}/word2id.pkl").format(dataset_file)
        word_embedding_file = os.path.abspath(os.path.join(os.getcwd(), "../..") + "/Dataset/MetaQA-Vanilla/{}/word_embeddings.npy").format(dataset_file)
        entity_neighours_path = os.path.abspath(os.path.join(os.getcwd(), "../..") + "/Dataset/MetaQA-Vanilla/{}/entity_neighours.pkl").format(dataset_file)
        
    
    elif args.dataset.startswith("PQ"):
        if args.dataset == "PQ-2H":
            args.max_hop = 2
            dataset_file = "PQ-2H"
    
        elif args.dataset == "PQ-3H":
            args.max_hop = 3
            dataset_file = "PQ-3H"
        
        elif args.dataset == "PQ-mix":
            args.max_hop = 3
            dataset_file = "PQ-Mix"
    
        elif args.dataset == "PQL-2H":
            args.max_hop = 2
            dataset_file = "PQL-2H"
        
        elif args.dataset == "PQL-3H":
            args.max_hop = 3
            dataset_file = "PQL-3H"
        
        elif args.dataset == "PQL-mix":
            args.max_hop = 3
            dataset_file = "PQL-Mix"
        
        
        kb_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Dataset/PathQuestion/{}/kb.txt".format(dataset_file)
        train_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Dataset/PathQuestion/{}/train_qa.txt".format(dataset_file)
        valid_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Dataset/PathQuestion/{}/valid_qa.txt".format(dataset_file)
        test_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Dataset/PathQuestion/{}/test_qa.txt".format(dataset_file)
        entity2id_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Dataset/PathQuestion/{}/entity2id.txt".format(dataset_file)
        relation2id_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Dataset/PathQuestion/{}/relation2id.txt".format(dataset_file)
        word2id_file = os.path.abspath(os.path.join(os.getcwd(), "../..") + "/Dataset/PathQuestion/{}/word2id.pkl").format(dataset_file)
        word_embedding_file = os.path.abspath(os.path.join(os.getcwd(), "../..") + "/Dataset/PathQuestion/{}/word_embeddings.npy").format(dataset_file)
        entity_neighours_path = os.path.abspath(os.path.join(os.getcwd(), "../..") + "/Dataset/PathQuestion/{}/entity_neighours.pkl").format(dataset_file)
    
    return kb_path, train_path, valid_path, test_path, entity2id_path, relation2id_path, word2id_file, word_embedding_file, entity_neighours_path


def get_id_vocab(entity2id_path, relation2id_path):
    if not os.path.isfile(entity2id_path):
        if args.add_reversed_edges:
            build_er2id(kb_path, entity2id_path, relation2id_path, PADDING_ENTITIES, PADDING_RELATIONS, reversed_edge=True)
        else:
            build_er2id(kb_path, entity2id_path, relation2id_path, PADDING_ENTITIES, PADDING_RELATIONS)
    
    d_entity2id = read_vocab(entity2id_path)
    d_relation2id = read_vocab(relation2id_path)

    return d_entity2id, d_relation2id

def Hits_1(final_score, batch_answers, hit_num, total_num):
    batch_size = batch_answers.size()[0]
    _, batch_hit_idx = torch.topk(final_score, k = 1) 
    batch_results = torch.gather(batch_answers, 1, batch_hit_idx).view(-1) 

    hit_num += batch_results.sum().item()
    total_num += batch_size
    return hit_num, total_num


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True

    if args.train:
        if args.dataset.startswith("MetaQA"):
            args.add_reversed_edges = True
        else:
            args.add_reversed_edges = False

        kb_path, train_path, valid_path, test_path, entity2id_path, relation2id_path, word2id_file, word_embedding_file, entity_neighours_path = get_dataset_path(args)

        d_entity2id, d_relation2id = get_id_vocab(entity2id_path, relation2id_path)

        if not os.path.isfile(word2id_file):
            d_word2id = build_qa_vocab(train_path, valid_path, word2id_file, args.min_freq, flag_words)
        else:
            d_word2id = pickle.load(open(word2id_file, 'rb'))

        train_data = process_text_file(train_path, d_entity2id, d_word2id)
        valid_data = process_text_file(valid_path, d_entity2id, d_word2id)
        test_data = process_text_file(test_path, d_entity2id, d_word2id)

        d_train_data = load_data_in_dict(train_data)
        d_valid_data = load_data_in_dict(valid_data)
        d_test_data = load_data_in_dict(test_data)

        
        triples = load_all_triples_from_txt(kb_path, d_entity2id, d_relation2id, args.add_reversed_edges)
        triple_dict = get_adjacent(triples) 
            
        if not os.path.exists(entity_neighours_path):
            d_entity_neighours = get_all_entity_neighours(train_data, valid_data, test_data, triple_dict, args)
            
            with open(entity_neighours_path, 'wb') as f:
                pickle.dump(d_entity_neighours, f)
        else:
            with open(entity_neighours_path, 'rb') as f:
                d_entity_neighours = pickle.load(f)

        d_entity2bucketid, d_action_space_buckets = initialize_action_space(len(d_entity2id), triple_dict, args.bucket_interval)
        o_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Output/4_A3C"
        if not os.path.exists(o_path):
            os.mkdir(o_path)

        base_output_path = os.path.join(o_path, args.dataset) 
        if not os.path.exists(base_output_path):
            os.mkdir(base_output_path)

        kge_output_path = os.path.join(base_output_path, args.KGE_model)
        if not os.path.exists(kge_output_path):
            os.mkdir(kge_output_path)
        
        output_path = os.path.join(kge_output_path, args.strategy)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        keqa_checkpoint_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Output/3_KEQA/{}/{}/keqa_best.ckpt".format(args.dataset, args.KGE_model)
        
    
        reqa_checkpoint_path = os.path.join(output_path, "best.ckpt")
        intermediate_result_path = os.path.join(output_path, "valid_results.csv")

        if not os.path.exists(reqa_checkpoint_path):
            best_episode, best_hits1_valid = run_train(args, d_train_data, d_valid_data, d_word2id, d_entity2id, d_relation2id, keqa_checkpoint_path, d_entity_neighours, d_entity2bucketid, d_action_space_buckets, reqa_checkpoint_path, intermediate_result_path)
    
    elif args.eval:
        d_datasets = {
            "PQ-2H": ("ConvE", "top1"),
            "PQ-3H": ("ConvE", "top1"),
            "PQ-mix": ("ConvE", "sample"),
            "PQL-2H": ("TuckER", "sample"),
            "PQL-3H": ("TuckER", "sample"),
            "PQL-mix": ("ConvE", "top1"),
            "MetaQA-1H": ("DistMult", "avg"),
            "MetaQA-2H": ("ComplEx", "top1"),
            "MetaQA-3H": ("ConvE", "top1"),
        }
        if args.dataset.startswith("MetaQA"):
                args.add_reversed_edges = True
        else:
            args.add_reversed_edges = False

        kb_path, train_path, valid_path, test_path, entity2id_path, relation2id_path, word2id_file, word_embedding_file, entity_neighours_path = get_dataset_path(args)

        d_entity2id, d_relation2id = get_id_vocab(entity2id_path, relation2id_path)

        if not os.path.isfile(word2id_file):
            d_word2id = build_qa_vocab(train_path, valid_path, word2id_file, args.min_freq, flag_words)
        else:
            d_word2id = pickle.load(open(word2id_file, 'rb'))

        test_data = process_text_file(test_path, d_entity2id, d_word2id)

        d_test_data = load_data_in_dict(test_data)

        triples = load_all_triples_from_txt(kb_path, d_entity2id, d_relation2id, args.add_reversed_edges)
        triple_dict = get_adjacent(triples) 
        
        if not os.path.exists(entity_neighours_path):
            d_entity_neighours = get_all_entity_neighours(None, None, test_data, triple_dict, args)
            
            with open(entity_neighours_path, 'wb') as f:
                pickle.dump(d_entity_neighours, f)
        else:
            with open(entity_neighours_path, 'rb') as f:
                d_entity_neighours = pickle.load(f)

        d_entity2bucketid, d_action_space_buckets = initialize_action_space(len(d_entity2id), triple_dict, args.bucket_interval) 
        
        args.KGE_model = d_datasets[args.dataset][0]
        args.strategy = d_datasets[args.dataset][1]

        keqa_checkpoint_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Output/3_KEQA/{}/{}/keqa_best.ckpt".format(args.dataset, args.KGE_model)
        reqa_checkpoint_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Output/4_A3C/{}/{}/reqa_best.ckpt".format(args.dataset, args.KGE_model)

        best_hits1_test = run_test(args, d_test_data, d_word2id, d_entity2id, d_relation2id, keqa_checkpoint_path, d_entity_neighours, d_entity2bucketid, d_action_space_buckets, reqa_checkpoint_path)
        print(args.dataset, args.KGE_model, args.beam_size, best_hits1_test)