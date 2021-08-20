import re
import os
import random
import torch
import pickle
import zipfile
import numpy as np
from collections import Counter
from collections import defaultdict
import operator
from tqdm import tqdm

EPSILON = float(np.finfo(float).eps)

def build_er2id(kb_file, e_output_file, r_output_file, pre_entities,pre_relations, reversed_edge = False):
    l_entities = []
    l_relations = []

    with open(kb_file) as fr:
        for i, line in enumerate(fr):
            line = line.strip()
            s, r, o = line.split("\t")
            s, r, o = s.strip(), r.strip(), o.strip()
            l_entities.append(s)
            l_relations.append(r)
            if reversed_edge == True:
                l_relations.append(r + "_inverse")
            l_entities.append(o)
    
    l_entities = list(set(l_entities))
    l_entities = pre_entities + l_entities

    l_relations = list(set(l_relations))
    l_relations = pre_relations + l_relations

    with open(e_output_file, "w") as f:
        for i , e in enumerate(l_entities):
            f.write(e + "\t" + str(i) + "\n")

    with open(r_output_file, "w") as f:
        for i , r in enumerate(l_relations):
            f.write(r + "\t" + str(i) + "\n")

def read_vocab(vocab_file):
    d_item2id = {}

    with open(vocab_file) as fr:
        for i, line in enumerate(fr):
            line = line.strip()
            items = line.split("\t")
            d_item2id[items[0]] = int(items[1])

    return d_item2id

def index2word(word2id):
    return {i: w for w, i in word2id.items()}

def flatten(l):
    flatten_l = []
    for c in l:
        if type(c) is list or type(c) is tuple:
            flatten_l.extend(flatten(c))
        else:
            flatten_l.append(c)
    return flatten_l

def load_all_triples_from_txt(data_path, d_entity2id, d_relation2id, add_reverse_relations=False):
    triples = []
    
    def triple2ids(s, r, o):
        return (d_entity2id[s], d_relation2id[r], d_entity2id[o])
    
    with open(data_path) as f:
        for line in f.readlines():
            s, r, o = line.strip().split("\t")
            s, r, o = s.strip(), r.strip(), o.strip()
            triples.append(triple2ids(s, r, o))
            if add_reverse_relations:
                triples.append(triple2ids(o, r + '_inverse', s))
    
    print('{} triples loaded from {}'.format(len(triples), data_path))
    return triples

def get_adjacent(triples):
    triple_dict = defaultdict(defaultdict)

    for triple in triples:
        s_id, r_id, o_id = triple
        
        if r_id not in triple_dict[s_id]:
            triple_dict[s_id][r_id] = set()
        triple_dict[s_id][r_id].add(o_id)

    return triple_dict


def print_all_model_parameters(model):
    print('\nModel Parameters')
    print('--------------------------')
    for name, param in model.named_parameters():
        print(name, param.numel(), 'requires_grad={}, device= {}'.format(param.requires_grad, param.device))
    param_sizes = [param.numel() for param in model.parameters()]
    print('Total # parameters = {}'.format(sum(param_sizes)))
    print('--------------------------')


def build_qa_vocab(train_file, valid_file, word2id_output_file, min_freq, flag_words):
    count = Counter()
    with open(train_file) as fr:
        for i, line in enumerate(fr):
            line = line.strip()
            items = line.split("\t")
            question_pattern = items[0].strip()
            words_pattern = [word for word in question_pattern.split(" ")]
            count.update(words_pattern)
    
    with open(valid_file) as fr:
        for i, line in enumerate(fr):
            line = line.strip()
            items = line.split("\t")
            question_pattern = items[0].strip()
            words_pattern = [word for word in question_pattern.split(" ")]
            count.update(words_pattern)

    count = {k: v for k, v in count.items()}
    count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)  
    vocab = [w[0] for w in count if w[1] >= min_freq]
    vocab = flag_words + vocab
    word2id = {k: v for k, v in zip(vocab, range(0, len(vocab)))}
    print("word len: ", len(word2id))
    assert word2id['<pad>'] == 0, "ValueError: '<pad>' id is not 0"

    with open(word2id_output_file, 'wb') as fw:
        pickle.dump(word2id, fw)

    return word2id

def initialize_word_embedding(word2id, glove_path, word_embedding_file):
    word_embeddings = np.random.uniform(-0.1, 0.1, (len(word2id), 300))
    seen_words = []

    gloves = zipfile.ZipFile(glove_path)
    for glove in gloves.infolist():
        with gloves.open(glove) as f:
            for line in f:
                if line != "":
                    splitline = line.split()
                    word = splitline[0].decode('utf-8')
                    embedding = splitline[1:]
                    if word in word2id and len(embedding) == 300:
                        temp = np.array([float(val) for val in embedding])
                        word_embeddings[word2id[word], :] = temp/np.sqrt(np.sum(temp**2))
                        seen_words.append(word)

    word_embeddings = word_embeddings.astype(np.float32)
    word_embeddings[0, :] = 0. 
    print("pretrained vocab %s among %s" %(len(seen_words), len(word_embeddings)))
    unseen_words = set([k for k in word2id]) - set(seen_words)
    print("unseen words = ", len(unseen_words), unseen_words) 
    np.save(word_embedding_file, word_embeddings)
    return word_embeddings

def token_to_id(token, token2id, flag_words = "<unk>"):
    return token2id[token] if token in token2id else token2id[flag_words]

def get_relation_signature(question_pattern_id):
    return " ".join(map(str, question_pattern_id))

def process_text_file(text_file, d_entity2id, d_word2id):
    l_data = []

    with open(text_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            question_pattern, topic_entity, answer_entities = line.split('\t')
            answer_entities = answer_entities.split("|")

            question_pattern_id = [token_to_id(word, d_word2id) for word in question_pattern.strip().split(" ")]
            topic_entity_id = d_entity2id[topic_entity]
            answer_entities_id = [d_entity2id[entity] for entity in answer_entities]

            l_data.append([topic_entity_id, question_pattern_id, answer_entities_id])
    
    return l_data


def getNeighbourhoodbyDict(entities, adj_list, hop, dataset, center = True):
    l_neighbourhood = []
    if type(entities) is not list:
        l_entities = [entities]
    else:
        l_entities = entities

    while(hop > 0):
        hop -= 1
        l_one_step = []
        for entity in l_entities:
            for edge in adj_list[entity]:
                l_one_step.extend([i for i in list(adj_list[entity][edge])])
                l_neighbourhood.extend([i for i in list(adj_list[entity][edge])])
        l_entities.clear()
        l_entities.extend([i for i in l_one_step])
    
    if re.search("\dH$", dataset): 
        l_neighbourhood = list(set(l_entities))
    else:
        l_neighbourhood = list(set(l_neighbourhood)) 
    
    if center == False: 
        if entities in l_neighbourhood:
            l_neighbourhood.remove(entities)

    return l_neighbourhood

def get_all_entity_neighours(train_data, valid_data, test_data, triple_dict, args):
    d_entity_neighours = {}
    if train_data == None and valid_data == None: 
        all_data = test_data
    else:
        all_data = train_data + valid_data + test_data
    
    all_data_tqdm = tqdm(all_data, total=len(all_data), unit="data")

    for i, row in enumerate(all_data_tqdm):
        topic_entity_id, _, _ = row
        if topic_entity_id not in d_entity_neighours:
            l_neighbourhood = getNeighbourhoodbyDict(topic_entity_id, triple_dict, args.max_hop, args.dataset)
            d_entity_neighours[topic_entity_id] = l_neighbourhood
        else:
            continue

    return d_entity_neighours

def getEntityActions(subject, triple_dict, NO_OP_RELATION = 2):
    action_space = []

    if subject in triple_dict:
        for relation in triple_dict[subject]:
            objects = triple_dict[subject][relation]
            for obj in objects: 
                action_space.append((relation, obj))
        
    action_space.insert(0, (NO_OP_RELATION, subject)) 

    return action_space

def vectorize_action_space(action_space_list, action_space_size, DUMMY_ENTITY = 0, DUMMY_RELATION = 0): 
    bucket_size = len(action_space_list)
    r_space = torch.zeros(bucket_size, action_space_size) + DUMMY_ENTITY 
    e_space = torch.zeros(bucket_size, action_space_size) + DUMMY_RELATION
    r_space = r_space.long()
    e_space = e_space.long()
    action_mask = torch.zeros(bucket_size, action_space_size)
    for i, action_space in enumerate(action_space_list): 
        for j, (r, e) in enumerate(action_space): 
            r_space[i, j] = r
            e_space[i, j] = e
            action_mask[i, j] = 1
    action_mask = action_mask.long()
    return (r_space, e_space), action_mask

def initialize_action_space(num_entities, triple_dict, bucket_interval):
    d_action_space_buckets = {}
    d_action_space_buckets_discrete = defaultdict(list)
    d_entity2bucketid = torch.zeros(num_entities, 2).long()
    num_facts_saved_in_action_table = 0

    for e1 in range(num_entities):
        action_space = getEntityActions(e1, triple_dict) 
        key = int(len(action_space) / bucket_interval) + 1 
        d_entity2bucketid[e1, 0] = key
        d_entity2bucketid[e1, 1] = len(d_action_space_buckets_discrete[key]) 
        d_action_space_buckets_discrete[key].append(action_space)
        num_facts_saved_in_action_table += len(action_space)
    
    print('{} facts saved in action table'.format(
        num_facts_saved_in_action_table - num_entities)) 
    for key in d_action_space_buckets_discrete: 
        d_action_space_buckets[key] = vectorize_action_space(
            d_action_space_buckets_discrete[key], key * bucket_interval) 
    return d_entity2bucketid, d_action_space_buckets
    
def safe_log(x):
    return torch.log(x + EPSILON)

def entropy(p):
    return torch.sum(-p * safe_log(p), 1)

def get_num_gpus():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        num_gpus = len(visible.split(','))
    else:
        num_gpus = torch.cuda.device_count()
    return num_gpus

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data_in_dict(data_list):
    d_data = {}
    for i, data in enumerate(data_list):
        question_ids = data[1] 
        head_id = data[0]
        tail_ids = data[2]
        d_data[i] = (question_ids, head_id, tail_ids)
    
    return d_data

def rearrange_vector_list(l, offset):
    for i, v in enumerate(l):
        l[i] = v[offset]