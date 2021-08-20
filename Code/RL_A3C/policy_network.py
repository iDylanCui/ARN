import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from util import entropy
from KEQA_framework import KEQA_framework

class Policy_Network(nn.Module):
    def __init__(self, args, word_num, entity_num, relation_num, keqa_checkpoint_path, gpu_id = -1):
        super(Policy_Network, self).__init__()
        self.gpu_id = gpu_id
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.relation_only = args.relation_only
        self.history_dim = args.history_dim
        self.history_layers = args.history_layers
        self.word_dim = args.word_dim
        self.use_keqa_vector = args.use_keqa_vector
        self.rl_dropout_rate = args.rl_dropout_rate
        self.history_layers = args.history_layers
        self.strategy = args.strategy
        self.max_hop = args.max_hop + 1 

        self.word_num = word_num 
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.keqa_checkpoint_path = keqa_checkpoint_path
        self.KEQA = KEQA_framework(args, self.word_num, self.entity_num, self.relation_num, self.gpu_id)
        self.KEQA.load(self.keqa_checkpoint_path)

        self.Transformer = copy.deepcopy(self.KEQA.transformer_model)
        self.Word_embedding = copy.deepcopy(self.KEQA.word_embedding_layer)
        self.Relation_embedding = copy.deepcopy(self.KEQA.KGE_framework.relation_embeddings)
        self.Entity_embedding = copy.deepcopy(self.KEQA.KGE_framework.entity_embeddings)

        self.step_aware_linear = nn.ModuleList([nn.Linear(self.word_dim, self.relation_dim) for _ in range(self.max_hop)]) 
        self.W_att = nn.Linear(self.relation_dim, 1)

        if self.use_keqa_vector:
            self.input_dim = self.history_dim + self.relation_dim + self.entity_dim 
        else:
            self.input_dim = self.history_dim + self.relation_dim
        
        if self.relation_only:
            self.action_dim = self.relation_dim
        else:
            self.action_dim = self.relation_dim + self.entity_dim
        
        self.lstm_input_dim = self.action_dim 
        
        self.W1 = nn.Linear(self.input_dim, self.action_dim)
        self.W2 = nn.Linear(self.action_dim, self.action_dim)
        self.W1Dropout = nn.Dropout(self.rl_dropout_rate)
        self.W2Dropout = nn.Dropout(self.rl_dropout_rate)
        self.W_value = nn.Linear(self.input_dim - self.relation_dim, 1)

        self.path_encoder = nn.LSTM(input_size=self.lstm_input_dim,
                                    hidden_size=self.history_dim,
                                    num_layers=self.history_layers,
                                    batch_first=True) 
        

        for param in self.KEQA.parameters():
            param.requires_grad = False
        self.KEQA.eval() 
        
        self.initialize_modules()

    def get_step_aware_representation(self, t, batch_question, batch_sent_len):
        batch_question_embedding = self.Word_embedding(batch_question) 
        mask = self.batch_sentence_mask(batch_sent_len) 
        transformer_output = self.Transformer(batch_question_embedding.permute(1, 0 ,2), mask) 
        
        transformer_output = transformer_output.permute(1, 0 ,2) 

        step_aware_question_vector = self.step_aware_linear[t](transformer_output) 

        return torch.tanh(step_aware_question_vector), mask 

    def get_anticipated_entity_vector(self, batch_e_head, batch_question, batch_sent_len, d_entity_neighours):
        batch_entity_scores = self.KEQA(batch_question, batch_sent_len, batch_e_head)
        entity_mask = self.KEQA.get_max_hop_Neighbours_mask(batch_e_head, d_entity_neighours) 
        final_score = entity_mask * batch_entity_scores
        
        if self.strategy == 'sample':
            batch_index = torch.multinomial(final_score, 1) 
            batch_pred_vector = self.Entity_embedding(batch_index.view(-1))
        
        elif self.strategy == 'avg':
            batch_pred_vector = torch.matmul(final_score, self.Entity_embedding.weight) / torch.sum(final_score, dim=1, keepdim=True) 
        
        elif self.strategy == 'top1':
            _, batch_index = torch.topk(final_score, k=1, dim=1) 
            batch_pred_vector = self.Entity_embedding(batch_index.view(-1))

        return batch_pred_vector
    
    def get_relation_aware_question_vector(self, step_aware_question_vector, question_mask):
        relation_num, _ = self.Relation_embedding.weight.shape
        batch_size, seq_len, _ = step_aware_question_vector.shape
        step_aware_question_vector = step_aware_question_vector.unsqueeze(1).repeat(1, relation_num, 1, 1).view(batch_size * relation_num, seq_len, -1) 
        question_mask = question_mask.unsqueeze(1).repeat(1, relation_num, 1).view(batch_size * relation_num, seq_len) 
        Relation_embedding = self.Relation_embedding.weight.unsqueeze(1).unsqueeze(0).repeat(batch_size, 1, seq_len, 1).view(batch_size * relation_num, seq_len, -1)

        dot_vector = step_aware_question_vector * Relation_embedding
        linear_result = self.W_att(dot_vector).squeeze(-1) 
        linear_result_masked = linear_result.masked_fill(question_mask, float('-inf'))
        matrix_alpha = F.softmax(linear_result_masked, 1).unsqueeze(1) 
        relation_aware_question_vector = torch.matmul(matrix_alpha, step_aware_question_vector).squeeze(1).view(batch_size, relation_num, -1)

        return relation_aware_question_vector 

        
    def get_action_space_in_buckets(self, batch_e_t, d_entity2bucketid, d_action_space_buckets):
        db_action_spaces, db_references = [], []

        entity2bucketid = d_entity2bucketid[batch_e_t.tolist()] 
        key1 = entity2bucketid[:, 0] 
        key2 = entity2bucketid[:, 1] 
        batch_ref = {} 

        for i in range(len(batch_e_t)):
            key = int(key1[i])
            if not key in batch_ref:
                batch_ref[key] = []
            batch_ref[key].append(i) 
        for key in batch_ref:
            action_space = d_action_space_buckets[key] 

            l_batch_refs = batch_ref[key] 
            g_bucket_ids = key2[l_batch_refs].tolist()
            r_space_b = action_space[0][0][g_bucket_ids] 
            e_space_b = action_space[0][1][g_bucket_ids]
            action_mask_b = action_space[1][g_bucket_ids]

            r_space_b = r_space_b.cuda(self.gpu_id)
            e_space_b = e_space_b.cuda(self.gpu_id)
            action_mask_b = action_mask_b.cuda(self.gpu_id)

            action_space_b = ((r_space_b, e_space_b), action_mask_b)
            db_action_spaces.append(action_space_b)
            db_references.append(l_batch_refs)

        return db_action_spaces, db_references
    
    def policy_linear(self, b_input_vector):
        X = self.W1(b_input_vector)
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2(X)
        X2 = self.W2Dropout(X) 
        return X2
    
    def get_action_embedding(self, action):
        r, e = action
        relation_embedding = self.Relation_embedding(r) 
        
        if self.relation_only:
            action_embedding = relation_embedding
        else:
            entity_embedding = self.Entity_embedding(e) 
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1) 
        return action_embedding
    
    def apply_action_masks(self, t, last_r, r_space, action_mask, NO_OP_RELATION = 2): 
        if t == 0: 
            judge = (r_space == NO_OP_RELATION).long()
            judge = 1 - judge
            action_mask = judge * action_mask
        
        elif t == self.max_hop - 1: 
            judge = (r_space == NO_OP_RELATION).long()
            action_mask = judge * action_mask
        
        else: 
            judge = (last_r == NO_OP_RELATION).long() 
            judge = 1 - judge
            action_mask = judge.unsqueeze(1) * action_mask 
            self_loop = torch.zeros(action_mask.size(), dtype= torch.long) 
            self_loop = self_loop.cuda(self.gpu_id)

            self_loop[:, 0] = 1 
            self_loop = (1 - judge).unsqueeze(1) * self_loop
            action_mask = action_mask + self_loop
        
        return action_mask

    def transit(self, t, batch_e_t, batch_question, batch_sent_len, batch_path_hidden, d_entity2bucketid, d_action_space_buckets, last_r, is_Train, batch_pred_vector = None):
        
        step_aware_question_vector, question_mask = self.get_step_aware_representation(t, batch_question, batch_sent_len)

        
        relation_aware_question_vector = self.get_relation_aware_question_vector(step_aware_question_vector, question_mask) 
        db_action_spaces, db_references = self.get_action_space_in_buckets(batch_e_t, d_entity2bucketid, d_action_space_buckets) 

        db_outcomes = []
        entropy_list = []
        references = []
        values_list = []

        for b_action_space, b_reference in zip(db_action_spaces, db_references): 
            (b_r_space, b_e_space), b_action_mask = b_action_space
            b_last_r = last_r[b_reference]
            
            b_relation_aware_q = relation_aware_question_vector[b_reference] 

            _, relation_num, _ = b_relation_aware_q.shape

            b_path_hidden = batch_path_hidden[b_reference] 
            b_path_hidden = b_path_hidden.unsqueeze(1).repeat(1, relation_num, 1)

            if self.use_keqa_vector:
                b_pred_vector = batch_pred_vector[b_reference] 
                b_pred_vector = b_pred_vector.unsqueeze(1).repeat(1, relation_num, 1)

                b_input_vector = torch.cat([b_path_hidden, b_relation_aware_q, b_pred_vector], -1) 
            else:
                b_input_vector = torch.cat([b_path_hidden, b_relation_aware_q], -1) 
            
            
            b_output_vector = self.policy_linear(b_input_vector) 
            b_action_embedding = self.get_action_embedding((b_r_space, b_e_space)) 
            _, action_num, _ = b_action_embedding.shape
            
            
            l_output_vector_small = []

            for batch_i in range(len(b_output_vector)):
                output_i = b_output_vector[batch_i] 
                relation_i = b_r_space[batch_i] 
                new_i = output_i[relation_i]
                l_output_vector_small.append(new_i)
            
            b_output_vector_small = torch.stack(l_output_vector_small, 0) 

            b_action_embedding = b_action_embedding.view(-1, self.action_dim).unsqueeze(1) 
            b_output_vector_small = b_output_vector_small.view(-1, self.action_dim).unsqueeze(-1) 
            b_action_score = torch.matmul(b_action_embedding, b_output_vector_small).squeeze(-1).view(-1, action_num) 
            if is_Train:
                b_action_mask = self.apply_action_masks(t, b_last_r, b_r_space, b_action_mask) 
            b_action_score_masked = b_action_score.masked_fill((1- b_action_mask).bool(), float('-inf'))
            b_action_dist = F.softmax(b_action_score_masked, 1) 
            b_entropy = entropy(b_action_dist) 

            if self.use_keqa_vector:
                b_value = self.W_value(torch.cat([batch_path_hidden[b_reference], batch_pred_vector[b_reference]], 1)).view(-1) 
            else:
                b_value = self.W_value(batch_path_hidden[b_reference]).view(-1)
            b_value = torch.sigmoid(b_value)

            b_action_space = ((b_r_space, b_e_space), b_action_mask) 
            references.extend(b_reference)
            db_outcomes.append((b_action_space, b_action_dist))
            entropy_list.append(b_entropy)
            values_list.append(b_value)

        inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])] 
        return db_outcomes, entropy_list, values_list, inv_offset

    def initialize_path(self, init_action): 
        init_action_embedding = self.get_action_embedding(init_action)
        init_action_embedding.unsqueeze_(1) 
        init_h = torch.zeros([self.history_layers, len(init_action_embedding), self.history_dim]) 
        init_c = torch.zeros([self.history_layers, len(init_action_embedding), self.history_dim])

        init_h = init_h.cuda(self.gpu_id)
        init_c = init_c.cuda(self.gpu_id)

        h_n, c_n = self.path_encoder(init_action_embedding, (init_h, init_c))[1] 
        return (h_n, c_n)

    def update_path(self, action, path_list, offset=None):
        
        def offset_path_history(p, offset): 
            for i, x in enumerate(p):
                if type(x) is tuple: 
                    new_tuple = tuple([_x[:, offset, :] for _x in x])
                    p[i] = new_tuple
                else:
                    p[i] = x[offset, :]

        action_embedding = self.get_action_embedding(action)
        action_embedding.unsqueeze_(1) 
        
        if offset is not None: 
            offset_path_history(path_list, offset)

        h_n, c_n = self.path_encoder(action_embedding, path_list[-1])[1]
        return path_list, (h_n, c_n) 


    def initialize_modules(self):
        for i in self.step_aware_linear:
            nn.init.xavier_uniform_(i.weight)
            nn.init.constant_(i.bias, 0.0)

        nn.init.xavier_uniform_(self.W_att.weight)
        nn.init.constant_(self.W_att.bias, 0.0)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.constant_(self.W1.bias, 0.0)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.constant_(self.W2.bias, 0.0)
        nn.init.xavier_uniform_(self.W_value.weight)
        nn.init.constant_(self.W_value.bias, 0.0)


        for name, param in self.path_encoder.named_parameters(): 
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
    
    def batch_sentence_mask(self, batch_sent_len):
        batch_size = len(batch_sent_len)
        max_sent_len = batch_sent_len[0]
        mask = torch.zeros(batch_size, max_sent_len, dtype=torch.long)

        for i in range(batch_size):
            sent_len = batch_sent_len[i]
            mask[i][sent_len:] = 1
        
        mask = (mask == 1) 
        mask = mask.cuda(self.gpu_id)
        return mask
        
    def load(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir, map_location=torch.device('cpu')))

    def save(self, checkpoint_dir):
        torch.save(self.state_dict(), checkpoint_dir)