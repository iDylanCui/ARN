from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from util import get_num_gpus, set_seed, safe_log
from policy_network import Policy_Network
import torch.nn as nn

class ValidWorker(mp.Process):
    def __init__(self, args, worker_id, done, shared_model, env, train_gstep, valid_gstep, d_entity_neighours, d_entity2bucketid, d_action_space_buckets, l_intimediate_result, d_ckpt, checkpoint_path,  word_num, entity_num, relation_num, keqa_checkpoint_path):
        super().__init__(name='valid-worker-%02d' % (worker_id))
        self.args = args
        self.fix_batch_size = args.batch_size
        self.early_stop_patience = args.early_stop_patience
        self.valid_episodes = args.valid_episodes 
        self.use_keqa_vector = args.use_keqa_vector
        self.max_hop = args.max_hop 
        self.beam_size = args.beam_size

        self.word_num = word_num 
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.keqa_checkpoint_path = keqa_checkpoint_path
        self.d_entity_neighours = d_entity_neighours
        self.d_entity2bucketid = d_entity2bucketid
        self.d_action_space_buckets = d_action_space_buckets

        self.worker_id = worker_id
        self.gpu_id = self.worker_id % get_num_gpus()
        self.seed = args.seed + worker_id
        self.done = done
        self.shared_model = shared_model
        self.env = env
        self.train_gstep = train_gstep
        self.valid_gstep = valid_gstep
        self.l_intimediate_result = l_intimediate_result
        self.d_ckpt = d_ckpt
        self.checkpoint_path = checkpoint_path

    def run(self):
        set_seed(self.seed)
        self.model = Policy_Network(self.args, self.word_num, self.entity_num, self.relation_num, self.keqa_checkpoint_path, self.gpu_id) 
        self.model.cuda(self.gpu_id)
        self.model.eval()
        self.env.set_model(self.model)
        self.env.set_gpu_id(self.gpu_id)
        total_data_num = len(self.env.d_dataset)

        best_hits_1 = -float('inf')
        iters_not_improved = 0
        best_model = None

        while not self.done.value:
            while True: 
                times = int(self.train_gstep.value / self.valid_episodes)
                if times != self.valid_gstep.value:
                    break
                if self.done.value: 
                    break
            
            hits_1_num = 0
            self.sync_model() 
            with torch.no_grad():
                for example_id in tqdm(range(0, len(self.env.d_dataset), self.fix_batch_size), desc=self.name, position=self.worker_id): 
                    idx = range(example_id, example_id + self.fix_batch_size) 
                    
                    self.env.reset(idx) 
                    self.batch_size = self.env.batch_size 
                    
                    batch_hits1 = self.rollout()
                    torch.cuda.empty_cache()
                    hits_1_num += batch_hits1
            
            self.valid_gstep.value += 1
            valid_episode_num = self.valid_gstep.value * self.valid_episodes
            hits_1_result = 1.0 * hits_1_num / total_data_num
            self.l_intimediate_result.append((valid_episode_num, hits_1_result)) 

            if hits_1_result > best_hits_1:
                best_hits_1 = hits_1_result
                iters_not_improved = 0
                best_model = self.model.state_dict()

                self.d_ckpt['best_episode_num'] = valid_episode_num 
                self.d_ckpt['best_hits@1'] = best_hits_1
            
            elif hits_1_result < best_hits_1 and iters_not_improved < self.early_stop_patience:
                iters_not_improved += 1

            elif iters_not_improved >= self.early_stop_patience:
                tqdm.write("Early Stop at {} eposide.".format(valid_episode_num))
                self.done.value = True 
            
            tqdm.write("Valid eposide: {}, Hits@1 = {}; Best_Hits@1 = {}, iters_not_improved = {}".format(valid_episode_num, round(hits_1_result, 4), round(best_hits_1, 4), iters_not_improved))
            
            if self.done.value:
                torch.save(best_model, self.checkpoint_path)

    def rollout(self): 
        batch_question, batch_question_len, batch_head, batch_answers = self.env.return_batch_data()

        batch_pred_vector = None
        if self.use_keqa_vector:
            batch_pred_vector = self.model.get_anticipated_entity_vector(batch_head, batch_question, batch_question_len, self.d_entity_neighours) 
        
        log_action_prob = torch.zeros(self.batch_size).cuda(self.gpu_id)
        for t in range(self.max_hop):
            path_trace, path_hidden = self.env.observe()
            last_r, e_t = path_trace[-1] 
            
            batch_path_hidden = path_hidden[-1][0][-1, :, :]
            
            k = int(e_t.size()[0] / self.batch_size)

            beam_question = batch_question.unsqueeze(1).repeat(1, k, 1).view(self.batch_size * k, -1) 
            beam_question_len = batch_question_len.unsqueeze(1).repeat(1, k).view(self.batch_size * k) 
            
            beam_pred_vector = None
            if self.use_keqa_vector:
                beam_pred_vector = batch_pred_vector.unsqueeze(1).repeat(1, k, 1).view(self.batch_size * k, -1) 
            
            db_outcomes, _, _, inv_offset = self.model.transit(t, e_t, beam_question, beam_question_len, batch_path_hidden, self.d_entity2bucketid, self.d_action_space_buckets, last_r, False, beam_pred_vector) 

            db_action_spaces = [action_space for action_space, _ in db_outcomes]
            db_action_dist = [action_dist for _, action_dist in db_outcomes]
             
            action_space = self.pad_and_cat_action_space(db_action_spaces, inv_offset) 
            action_dist = self.pad_and_cat(db_action_dist, padding_value=0)[inv_offset]

            log_action_dist = log_action_prob.view(-1, 1) + safe_log(action_dist)

            if t == self.max_hop - 1:
                action, log_action_prob, action_offset = self.top_k_answer_unique(log_action_dist, action_space)
            else:
                action, log_action_prob, action_offset = self.top_k_action(log_action_dist, action_space)
            
            path_list, (h_t, c_t) = self.model.update_path(action, path_hidden, offset = action_offset) 
            self.env.step(action, path_list, (h_t, c_t))
        
        batch_pred_e2 = action[1].view(self.batch_size, -1) 
        batch_pred_e2_top1 = batch_pred_e2[:, 0].view(self.batch_size, -1)
        
        batch_hits1 = torch.sum(torch.gather(batch_answers, 1, batch_pred_e2_top1).view(-1)).item()
            
        return batch_hits1


    def top_k_action(self, log_action_dist, action_space):
        full_size = len(log_action_dist)
        last_k = int(full_size / self.batch_size)

        (r_space, e_space), _ = action_space
        action_space_size = r_space.size()[1]

        log_action_dist = log_action_dist.view(self.batch_size, -1)
        beam_action_space_size = log_action_dist.size()[1]
        k = min(self.beam_size, beam_action_space_size) 

        log_action_prob, action_ind = torch.topk(log_action_dist, k)
        next_r = torch.gather(r_space.view(self.batch_size, -1), 1, action_ind).view(-1) 
        next_e = torch.gather(e_space.view(self.batch_size, -1), 1, action_ind).view(-1) 
        log_action_prob = log_action_prob.view(-1) 
        action_beam_offset = action_ind // action_space_size 
        action_batch_offset = (torch.arange(self.batch_size).cuda(self.gpu_id) * last_k).unsqueeze(1) 
        action_offset = (action_batch_offset + action_beam_offset).view(-1) 

        return (next_r, next_e), log_action_prob, action_offset 
    
    def top_k_answer_unique(self, log_action_dist, action_space):
        full_size = len(log_action_dist)
        last_k = int(full_size / self.batch_size)
        (r_space, e_space), _ = action_space
        action_space_size = r_space.size()[1]

        r_space = r_space.view(self.batch_size, -1) 
        e_space = e_space.view(self.batch_size, -1)
        log_action_dist = log_action_dist.view(self.batch_size, -1)
        beam_action_space_size = log_action_dist.size()[1]
        
        k = min(self.beam_size, beam_action_space_size)
        next_r_list, next_e_list = [], []
        log_action_prob_list = []
        action_offset_list = []

        for i in range(self.batch_size):
            log_action_dist_b = log_action_dist[i]
            r_space_b = r_space[i]
            e_space_b = e_space[i]
            unique_e_space_b = torch.unique(e_space_b.data.cpu()).cuda(self.gpu_id) 
            unique_log_action_dist, unique_idx = self.unique_max(unique_e_space_b, e_space_b, log_action_dist_b) 
            k_prime = min(len(unique_e_space_b), k)
            top_unique_log_action_dist, top_unique_idx2 = torch.topk(unique_log_action_dist, k_prime)
            top_unique_idx = unique_idx[top_unique_idx2]
            top_unique_beam_offset = top_unique_idx // action_space_size
            top_r = r_space_b[top_unique_idx]
            top_e = e_space_b[top_unique_idx]
            next_r_list.append(top_r.unsqueeze(0))
            next_e_list.append(top_e.unsqueeze(0))
            log_action_prob_list.append(top_unique_log_action_dist.unsqueeze(0))
            top_unique_batch_offset = i * last_k
            top_unique_action_offset = top_unique_batch_offset + top_unique_beam_offset
            action_offset_list.append(top_unique_action_offset.unsqueeze(0))
        next_r = self.pad_and_cat(next_r_list, padding_value=0).view(-1)
        next_e = self.pad_and_cat(next_e_list, padding_value=0).view(-1)
        log_action_prob = self.pad_and_cat(log_action_prob_list, padding_value = -float("inf"))
        action_offset = self.pad_and_cat(action_offset_list, padding_value=-1)
        return (next_r, next_e), log_action_prob.view(-1), action_offset.view(-1)

    def sync_model(self):
        self.model.load_state_dict(self.shared_model.state_dict()) 
    
    def pad_and_cat_action_space(self, action_spaces, inv_offset):
        db_r_space, db_e_space, db_action_mask = [], [], []
        for (r_space, e_space), action_mask in action_spaces:
            db_r_space.append(r_space)
            db_e_space.append(e_space)
            db_action_mask.append(action_mask)
        r_space = self.pad_and_cat(db_r_space, padding_value=0)[inv_offset]
        e_space = self.pad_and_cat(db_e_space, padding_value=0)[inv_offset]
        action_mask = self.pad_and_cat(db_action_mask, padding_value=0)[inv_offset]
        action_space = ((r_space, e_space), action_mask)
        return action_space
    
    def pad_and_cat(self, a, padding_value, padding_dim=1):
        max_dim_size = max([x.size()[padding_dim] for x in a])
        padded_a = []
        for x in a:
            if x.size()[padding_dim] < max_dim_size:
                res_len = max_dim_size - x.size()[1]
                pad = nn.ConstantPad1d((0, res_len), padding_value)
                padded_a.append(pad(x))
            else:
                padded_a.append(x)
        return torch.cat(padded_a, dim=0).cuda(self.gpu_id)
    

    def unique_max(self, unique_x, x, values, marker_2D=None):
        unique_interval = 100
        HUGE_INT = 1e31

        unique_values, unique_indices = [], []
        for i in range(0, len(unique_x), unique_interval):
            unique_x_b = unique_x[i:i+unique_interval]
            marker_2D = (unique_x_b.unsqueeze(1) == x.unsqueeze(0)).float() 
            values_2D = marker_2D * values.unsqueeze(0) - (1 - marker_2D) * HUGE_INT 
            unique_values_b, unique_idx_b = values_2D.max(dim=1)
            unique_values.append(unique_values_b)
            unique_indices.append(unique_idx_b)
        unique_values = torch.cat(unique_values).cuda(self.gpu_id)
        unique_idx = torch.cat(unique_indices).cuda(self.gpu_id)
        return unique_values, unique_idx 