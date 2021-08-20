import time
import math
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from util import get_num_gpus, set_seed
from policy_network import Policy_Network
from util import safe_log


class TrainWorker(mp.Process):
    def __init__(self, args, worker_id, done, shared_model, optim, env, train_gstep, valid_gstep, d_entity_neighours, d_entity2bucketid, d_action_space_buckets, num_episodes, word_num, entity_num, relation_num, keqa_checkpoint_path):
        super().__init__(name='train-worker-%02d' % (worker_id))
        self.args = args
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.valid_episodes = args.valid_episodes
        self.use_reward_shaping = args.use_reward_shaping 
        self.use_keqa_vector = args.use_keqa_vector
        self.num_episodes = num_episodes
        self.max_hop = args.max_hop + 1 

        self.gamma = args.gamma 
        self.tau = args.tau 
        self.entropy_coef = args.entropy_coef 
        self.value_loss_coef = args.value_loss_coef 
        self.max_grad_norm = args.grad_norm
        self.num_workers = args.num_workers

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
        self.optim = optim
        self.env = env 
        self.train_gstep = train_gstep
        self.valid_gstep = valid_gstep
    
    def sync_model(self):
        self.model.load_state_dict(self.shared_model.state_dict()) 

    def ensure_shared_grads(self):
        for param, shared_param in zip(self.model.parameters(),
                                       self.shared_model.parameters()):  
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def run(self):
        set_seed(self.seed) 
        self.model = Policy_Network(self.args, self.word_num, self.entity_num, self.relation_num, self.keqa_checkpoint_path, self.gpu_id) 
        self.model.train()
        self.env.set_gpu_id(self.gpu_id) 
        self.env.set_model(self.model) 

        init_approx = self.train_gstep.value // self.num_workers
        total_approx = math.ceil(self.num_episodes / self.num_workers) 
        with tqdm(desc=self.name, initial=init_approx, total=total_approx,
                  position=self.worker_id) as pbar:
            while not self.done.value:
                t1 = time.time()
                times = int(self.train_gstep.value / self.valid_episodes)
                while self.valid_gstep.value < times: 
                    if self.valid_gstep.value == times: 
                        break
                    if self.done.value: 
                        break

                self.sync_model() 
                self.model.cuda(self.gpu_id)
                t1_1 = time.time()
                self.env.reset() 
                t2 = time.time()
                self.rollout()
                t3 = time.time()
                pbar.update()
                torch.cuda.empty_cache()

                if self.train_gstep.value >= self.num_episodes:
                    self.done.value = True
                else:
                    self.train_gstep.value += 1


    def rollout(self):
        batch_question, batch_question_len, batch_head, batch_answers = self.env.return_batch_data()
        log_action_probs = []
        action_entropy = []
        action_value = []


        batch_pred_vector = None
        if self.use_keqa_vector:
            batch_pred_vector = self.model.get_anticipated_entity_vector(batch_head, batch_question, batch_question_len, self.d_entity_neighours) 

        for t in range(self.max_hop):
            path_trace, path_hidden = self.env.observe()
            last_r, e_t = path_trace[-1]
            batch_path_hidden = path_hidden[-1][0][-1, :, :]
            db_outcomes, entropy_list, values_list, inv_offset = self.model.transit(t, e_t, batch_question, batch_question_len, batch_path_hidden, self.d_entity2bucketid, self.d_action_space_buckets, last_r, True, batch_pred_vector)
            entropy = torch.cat(entropy_list, dim=0)[inv_offset]
            values = torch.cat(values_list, dim=0)[inv_offset]
            action_sample, action_prob = self.sample_action(db_outcomes, inv_offset) 
            log_action_probs.append(safe_log(action_prob)) 
            action_entropy.append(entropy) 
            action_value.append(values)

            path_list, (h_t, c_t) = self.model.update_path(action_sample, path_hidden)
            self.env.step(action_sample, path_list, (h_t, c_t))

        batch_pred_e2 = self.env.get_pred_entities()
        batch_rewards = self.reward_fun(batch_head, batch_question, batch_question_len, batch_answers, batch_pred_e2)

        cum_rewards = [0] * self.max_hop
        cum_rewards[-1] = batch_rewards

        R = torch.zeros(self.batch_size).cuda(self.gpu_id)
        action_value.append(R)

        policy_loss = torch.zeros(self.batch_size).cuda(self.gpu_id)
        value_loss = torch.zeros(self.batch_size).cuda(self.gpu_id)
        gae = torch.zeros(self.batch_size).cuda(self.gpu_id)

        for i in reversed(range(self.max_hop)):
            R = self.gamma * R + cum_rewards[i]
            advantage = R - action_value[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            delta_t = cum_rewards[i] + self.gamma * action_value[i+1] - action_value[i]
            gae = gae * self.gamma * self.tau + delta_t
            policy_loss = policy_loss - gae.detach() * log_action_probs[i] - self.entropy_coef * action_entropy[i]

        rl_loss = policy_loss + self.value_loss_coef * value_loss
        rl_loss = rl_loss.mean()

        self.optim.zero_grad()
        rl_loss.backward()

        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.model.cpu()
        self.ensure_shared_grads()
        self.optim.step()
    
    def sample(self, action_space, action_dist):
        sample_outcome = {}
        (r_space, e_space), action_mask = action_space
        idx = torch.multinomial(action_dist, 1, replacement=True)
        next_r = torch.gather(r_space, 1, idx).view(-1) 
        next_e = torch.gather(e_space, 1, idx).view(-1) 
        action_prob = torch.gather(action_dist, 1, idx).view(-1) 
        sample_outcome['action_sample'] = (next_r, next_e)
        sample_outcome['action_prob'] = action_prob
        return sample_outcome

    def sample_action(self, db_outcomes, inv_offset=None):
        next_r_list = []
        next_e_list = []
        action_prob_list = []
        for action_space, action_dist in db_outcomes: 
            sample_outcome = self.sample(action_space, action_dist)
            next_r_list.append(sample_outcome['action_sample'][0])
            next_e_list.append(sample_outcome['action_sample'][1])
            action_prob_list.append(sample_outcome['action_prob'])
        next_r = torch.cat(next_r_list, dim=0)[inv_offset] 
        next_e = torch.cat(next_e_list, dim=0)[inv_offset]
        action_sample = (next_r, next_e)
        action_prob = torch.cat(action_prob_list, dim=0)[inv_offset]

        return action_sample, action_prob
    
    def reward_fun(self, batch_head, batch_question, batch_seq_len, batch_answers, batch_pred_e2):
        binary_reward = torch.gather(batch_answers, 1, batch_pred_e2.unsqueeze(-1)).squeeze(-1).float() 

        if self.use_reward_shaping: 
            shaping_reward = self.model.KEQA.forward_fact(batch_question, batch_seq_len, batch_head, batch_pred_e2)
            return binary_reward + (1 - binary_reward) * shaping_reward 
        else:
            return binary_reward
