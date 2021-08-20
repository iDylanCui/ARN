import numpy as np
import torch
import time

class Environment(object):
    def __init__(self, args, d_dataset, entity_num, shuffle, dummy_start_r = 1):
        self.args = args
        self.batch_size = args.batch_size
        self.entity_num = entity_num
        self.shuffle = shuffle
        self.dummy_start_r = dummy_start_r
        self.d_dataset = d_dataset
        self._model = None

    def set_model(self, model):
        self._model = model

    def set_gpu_id(self, gpu_id):
        self._gpu_id = gpu_id

    def reset(self, data_indexes=None):
        self.data = [] 
        if data_indexes is not None:
            self.data_indexes = data_indexes
        elif self.shuffle:
            self.data_indexes = np.random.randint(0, len(self.d_dataset), self.batch_size)
        
        
        for idx in self.data_indexes: 
            if idx < len(self.d_dataset):
                self.data.append(self.d_dataset[idx])
        
        self.batch_size = len(self.data) 
        
        self.batch_question, self.batch_question_len, self.batch_head, self.batch_answers = self.get_batch_data() 
        r_s = torch.zeros(self.batch_size, dtype= torch.long) + self.dummy_start_r
        r_s = r_s.cuda(self._gpu_id)
        self.path_trace = [(r_s, self.batch_head)] 
        self.path_hidden = [self._model.initialize_path(self.path_trace[0])]
    
    def return_batch_data(self):
        return self.batch_question, self.batch_question_len, self.batch_head, self.batch_answers

    def observe(self):
        return self.path_trace, self.path_hidden

    def step(self, new_action, path_hidden, new_hidden):
        self.path_trace.append(new_action)
        self.path_hidden = path_hidden
        self.path_hidden.append(new_hidden)
    
    def get_pred_entities(self):
        return self.path_trace[-1][1]
    
    def toOneHot(self, indices, vec_len):
        indices = torch.LongTensor(indices).cuda(self._gpu_id)
        one_hot = torch.LongTensor(vec_len).cuda(self._gpu_id)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def get_batch_data(self):

        sorted_seq = sorted(self.data, key=lambda sample: len(sample[0]), reverse=True)
        sorted_seq_lengths = [len(i[0]) for i in sorted_seq]
        longest_sample = sorted_seq_lengths[0]

        inputs = torch.zeros(self.batch_size, longest_sample, dtype=torch.long).cuda(self._gpu_id)
        input_lengths = torch.zeros(self.batch_size, dtype=torch.long).cuda(self._gpu_id)
        p_head = torch.zeros(self.batch_size, dtype=torch.long).cuda(self._gpu_id)
        p_tail = torch.zeros((self.batch_size, self.entity_num), dtype=torch.long).cuda(self._gpu_id)

        for x in range(self.batch_size):
            l_question, head_entity, l_answer_entities = sorted_seq[x]
            seq_len = len(l_question)
            l_question = torch.LongTensor(l_question).cuda(self._gpu_id)
            inputs[x].narrow(0, 0, seq_len).copy_(l_question)
            input_lengths[x] = seq_len
            p_head[x] = head_entity
            l_answer_entities = self.toOneHot(l_answer_entities, self.entity_num)
            p_tail[x] = l_answer_entities

        return inputs, input_lengths, p_head, p_tail