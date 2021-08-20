import torch
import torch.nn as nn
import torch.nn.functional as F
from Embedding import Embedding
from KGE_framework import KGE_framework
from Transformer import TransformerModel

class KEQA_framework(nn.Module):
    def __init__(self, args, word_num, num_entities, num_relations, gpu_id = -1):
        super(KEQA_framework, self).__init__()
        self.gpu_id = gpu_id
        self.batch_size = args.batch_size
        self.entity_dim = args.entity_dim
        self.max_hop = args.max_hop
        self.word_dim = args.word_dim
        self.word_padding_idx = args.word_padding_idx
        self.word_dropout_rate = args.word_dropout_rate
        self.is_train_word_emb = args.is_train_word_emb

        self.encoder_dropout_rate = args.encoder_dropout_rate
        self.head_num = args.head_num
        self.hidden_dim = args.hidden_dim
        self.encoder_layers = args.encoder_layers
        self.encoder_dropout_rate = args.encoder_dropout_rate

        self.num_entities = num_entities

        self.word_embedding_layer = Embedding(word_num, self.word_dim, self.word_padding_idx, self.word_dropout_rate, self.is_train_word_emb)

        self.transformer_model = TransformerModel(self.word_dim, self.head_num, self.hidden_dim, self.encoder_layers, self.encoder_dropout_rate)

        self.KGE_framework = KGE_framework(args, num_entities, num_relations)
        
        if args.KGE_model == "ComplEx":
            self.linear = nn.Linear(self.word_dim, 2 * self.entity_dim)
        else:
            self.linear = nn.Linear(self.word_dim, self.entity_dim)

    
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

    def question_max_pooling(self, transformer_out, question_mask):
        _, _, output_dim = transformer_out.shape
        question_mask = question_mask.unsqueeze(-1).repeat(1,1, output_dim)
        transformer_out_masked = transformer_out.masked_fill(question_mask, float('-inf'))

        question_transformer_masked = transformer_out_masked.transpose(1, 2) 
        question_mp = F.max_pool1d(question_transformer_masked, question_transformer_masked.size(2)).squeeze(2)

        return question_mp 

    def get_question_vector(self, batch_question, batch_sent_len):
        batch_question_embedding = self.word_embedding_layer(batch_question) 
        mask = self.batch_sentence_mask(batch_sent_len) 
        transformer_output = self.transformer_model(batch_question_embedding.permute(1, 0 ,2), mask) 
        
        transformer_output = transformer_output.permute(1, 0 ,2) 
        
        transformer_output_mp = self.question_max_pooling(transformer_output, mask) 
        question_vector = self.linear(transformer_output_mp)
        return question_vector


    def forward(self, batch_question, batch_sent_len, batch_topic_entity):
        question_vector = self.get_question_vector(batch_question, batch_sent_len) 
        return self.KGE_framework(batch_topic_entity, question_vector)
    
    def forward_fact(self, batch_question, batch_sent_len, batch_topic_entity, batch_pred_entity):
        question_vector = self.get_question_vector(batch_question, batch_sent_len) 
        return self.KGE_framework.forward_fact(batch_topic_entity, question_vector, batch_pred_entity)
    
    def loss(self, batch_question, batch_sent_len, batch_topic_entity, batch_answers):
        
        question_vector = self.get_question_vector(batch_question, batch_sent_len) 
        return self.KGE_framework.loss(batch_question, batch_sent_len, batch_topic_entity, question_vector, batch_answers)
    

    def get_max_hop_Neighbours_mask(self, batch_topic_entity, d_entity_neighours): 
        batch_topic_entity = batch_topic_entity.cpu().numpy().tolist()
        batch_size = len(batch_topic_entity)

        entity_mask = torch.zeros(batch_size, self.num_entities, dtype=torch.long)
        entity_mask = entity_mask.cuda(self.gpu_id)

        for i in range(batch_size):
            topic_entity = batch_topic_entity[i]
            max_hop_neighours = d_entity_neighours[topic_entity]
            entity_mask[i][max_hop_neighours] = 1
        
        return entity_mask
    
    def load(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir, map_location=torch.device('cpu')))

    def save(self, checkpoint_dir):
        torch.save(self.state_dict(), checkpoint_dir)