import random
import torch
import torch.nn as nn
from TransE import TransE
from ConvE import ConvE
from ComplEx import ComplEx
from DistMult import DistMult
from TuckER import TuckER
from util import get_relation_signature

class KGE_framework(nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(KGE_framework, self).__init__()
        self.batch_size = args.batch_size
        self.label_smoothing_epsilon = args.label_smoothing_epsilon
        self.margin = args.margin
        self.model = args.KGE_model
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.emb_dropout_rate = args.emb_dropout_rate

        self.entity_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
        self.EDropout = nn.Dropout(self.emb_dropout_rate)
        
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_dim)
        self.RDropout = nn.Dropout(self.emb_dropout_rate)

        if self.model == 'ComplEx':
            self.entity_img_embeddings = nn.Embedding(self.num_entities, self.entity_dim)
            self.relation_img_embeddings = nn.Embedding(self.num_relations, self.relation_dim)

        self.initialize_modules() 
        if self.model == 'TransE':
            self.loss_fun = nn.MarginRankingLoss(self.margin, False)
        else:
            self.loss_fun = nn.BCELoss()
        
        if self.model == 'TransE':
            self.kge_model = TransE()
        elif self.model == 'ConvE':
            self.kge_model = ConvE(args)
        elif self.model == 'ComplEx':
            self.kge_model = ComplEx()
        elif self.model == 'DistMult':
            self.kge_model = DistMult()
        elif self.model == 'TuckER':
            self.kge_model = TuckER(args)
    
    def initialize_modules(self):
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        if self.model == 'ComplEx':
            nn.init.xavier_normal_(self.entity_img_embeddings.weight)
            nn.init.xavier_normal_(self.relation_img_embeddings.weight)
    
    def predict(self, batch_topic_entity, question_vector): 
        E1 = self.EDropout(self.entity_embeddings(batch_topic_entity)) 
        E2 = self.EDropout(self.entity_embeddings.weight)
        
        if self.model == 'ComplEx':
            E1_img = self.EDropout(self.entity_img_embeddings(batch_topic_entity))
            R, R_img = torch.chunk(question_vector, 2, dim=1)
            R = self.RDropout(R)
            R_img = self.RDropout(R_img)
            E2_img = self.EDropout(self.entity_img_embeddings.weight)
            pred_scores = self.kge_model.forward(E1, R, E2, E1_img, R_img, E2_img) 
        else:
            R = self.RDropout(question_vector) 
            pred_scores = self.kge_model.forward(E1, R, E2)
        
        return pred_scores

    def forward(self, batch_topic_entity, question_vector): 
        pred_score = self.predict(batch_topic_entity, question_vector)
        return pred_score
    
    def forward_fact(self, batch_topic_entity, question_vector, batch_pred_entity):
        E1 = self.EDropout(self.entity_embeddings(batch_topic_entity)) 
        E2 = self.EDropout(self.entity_embeddings(batch_pred_entity))
        
        if self.model == 'ComplEx':
            E1_img = self.EDropout(self.entity_img_embeddings(batch_topic_entity))
            R, R_img = torch.chunk(question_vector, 2, dim=1)
            R = self.RDropout(R)
            R_img = self.RDropout(R_img)
            E2_img = self.EDropout(self.entity_img_embeddings(batch_pred_entity))
            pred_scores = self.kge_model.forward_fact(E1, R, E2, E1_img, R_img, E2_img) 
        else:
            R = self.RDropout(question_vector) 
            pred_scores = self.kge_model.forward_fact(E1, R, E2)
        
        pred_scores = pred_scores.view(-1)
        return pred_scores