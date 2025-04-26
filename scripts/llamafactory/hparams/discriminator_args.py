import torch
import torch.nn as nn
from transformers import LongformerModel  
import torch.nn.functional as F

class DiscriminatorArguments(nn.Module):
    def __init__(self):
        super(DiscriminatorArguments, self).__init__()
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')  
        self.fc_classify = nn.Linear(self.longformer.config.hidden_size, 1)  
        self.cosine_similarity = nn.CosineSimilarity(dim=1)  
        self.loss_fn = nn.BCELoss()  

    def forward(self, text1, text2, labels):
        
        text1 = text1.long()  
        text2 = text2.long()  

        
        outputs1 = self.longformer(text1)  
        outputs2 = self.longformer(text2)  

        
        cls_embedding1 = outputs1.last_hidden_state[:, 0, :]
        cls_embedding2 = outputs2.last_hidden_state[:, 0, :]

        
        similarity_score = self.cosine_similarity(cls_embedding1, cls_embedding2)

        
        real_fake_logits1 = self.fc_classify(cls_embedding1)  
        real_fake_logits2 = self.fc_classify(cls_embedding2)  

        
        real_labels = labels[:, 0]  
        fake_labels = labels[:, 1]  

        loss1 = self.loss_fn(torch.sigmoid(real_fake_logits1), real_labels)  
        loss2 = self.loss_fn(torch.sigmoid(real_fake_logits2), fake_labels)  

        total_loss = loss1 + loss2  

        return total_loss, similarity_score, torch.sigmoid(real_fake_logits1), torch.sigmoid(real_fake_logits2)
