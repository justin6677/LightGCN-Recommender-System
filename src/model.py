import torch
from torch import nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.num_users, self.num_items = dataset.n_users, dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.n_layers   = config['lightGCN_n_layers']
        self.A_split    = config['A_split']

        self.Graph = dataset.getSparseGraph()

        self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        all_emb = torch.cat([users_emb, items_emb], dim=0)
        embs = [all_emb]

        g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = [torch.sparse.mm(f, all_emb) for f in g_droped]
                all_emb = torch.cat(temp_emb, dim=0)
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = torch.matmul(users_emb, items_emb.t())
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        return (
            all_users[users],
            all_items[pos_items],
            all_items[neg_items],
            self.embedding_user(users),
            self.embedding_item(pos_items),
            self.embedding_item(neg_items),
        )

    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0 = \
            self.getEmbedding(users, pos, neg)

        reg_loss = (1 / 2) * (
            userEmb0.norm(2).pow(2) +
            posEmb0.norm(2).pow(2) +
            negEmb0.norm(2).pow(2)
        ) / float(len(users))

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss, reg_loss