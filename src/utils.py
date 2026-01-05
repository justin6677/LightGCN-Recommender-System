import random
import numpy as np
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F

# ================= 1. 基礎設定與工具 =================
def set_seed(seed=2024):
    """ 鎖定隨機因素，確保實驗可重現 """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random Seed set to {seed}")

def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")

# ================= 2. 損失函數 (BPR Loss) =================
class BPRLoss:
    def __init__(self, recmodel, config):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.opt = optim.Adam(recmodel.parameters(), lr=config['lr'])

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        total_loss = loss + reg_loss
        
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()
        
        return total_loss.cpu().item()

# ================= 3. 訓練邏輯 (Cell 6 ) =================
# 這裡處理 User 的隨機負採樣
def random_neg_for_user(u, n_items, pos_set, rng):
    if len(pos_set) >= n_items - 1:
        return 0
    while True:
        cand = int(rng.randint(0, n_items))
        if cand not in pos_set:
            return cand

# 這裡處理訓練資料的配對 (User, Positive Item)
def sample_user_pos_pairs(dataset, rng):
    allPos = dataset.allPos
    n_users = dataset.n_users
    try:
        train_size = dataset.trainDataSize
    except AttributeError:
        train_size = int(sum(len(p) for p in allPos))

    users = np.empty(train_size, dtype=np.int64)
    pos_items = np.empty(train_size, dtype=np.int64)

    idx = 0
    while idx < train_size:
        u = rng.randint(0, n_users)
        pos_u = allPos[u]
        if len(pos_u) == 0:
            continue
        i_pos = np.random.choice(pos_u)
        users[idx] = u
        pos_items[idx] = i_pos
        idx += 1
    return users, pos_items

# 核心訓練函數
def BPR_train_with_sampler(dataset, Recmodel, loss_class, epoch,
                           sampler=None,
                           batch_size=2048,
                           seed=2025):
    # 自動獲取模型所在的 device (CPU 或 GPU)
    device = next(Recmodel.parameters()).device
    
    Recmodel.train()
    rng = np.random.RandomState(seed + epoch)
    allPos = dataset.allPos
    n_items = dataset.m_items

    users_np, pos_np = sample_user_pos_pairs(dataset, rng)
    n_samples = len(users_np)

    total_batch = n_samples // batch_size + 1
    aver_loss = 0.0

    for batch_idx in range(total_batch):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, n_samples)
        if start >= end:
            break

        batch_users = users_np[start:end]
        batch_pos   = pos_np[start:end]

        # 如果有 Sampler 就用 Sampler (Hard Negative)，否則用隨機
        if sampler is None:
            neg_items = np.empty_like(batch_users)
            for i, u in enumerate(batch_users):
                pos_set = set(allPos[u])
                neg_items[i] = random_neg_for_user(u, n_items, pos_set, rng)
            batch_neg = neg_items
        else:
            batch_neg = sampler.sample_batch(batch_users, batch_pos)

        users_t = torch.from_numpy(batch_users).long().to(device)
        pos_t   = torch.from_numpy(batch_pos).long().to(device)
        neg_t   = torch.from_numpy(batch_neg).long().to(device)

        loss = loss_class.stageOne(users_t, pos_t, neg_t)
        aver_loss += loss

    # 防止除以零
    if total_batch == 0: 
        return 0.0
    return float(aver_loss / total_batch)

# ================= 4. 評估函數 (Test / Recall / NDCG) =================
def _bool_hits(test_data, topk_items):
    B, K = topk_items.shape
    r = np.zeros((B, K), dtype=np.float32)
    for i in range(B):
        gt = set(test_data[i])
        if not gt:
            continue
        for j in range(K):
            if topk_items[i, j] in gt:
                r[i, j] = 1.0
    return r

def _recall_precision_at_k(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    # 防止分母為 0
    recall_n = np.array([max(1, len(test_data[i])) for i in range(len(test_data))],
                        dtype=np.float32)
    recall = float(np.sum(right_pred / recall_n))
    precis = float(np.sum(right_pred) / (k * len(test_data)))
    return {"recall": recall, "precision": precis}

def _ndcg_at_k(test_data, r, k):
    denom = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (r[:, :k] * denom[None, :]).sum(1)

    idcg = []
    for items in test_data:
        L = min(k, len(items))
        if L == 0:
            idcg.append(1.0)
        else:
            idcg.append(np.sum(denom[:L]))
    idcg = np.array(idcg, dtype=np.float32)
    return float(np.sum(dcg / idcg))

@torch.no_grad()
def Test(dataset, Recmodel, K=20):
    # 自動獲取 device
    device = next(Recmodel.parameters()).device
    
    Recmodel.eval()
    testDict = dataset.testDict
    users = list(testDict.keys())
    result = {"recall": 0.0, "precision": 0.0, "ndcg": 0.0}

    # 測試時的 Batch Size (可以調小一點避免 OOM)
    ub = 100 
    
    for i in range(0, len(users), ub):
        batch_users = users[i: i + ub]
        allPos = dataset.getUserPosItems(batch_users)
        groundTrue = [testDict[u] for u in batch_users]

        batch_users_gpu = torch.tensor(batch_users, dtype=torch.long, device=device)
        rating = Recmodel.getUsersRating(batch_users_gpu).detach().cpu()

        for row, items in enumerate(allPos):
            if len(items) == 0:
                continue
            # 把訓練集裡已經看過的 item 分數設為極低，避免重複推薦
            valid_items = [it for it in items if 0 <= it < rating.shape[1]]
            if len(valid_items) > 0:
                rating[row, valid_items] = -1e9

        _, topk = torch.topk(rating, k=K, dim=1)
        topk_np = topk.cpu().numpy()

        r = _bool_hits(groundTrue, topk_np)
        ret = _recall_precision_at_k(groundTrue, r, K)
        result["recall"]   += ret["recall"] / len(users)
        result["precision"]+= ret["precision"] / len(users)
        result["ndcg"]     += _ndcg_at_k(groundTrue, r, K) / len(users)
    return result