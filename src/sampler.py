import numpy as np
import torch
from tqdm.auto import tqdm

class HardNegativeSampler:
    """
    修正版 Hard Negative Sampler:

    - hard_ratio > 0:
        * 每個 user 有一個負樣本池 pool[u]，大小 pool_size
        * pool = 一部分 hard（分數高） + 一部分 random
        * 每隔 update_every 個 epoch，重新用最新模型更新 pool

    - hard_ratio <= 0:
        * 不建 pool、不更新 pool
        * sample_batch 時直接 uniform random 負樣本
          → 理論上等價於 baseline 的 uniform 負採樣
    """
    def __init__(self,
                 model,
                 dataset,
                 pool_size=100,
                 candidate_size=1000,
                 update_every=10,
                 device=None,
                 user_batch=512,
                 hard_ratio=0.5,
                 seed=42):
        self.model = model
        self.dataset = dataset
        self.pool_size = pool_size
        self.candidate_size = candidate_size
        self.update_every = update_every
        self.device = device or torch.device("cpu")
        self.user_batch = user_batch
        self.hard_ratio = float(hard_ratio)
        self.rng = np.random.RandomState(seed)

        self.n_users = dataset.n_users
        self.n_items = dataset.m_items

        self.pos_items = [set(arr.tolist()) for arr in dataset.allPos]

        self.use_pool = self.hard_ratio > 0.0

        if self.use_pool:
            self.pools = [np.array([], dtype=np.int64) for _ in range(self.n_users)]

            print(f"Initializing HardNegativeSampler with HARD POOLS: "
                  f"pool={pool_size}, candidates={candidate_size}, "
                  f"update_every={update_every}, hard_ratio={hard_ratio}")
            print("Building initial hard-negative pools...")
            self._build_pools(first_time=True)
            print("✔ Initial hard-negative pools ready.")
        else:
            self.pools = None
            print("Initializing HardNegativeSampler in **UNIFORM** mode "
                  "(hard_ratio <= 0 → no pools, pure random negatives).")

    def _random_neg(self, u):
        pos = self.pos_items[u]
        if len(pos) >= self.n_items - 1:
            return 0
        while True:
            cand = int(self.rng.randint(0, self.n_items))
            if cand not in pos:
                return cand

    def _build_pools(self, first_time=False):
        if not self.use_pool:
            return

        self.model.eval()
        n_users = self.n_users
        B = self.user_batch
        hard_ratio = self.hard_ratio

        with torch.no_grad():
            for start in tqdm(range(0, n_users, B), disable=False):
                end = min(start + B, n_users)
                if start >= end:
                    break
                batch_users = np.arange(start, end, dtype=np.int64)
                users_t = torch.from_numpy(batch_users).long().to(self.device)
                rating = self.model.getUsersRating(users_t).detach().cpu().numpy()

                for idx, u in enumerate(batch_users):
                    pos_set = self.pos_items[u]
                    if len(pos_set) == 0:
                        pool = [self._random_neg(u) for _ in range(max(1, self.pool_size))]
                        self.pools[u] = np.array(pool, dtype=np.int64)
                        continue

                    max_neg = max(1, self.n_items - len(pos_set))
                    num_cand = min(self.candidate_size, max_neg)

                    neg_cands = []
                    neg_set = set()
                    while len(neg_cands) < num_cand:
                        cand = int(self.rng.randint(0, self.n_items))
                        if cand in pos_set or cand in neg_set:
                            continue
                        neg_cands.append(cand)
                        neg_set.add(cand)
                    neg_cands = np.array(neg_cands, dtype=np.int64)

                    scores = rating[idx]
                    cand_scores = scores[neg_cands]
                    order = np.argsort(-cand_scores)
                    neg_cands_sorted = neg_cands[order]

                    K_hard = int(self.pool_size * hard_ratio)
                    K_hard = min(K_hard, len(neg_cands_sorted))
                    hard_negs = neg_cands_sorted[:K_hard]

                    remain = neg_cands_sorted[K_hard:]
                    remain = np.array(remain, dtype=np.int64)

                    rest = self.pool_size - len(hard_negs)
                    if rest > 0 and len(remain) > 0:
                        if len(remain) <= rest:
                            rand_negs = remain
                        else:
                            idxs = self.rng.choice(len(remain), size=rest, replace=False)
                            rand_negs = remain[idxs]
                    else:
                        rand_negs = np.array([], dtype=np.int64)

                    pool = np.concatenate([hard_negs, rand_negs], axis=0)

                    while len(pool) < self.pool_size:
                        pool = np.append(pool, self._random_neg(u))

                    if len(pool) > self.pool_size:
                        idxs = self.rng.choice(len(pool), size=self.pool_size, replace=False)
                        pool = pool[idxs]

                    pool = np.array([it for it in pool if it not in pos_set], dtype=np.int64)
                    if len(pool) == 0:
                        pool = np.array(
                            [self._random_neg(u) for _ in range(self.pool_size)],
                            dtype=np.int64
                        )

                    self.pools[u] = pool

    def update_pool(self, model=None):
        if not self.use_pool:
            return
        if model is not None:
            self.model = model
        print("Updating hard-negative pools...")
        self._build_pools(first_time=False)
        print("✔ Pools updated.")

    def sample_batch(self, users, pos_items):
        users = np.asarray(users, dtype=np.int64)
        B = len(users)
        neg_items = np.empty(B, dtype=np.int64)

        for idx in range(B):
            u = int(users[idx])
            pos_set = self.pos_items[u]

            if not self.use_pool:
                neg = self._random_neg(u)
            else:
                pool = self.pools[u]
                if len(pool) == 0:
                    neg = self._random_neg(u)
                else:
                    neg = int(self.rng.choice(pool))
                    if neg in pos_set:
                        neg = self._random_neg(u)

            neg_items[idx] = neg

        return neg_items