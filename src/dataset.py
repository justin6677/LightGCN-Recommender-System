# src/dataset.py (修正版)
import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
from .utils import cprint

class Loader:
    """ Gowalla dataset Loader """
    # 這裡修正了：增加 config, max_users, max_items 參數
    def __init__(self, config, path="./data/gowalla", max_users=None, max_items=None):
        cprint(f'loading [{path}]')
        
        # 接收傳入的 config
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.path = path
        
        train_file = os.path.join(path, 'train.txt')
        test_file  = os.path.join(path, 'test.txt')

        trainUser, trainItem = [], []
        testUser,  testItem  = [], []

        self.traindataSize = 0
        self.testDataSize  = 0
        self.n_user = 0
        self.m_item = 0

        # ---- 讀 train.txt ----
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) == 0: continue
                l = l.strip().split(' ')
                uid = int(l[0])
                
                # 如果有設定 max_users，超過就跳過
                if max_users is not None and uid >= max_users:
                    continue
                    
                items = [int(i) for i in l[1:]]
                # 如果有設定 max_items，超過就過濾
                if max_items is not None:
                    items = [i for i in items if i < max_items]
                
                if not items: continue
                
                trainUser.extend([uid] * len(items))
                trainItem.extend(items)
                self.m_item = max(self.m_item, max(items))
                self.n_user = max(self.n_user, uid)
                self.traindataSize += len(items)

        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        # ---- 讀 test.txt ----
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) == 0: continue
                l = l.strip().split(' ')
                uid = int(l[0])
                
                if max_users is not None and uid >= max_users:
                    continue
                
                items = [int(i) for i in l[1:]]
                if max_items is not None:
                    items = [i for i in items if i < max_items]
                    
                if not items: continue
                
                testUser.extend([uid] * len(items))
                testItem.extend(items)
                self.m_item = max(self.m_item, max(items))
                self.n_user = max(self.n_user, uid)
                self.testDataSize += len(items)

        self.m_item += 1
        self.n_user += 1
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        cprint(f"✅ {self.traindataSize} train, {self.testDataSize} test interactions, "
               f"{self.n_user} users, {self.m_item} items")

        # ---- 建使用者-物品稀疏矩陣 ----
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item),
        )

        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        self.Graph = None

    @property
    def n_users(self): return self.n_user
    @property
    def m_items(self): return self.m_item
    @property
    def trainDataSize(self): return self.traindataSize
    @property
    def testDict(self): return self.__testDict
    @property
    def allPos(self): return self._allPos

    def __build_test(self):
        test_data = {}
        for u, i in zip(self.testUser, self.testItem):
            if u in test_data: test_data[u].append(i)
            else: test_data[u] = [i]
        return test_data

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.LongTensor(coo.row)
        col = torch.LongTensor(coo.col)
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

# 請把原本的 getSparseGraph 刪掉，換成這個優化版
    def getSparseGraph(self):
        # 自動偵測 device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.Graph is not None:
            return self.Graph

        cprint("generating adjacency matrix")
        
        n_users, m_items = self.n_user, self.m_item
        
        # ==========================================
        # 💡 記憶體優化重點 (Memory Optimization)
        # 不要用 adj_mat[slice] = R 的方式，因為會觸發記憶體複製。
        # 我們改用 hstack, vstack 直接拼接稀疏區塊。
        # 矩陣結構 A = [   0,    R ]
        #             [ R.T,    0 ]
        # ==========================================
        
        from scipy.sparse import csr_matrix, hstack, vstack
        
        R = self.UserItemNet.tocsr()
        RT = R.T
        
        # 1. 建立左上角和右下角的 0 矩陣 (使用 CSR 格式極省空間)
        # 注意：不需要真的分配記憶體給 0，CSR 格式只會存非 0 的值
        zero_user = csr_matrix((n_users, n_users), dtype=np.float32)
        zero_item = csr_matrix((m_items, m_items), dtype=np.float32)
        
        # 2. 拼接矩陣
        # 上半部: [0, R]
        upper = hstack([zero_user, R])
        # 下半部: [R.T, 0]
        lower = hstack([RT, zero_item])
        
        # 整個 A
        adj_mat = vstack([upper, lower])
        adj_mat = adj_mat.tocsr()

        # ==========================================
        # 以下是標準的 Normalization (D^-1/2 * A * D^-1/2)
        # ==========================================
        
        rowsum = np.array(adj_mat.sum(axis=1))
        
        # 防止除以 0 (加上 1e-7 或直接設 0)
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        norm_adj = norm_adj.tocsr()
        
        self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.Graph = self.Graph.coalesce().to(device)
        
        cprint("✅ Adjacency matrix generated (Memory Optimized)!")
        return self.Graph