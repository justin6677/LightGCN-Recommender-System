import torch
import numpy as np
from src.model import LightGCN
from src.dataset import Loader

# 設定參數 (必須跟訓練時一樣)
config = {
    'latent_dim_rec': 64,
    'lightGCN_n_layers': 3,
    'A_n_fold': 100,
    'A_split': False,
    # 這些是為了初始化 Dataset 用的，推論時其實用不到太多
    'bpr_batch_size': 2048, 
    'dropout': False,
    'keep_prob': 0.6,
    'test_u_batch_size': 100,
    'multicore': 0,
    'lr': 0.001,
    'decay': 1e-4,
    'pretrain': 0
}

def get_recommendation(user_id, k=10):
    """
    輸入 User ID，回傳 Top-K 推薦的 Item ID
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 載入資料 (為了建立圖結構，LightGCN 需要 Graph)
    # 注意：這裡我們需要全量資料來建立正確的 Graph
    # 如果只是測試，可以用 max_users=1000，但 ID 必須在範圍內
    print("Loading Data for Inference...")
    dataset = Loader(config, "./data/gowalla", max_users=1000, max_items=5000)
    
    # 2. 初始化模型
    model = LightGCN(config, dataset).to(device)
    
    # 3. 載入權重 (Load Weights)
    try:
        model.load_state_dict(torch.load("lightgcn_model.pth", map_location=device))
        print("✅ 成功載入模型權重！")
    except FileNotFoundError:
        print("❌ 找不到模型檔案，請先執行 main.py 進行訓練！")
        return []

    model.eval() # 切換到評估模式

    # 4. 進行預測
    with torch.no_grad():
        # 取得該 User 的 Embedding
        # 因為 LightGCN 需要全圖卷積，所以我們要呼叫 getUsersRating 算出所有分數
        # (這裡有優化空間，但先求有)
        user_tensor = torch.tensor([user_id], dtype=torch.long).to(device)
        
        # 取得對所有物品的評分
        ratings = model.getUsersRating(user_tensor)
        
        # 排除訓練集裡已經看過的 (Optional, 這裡先簡化不排除)
        
        # 取 Top-K
        _, topk_indices = torch.topk(ratings, k=k, dim=1)
        
        recommendations = topk_indices.cpu().numpy().flatten().tolist()
        return recommendations

if __name__ == "__main__":
    # 測試：推薦給 User ID = 10
    target_user = 10
    print(f"正在計算 User {target_user} 的推薦結果...")
    
    recs = get_recommendation(target_user, k=5)
    
    print("="*30)
    print(f"👤 User ID: {target_user}")
    print(f"📦 推薦商品 (Top-5): {recs}")
    print("="*30)