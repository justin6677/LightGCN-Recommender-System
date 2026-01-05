from fastapi import FastAPI, HTTPException
import torch
import uvicorn
from contextlib import asynccontextmanager

# 引入你的模組
from src.model import LightGCN
from src.dataset import Loader

# 全域變數 (用來放載入好的模型，避免每次請求都重新載入)
model_cache = {}

# 設定參數 (必須跟訓練時一樣)
config = {
    'latent_dim_rec': 64,
    'lightGCN_n_layers': 3,
    'A_n_fold': 100,
    'A_split': False,
    'bpr_batch_size': 2048,
    'dropout': False,
    'keep_prob': 0.6,
    'test_u_batch_size': 100,
    'multicore': 0,
    'lr': 0.001,
    'decay': 1e-4,
    'pretrain': 0
}

# === 1. 定義伺服器啟動時的行為 (Lifespan) ===
# 這段程式碼保證模型只會在伺服器啟動時載入一次 (省時間！)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 正在啟動推薦系統 API...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"運行裝置: {device}")

    # 1. 載入資料 (建立 Graph)
    print("Loading Data...")
    # 注意：這裡使用 max_users 來加快啟動速度，正式環境請拿掉
    dataset = Loader(config, "./data/gowalla", max_users=1000, max_items=5000)
    
    # 2. 初始化模型
    model = LightGCN(config, dataset).to(device)
    
    # 3. 載入權重
    try:
        model.load_state_dict(torch.load("lightgcn_model.pth", map_location=device))
        print("✅ 模型權重載入成功！")
        model.eval() # 切換到推論模式
    except FileNotFoundError:
        print("❌ 錯誤：找不到 lightgcn_model.pth，請先執行 main.py 訓練模型！")
    
    # 把模型存到快取中
    model_cache['model'] = model
    model_cache['device'] = device
    
    yield # 這裡代表伺服器開始運作
    
    print("🛑 伺服器正在關閉...")
    model_cache.clear()

# === 2. 建立 FastAPI APP ===
app = FastAPI(title="LightGCN Recommender API", lifespan=lifespan)

# === 3. 定義 API 路徑 (Endpoint) ===

@app.get("/")
def read_root():
    return {"message": "歡迎來到 LightGCN 推薦系統 API！請訪問 /docs 查看使用說明。"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 5):
    """
    輸入 User ID，回傳 Top-K 推薦列表
    例如: /recommend/10?k=5
    """
    model = model_cache.get('model')
    device = model_cache.get('device')
    
    if model is None:
        raise HTTPException(status_code=500, detail="模型未載入")

    # 檢查 User ID 是否在範圍內
    if user_id >= model.num_users:
        raise HTTPException(status_code=404, detail=f"User ID {user_id} 不存在 (超出範圍)")

    try:
        with torch.no_grad():
            user_tensor = torch.tensor([user_id], dtype=torch.long).to(device)
            ratings = model.getUsersRating(user_tensor)
            
            # 取 Top-K
            _, topk_indices = torch.topk(ratings, k=k, dim=1)
            recs = topk_indices.cpu().numpy().flatten().tolist()
            
        return {
            "user_id": user_id,
            "top_k": k,
            "recommendations": recs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === 4. 如果直接執行此檔案，則啟動伺服器 ===
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)