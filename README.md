High-Performance LightGCN Recommender System Microservice

這是一個基於 LightGCN (Light Graph Convolutional Network) 的高效能推薦系統微服務。
專案整合了 Hard Negative Sampling 演算法優化、Scipy 稀疏矩陣記憶體優化，並使用 FastAPI 與 Docker 實現容器化部署，是一套完整的 MLOps 實踐案例。

📂 專案結構與檔案說明 (Project Structure)

本專案採用模組化設計，各檔案功能如下：

LightGCN_Recommender/
├── src/                        # [核心模組]
│   ├── dataset.py              # 資料處理層
│   │   - 負責讀取 Gowalla 數據集 (train.txt/test.txt)。
│   │   - ★ 關鍵優化：使用 scipy.sparse (vstack/hstack) 重構稀疏矩陣，
│   │     解決大規模圖資料 (30k User, 40k Item) 在文書筆電上的 OOM 問題。
│   │
│   ├── model.py                # 模型架構層
│   │   - 定義 LightGCN 類別 (繼承 nn.Module)。
│   │   - 實作圖卷積 (Graph Convolution) 與 Embedding 傳播機制。
│   │
│   ├── sampler.py              # 採樣策略層
│   │   - ★ 關鍵演算法：實作 Hard Negative Sampling (困難負採樣)。
│   │   - 動態挑選分數高但非正樣本的項目進行訓練，提升模型區別能力。
│   │
│   └── utils.py                # 工具層
│       - 包含 BPR Loss (Bayesian Personalized Ranking) 損失函數。
│       - 實作評估指標：Recall@K 與 NDCG@K。
│       - 設定 Random Seed 確保實驗可重現。
│
├── main.py                     # [訓練入口]
│   - 負責整合 Dataset, Model, Sampler 進行模型訓練。
│   - 執行 Uniform vs Hard Negative 的對照實驗。
│   - 繪製比較圖表 (comparison_result.png) 並儲存模型權重 (.pth)。
│   
├── server.py                   # [微服務入口]
│   - 使用 FastAPI 建立 RESTful API。
│   - 實作 Lifespan 機制：伺服器啟動時預載入模型，推論時不需重複讀檔。
│   - 提供 /recommend/{user_id} 接口，回傳 Top-K 推薦結果。
│   
├── Dockerfile                  # [容器化設定]
│   - 定義 Python 3.11 環境、安裝 PyTorch 等依賴。
│   - 設定啟動指令，實現一鍵部署。
│   
├── requirements.txt            # [套件清單]
└── comparison_result.png       # [實驗結果圖]


⚙️ 系統運作原理 (How it Works)

1. 模型訓練流程 (main.py)

資料載入: Loader 讀取 Gowalla 數據，建構 User-Item 二分圖的鄰接矩陣 (Adjacency Matrix)。

圖卷積: LightGCN 模型將 User 和 Item 的 Embedding 在圖上傳播 (Propagation)，聚合鄰居特徵。

負採樣: HardNegativeSampler 每隔 N 個 Epoch 計算當前預測分數，挑選「難以區分」的負樣本放入訓練池。

優化: 計算 BPR Loss，透過 Adam 優化器更新 Embedding。

評估與存檔: 每隔數個 Epoch 計算 Recall/NDCG，訓練結束後儲存 lightgcn_model.pth。

2. 推論服務流程 (server.py)

啟動: Docker 啟動時，lifespan 函數會自動載入 lightgcn_model.pth 到記憶體 (GPU/CPU)。

請求: 當使用者呼叫 GET /recommend/10。

計算: 模型計算該 User 與所有 Items 的內積 (Dot Product) 分數。

排序: 取分數最高的 Top-K 物品。

回傳: 以 JSON 格式回傳推薦列表。

🚀 快速開始 (Quick Start)

步驟 0: 準備資料集

由於資料集較大未上傳至 GitHub，請先下載 Gowalla 資料集並放入 data 資料夾：

建立資料夾：mkdir -p data/gowalla

將 train.txt 與 test.txt 放入 data/gowalla/ 中。

方法 1: 使用 Docker 啟動 (推薦)

這是最簡單的方法，不需要在本地安裝 Python 環境。

# 1. 建置映像檔 (這會讀取 Dockerfile 並安裝所有套件)
docker build -t lightgcn-app .

# 2. 啟動容器 (將容器的 8000 port 對應到本機的 8000 port)
docker run -p 8000:8000 lightgcn-app


方法 2: 本機開發執行

# 1. 建立虛擬環境 & 安裝依賴
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt

# 2. 執行訓練 (會產出 lightgcn_model.pth 與比較圖表)
python main.py

# 3. 啟動 API Server
python server.py


📊 效能比較 (Performance)

本專案比較了 Hard Negative Sampling (紅線) 與傳統 Uniform Sampling (灰線) 的訓練效果。
可以看到改進後的方法在 Recall@20 與 NDCG@20 均有顯著提升，證明困難負採樣能有效幫助模型學習細微特徵。

(執行 main.py 後會自動生成此圖表)

🔗 API 文件

啟動服務後，可訪問 Swagger UI 進行互動式測試：

文件網址: http://localhost:8000/docs

推論接口: GET /recommend/{user_id}

Example Response:

{
  "user_id": 10,
  "top_k": 5,
  "recommendations": [1302, 1292, 1307, 1638, 211]
}
