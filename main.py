import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# 導入我們拆分好的模組
from src.dataset import Loader
from src.model import LightGCN
from src.utils import set_seed, BPRLoss, Test, cprint, BPR_train_with_sampler
from src.sampler import HardNegativeSampler

# ================= 輔助函數 =================
def fmt_secs(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    if h > 0: return f"{h}h {m}m {sec}s"
    elif m > 0: return f"{m}m {sec}s"
    else: return f"{sec}s"

def save_comparison_plot(epochs, results_uniform, results_hard, filename="comparison_result.png"):
    """ 畫出對比圖並存檔 """
    E = np.arange(1, epochs + 1)
    plt.figure(figsize=(18, 5))

    # 1. Loss 比較
    plt.subplot(1, 3, 1)
    plt.plot(E, results_uniform["loss"], label="Uniform (Baseline)", color='gray', linestyle='--')
    plt.plot(E, results_hard["loss"],    label="Hard Negative", color='red')
    plt.title("BPR Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 2. Recall 比較
    plt.subplot(1, 3, 2)
    plt.plot(E, results_uniform["recall"], label="Uniform (Baseline)", color='gray', linestyle='--')
    plt.plot(E, results_hard["recall"],    label="Hard Negative", color='blue')
    plt.title("Recall@20 Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 3. NDCG 比較
    plt.subplot(1, 3, 3)
    plt.plot(E, results_uniform["ndcg"], label="Uniform (Baseline)", color='gray', linestyle='--')
    plt.plot(E, results_hard["ndcg"],    label="Hard Negative", color='green')
    plt.title("NDCG@20 Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("NDCG")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"\n✅ 比較圖表已儲存至: {filename}")
    # plt.show() # 在 VS Code 建議用存檔的方式查看

# ================= 主程式 =================
if __name__ == "__main__":
    # 1. 設定全域參數
    TRAIN_epochs = 10  # 根據你的需求調整
    
    SEED = 2024
    set_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cprint(f"Running on Device: {device}")

    # 你的最佳參數設定
    BEST_HARD_RATIO   = 0.5
    BEST_UPDATE_EVERY = 10
    BEST_POOL_SIZE    = 150

    # 模型 Config
    config = {
        'bpr_batch_size': 2048,
        'latent_dim_rec': 64,
        'lightGCN_n_layers': 3,
        'dropout': False,
        'keep_prob': 0.6,
        'A_n_fold': 100,
        'test_u_batch_size': 100,
        'multicore': 0,
        'lr': 0.001,
        'decay': 1e-4,
        'A_split': False,
        'A_n_fold': 5
    }

    # 2. 載入資料 (只載入一次，兩個模型共用)
    # 注意：測試時可以保留 max_users=1000 加快速度，正式跑請拿掉 max_users
    print("Loading Data...")
    dataset = Loader(config, "./data/gowalla", max_users=1000, max_items=5000) # 測試用
    #dataset = Loader(config, "./data/gowalla") # 正式用 (全量資料)
    
    all_results = {}

    # ==========================================
    # 實驗 1: Uniform Baseline (原本的方法)
    # ==========================================
    print("\n" + "="*60)
    print(f"[Uniform] Start Baseline Experiment (Epochs={TRAIN_epochs})")
    print("="*60)

    # 重新初始化模型 (確保權重重置)
    set_seed(SEED) 
    Rec_uniform = LightGCN(config, dataset).to(device)
    bpr_uniform = BPRLoss(Rec_uniform, config)

    u_losses, u_recalls, u_ndcgs = [], [], []
    u_times = []
    start_time = time.time()

    for epoch in range(1, TRAIN_epochs + 1):
        ep_start = time.time()
        
        # 訓練：sampler=None 代表使用 Uniform
        loss_val = BPR_train_with_sampler(
            dataset, Rec_uniform, bpr_uniform, 
            epoch=epoch, sampler=None, batch_size=config['bpr_batch_size']
        )
        u_losses.append(loss_val)

        # 測試
        if epoch % 5 == 0 or epoch == 1 or epoch == TRAIN_epochs:
            metrics = Test(dataset, Rec_uniform, K=20)
            u_recalls.append(metrics["recall"])
            u_ndcgs.append(metrics["ndcg"])
            
            print(f"[Uniform] Epoch {epoch:3d}: Loss={loss_val:.4f} | "
                  f"R@20={metrics['recall']:.4f} NDCG@20={metrics['ndcg']:.4f}")
        else:
            # 沒測試時補上一個值方便畫圖
            u_recalls.append(u_recalls[-1] if u_recalls else 0)
            u_ndcgs.append(u_ndcgs[-1] if u_ndcgs else 0)
            
        u_times.append(time.time() - ep_start)

    all_results["uniform"] = {"loss": u_losses, "recall": u_recalls, "ndcg": u_ndcgs}
    print(f"[Uniform] Done. Total Time: {fmt_secs(sum(u_times))}")


    # ==========================================
    # 實驗 2: Hard Negative Sampling (你的改進)
    # ==========================================
    print("\n" + "="*60)
    print(f"[HardNeg] Start Experiment (Ratio={BEST_HARD_RATIO}, Pool={BEST_POOL_SIZE})")
    print("="*60)

    # 重新初始化模型 (這很重要，不能用剛剛訓練過的)
    set_seed(SEED)
    Rec_hard = LightGCN(config, dataset).to(device)
    bpr_hard = BPRLoss(Rec_hard, config)

    # 初始化 Sampler
    sampler = HardNegativeSampler(
        model=Rec_hard,
        dataset=dataset,
        pool_size=BEST_POOL_SIZE,
        candidate_size=1000,
        update_every=BEST_UPDATE_EVERY,
        device=device,
        user_batch=config['bpr_batch_size'],
        hard_ratio=BEST_HARD_RATIO
    )

    h_losses, h_recalls, h_ndcgs = [], [], []
    h_times = []
    
    for epoch in range(1, TRAIN_epochs + 1):
        ep_start = time.time()

        # 更新 Hard Negative Pool
        if (epoch % sampler.update_every) == 0:
            # 這裡小技巧：通常在訓練前更新，或者每隔 N epoch 更新
            sampler.update_pool(Rec_hard)

        # 訓練：傳入 sampler
        loss_val = BPR_train_with_sampler(
            dataset, Rec_hard, bpr_hard, 
            epoch=epoch, sampler=sampler, batch_size=config['bpr_batch_size']
        )
        h_losses.append(loss_val)

        # 測試
        if epoch % 5 == 0 or epoch == 1 or epoch == TRAIN_epochs:
            metrics = Test(dataset, Rec_hard, K=20)
            h_recalls.append(metrics["recall"])
            h_ndcgs.append(metrics["ndcg"])
            
            print(f"[HardNeg] Epoch {epoch:3d}: Loss={loss_val:.4f} | "
                  f"R@20={metrics['recall']:.4f} NDCG@20={metrics['ndcg']:.4f}")
        else:
            h_recalls.append(h_recalls[-1] if h_recalls else 0)
            h_ndcgs.append(h_ndcgs[-1] if h_ndcgs else 0)
            
        h_times.append(time.time() - ep_start)

    all_results["hard"] = {"loss": h_losses, "recall": h_recalls, "ndcg": h_ndcgs}
    print(f"[HardNeg] Done. Total Time: {fmt_secs(sum(h_times))}")

# 儲存訓練好的模型參數 (Save Model)
    # 我們存 Hard Negative 的那個版本，因為效果通常比較好
    save_path = "lightgcn_model.pth"
    torch.save(Rec_hard.state_dict(), save_path)
    print(f"✅ 模型已儲存至: {save_path}")

    # ==========================================
    # 3. 畫圖並比較
    # ==========================================
    print("\n=== Summary ===")
    print(f"Uniform Final Recall: {all_results['uniform']['recall'][-1]:.4f}")
    print(f"HardNeg Final Recall: {all_results['hard']['recall'][-1]:.4f}")
    
    save_comparison_plot(TRAIN_epochs, all_results["uniform"], all_results["hard"])