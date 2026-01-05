# 1. 選擇基底映像檔 (我們用官方輕量版的 Python 3.11)
FROM python:3.11-slim

# 2. 設定工作目錄
WORKDIR /app

# 3. 複製 requirements.txt
COPY requirements.txt .

# 4. 安裝套件
RUN pip install --no-cache-dir -r requirements.txt

# 5. 複製所有程式碼
COPY . .

# 6. 開啟 Port
EXPOSE 8000

# 7. 啟動指令
CMD ["python", "server.py"]