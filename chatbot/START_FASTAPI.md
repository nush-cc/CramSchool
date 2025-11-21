# FastAPI RAG 服務啟動指南

## 📋 前置準備

### 1. 安裝依賴

```bash
# 在專案根目錄（D:\NCKU\cram）
pip install -r chatbot/requirements_fastapi.txt
```

### 2. 檢查必要檔案

確認以下檔案存在：
- ✅ `.env` - 包含 OPENAI_API_KEY
- ✅ `chatbot/dataset/handouts_data/*.pdf` - 教學資料
- ✅ `chatbot/dataset/raw_data/add_id_data/question_math_id.json` - 練習題
- ✅ `chatbot/faiss_index_teaching/` - 教學向量庫
- ✅ `chatbot/faiss_index_exercise/` - 練習題向量庫

---

## 🚀 啟動服務

### 方法 1: 使用 uvicorn 指令（推薦）

```bash
# 在專案根目錄（D:\NCKU\cram）
cd D:\NCKU\cram
uvicorn chatbot.fastapi_app:app --host 0.0.0.0 --port 8001 --reload
```

### 方法 2: 直接執行 Python

```bash
cd D:\NCKU\cram\chatbot
python fastapi_app.py
```

---

## 📝 啟動後應該看到

```
============================================================
🚀 FastAPI RAG 服務啟動中...
============================================================

[1/4] 檢查資料檔案...
   ✅ 教學向量庫: D:\NCKU\cram\chatbot\faiss_index_teaching
   ✅ 練習題向量庫: D:\NCKU\cram\chatbot\faiss_index_exercise
   ✅ 練習題資料: D:\NCKU\cram\chatbot\dataset\raw_data\add_id_data\question_math_id.json

[2/4] 初始化 RAG 處理器...
   ✅ RAG 處理器初始化完成

[3/4] 載入向量資料庫...
   📚 找到 2 個教學檔案
   ✅ 向量資料庫載入完成

[4/4] 檢查 OpenAI API...
   ✅ OpenAI API Key 已設定

============================================================
✅ RAG 服務啟動成功！
============================================================

📝 API 文件: http://localhost:8001/docs
🔍 健康檢查: http://localhost:8001/health
```

---

## 🧪 測試 API

### 1. 健康檢查

在瀏覽器開啟：
```
http://localhost:8001/health
```

應該看到：
```json
{
  "status": "ok",
  "rag_loaded": true,
  "message": "RAG 服務運行正常"
}
```

### 2. 查看 API 文件（Swagger UI）

在瀏覽器開啟：
```
http://localhost:8001/docs
```

你會看到完整的 API 文件，可以直接在瀏覽器測試！

### 3. 測試基本聊天（使用 curl）

```bash
curl -X POST "http://localhost:8001/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "什麼是三角形全等？",
    "search_type": "teaching",
    "learner_style": "標準級"
  }'
```

### 4. 測試帶記憶的聊天

```bash
curl -X POST "http://localhost:8001/chat_with_history" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "第一個是什麼？",
    "history": [
      {"role": "user", "content": "三角形全等有哪些判定方法？"},
      {"role": "assistant", "content": "有 SSS、SAS、ASA、AAS 四種判定方法"}
    ],
    "search_type": "teaching",
    "learner_style": "基礎級"
  }'
```

### 5. 測試練習題模式

```bash
curl -X POST "http://localhost:8001/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "給我一題關於三角形的練習題",
    "search_type": "exercise",
    "learner_style": "標準級"
  }'
```

---

## 📡 API 端點說明

### `GET /health`
健康檢查，確認服務是否正常

### `POST /chat`
基本問答（無記憶）
- **request body:**
  ```json
  {
    "message": "學生的問題",
    "search_type": "teaching|exercise|hybrid",
    "learner_style": "基礎級|標準級|進階級",
    "course_id": 123  // 可選
  }
  ```

### `POST /chat_with_history`
帶記憶的問答
- **request body:**
  ```json
  {
    "message": "學生的問題",
    "history": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ],
    "search_type": "teaching|exercise|hybrid",
    "learner_style": "基礎級|標準級|進階級"
  }
  ```

---

## 🐛 常見問題

### 問題 1: ModuleNotFoundError
```
解決方法：確保在專案根目錄（D:\NCKU\cram）執行指令
cd D:\NCKU\cram
uvicorn chatbot.fastapi_app:app --port 8001
```

### 問題 2: CUDA out of memory
```
解決方法：修改 config.py
model_device = "cpu"  # 改用 CPU
```

### 問題 3: 找不到 OPENAI_API_KEY
```
解決方法：檢查 D:\NCKU\cram\.env 檔案
確保有這一行：
OPENAI_API_KEY=sk-proj-...
```

### 問題 4: 向量資料庫載入失敗
```
解決方法：重新建立向量資料庫
1. 刪除 faiss_index_teaching 和 faiss_index_exercise 資料夾
2. 執行 Rag_model.py 重新建立
```

---

## 🔧 開發模式

FastAPI 使用 `--reload` 參數會自動偵測檔案變更並重新載入：

```bash
uvicorn chatbot.fastapi_app:app --port 8001 --reload
```

修改 `fastapi_app.py` 後，服務會自動重啟（但向量資料庫會重新載入，需要等待）

---

## 📊 效能監控

查看處理時間：
```json
{
  "answer": "...",
  "processing_time": 2.35  // 秒
}
```

一般來說：
- **檢索**: 0.5-1 秒
- **LLM 生成**: 1-3 秒
- **總計**: 2-5 秒

---

## 🎯 下一步

✅ **Phase 1 完成** - FastAPI 基本功能
- [x] 基本問答 endpoint
- [x] 帶記憶的問答 endpoint
- [x] 三種檢索模式
- [x] 三種學習風格

🔜 **Phase 2** - 整合到 Django
- [ ] Django view 呼叫 FastAPI
- [ ] 前端 JavaScript 修改
- [ ] 對話歷史記錄到資料庫
- [ ] 錯誤處理和 timeout

---

## 📞 測試完成後

如果測試成功，你應該能：
1. ✅ 在瀏覽器看到 Swagger UI
2. ✅ 健康檢查返回正常
3. ✅ 使用 Swagger UI 測試聊天功能
4. ✅ 得到 RAG 生成的答案

**測試成功後，告訴我結果，我們就可以進入 Phase 2（整合到 Django）！** 🚀
