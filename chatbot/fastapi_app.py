"""
FastAPI 服務 - RAG 聊天機器人
用於學生問答系統的後端 API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import sys
import time
from dotenv import load_dotenv

# 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

# 載入環境變數
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

# 導入 RAG 相關模組
from chatbot.rag_pipeline.RAG_function import rag_process

def get_drawing_info(retrieved_docs):
    import json
    
    # 路徑設定
    DRAWING_DIR = os.path.join(project_root, "chatbot", "dataset", "llama_drawing_steps")
    
    for doc in retrieved_docs:
        # 1. 從 FAISS Metadata 取得 ID
        if hasattr(doc, "metadata"):
            doc_id = doc.metadata.get("id") # 這裡拿到的是 "2907"
            
            if doc_id:
                # 2. 拼湊檔名：目標是 "2907_layout.json"
                # 注意：這裡要根據你的截圖調整，只找 _layout.json
                target_filename = f"{doc_id}_layout.json"
                full_path = os.path.join(DRAWING_DIR, target_filename)
                
                # 3. 檢查檔案是否存在
                if os.path.exists(full_path):
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # 計算總步數
                            steps = len(data.get("steps", []))
                            # 回傳 ID (2907) 和總步數
                            return str(doc_id), steps
                    except Exception as e:
                        print(f"讀取 Layout JSON 失敗: {e}")
                        continue
                        
    return None, 0

# ==================== Pydantic Models ====================

class ChatMessage(BaseModel):
    """聊天訊息模型"""
    role: str = Field(..., description="user 或 assistant")
    content: str = Field(..., description="訊息內容")


class ChatRequest(BaseModel):
    """聊天請求模型"""
    message: str = Field(..., description="學生的問題", min_length=1)
    search_type: str = Field(
        default="teaching",
        description="檢索類型: teaching(教學), exercise(練習題), hybrid(混合)"
    )
    learner_style: str = Field(
        default="標準級",
        description="學習風格: 基礎級, 標準級, 進階級"
    )
    course_id: Optional[int] = Field(default=None, description="課程 ID（可選）")
    course_title: Optional[str] = Field(
        default=None,
        description="課程標題/主題（如：一元一次方程式、比例等）"
    )
    history: Optional[List[ChatMessage]] = Field(
        default=[],
        description="對話歷史（用於記憶功能）"
    )
    # 新增：重試和降級相關參數
    is_retry: bool = Field(
        default=False,
        description="是否為重試請求（使用者點擊「我還是不懂」）"
    )
    retry_count: int = Field(
        default=0,
        description="重試次數（用於基礎級多角度解釋）"
    )
    use_alternative: bool = Field(
        default=False,
        description="是否使用替代解釋方法（基礎級換角度）"
    )


class ChatResponse(BaseModel):
    """聊天回應模型"""
    answer: str = Field(..., description="AI 生成的答案（完整）")
    segments: List[str] = Field(
        default=[],
        description="答案分段（3段，用於點選深入追問）"
    )
    retrieved_docs: List[Dict[str, Any]] = Field(
        default=[],
        description="檢索到的相關文件"
    )
    processing_time: float = Field(..., description="處理時間（秒）")
    search_type: str = Field(..., description="使用的檢索類型")
    learner_style: str = Field(..., description="使用的學習風格")
    exercise_question: Optional[str] = Field(
        default=None,
        description="練習題的題目部分（僅練習題模式）"
    )
    exercise_answer: Optional[str] = Field(
        default=None,
        description="練習題的答案部分（僅練習題模式，用於遮罩）"
    )
    drawing_id: Optional[str] = Field(default=None, description="對應的畫圖 ID")
    drawing_total_steps: int = Field(default=0, description="畫圖總步數")


class ClarifyRequest(BaseModel):
    """深入追問請求模型"""
    selected_text: str = Field(..., description="學生選中的文字片段", min_length=1)
    original_query: str = Field(..., description="原始問題")
    learner_style: str = Field(
        default="標準級",
        description="學習風格: 基礎級, 標準級, 進階級"
    )
    original_context: Optional[str] = Field(
        default=None,
        description="原始答案的上下文（可選）"
    )


class ClarifyResponse(BaseModel):
    """深入追問回應模型"""
    clarification: str = Field(..., description="針對選中文字的深入解釋")
    processing_time: float = Field(..., description="處理時間（秒）")


class HealthResponse(BaseModel):
    """健康檢查回應"""
    status: str
    rag_loaded: bool
    message: str


# ==================== FastAPI App ====================

app = FastAPI(
    title="RAG 聊天機器人 API",
    description="基於 RAG 的學生問答系統後端服務",
    version="1.0.0"
)

# CORS 設定（允許 Django 前端呼叫）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開發時允許所有來源，正式環境應該限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 全域變數 ====================

rag_service = None
rag_initialized = False

# 資料路徑配置
CHATBOT_DIR = current_dir
DATASET_DIR = os.path.join(CHATBOT_DIR, "dataset")
TEACHING_DATA_DIR = os.path.join(DATASET_DIR, "handouts_data")
EXERCISE_DATA_PATH = os.path.join(DATASET_DIR, "raw_data", "add_id_data", "question_math_id.json")
FAISS_INDEX_TEACHING = os.path.join(CHATBOT_DIR, "faiss_index_teaching")
FAISS_INDEX_EXERCISE = os.path.join(CHATBOT_DIR, "faiss_index_exercise")


# ==================== 啟動事件 ====================

@app.on_event("startup")
async def startup_event():
    """應用啟動時初始化 RAG 系統"""
    global rag_service, rag_initialized

    print("\n" + "=" * 60)
    print("🚀 FastAPI RAG 服務啟動中...")
    print("=" * 60)

    try:
        # 檢查必要檔案
        print("\n[1/4] 檢查資料檔案...")

        if not os.path.exists(FAISS_INDEX_TEACHING):
            raise FileNotFoundError(f"找不到教學向量資料庫: {FAISS_INDEX_TEACHING}")
        if not os.path.exists(FAISS_INDEX_EXERCISE):
            raise FileNotFoundError(f"找不到練習題向量資料庫: {FAISS_INDEX_EXERCISE}")
        if not os.path.exists(EXERCISE_DATA_PATH):
            raise FileNotFoundError(f"找不到練習題資料: {EXERCISE_DATA_PATH}")

        print(f"   ✅ 教學向量庫: {FAISS_INDEX_TEACHING}")
        print(f"   ✅ 練習題向量庫: {FAISS_INDEX_EXERCISE}")
        print(f"   ✅ 練習題資料: {EXERCISE_DATA_PATH}")

        # 初始化 RAG 處理器
        print("\n[2/4] 初始化 RAG 處理器...")
        rag_service = rag_process()
        print("   ✅ RAG 處理器初始化完成")

        # 載入向量資料庫
        print("\n[3/4] 載入向量資料庫...")

        # 直接載入已存在的 FAISS 索引（不重新建立）
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        from langchain_community.vectorstores import FAISS
        from chatbot.config import embedding_model_name, model_device

        # 初始化 Embedding 模型
        model_kwargs = {"device": model_device}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
        print("   ✅ Embedding 模型載入完成")

        # 載入教學向量庫
        print(f"   📚 載入教學向量庫: {FAISS_INDEX_TEACHING}")
        rag_service.teaching_vs = FAISS.load_local(
            FAISS_INDEX_TEACHING,
            embeddings,
            allow_dangerous_deserialization=True
        )
        rag_service.teaching_ds = []  # 暫時設為空列表（不影響檢索）

        # 載入練習題向量庫
        print(f"   📚 載入練習題向量庫: {FAISS_INDEX_EXERCISE}")
        rag_service.exercise_vs = FAISS.load_local(
            FAISS_INDEX_EXERCISE,
            embeddings,
            allow_dangerous_deserialization=True
        )
        rag_service.exercise_ds = []  # 暫時設為空列表（不影響檢索）

        print("   ✅ 向量資料庫載入完成")

        # 檢查 OpenAI API Key
        print("\n[4/4] 檢查 OpenAI API...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("找不到 OPENAI_API_KEY，請檢查 .env 檔案")
        print(f"   ✅ OpenAI API Key 已設定 (開頭: {api_key[:8]}...)")

        rag_initialized = True

        print("\n" + "=" * 60)
        print("✅ RAG 服務啟動成功！")
        print("=" * 60)
        print(f"\n📝 API 文件: http://localhost:8001/docs")
        print(f"🔍 健康檢查: http://localhost:8001/health\n")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ RAG 服務啟動失敗: {str(e)}")
        print("=" * 60)
        rag_initialized = False
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """應用關閉時的清理"""
    print("\n🛑 FastAPI RAG 服務關閉中...")
    print("✅ 清理完成\n")


# ==================== API Endpoints ====================

@app.get("/", tags=["根路徑"])
async def root():
    """根路徑"""
    return {
        "message": "RAG 聊天機器人 API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["健康檢查"])
async def health_check():
    """
    健康檢查端點

    檢查 RAG 服務是否正常運行
    """
    return HealthResponse(
        status="ok" if rag_initialized else "error",
        rag_loaded=rag_initialized,
        message="RAG 服務運行正常" if rag_initialized else "RAG 服務未初始化"
    )


@app.post("/chat", response_model=ChatResponse, tags=["聊天"])
async def chat(request: ChatRequest):
    """
    基本聊天端點（無記憶）

    - **message**: 學生的問題
    - **search_type**: 檢索類型（teaching/exercise/hybrid）
    - **learner_style**: 學習風格（基礎級/標準級/進階級）
    - **course_id**: 課程 ID（可選）
    """
    if not rag_initialized:
        raise HTTPException(status_code=503, detail="RAG 服務尚未初始化完成")

    start_time = time.time()

    try:
        # 驗證參數
        if request.search_type not in ["teaching", "exercise", "hybrid"]:
            raise HTTPException(
                status_code=400,
                detail=f"無效的 search_type: {request.search_type}"
            )

        if request.learner_style not in ["基礎級", "標準級", "進階級"]:
            raise HTTPException(
                status_code=400,
                detail=f"無效的 learner_style: {request.learner_style}"
            )

        # 設定檢索數量
        if request.search_type == "teaching":
            top_n = 3
        elif request.search_type == "exercise":
            top_n = 1
        else:  # hybrid
            top_n = 4

        # 1. 檢索相關文件（加入課程過濾）
        retrieved = rag_service.retrival_step(
            [request.message],
            request.search_type,
            (rag_service.teaching_vs, rag_service.teaching_ds),
            (rag_service.exercise_vs, rag_service.exercise_ds),
            top_n=top_n,
            course_filter=request.course_title  # 傳入課程標題進行過濾
        )

        retrieved_docs = retrieved.get(request.message, [])

        # [新增] 檢查是否有畫圖 ID
        drawing_id, total_steps = get_drawing_info(retrieved_docs)

        if not drawing_id and request.search_type != "teaching": # 如果是純教學模式就不找
            try:
                # 專門針對練習題庫 (Exercise) 搜 1 筆
                extra_retrieval = rag_service.retrival_step(
                    [request.message],
                    "exercise", # 強制搜練習題
                    (rag_service.teaching_vs, rag_service.teaching_ds),
                    (rag_service.exercise_vs, rag_service.exercise_ds),
                    top_n=1,
                    course_filter=None # 為了提高命中率，可以先不過濾課程
                )
                extra_docs = extra_retrieval.get(request.message, [])
                
                # 檢查這外搜出來的一題有沒有圖
                extra_id, extra_steps = get_drawing_info(extra_docs)
                
                if extra_id:
                    print(f"💡 [側面推薦] 主要回答沒圖，但從練習題庫找到了相關圖表 ID: {extra_id}")
                    drawing_id = extra_id
                    total_steps = extra_steps
                    # 選擇性：你要不要把這題的題目/答案也覆蓋過去？
                    # 如果你只想顯示圖，保留原本的回答，就這樣就好。
                    # 如果你想讓 AI 順便提到這題，你可以把 extra_docs 加進 context。
            except Exception as e:
                print(f"側面推薦檢索失敗: {e}")

        # 2. 建立上下文
        matched_context = "\n".join([
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in retrieved_docs
        ])

        # 3. 生成答案（目前無記憶，memory_chunk 為空）
        memory_chunk = ""
        is_exercise_mode = (request.search_type == "exercise")
        answer = rag_service.generate_answer(
            matched_context,
            request.message,
            request.learner_style,
            memory_chunk,
            is_exercise_mode=is_exercise_mode,
            course_title=request.course_title,  # 傳遞課程標題
            use_alternative=request.use_alternative,  # 是否換角度
            retry_count=request.retry_count  # 重試次數
        )

        # 4. 解析練習題的題目和答案（僅在練習題模式）
        exercise_question = None
        exercise_answer = None
        segments = []

        if is_exercise_mode:
            # 解析【題目】和【答案】
            import re
            question_match = re.search(r'【題目】\s*(.*?)\s*【答案】', answer, re.DOTALL)
            answer_match = re.search(r'【答案】\s*(.*)', answer, re.DOTALL)

            if question_match and answer_match:
                exercise_question = question_match.group(1).strip()
                exercise_answer = answer_match.group(1).strip()
            else:
                # 如果沒有匹配到格式，整個當作題目
                exercise_question = answer
                exercise_answer = "（AI 未提供標準答案格式）"

            # 練習題模式不分段
            segments = []
        else:
            # 非練習題模式才分段
            from chatbot.rag_pipeline.post_process import Post_process
            post_processor = Post_process()
            segments = post_processor.split_answer(answer)

        # 5. 整理檢索文件資訊
        docs_info = []
        for doc in retrieved_docs[:3]:  # 只返回前 3 個
            doc_info = {
                "content": doc.page_content if hasattr(doc, "page_content") else str(doc),
                "metadata": doc.metadata if hasattr(doc, "metadata") else {}
            }
            docs_info.append(doc_info)

        processing_time = time.time() - start_time

        return ChatResponse(
            answer=answer,
            segments=segments,
            retrieved_docs=docs_info,
            processing_time=round(processing_time, 2),
            search_type=request.search_type,
            learner_style=request.learner_style,
            exercise_question=exercise_question,
            exercise_answer=exercise_answer,
            drawing_id=drawing_id,          # 回傳 ID
            drawing_total_steps=total_steps # 回傳總步數
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理請求時發生錯誤: {str(e)}")


@app.post("/chat_with_history", response_model=ChatResponse, tags=["聊天"])
async def chat_with_history(request: ChatRequest):
    """
    帶記憶的聊天端點

    支援對話歷史，實現多輪對話

    - **message**: 學生的問題
    - **history**: 對話歷史（最近 3-5 輪）
    - **search_type**: 檢索類型
    - **learner_style**: 學習風格
    """
    if not rag_initialized:
        raise HTTPException(status_code=503, detail="RAG 服務尚未初始化完成")

    start_time = time.time()

    try:
        # 驗證參數（同上）
        if request.search_type not in ["teaching", "exercise", "hybrid"]:
            raise HTTPException(
                status_code=400,
                detail=f"無效的 search_type: {request.search_type}"
            )

        if request.learner_style not in ["基礎級", "標準級", "進階級"]:
            raise HTTPException(
                status_code=400,
                detail=f"無效的 learner_style: {request.learner_style}"
            )

        # 設定檢索數量
        if request.search_type == "teaching":
            top_n = 3
        elif request.search_type == "exercise":
            top_n = 1
        else:  # hybrid
            top_n = 4

        # 1. 檢索相關文件（加入課程過濾）
        retrieved = rag_service.retrival_step(
            [request.message],
            request.search_type,
            (rag_service.teaching_vs, rag_service.teaching_ds),
            (rag_service.exercise_vs, rag_service.exercise_ds),
            top_n=top_n,
            course_filter=request.course_title  # 傳入課程標題進行過濾
        )

        retrieved_docs = retrieved.get(request.message, [])

        # [新增] 檢查是否有畫圖 ID
        drawing_id, total_steps = get_drawing_info(retrieved_docs)

        # === [新增] 側面推薦邏輯 ===
        if not drawing_id and request.search_type != "teaching":
            try:
                # 專門針對練習題庫 (Exercise) 搜 1 筆
                extra_retrieval = rag_service.retrival_step(
                    [request.message],
                    "exercise", 
                    (rag_service.teaching_vs, rag_service.teaching_ds),
                    (rag_service.exercise_vs, rag_service.exercise_ds),
                    top_n=1,
                    course_filter=None # 不過濾課程以提高命中率
                )
                extra_docs = extra_retrieval.get(request.message, [])
                
                extra_id, extra_steps = get_drawing_info(extra_docs)
                
                if extra_id:
                    print(f"💡 [側面推薦] 主要回答沒圖，但從練習題庫找到了相關圖表 ID: {extra_id}")
                    drawing_id = extra_id
                    total_steps = extra_steps
            except Exception as e:
                print(f"側面推薦檢索失敗: {e}")

        # 2. 建立上下文
        matched_context = "\n".join([
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in retrieved_docs
        ])

        # 3. 建立記憶（從對話歷史）
        memory_chunk = ""
        if request.history and len(request.history) > 0:
            # 只取最近 5 輪對話（控制 token 數量）
            recent_history = request.history[-10:]  # 最近 5 輪 = 10 條訊息

            memory_lines = []
            for msg in recent_history:
                if msg.role == "user":
                    memory_lines.append(f"學生問: {msg.content}")
                else:
                    memory_lines.append(f"助教答: {msg.content}")

            memory_chunk = "\n".join(memory_lines)

        # 4. 生成答案（帶記憶）
        is_exercise_mode = (request.search_type == "exercise")
        answer = rag_service.generate_answer(
            matched_context,
            request.message,
            request.learner_style,
            memory_chunk,
            is_exercise_mode=is_exercise_mode,
            course_title=request.course_title,  # 傳遞課程標題
            use_alternative=request.use_alternative,  # 是否換角度
            retry_count=request.retry_count,  # 重試次數
        )

        # 5. 解析練習題的題目和答案（僅在練習題模式）
        exercise_question = None
        exercise_answer = None
        segments = []

        if is_exercise_mode:
            # 解析【題目】和【答案】
            import re
            question_match = re.search(r'【題目】\s*(.*?)\s*【答案】', answer, re.DOTALL)
            answer_match = re.search(r'【答案】\s*(.*)', answer, re.DOTALL)

            if question_match and answer_match:
                exercise_question = question_match.group(1).strip()
                exercise_answer = answer_match.group(1).strip()
            else:
                # 如果沒有匹配到格式，整個當作題目
                exercise_question = answer
                exercise_answer = "（AI 未提供標準答案格式）"

            # 練習題模式不分段
            segments = []
        else:
            # 非練習題模式才分段
            from chatbot.rag_pipeline.post_process import Post_process
            post_processor = Post_process()
            segments = post_processor.split_answer(answer)

        # 6. 整理檢索文件資訊
        docs_info = []
        for doc in retrieved_docs[:3]:
            doc_info = {
                "content": doc.page_content if hasattr(doc, "page_content") else str(doc),
                "metadata": doc.metadata if hasattr(doc, "metadata") else {}
            }
            docs_info.append(doc_info)

        processing_time = time.time() - start_time

        return ChatResponse(
            answer=answer,
            segments=segments,
            retrieved_docs=docs_info,
            processing_time=round(processing_time, 2),
            search_type=request.search_type,
            learner_style=request.learner_style,
            exercise_question=exercise_question,
            exercise_answer=exercise_answer,
            drawing_id=drawing_id,          # 回傳 ID
            drawing_total_steps=total_steps # 回傳總步數
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理請求時發生錯誤: {str(e)}")


@app.post("/clarify", response_model=ClarifyResponse, tags=["深入追問"])
async def clarify_segment(request: ClarifyRequest):
    """
    深入追問端點

    當學生點選答案中的某一段文字時，提供更詳細的解釋

    - **selected_text**: 學生選中的文字片段
    - **original_query**: 原始問題
    - **learner_style**: 學習風格
    - **original_context**: 原始答案的上下文（可選）
    """
    if not rag_initialized:
        raise HTTPException(status_code=503, detail="RAG 服務尚未初始化完成")

    start_time = time.time()

    try:
        # 驗證學習風格
        if request.learner_style not in ["基礎級", "標準級", "進階級"]:
            raise HTTPException(
                status_code=400,
                detail=f"無效的 learner_style: {request.learner_style}"
            )

        # 使用 RAG 的 generate_clarification 功能
        # original_docs 可以用原始上下文，或設為空列表
        original_docs = request.original_context if request.original_context else ""

        clarification = rag_service.generate_clarification(
            request.selected_text,
            request.original_query,
            original_docs,
            request.learner_style
        )

        processing_time = time.time() - start_time

        return ClarifyResponse(
            clarification=clarification,
            processing_time=round(processing_time, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理深入追問時發生錯誤: {str(e)}")


# ==================== 啟動說明 ====================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("🚀 啟動 FastAPI RAG 服務")
    print("=" * 60)
    print("\n請使用以下指令啟動:")
    print("uvicorn chatbot.fastapi_app:app --host 0.0.0.0 --port 8001 --reload\n")

    # 直接啟動（開發用）
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
