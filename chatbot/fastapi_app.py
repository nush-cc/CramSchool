"""
FastAPI 服務 - RAG 聊天機器人 (多學科支援版 + 外部連結整合)
用於學生問答系統的後端 API
"""

import os
import random
import sys
import time
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

# 載入環境變數
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

# 導入 RAG 相關模組
from chatbot.rag_pipeline.RAG_function import rag_process


# ==================== Helper Functions ====================

def get_drawing_info(retrieved_docs):
    import json

    DRAWING_DIR = os.path.join(
        project_root, "chatbot", "dataset", "llama_drawing_steps"
    )

    for doc in retrieved_docs:
        if hasattr(doc, "metadata"):
            doc_id = doc.metadata.get("id")
            if doc_id:
                target_filename = f"{doc_id}_layout.json"
                full_path = os.path.join(DRAWING_DIR, target_filename)

                if os.path.exists(full_path):
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            steps = len(data.get("steps", []))
                            return str(doc_id), steps
                    except Exception as e:
                        print(f"讀取 Layout JSON 失敗: {e}")
                        continue
    return None, 0


def get_simulation_info(retrieved_docs):
    """
    從檢索到的文件中提取模擬網站 URL
    支援: Dictionary 物件 或 JSON String
    """
    import json  # 確保有引入 json

    for doc in retrieved_docs:
        sim_data = None

        # 1. 取得原始資料
        if hasattr(doc, "metadata"):
            sim_data = doc.metadata.get("simulation")
        elif isinstance(doc, dict):
            # 防呆: 如果 doc 本身就是 dict (非 Document 物件)
            sim_data = doc.get("metadata", {}).get("simulation")

        if not sim_data:
            continue

        # 2. 關鍵修正: 如果是字串，先轉成 Dict
        if isinstance(sim_data, str):
            try:
                sim_data = json.loads(sim_data)
            except Exception as e:
                print(f"⚠️ Simulation JSON 解析失敗: {e}")
                continue

        # 3. 現在 sim_data 肯定是 Dict 了，放心取值
        if isinstance(sim_data, dict):
            url = sim_data.get("url")
            if url:
                print(f"🎯 [get_simulation_info] 成功提取 URL: {url}")
                return url

    return None


def get_random_exercise(subject, course_filter=None):
    """
    從練習題庫中隨機抽取一題
    """
    # 1. 取得該科目的練習題 vectorstore
    vs = vector_stores.get(subject, {}).get("exercise")
    if not vs:
        return []

    # 2. 取得所有文件 (從 docstore)
    # FAISS 的 docstore._dict 存有所有 Document 物件
    all_docs = list(vs.docstore._dict.values())

    candidates = []

    # 3. 過濾單元 (如果有指定)
    if course_filter:
        print(f"🎲 [Random] 正在篩選單元: {course_filter}")
        for doc in all_docs:
            category = doc.metadata.get("category", "")
            # 這裡做簡單的字串包含比對
            if course_filter in category:
                candidates.append(doc)
    else:
        # 沒指定單元就全梭了
        candidates = all_docs

    print(f"🎲 [Random] 候選題數: {len(candidates)}")

    # 4. 隨機抽一題
    if candidates:
        selected = random.choice(candidates)
        return [selected]  # 回傳 list 格式以配合原本流程

    return []


# ==================== Pydantic Models ====================


class ChatMessage(BaseModel):
    role: str = Field(..., description="user 或 assistant")
    content: str = Field(..., description="訊息內容")


class ChatRequest(BaseModel):
    """聊天請求模型"""

    message: str = Field(..., description="學生的問題", min_length=1)
    subject: str = Field(
        default="math", description="科目: math (數學) 或 science (自然)"
    )
    search_type: str = Field(
        default="teaching", description="檢索類型: teaching, exercise, hybrid"
    )
    learner_style: str = Field(
        default="標準級", description="學習風格: 基礎級, 標準級, 進階級"
    )
    course_id: Optional[int] = Field(default=None, description="課程 ID（可選）")
    course_title: Optional[str] = Field(default=None, description="課程標題/主題")
    history: Optional[List[ChatMessage]] = Field(default=[], description="對話歷史")
    is_retry: bool = Field(default=False, description="是否為重試請求")
    retry_count: int = Field(default=0, description="重試次數")
    use_alternative: bool = Field(default=False, description="是否使用替代解釋")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="AI 生成的答案")
    segments: List[str] = Field(default=[], description="答案分段")
    retrieved_docs: List[Dict[str, Any]] = Field(
        default=[], description="檢索到的相關文件"
    )
    processing_time: float = Field(..., description="處理時間")
    search_type: str = Field(..., description="使用的檢索類型")
    learner_style: str = Field(..., description="使用的學習風格")
    exercise_question: Optional[str] = Field(default=None)
    exercise_answer: Optional[str] = Field(default=None)

    # 畫圖相關
    drawing_id: Optional[str] = Field(default=None)
    drawing_total_steps: int = Field(default=0)

    # 新增: 外部連結相關
    simulation_url: Optional[str] = Field(default=None, description="外部模擬網站 URL")


class ClarifyRequest(BaseModel):
    selected_text: str = Field(..., description="學生選中的文字片段")
    original_query: str = Field(..., description="原始問題")
    learner_style: str = Field(default="標準級")
    original_context: Optional[str] = Field(default=None)


class ClarifyResponse(BaseModel):
    clarification: str = Field(..., description="深入解釋")
    processing_time: float = Field(..., description="處理時間")


class HealthResponse(BaseModel):
    status: str
    rag_loaded: bool
    message: str


# ==================== FastAPI App ====================

app = FastAPI(
    title="RAG 聊天機器人 API (多學科)",
    description="支援數學與自然科的問答系統",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 全域變數 ====================

rag_service = None
rag_initialized = False

# 使用字典來管理不同科目的 VectorStore
# 結構: { "math": {"teaching": VS, "exercise": VS}, "science": ... }
vector_stores = {
    "math": {"teaching": None, "exercise": None},
    "science": {"teaching": None, "exercise": None},
}

# 資料路徑配置 (需與 build_faiss.py 對應)
PATH_CONFIG = {
    "math": {
        "teaching": "faiss_index_teaching",
        "exercise": "faiss_index_exercise",
        "ex_json": os.path.join(
            DATASET_DIR := os.path.join(current_dir, "dataset"),
            "raw_data",
            "add_id_data",
            "question_math_id.json",
        ),
    },
    "science": {
        "teaching": "faiss_index_science_teaching",
        "exercise": "faiss_index_science_exercise",
        "ex_json": os.path.join(
            DATASET_DIR, "raw_data", "add_id_data", "question_science_id.json"
        ),
    },
}


# ==================== 啟動事件 ====================


@app.on_event("startup")
async def startup_event():
    global rag_service, rag_initialized, vector_stores
    print("\n" + "=" * 60)
    print("🚀 FastAPI RAG 服務啟動中 (多學科模式)...")
    print("=" * 60)

    try:
        # 初始化 RAG 處理器
        print("\n[1/3] 初始化 RAG 處理器...")
        rag_service = rag_process()

        # 準備 Embedding 模型
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        from chatbot.config import embedding_model_name, model_device

        model_kwargs = {"device": model_device}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
        print("   ✅ Embedding 模型載入完成")

        # 載入所有科目的向量庫
        print("\n[2/3] 載入向量資料庫...")

        for subj, paths in PATH_CONFIG.items():
            t_path = os.path.join(current_dir, paths["teaching"])
            e_path = os.path.join(current_dir, paths["exercise"])

            # 載入教學庫
            if os.path.exists(t_path):
                print(f"   📚 [{subj}] 載入教學向量庫: {paths['teaching']}")
                vector_stores[subj]["teaching"] = FAISS.load_local(
                    t_path, embeddings, allow_dangerous_deserialization=True
                )
            else:
                print(f"   ⚠️ [{subj}] 找不到教學庫: {paths['teaching']}")

            # 載入練習庫
            if os.path.exists(e_path):
                print(f"   📚 [{subj}] 載入練習向量庫: {paths['exercise']}")
                vector_stores[subj]["exercise"] = FAISS.load_local(
                    e_path, embeddings, allow_dangerous_deserialization=True
                )
            else:
                print(f"   ⚠️ [{subj}] 找不到練習庫: {paths['exercise']}")

        # 檢查 OpenAI API Key
        print("\n[3/3] 檢查 OpenAI API...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("找不到 OPENAI_API_KEY")

        rag_initialized = True
        print("\n✅ 服務啟動成功！")

    except Exception as e:
        print(f"\n❌ 啟動失敗: {str(e)}")
        rag_initialized = False
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok" if rag_initialized else "error",
        rag_loaded=rag_initialized,
        message="RAG 服務運行正常" if rag_initialized else "RAG 服務未初始化",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not rag_initialized:
        raise HTTPException(status_code=503, detail="RAG 服務尚未初始化完成")

    start_time = time.time()

    try:
        # 1. 決定科目 (防呆)
        current_subject = (
            request.subject if request.subject in ["math", "science"] else "math"
        )

        # 2. 取出對應的 VectorStore
        selected_vs = vector_stores[current_subject]
        if not selected_vs["teaching"] or not selected_vs["exercise"]:
            # 如果該科目的資料庫沒載入，回退到 math 或報錯
            if current_subject == "science" and vector_stores["math"]["teaching"]:
                print("Warning: Science DB not found, fallback to Math")
                selected_vs = vector_stores["math"]
                current_subject = "math"
            else:
                raise HTTPException(
                    status_code=400, detail=f"科目 {current_subject} 的資料庫未載入"
                )

        # 構建 DB Tuple (VS, DS) - 這裡簡化 DS 為空列表
        teaching_db = (selected_vs["teaching"], [])
        exercise_db = (selected_vs["exercise"], [])

        if request.search_type == "teaching":
            top_n = 3
        elif request.search_type == "exercise":
            top_n = 1
        else:
            top_n = 4

        # 3. 檢索
        retrieved = rag_service.retrival_step(
            [request.message],
            request.search_type,
            teaching_db,
            exercise_db,
            top_n=top_n,
            course_filter=request.course_title,
        )

        retrieved_docs = retrieved.get(request.message, [])

        # --- 整合額外資源 (畫圖 & 外部網站) ---
        drawing_id, total_steps = get_drawing_info(retrieved_docs)
        simulation_url = get_simulation_info(retrieved_docs)  # <--- 新增此行

        # 4. 生成答案
        matched_context = "\n".join(
            [
                doc.page_content if hasattr(doc, "page_content") else str(doc)
                for doc in retrieved_docs
            ]
        )

        memory_chunk = ""  # 此端點無記憶
        is_exercise_mode = request.search_type == "exercise"

        answer = rag_service.generate_answer(
            matched_context,
            request.message,
            request.learner_style,
            memory_chunk,
            subject=current_subject,  # 傳入科目
            is_exercise_mode=is_exercise_mode,
            course_title=request.course_title,
            use_alternative=request.use_alternative,
            retry_count=request.retry_count,
        )

        # 5. 後處理 (練習題解析/分段)
        exercise_question = None
        exercise_answer = None
        segments = []

        if is_exercise_mode:
            import re

            question_match = re.search(
                r"【題目】\s*(.*?)\s*【答案】", answer, re.DOTALL
            )
            answer_match = re.search(r"【答案】\s*(.*)", answer, re.DOTALL)

            if question_match and answer_match:
                exercise_question = question_match.group(1).strip()
                exercise_answer = answer_match.group(1).strip()
            else:
                exercise_question = answer
                exercise_answer = "（AI 未提供標準答案格式）"
        else:
            from chatbot.rag_pipeline.post_process import Post_process

            post_processor = Post_process()
            segments = post_processor.split_answer(answer)

        # 整理文件資訊
        docs_info = []
        for doc in retrieved_docs[:3]:
            doc_info = {
                "content": doc.page_content
                if hasattr(doc, "page_content")
                else str(doc),
                "metadata": doc.metadata if hasattr(doc, "metadata") else {},
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
            drawing_id=drawing_id,
            drawing_total_steps=total_steps,
            simulation_url=simulation_url,  # <--- 新增此行
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理請求錯誤: {str(e)}")


@app.post("/chat_with_history", response_model=ChatResponse)
async def chat_with_history(request: ChatRequest):
    """帶記憶的聊天端點"""
    if not rag_initialized:
        raise HTTPException(status_code=503, detail="RAG 服務尚未初始化完成")

    start_time = time.time()
    try:
        # 1. 決定科目
        current_subject = (
            request.subject if request.subject in ["math", "science"] else "math"
        )
        selected_vs = vector_stores[current_subject]

        if not selected_vs["teaching"] or not selected_vs["exercise"]:
            if current_subject == "science" and vector_stores["math"]["teaching"]:
                selected_vs = vector_stores["math"]
                current_subject = "math"
            else:
                raise HTTPException(
                    status_code=400, detail=f"科目 {current_subject} 的資料庫未載入"
                )

        teaching_db = (selected_vs["teaching"], [])
        exercise_db = (selected_vs["exercise"], [])

        if request.search_type == "teaching":
            top_n = 3
        elif request.search_type == "exercise":
            top_n = 1
        else:
            top_n = 4

        # 2. 檢索
        retrieved = rag_service.retrival_step(
            [request.message],
            request.search_type,
            teaching_db,
            exercise_db,
            top_n=top_n,
            course_filter=request.course_title,
        )
        retrieved_docs = retrieved.get(request.message, [])

        # ========== DEBUG 區塊 ==========
        # print("\n" + "=" * 40)
        # print(f"🧐 [DEBUG] 收到問題: {request.message}")
        # print(f"⚙️ [DEBUG] 參數設定: Subject={current_subject}, SearchType={request.search_type}")
        # print(f"🏷️ [DEBUG] 課程過濾: {request.course_title} (若非 None 則會進行過濾)")
        # print(f"📊 [DEBUG] 檢索結果數量: {len(retrieved_docs)} 筆")
        #
        # if len(retrieved_docs) == 0:
        #     print("⚠️ 警告：檢索結果為空！請檢查 Subject 或 Course Title 是否設限過嚴。")
        # else:
        #     for i, doc in enumerate(retrieved_docs):
        #         meta = doc.metadata if hasattr(doc, "metadata") else {}
        #         sim = meta.get("simulation")
        #
        #         print(f"--- 文件 {i + 1} ---")
        #         print(f"   📂 Category: {meta.get('category', 'N/A')}")
        #         if sim:
        #             print(f"   ✅ 找到 Simulation (Type: {type(sim)})")
        #             print(f"   👀內容: {sim}")
        #         else:
        #             print(f"   ❌ 無 Simulation")
        #
        # sim_url = get_simulation_info(retrieved_docs)
        # print(f"🚀 [DEBUG] 最終提取的 URL: {sim_url}")
        # print("=" * 40 + "\n")
        # ===============================        # =========================================

        # --- 整合額外資源 (畫圖 & 外部網站) ---
        drawing_id, total_steps = get_drawing_info(retrieved_docs)
        simulation_url = get_simulation_info(retrieved_docs)  # <--- 新增此行

        # 3. 記憶處理
        matched_context = "\n".join([doc.page_content for doc in retrieved_docs])
        memory_chunk = ""
        if request.history:
            recent_history = request.history[-10:]
            memory_lines = []
            for msg in recent_history:
                role = "學生問" if msg.role == "user" else "助教答"
                memory_lines.append(f"{role}: {msg.content}")
            memory_chunk = "\n".join(memory_lines)

        # 4. 生成
        is_exercise_mode = request.search_type == "exercise"
        answer = rag_service.generate_answer(
            matched_context,
            request.message,
            request.learner_style,
            memory_chunk,
            subject=current_subject,  # 傳入科目
            is_exercise_mode=is_exercise_mode,
            course_title=request.course_title,
            use_alternative=request.use_alternative,
            retry_count=request.retry_count,
        )

        # 5. 後處理
        exercise_question = None
        exercise_answer = None
        segments = []
        if is_exercise_mode:
            import re

            question_match = re.search(
                r"【題目】\s*(.*?)\s*【答案】", answer, re.DOTALL
            )
            answer_match = re.search(r"【答案】\s*(.*)", answer, re.DOTALL)
            if question_match and answer_match:
                exercise_question = question_match.group(1).strip()
                exercise_answer = answer_match.group(1).strip()
            else:
                exercise_question = answer
                exercise_answer = "（AI 未提供標準答案格式）"
        else:
            from chatbot.rag_pipeline.post_process import Post_process

            post_processor = Post_process()
            segments = post_processor.split_answer(answer)

        docs_info = [
            {"content": d.page_content, "metadata": d.metadata}
            for d in retrieved_docs[:3]
        ]
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
            drawing_id=drawing_id,
            drawing_total_steps=total_steps,
            simulation_url=simulation_url,  # <--- 新增此行
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理請求錯誤: {str(e)}")


@app.post("/clarify", response_model=ClarifyResponse)
async def clarify_segment(request: ClarifyRequest):
    if not rag_initialized:
        raise HTTPException(status_code=503, detail="RAG 服務尚未初始化完成")
    start_time = time.time()
    try:
        original_docs = request.original_context if request.original_context else ""
        clarification = rag_service.generate_clarification(
            request.selected_text,
            request.original_query,
            original_docs,
            request.learner_style,
        )
        processing_time = time.time() - start_time
        return ClarifyResponse(
            clarification=clarification, processing_time=round(processing_time, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8001, reload=True)
