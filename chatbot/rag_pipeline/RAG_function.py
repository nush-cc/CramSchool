from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
import torch.nn.functional as F

# ==================Load Function=============
from .data_loader import Data_loader
from .keyword_match import Keyword_matching
from .post_process import Post_process
from .semantic_chunk import Chunking

# import model
from ..config import (
    embedding_model_name,
    model_device,
    reranker_model_name,
    gpt_model,
    StyleClassifer_model_name,
)

# ========================Setting======================
ROLEPLAY_STYLES = {
    "基礎級": """
    【角色設定】
    你是一位極具耐心、親和力十足的「同儕小老師」。你的對象是完全聽不懂、對這個單元感到挫折的學生。
    
    【核心規則】
    1. 風格：必須像「聊天」一樣口語化，多用「我們」、「你看」這種拉近距離的詞。
    2. 絕對禁止：『嚴格禁止』使用任何教科書的專有名詞（例如：不能說「一元一次方程式」，要說「想要算出的未知數字」）。
    3. 輸出重點：請提供一個「一步一步照著做就能算出答案」的傻瓜式 SOP。
    4. 目標：讓學生覺得「原來這麼簡單」，建立信心。
    """,
    "標準級": """
    【角色設定】
    你是一位嚴謹、講求效率的「補習班王牌名師」。你的對象是正在準備考試、需要標準答案的普通學生。
    
    【核心規則】
    1. 風格：像教科書或參考書詳解一樣嚴謹、條理分明。
    2. 必要條件：『必須』精確使用所有相關的數學/學術專有名詞、定義與公式。
    3. 輸出重點：提供最標準、考試最常規的解題步驟。
    4. 目標：讓學生能夠在考試中規範作答，拿到滿分。
    """,
    "進階級": """
    【角色設定】
    你是一位啟發思考的「奧林匹亞競賽教練」或「大學教授」。你的對象是想要考高分、追求真理的頂尖學生。
    
    【核心規則】
    1. 風格：引導式、啟發式，著重於「數學思維」與「邏輯本質」。
    2. 差異化要求：你的回答必須與標準教科書解法有「巨大鑑別度」。
    
    【必須執行的五大任務】
    1. 原理解析：深入分析「為什麼」這個解法可行？背後的數學原理是什麼？
    2. 替代解法：提供至少一種「不同於標準解法」的路徑（如：幾何圖解、速解法、逆推法、特殊值法）。
    3. 觀念延伸：將此問題連結到「其他相關單元」或「高中/大學」的更高階知識點。
    4. 易錯點提示：指出這個題型最容易掉進的陷阱。
    5. 總結：嚴格禁止只是換句話說，必須提供高維度的視角。
    """,
}

load_dotenv()
key = os.getenv("OPENAI_API_KEY")
if not key:
    print("錯誤：找不到 OPENAI_API_KEY。請檢查你的 .env 檔案。")
    exit()

client = OpenAI(api_key=key)


# --------------
class StyleClassifier:
    def __init__(self, model_name=StyleClassifer_model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            "cuda"
        )
        self.label_map = {0: "基礎級", 1: "標準級", 2: "進階級"}

    def predict(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to("cuda")
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
        return self.label_map.get(pred_idx, "基礎級")


# -------------


class Vectorize:
    def __init__(self, model_name):
        self.model_name = model_name

    # 建立向量資料庫
    def vector_store(self, doc, save_path="faiss_index"):
        model_kwargs = {"device": model_device}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )  # 引入向量EMBEDDINGS
        vectorstore = FAISS.from_documents(doc, embeddings)
        vectorstore.save_local(save_path)

        return vectorstore


class rag_process:
    def __init__(self):
        print("[rag_process] 正在初始化...")
        self.keyword_match = Keyword_matching()
        self.post_processor = Post_process()
        self.vectorize_processor = Vectorize(embedding_model_name)
        print("[rag_process] 正在載入 StyleClassifier 模型 (第一次執行會下載)...")
        self.style_classifier = StyleClassifier()

    def vectorize_workflow(self, teaching_path, exercise_path):
        loader = Data_loader()

        # 初始化 Embedding 模型 ---
        model_kwargs = {"device": model_device}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[vectorize_workflow] Embedding 模型載入完成。")

        # ==========================================
        # 處理教學資料 (Teaching Data)
        # ==========================================
        teacher_save_path = "faiss_index_teaching"

        final_teaching_docs = []

        if isinstance(teaching_path, list):
            print(
                f"[Info] 偵測到 teaching_path 為列表，將啟用 Semantic Chunking 處理..."
            )
            loader.input_path_list = teaching_path
            teaching_data = loader.load_file_list()
            raw_dicts = loader.get_page_content(teaching_data)

            chunker = Chunking()
            chunk_result = chunker.semantic_chunk(raw_dicts)

            for chunk in chunk_result:
                new_doc = Document(
                    page_content=chunk["text"],
                    metadata={
                        "category": str(chunk["category"]),
                        "id": str(chunk["id"]),
                    },
                )
                final_teaching_docs.append(new_doc)
        else:
            print(f"[Info] teaching_path 為單一路徑，執行一般載入...")
            loader.input_path = teaching_path
            teaching_data = loader.load_file()  # 單一檔案讀取
            # 取得標準化字典列表
            raw_dicts = loader.get_page_content(teaching_data)

            for item in raw_dicts:
                new_doc = Document(
                    page_content=item["content"],
                    metadata={
                        "category": item.get("category", "Teaching_Material"),
                        "id": item.get("id", ""),
                    },
                )
                final_teaching_docs.append(new_doc)

        # 建立或載入 FAISS (教學)
        if os.path.exists(teacher_save_path):
            print(f"載入現有教學向量庫: {teacher_save_path}")
            teaching_vectorstore = FAISS.load_local(
                teacher_save_path, embeddings, allow_dangerous_deserialization=True
            )
        else:
            print(f"建立新教學向量庫...")
            teaching_vectorstore = self.vectorize_processor.vector_store(
                final_teaching_docs, save_path=teacher_save_path
            )
        print(f"教學資料庫準備完成。")

        # ==========================================
        # 2. 處理練習題資料 (Exercise Data)
        # ==========================================
        exercise_save_path = "faiss_index_exercise"
        print(f"\n--- 開始處理練習題資料 ---")

        loader.input_path = exercise_path
        exercise_data = loader.load_file()
        raw_ex_dicts = loader.get_page_content(exercise_data)

        final_exercise_docs = []  # 最終要給 FAISS 的 Document 列表

        # 【修正點 1】: 將字典轉為 Document 物件
        for item in raw_ex_dicts:
            new_doc = Document(
                page_content=item["content"],
                metadata={
                    "category": item.get("category", "Exercise"),
                    "id": item.get("id", ""),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                },
            )
            final_exercise_docs.append(new_doc)

        # 【修正點 2】: 修正檢查路徑變數 (teacher_save_path -> exercise_save_path)
        if os.path.exists(exercise_save_path):
            print(f"載入現有練習題向量庫: {exercise_save_path}")
            exercise_vectorstore = FAISS.load_local(
                exercise_save_path, embeddings, allow_dangerous_deserialization=True
            )
        else:
            print(f"建立新練習題向量庫...")
            exercise_vectorstore = self.vectorize_processor.vector_store(
                final_exercise_docs, save_path=exercise_save_path
            )

        print(f"練習題資料庫準備完成。")

        # 回傳 (VectorStore, Documents列表)
        return (teaching_vectorstore, final_teaching_docs), (
            exercise_vectorstore,
            final_exercise_docs,
        )

    def retrival_step(
        self,
        queries: list[str],
        search_type: str,
        teaching_db,
        exercise_db,
        top_n,
        course_filter=None,
    ):
        """
        檢索步驟，支援根據課程過濾結果

        Args:
            queries: 查詢問題列表
            search_type: 檢索類型 (teaching/exercise/hybrid)
            teaching_db: 教學資料庫
            exercise_db: 練習題資料庫
            top_n: 檢索數量
            course_filter: 課程過濾字串（可選），用於過濾 metadata['category']
        """
        teaching_vs, teaching_ds = teaching_db
        exercise_vs, exercise_ds = exercise_db

        vectorstores_to_search = []
        docs_to_search = []

        # 根據 search_type 決定要搜尋的資料庫
        if search_type == "teaching":
            vectorstores_to_search.append(teaching_vs)
            docs_to_search.extend(teaching_ds)
        elif search_type == "exercise":
            vectorstores_to_search.append(exercise_vs)
            docs_to_search.extend(exercise_ds)
        elif search_type == "hybrid":
            vectorstores_to_search.append(teaching_vs)
            vectorstores_to_search.append(exercise_vs)
            docs_to_search.extend(teaching_ds)
            docs_to_search.extend(exercise_ds)
        else:  # 預設為 hybrid
            print(f"未知的 search_type '{search_type}'，將使用 hybrid 模式。")
            vectorstores_to_search.append(teaching_vs)
            vectorstores_to_search.append(exercise_vs)
            docs_to_search.extend(teaching_ds)
            docs_to_search.extend(exercise_ds)

        results = {}
        # ---------- 關鍵字檢索 ----------
        kw_map = {}
        for q in queries:
            keywords = self.keyword_match.extract_keywords(q)
            kw_map[q] = self.keyword_match.keyword_match(docs_to_search, keywords)

        # ---------- 向量檢索 (批次) ----------
        emb = vectorstores_to_search[0].embedding_function
        if hasattr(emb, "embed_documents"):
            q_vecs = emb.embed_documents(queries)
        else:
            q_vecs = [emb.embed_query(q) for q in queries]
        q_mat = np.asarray(q_vecs, dtype="float32")

        # L2 normalize
        q_mat /= np.linalg.norm(q_mat, axis=1, keepdims=True)

        # 直接調 FAISS index（檢索更多結果以便後續過濾）
        # 如果有課程過濾，檢索 top_n * 5 個結果，過濾後再取 top_n
        search_k = top_n * 5 if course_filter else top_n
        faiss_hits_all_queries = {q: [] for q in queries}

        for vs in vectorstores_to_search:
            D, I = vs.index.search(q_mat, search_k)
            id_map = vs.index_to_docstore_id
            store = vs.docstore._dict

            for qi, idx_row in enumerate(I):
                hits = []
                for idx in idx_row:
                    if idx == -1:
                        continue
                    doc = store[id_map[idx]]

                    # 課程過濾：檢查 metadata['category'] 是否包含 course_filter
                    if course_filter:
                        if hasattr(doc, "metadata") and "category" in doc.metadata:
                            category = doc.metadata["category"]
                            # 模糊匹配：course_filter 包含在 category 中
                            if course_filter.lower() not in category.lower():
                                continue  # 跳過不符合的文檔
                        else:
                            # 沒有 metadata 或 category，跳過
                            continue

                    hits.append(doc)

                    # 如果有課程過濾，收集到 top_n 個就停止
                    if course_filter and len(hits) >= top_n:
                        break

                faiss_hits_all_queries[queries[qi]].extend(hits)

        # ---------- 合併 & rerank ----------
        for q in queries:
            # 結合關鍵字和所有向量檢索的結果
            matched_chunks = list(faiss_hits_all_queries[q]) + list(kw_map[q])

            # 如果有課程過濾，也過濾關鍵字檢索的結果
            if course_filter:
                filtered_chunks = []
                for chunk in matched_chunks:
                    # 處理 Document 物件
                    if hasattr(chunk, "metadata") and "category" in chunk.metadata:
                        if course_filter.lower() in chunk.metadata["category"].lower():
                            filtered_chunks.append(chunk)
                    # 處理字典物件
                    elif isinstance(chunk, dict) and "category" in chunk:
                        if course_filter.lower() in chunk["category"].lower():
                            filtered_chunks.append(chunk)
                matched_chunks = filtered_chunks

            # rerank
            reranked_chunks = self.rerank(q, matched_chunks, top_k=top_n)
            results[q] = reranked_chunks

        return results

    def rerank(self, translated_query, matched_chunks, top_k=20):
        reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(
            reranker_model_name
        ).to("cuda")
        reranker_model.eval()

        def get_text_content(chunk):
            # 如果是 LangChain 的 Document 物件 (FAISS 回傳)
            if hasattr(chunk, "page_content"):
                return chunk.page_content
            # 如果是 Dictionary (關鍵字搜尋或 Semantic Chunking 回傳)
            elif isinstance(chunk, dict):
                return chunk.get("content", "") or chunk.get("text", "")
            # 其他情況轉字串
            return str(chunk)

        unique_chunks = []
        seen_content = set()

        for chunk in matched_chunks:
            # 修改：直接使用上面的 helper function
            content = get_text_content(chunk)
            if content and content not in seen_content:
                unique_chunks.append(chunk)
                seen_content.add(content)

        if not unique_chunks:
            return []

        # 修改：建立 pairs 時也使用 get_text_content
        pairs = [[translated_query, get_text_content(chunk)] for chunk in unique_chunks]

        inputs = reranker_tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to("cuda")

        with torch.no_grad():
            scores = (
                reranker_model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .cpu()
                .numpy()
            )

        reranked = sorted(zip(unique_chunks, scores), key=lambda x: x[1], reverse=True)
        top_reranked_chunks = [chunk for chunk, _ in reranked[:top_k]]

        return top_reranked_chunks

    def generate_answer(
        self,
        context,
        query,
        learner_style,
        memory_chunk,
        is_exercise_mode=False,
        course_title=None,
        use_alternative=False,
        retry_count=0,
    ):
        # 根據是否為重試調整 temperature
        # 重試時使用較高的 temperature 以產生不同的回答
        # 正常對話使用 0.6-0.7 來增加多樣性，重試時使用 0.8 來換角度
        temperature = 0.8 if (use_alternative or retry_count > 0) else 0.7

        llm = ChatOpenAI(
            model=gpt_model,
            temperature=temperature,
            top_p=1.0,
            api_key=key,
            max_tokens=750,
            timeout=60,
            max_retries=0,
        )

        # 課程主題限制
        course_context = ""
        if course_title:
            course_context = f"""

【當前課程主題】：{course_title}

【回答指引】：
1. 優先使用「{course_title}」相關的例子和概念來回答
2. 如果學生問的問題是相關的數學概念（例如：在學一元一次方程式時問函數、變數、常數等概念），你可以回答，但要盡量連結回「{course_title}」
3. 例如：在學一元一次方程式時，如果問「函數是什麼」，你可以簡單說明函數的概念，並說明一元一次方程式與函數的關係
4. 如果問題完全無關（例如：在數學課問歷史問題），才需要溫和提醒學生當前課程主題
5. 整體原則：採取較寬鬆的解釋，只要是數學相關且有助於理解當前主題的問題，都可以回答
"""

        # 針對練習題模式的特殊指示
        exercise_instruction = ""
        if is_exercise_mode:
            topic_hint = f"關於「{course_title}」的" if course_title else ""
            exercise_instruction = f"""

【重要】你正在「練習題模式」，請嚴格遵守以下格式輸出：

【題目】
（在這裡寫出完整的{topic_hint}題目內容）

【答案】
（在這裡提供詳細的解答步驟和最終答案）

規則：
1. 必須使用「【題目】」和「【答案】」這兩個標記，不可省略或更改
2. 必須在【答案】部分提供完整解答，不可說「我計算一下再告訴你」
3. 答案部分要包含解題步驟和最終答案
4. 兩個部分之間空一行
5. 題目必須嚴格圍繞「{course_title}」主題
"""

        # 重試時換角度指示（所有級別都適用）
        alternative_instruction = ""
        if use_alternative or retry_count > 0:
            # 基礎級的換角度方法
            if learner_style == "基礎級":
                alternative_methods = [
                    "用「生活情境類比」：用日常生活中具體的例子來比喻（例如：買東西、分東西、排隊等）",
                    "用「視覺化描述」：描述圖像、動作、具體步驟，讓學生能在腦海中「看見」過程",
                    "用「反向思考」：從答案往回推導，或從常見錯誤中學習正確觀念",
                    "用「極簡化拆解」：把問題拆成最小的單位，一次只處理一個概念",
                ]
                selected_method = alternative_methods[
                    min(retry_count, len(alternative_methods) - 1)
                ]

                alternative_instruction = f"""

【特別指示：換個角度解釋（第 {retry_count + 1} 次重試）】
學生表示還是不理解，請用完全不同的方式重新解釋：

本次使用方法：{selected_method}

重要原則：
1. 不要重複之前的說法或例子
2. 保持基礎級的語言風格（口語化、無專有名詞）
3. 多用「就像...」、「想像一下...」、「你可以這樣想...」等引導詞
4. 確保學生能「感受到」或「看到」你在說什麼
"""
            else:
                # 標準級和進階級的換角度指示
                alternative_instruction = f"""

【特別指示：換個角度解釋（重試請求）】
學生表示需要更多解釋，請提供不同的視角或例子：

1. 使用不同的例題或應用場景
2. 從不同的角度切入（例如：先前從定義出發，現在從應用出發）
3. 補充更多細節或背景知識
4. 提供對比說明（與相關概念的異同）
5. 確保這次的解釋與上次有明顯差異
"""

        # 轉義 memory_chunk 和 course_context 中的大括號，避免被 LangChain 誤認為變數
        def escape_braces(text):
            """將單個大括號轉義為雙大括號"""
            if not text:
                return ""
            return text.replace("{", "{{").replace("}", "}}")

        escaped_memory = escape_braces(memory_chunk)
        escaped_course = escape_braces(course_context)
        escaped_alternative = escape_braces(alternative_instruction)
        escaped_exercise = escape_braces(exercise_instruction)

        # 對於 ROLEPLAY_STYLES 也需要轉義
        escaped_roleplay = escape_braces(ROLEPLAY_STYLES[learner_style])

        # 多樣性指示（讓相同問題有不同回答）
        diversity_instruction = """
【回答多樣性要求】：
即使是相同或類似的問題，也請嘗試從不同角度切入，使用不同的例子或解釋方式。
你可以：
- 使用不同的生活例子或情境
- 從不同的數學角度解釋（幾何、代數、應用等）
- 採用不同的教學順序（由淺入深、由具體到抽象、或反過來）
- 每次回答時展現創意，讓學生感受到新鮮感
"""
        escaped_diversity = escape_braces(diversity_instruction)

        # 構建系統提示詞 - 先組合動態內容部分
        system_prompt_parts = [
            "你是一個專業的學術助教：",
            "這是使用者最近的對話相關內容，請重點結合參考。",
            escaped_memory,
            escaped_course,
            "",
            f"你的個性設定是：{escaped_roleplay}，請嚴格遵守該等級的【角色設定】與【核心規則】。",
            escaped_diversity,
            escaped_alternative,
            "盡量依據提供的上下文回答問題，若搜不到合適的文句可以根據語言模型的內建知識庫回答問題。",
            "你只能使用繁體中文回答，不要有簡體中文字和其他語言。",
            "",
            "【數學公式格式規範】",
            "- 如果上下文中出現 MATH_INLINE_0、MATH_INLINE_1 等佔位符，這代表原始文件中有數學公式但未正確提取",
            "- 請根據上下文推斷這些佔位符代表的數學內容，並用 LaTeX 格式重新表達",
            "- 行內公式使用單個 $ 符號包圍，例如：$x^2 + y^2 = r^2$",
            "- 區塊公式使用雙 $ 符號包圍，例如：$$\\\\frac{{a}}{{b}} = c$$",
            "- 絕對不要在回答中直接輸出 MATH_INLINE_X 或 MATH_BLOCK_X 這樣的佔位符文字",
            "",
            escaped_exercise,
            "以下是與你需要答的問題相關的內容:",
            "上下文如下：\n\n{context}",  # LangChain 變數不需要轉義，因為不在 f-string 中
        ]

        system_prompt = "\n".join(system_prompt_parts)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "question: {input}"),  # LangChain 變數
            ]
        )  # 提示模板

        chain = prompt_template | llm
        try:
            result = chain.invoke(
                {"input": query, "context": context}
            )  # 用 document_chain 產生回答
            # 正確提取純文字內容（避免 metadata）
            if hasattr(result, "content"):
                answer = result.content  # AIMessage 的純文字內容
            else:
                answer = str(result)  # 備用：直接轉字串
        except Exception as e:
            print(f"錯誤訊息:{e}")
            answer = "抱歉，生成答案時發生錯誤。"  # 提供一個錯誤回覆

        return str(answer).strip()

    # 新的深入回覆
    def generate_clarification(
        self,
        selected_text: str,
        original_query: str,
        original_docs: list,
        learner_style: str,
    ):
        llm = ChatOpenAI(
            model=gpt_model,
            temperature=0.1,
            top_p=1.0,
            api_key=key,
            max_tokens=750,
            timeout=60,
            max_retries=0,
        )

        if isinstance(original_docs, str):
            # 如果傳進來原本就是純文字字串，直接使用
            formatted_context = original_docs
        else:
            # 如果是列表，則進行迭代處理
            context_parts = []
            for doc in original_docs:
                if isinstance(doc, str):
                    # 列表裡面的項目是字串
                    context_parts.append(doc)
                elif isinstance(doc, tuple):
                    # 檢查 tuple 的第一個元素是否有 page_content 屬性 (是 Document 物件)
                    if hasattr(doc[0], "page_content"):
                        context_parts.append(doc[0].page_content)
                    else:
                        # 如果是字串或其他型態，直接轉字串使用
                        context_parts.append(str(doc[0]))
                elif hasattr(doc, "page_content"):
                    # 列表裡面的項目是 Document 物件
                    context_parts.append(doc.page_content)
                else:
                    # 其他情況，強轉字串
                    context_parts.append(str(doc))

            formatted_context = "\n\n".join(context_parts)

        # 轉義函數
        def escape_braces(text):
            """將單個大括號轉義為雙大括號"""
            if not text:
                return ""
            return text.replace("{", "{{").replace("}", "}}")

        # 轉義所有可能包含大括號的內容
        escaped_roleplay = escape_braces(ROLEPLAY_STYLES[learner_style])
        escaped_query = escape_braces(original_query)
        escaped_selected = escape_braces(selected_text)
        escaped_context = escape_braces(formatted_context)

        # --- 關鍵：設計新的提示詞 (Prompt) ---
        system_prompt = f"""
        你是一個專業的學術助教，你的個性設定是：{escaped_roleplay}。
        你只能使用繁體中文回答。

        任務：
        使用者先前詢問了一個問題，並得到了答案。現在，使用者從該答案中選取了一段文字，希望你針對這段「選取的文字」提供更深入、更清晰的解釋。

        1.  使用者「原始的問題」是：{escaped_query}
        2.  使用者「選取的文字」是：{escaped_selected}
        3.  當初回答時所參考的「原始資料」如下：
            ---
            {escaped_context}
            ---

        請你基於你的知識庫，並參考「原始資料」，專注地對「選取的文字」進行詳細說明。
        請解釋這段文字的**核心概念**、它在「原始的問題」中的**重要性**，並**舉一個例子**來幫助理解。

        你的回答應該是針對「{escaped_selected}」這句話的深入解析，而不是重新回答原始問題。
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", f"請幫我詳細解釋「{escaped_selected}」這句話的意思。"),
            ]
        )

        llm_chain = prompt_template | llm

        try:
            result = llm_chain.invoke({})  # 傳入空 dict，因為 input 已經在 prompt 裡
            clarification = result.content
        except Exception as e:
            print(f"錯誤訊息 (generate_clarification): {e}")
            clarification = "抱歉，生成闡釋時發生錯誤。"

        return str(clarification).strip()

    """新增完整使用process"""

    def clarification_main_process(
        self, segments, user_query, retrieved_docs, learner_style
    ):
        if segments and len(segments) > 1:
            print(f"\n{'-' * 30}")
            print("想針對以上回答的某個部分深入了解嗎？請選擇：")
            for i, seg in enumerate(segments):
                print(f"({i + 1}) {seg}")
            print("(N) 跳過")

            selection = input(">").strip().lower()

            if selection in ["1", "2", "3"] and int(selection) <= len(segments):
                idx = int(selection) - 1
                selected_text = segments[idx]
                print("\n正在生成深入解析，請稍候...\n")
                clarification_ans = self.generate_clarification(
                    selected_text, user_query, retrieved_docs, learner_style
                )
                print(f"★ 深入解析：\n{clarification_ans}\n")
            else:
                print("已跳過深入解析。")
