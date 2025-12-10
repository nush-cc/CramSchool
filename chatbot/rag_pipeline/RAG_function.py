from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os
import gc
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

SUBJECT_PROMPTS = {
    "math": """
    【數學科回答原則】
    1. 邏輯推導：強調「因為...所以...」的邏輯鏈條。
    2. 計算步驟：若涉及計算，必須列出詳細步驟，並使用 LaTeX 格式 (如 $x^2$)。
    3. 定義精確：使用精確的數學術語（如：係數、變數、函數）。
    4. 幾何視覺：如果問題涉及幾何，請嘗試描述圖形特徵。
    """,
    "science": """
    【自然科 (理化/生物/地科) 回答原則】
    1. 現象優先：先描述「自然現象」或「實驗觀察」，再解釋背後的原理。
    2. 因果關係：強調「變因」與「結果」的關係（例如：溫度升高導致分子動能增加）。
    3. 實例連結：必須舉出日常生活中的例子（例如：熱脹冷縮就像...）。
    4. 實驗思維：如果適合，請用「假設 -> 實驗 -> 結論」的科學方法來解釋。
    5. 化學式規範：若涉及化學反應，請正確顯示化學式（如 $H_2O$, $CO_2$）。
    """,
}

load_dotenv()
key = os.getenv("OPENAI_API_KEY")
if not key:
    print("錯誤：找不到 OPENAI_API_KEY。請檢查你的 .env 檔案。")

client = OpenAI(api_key=key)


class StyleClassifier:
    def __init__(self, model_name=StyleClassifer_model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            model_device
        )
        self.label_map = {0: "基礎級", 1: "標準級", 2: "進階級"}

    def predict(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(model_device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
        return self.label_map.get(pred_idx, "基礎級")


class Vectorize:
    def __init__(self):
        # 不再需要在這裡初始化模型名稱
        pass

    # 修改：直接接收已經載入好的 embeddings 物件
    def vector_store(self, doc, embeddings, save_path="faiss_index"):
        if not doc:
            print(f"  ⚠️ 警告: 沒有文件可建立索引 ({save_path})")
            return None

        # 使用傳入的 embeddings，不再重新載入
        vectorstore = FAISS.from_documents(doc, embeddings)
        vectorstore.save_local(save_path)
        return vectorstore


class rag_process:
    def __init__(self):
        print("[rag_process] 正在初始化...")
        self.keyword_match = Keyword_matching()
        self.post_processor = Post_process()
        self.vectorize_processor = Vectorize()

        # 1. 在這裡只載入一次 Embedding 模型
        print(f"  ⚡ [Init] 正在載入 Embedding 模型 ({embedding_model_name})...")
        model_kwargs = {"device": model_device}
        encode_kwargs = {"normalize_embeddings": True}
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        print("  ✅ Embedding 模型載入完成 (單例模式)。")

        print("[rag_process] 正在載入 StyleClassifier 模型 (第一次執行會下載)...")
        try:
            self.style_classifier = StyleClassifier()
        except Exception as e:
            print(f"StyleClassifier 載入失敗 (可能顯存不足，將略過): {e}")
            self.style_classifier = None

    def vectorize_workflow(
        self,
        teaching_path,
        exercise_path,
        save_path_teaching="faiss_index_teaching",
        save_path_exercise="faiss_index_exercise",
    ):
        # 強制垃圾回收，避免記憶體殘留
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        loader = Data_loader()

        # ==========================================
        # 1. 處理教學資料 (Teaching Data)
        # ==========================================
        final_teaching_docs = []

        if isinstance(teaching_path, list):
            print(f"  [Info] 偵測到 teaching_path 為列表，啟用 Semantic Chunking...")
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
            print(f"  [Info] 一般載入教學檔案...")
            loader.input_path = teaching_path
            teaching_data = loader.load_file()
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
        # 注意：這裡統一使用「建立」邏輯，因為 build_faiss.py 已經負責清理舊檔了
        # 且為了確保模型一致性，建議重建
        print(f"  建立新教學向量庫 ({save_path_teaching})...")
        # 傳入 self.embeddings (重複使用模型)
        teaching_vectorstore = self.vectorize_processor.vector_store(
            final_teaching_docs, self.embeddings, save_path=save_path_teaching
        )
        print(f"  教學資料庫準備完成。")

        # ==========================================
        # 2. 處理練習題資料 (Exercise Data)
        # ==========================================
        print(f"  --- 開始處理練習題資料 ---")

        loader.input_path = exercise_path
        exercise_data = loader.load_file()
        raw_ex_dicts = loader.get_page_content(exercise_data)

        final_exercise_docs = []

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

        print(f"  建立新練習題向量庫 ({save_path_exercise})...")
        # 傳入 self.embeddings (重複使用模型)
        exercise_vectorstore = self.vectorize_processor.vector_store(
            final_exercise_docs, self.embeddings, save_path=save_path_exercise
        )

        print(f"  練習題資料庫準備完成。")

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
        """
        teaching_vs, teaching_ds = teaching_db
        exercise_vs, exercise_ds = exercise_db

        vectorstores_to_search = []
        docs_to_search = []

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
        else:
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
        if not vectorstores_to_search or all(v is None for v in vectorstores_to_search):
            return {q: [] for q in queries}

        # 使用 self.embeddings (因為我們在 load_local 時已經綁定了，或者我們可以直接用)
        # 通常 vectorstore.embedding_function 就是我們傳進去的那個
        emb = vectorstores_to_search[0].embedding_function

        if hasattr(emb, "embed_documents"):
            q_vecs = emb.embed_documents(queries)
        else:
            q_vecs = [emb.embed_query(q) for q in queries]
        q_mat = np.asarray(q_vecs, dtype="float32")

        q_mat /= np.linalg.norm(q_mat, axis=1, keepdims=True)

        search_k = top_n * 5 if course_filter else top_n
        faiss_hits_all_queries = {q: [] for q in queries}

        for vs in vectorstores_to_search:
            if vs is None or vs.index is None:
                continue

            D, I = vs.index.search(q_mat, search_k)
            id_map = vs.index_to_docstore_id
            store = vs.docstore._dict

            for qi, idx_row in enumerate(I):
                hits = []
                for idx in idx_row:
                    if idx == -1:
                        continue
                    if idx not in id_map:
                        continue
                    doc = store[id_map[idx]]

                    if course_filter:
                        if hasattr(doc, "metadata") and "category" in doc.metadata:
                            category = doc.metadata["category"]
                            if course_filter.lower() not in category.lower():
                                continue
                        else:
                            continue

                    hits.append(doc)

                    if course_filter and len(hits) >= top_n:
                        break

                faiss_hits_all_queries[queries[qi]].extend(hits)

        # ---------- 合併 & rerank ----------
        for q in queries:
            matched_chunks = list(faiss_hits_all_queries[q]) + list(kw_map[q])

            if course_filter:
                filtered_chunks = []
                for chunk in matched_chunks:
                    if hasattr(chunk, "metadata") and "category" in chunk.metadata:
                        if course_filter.lower() in chunk.metadata["category"].lower():
                            filtered_chunks.append(chunk)
                    elif isinstance(chunk, dict) and "category" in chunk:
                        if course_filter.lower() in chunk["category"].lower():
                            filtered_chunks.append(chunk)
                matched_chunks = filtered_chunks

            reranked_chunks = self.rerank(q, matched_chunks, top_k=top_n)
            results[q] = reranked_chunks

        return results

    def rerank(self, translated_query, matched_chunks, top_k=20):
        reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(
            reranker_model_name
        ).to(model_device)
        reranker_model.eval()

        def get_text_content(chunk):
            if hasattr(chunk, "page_content"):
                return chunk.page_content
            elif isinstance(chunk, dict):
                return chunk.get("content", "") or chunk.get("text", "")
            return str(chunk)

        unique_chunks = []
        seen_content = set()

        for chunk in matched_chunks:
            content = get_text_content(chunk)
            if content and content not in seen_content:
                unique_chunks.append(chunk)
                seen_content.add(content)

        if not unique_chunks:
            return []

        pairs = [[translated_query, get_text_content(chunk)] for chunk in unique_chunks]

        inputs = reranker_tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(model_device)

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
        subject="math",
        is_exercise_mode=False,
        course_title=None,
        use_alternative=False,
        retry_count=0,
    ):
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

        subject_instruction = SUBJECT_PROMPTS.get(subject, SUBJECT_PROMPTS["math"])

        course_context = ""
        if course_title:
            course_context = f"""
【當前課程主題】：{course_title}
【回答指引】：
1. 優先使用「{course_title}」相關的例子和概念來回答
2. 如果問題完全無關，才需要溫和提醒學生當前課程主題
"""

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
1. 必須使用「【題目】」和「【答案】」這兩個標記
2. 兩個部分之間空一行
3. 題目必須嚴格圍繞「{course_title}」主題
"""

        alternative_instruction = ""
        if use_alternative or retry_count > 0:
            if learner_style == "基礎級":
                alternative_methods = [
                    "用「生活情境類比」：用日常生活中具體的例子來比喻",
                    "用「視覺化描述」：描述圖像、動作、具體步驟",
                    "用「反向思考」：從答案往回推導",
                    "用「極簡化拆解」：把問題拆成最小的單位",
                ]
                selected_method = alternative_methods[
                    min(retry_count, len(alternative_methods) - 1)
                ]
                alternative_instruction = f"""
【特別指示：換個角度解釋（第 {retry_count + 1} 次重試）】
學生表示還是不理解，請用完全不同的方式重新解釋：
本次使用方法：{selected_method}
"""
            else:
                alternative_instruction = f"""
【特別指示：換個角度解釋（重試請求）】
學生表示需要更多解釋，請提供不同的視角或例子，與上次解釋保持差異。
"""

        def escape_braces(text):
            if not text:
                return ""
            return text.replace("{", "{{").replace("}", "}}")

        escaped_memory = escape_braces(memory_chunk)
        escaped_course = escape_braces(course_context)
        escaped_alternative = escape_braces(alternative_instruction)
        escaped_exercise = escape_braces(exercise_instruction)
        escaped_roleplay = escape_braces(
            ROLEPLAY_STYLES.get(learner_style, ROLEPLAY_STYLES["標準級"])
        )
        escaped_subject_prompt = escape_braces(subject_instruction)

        diversity_instruction = """
【回答多樣性要求】：
即使是相同或類似的問題，也請嘗試從不同角度切入，使用不同的例子或解釋方式。
"""
        escaped_diversity = escape_braces(diversity_instruction)

        system_prompt_parts = [
            "你是一個專業的學術助教：",
            "這是使用者最近的對話相關內容，請重點結合參考。",
            escaped_memory,
            escaped_course,
            "",
            f"你的個性設定是：{escaped_roleplay}，請嚴格遵守該等級的【角色設定】與【核心規則】。",
            "",
            "=== 學科專屬指導原則 ===",
            escaped_subject_prompt,
            "======================",
            "",
            escaped_diversity,
            escaped_alternative,
            "盡量依據提供的上下文回答問題，若搜不到合適的文句可以根據語言模型的內建知識庫回答問題。",
            "你只能使用繁體中文回答。",
            "",
            "【數學公式格式規範】",
            "- 行內公式使用單個 $ 符號包圍，例如：$x^2 + y^2 = r^2$",
            "- 區塊公式使用雙 $ 符號包圍，例如：$$\\frac{{a}}{{b}} = c$$",
            "",
            escaped_exercise,
            "以下是與你需要答的問題相關的內容:",
            "上下文如下：\n\n{context}",
        ]

        system_prompt = "\n".join(system_prompt_parts)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "question: {input}"),
            ]
        )

        chain = prompt_template | llm
        try:
            result = chain.invoke({"input": query, "context": context})
            if hasattr(result, "content"):
                answer = result.content
            else:
                answer = str(result)
        except Exception as e:
            print(f"錯誤訊息:{e}")
            answer = "抱歉，生成答案時發生錯誤。"

        return str(answer).strip()

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
            formatted_context = original_docs
        else:
            context_parts = []
            for doc in original_docs:
                if isinstance(doc, str):
                    context_parts.append(doc)
                elif isinstance(doc, tuple):
                    if hasattr(doc[0], "page_content"):
                        context_parts.append(doc[0].page_content)
                    else:
                        context_parts.append(str(doc[0]))
                elif hasattr(doc, "page_content"):
                    context_parts.append(doc.page_content)
                else:
                    context_parts.append(str(doc))

            formatted_context = "\n\n".join(context_parts)

        def escape_braces(text):
            if not text:
                return ""
            return text.replace("{", "{{").replace("}", "}}")

        escaped_roleplay = escape_braces(
            ROLEPLAY_STYLES.get(learner_style, ROLEPLAY_STYLES["標準級"])
        )
        escaped_query = escape_braces(original_query)
        escaped_selected = escape_braces(selected_text)
        escaped_context = escape_braces(formatted_context)

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
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", f"請幫我詳細解釋「{escaped_selected}」這句話的意思。"),
            ]
        )

        llm_chain = prompt_template | llm

        try:
            result = llm_chain.invoke({})
            clarification = result.content
        except Exception as e:
            print(f"錯誤訊息 (generate_clarification): {e}")
            clarification = "抱歉，生成闡釋時發生錯誤。"

        return str(clarification).strip()

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
