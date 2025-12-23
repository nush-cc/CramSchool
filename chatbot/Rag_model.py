import os
import sys

from .rag_pipeline.RAG_function import rag_process
from .rag_pipeline.post_process import Post_process

current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 取得上一層目錄 (即專案根目錄，例如 'C:\Users\USER\補習班')
project_root = os.path.dirname(current_dir)

# 3. 將根目錄加入 Python 的搜尋路徑中
if project_root not in sys.path:
    sys.path.append(project_root)

from .draw_package.step_viewer import StepViewer
from .solve_generator.difficuly_prompt_generation import (
    Difficulty_rating,
    PROMPT_DIFFICULY,
    SYSTEM_PROMPT,
)


class Rag_Main:
    def __init__(self):
        self.rag_processor = rag_process()
        # 初始化儲存向量資料庫和 dataset 的變數
        self.teaching_vs = None
        self.teaching_ds = None
        self.exercise_vs = None
        self.exercise_ds = None
        self.drawing_steps_dir = os.path.join(
            project_root, "dataset", "llama_drawing_steps"
        )
        self.difficult_rator = Difficulty_rating(PROMPT_DIFFICULY, SYSTEM_PROMPT)

        self.post_processor = Post_process()

    def rag_main_process(self, teaching_path, exercise_path, learner_style):
        # 呼叫修改後的 vectorize_workflow
        (self.teaching_vs, self.teaching_ds), (self.exercise_vs, self.exercise_ds) = (
            self.rag_processor.vectorize_workflow(teaching_path, exercise_path)
        )

        print("\n=== 教學與練習題資料庫均已載入完成 ===")

        while True:
            print(f"\n{'-' * 60}\n")
            # 1. 根據圖片內容，讓使用者選擇檢索模式
            print("請選擇檢索模式：")
            print("1. 概念講解 / 觀念學習 (僅搜尋教學資料庫)")
            print("2. 練習題 / 出題練習 (僅搜尋練習題資料庫)")
            print("3. 解題與原理 / 觀念追溯 (混合搜尋)")

            mode_choice = input("請輸入 1, 2, 或 3 (輸入 exit 結束)：\n> ").strip()

            if mode_choice.lower() in ["exit", "quit", "q"]:
                print("結束互動！")
                break

            if mode_choice == "1":
                search_type = "teaching"
                prompt_text = "你選擇了 [教學資料庫]，請輸入你的問題："
                current_k = 3
            elif mode_choice == "2":
                search_type = "exercise"
                prompt_text = "你選擇了 [練習題資料庫]，請輸入你的問題："
                current_k = 1
            elif mode_choice == "3":
                search_type = "hybrid"
                prompt_text = "你選擇了 [混合搜尋]，請輸入你的問題："
                current_k = 4
            else:
                print("輸入無效，將使用預設的 [混合搜尋] 模式。")
                search_type = "hybrid"
                prompt_text = "請輸入你的問題："
                current_k = 2

            # 2. 取得使用者問題
            user_query = input(f"{prompt_text}\n> ").strip()
            if not user_query:
                print("請勿輸入空問題。")
                continue
            if user_query.lower() in ["exit", "quit", "q"]:
                print("結束互動！")
                break

            # 3. 檢索 (傳入 search_type 和兩個資料庫)
            retrieved = self.rag_processor.retrival_step(
                [user_query],
                search_type,
                (self.teaching_vs, self.teaching_ds),
                (self.exercise_vs, self.exercise_ds),
                top_n=current_k,
            )

            retrieved_docs = retrieved.get(user_query, [])

            # print(f"檢索內容:{retrieved_docs}")

            matched_context = "\n".join(
                [
                    doc.page_content if hasattr(doc, "page_content") else str(doc)
                    for doc in retrieved[user_query]
                ]
            )

            if search_type == "exercise":
                if retrieved_docs:
                    first_doc = retrieved_docs[0]

                    # 從 metadata 取得乾淨的題目與詳解
                    if hasattr(first_doc, "metadata"):
                        question_text = first_doc.metadata.get("question")
                        solution_text = first_doc.metadata.get("answer")

                # dict
                difficult_rating_content = self.difficult_rator.difficulty_main_process(
                    question_text, solution_text
                )

                print("==============================================\n")
                print(f"\n題目:{difficult_rating_content['problem']}\n")
                print(f"\n詳解L1:{difficult_rating_content['L1']}\n")
                print(f"\n詳解L2:{difficult_rating_content['L2']}\n")
                print(f"\n詳解L3:{difficult_rating_content['L3']}\n")
                print("==============================================\n")
                # 這裡不呼叫 generate_answer，直接結束這回合的文字輸出

                """題目的深入回覆"""
                # 統一為list
                segments = [
                    difficult_rating_content["L1"],
                    difficult_rating_content["L2"],
                    difficult_rating_content["L3"],
                ]
                self.rag_processor.clarification_main_process(
                    segments, user_query, [first_doc], learner_style
                )

            else:
                # 4. 預測使用者風格 & LLM 生成 (適用於教學或混合模式)
                memory_chunk = ""
                answer = self.rag_processor.generate_answer(
                    matched_context, user_query, learner_style, memory_chunk
                )

                processed_answer = self.post_processor.clean_redundant_text(answer)
                print(f"\n答案: {processed_answer}\n")

                """教學/混用的深入回覆"""
                # 分段的list
                segments = self.post_processor.split_answer(answer)

                self.rag_processor.clarification_main_process(
                    segments, user_query, retrieved_docs, learner_style
                )

            self.check_and_show_drawing(retrieved_docs)

    def check_and_show_drawing(self, docs):
        """
        檢查檢索到的文檔中是否有 metadata['id']
        若有，且在 llama_drawing_steps 資料夾中有對應 JSON，則啟動視覺化
        """
        found_id = None

        # 找 id
        for doc in docs:
            if hasattr(doc, "metadata"):
                # 假設 metadata 裡的 key 是 'id' 或 'question_id'
                doc_id = doc.metadata.get("id")
                if doc_id:
                    found_id = str(doc_id)
                    print(f"(系統偵測到題目 ID: {found_id})")
                    break  # 找到最相關的一個就停止

        if found_id:
            possible_filenames = [f"{found_id}.json", f"{found_id}_layout.json"]
            found_path = None

            for fname in possible_filenames:
                full_path = os.path.join(self.drawing_steps_dir, fname)
                if os.path.exists(full_path):
                    found_path = full_path
                    break

            if found_path:
                try:
                    viewer = StepViewer(found_path)
                    viewer.show()
                    print(">>> 演示結束。\n")
                except Exception as e:
                    print(f"開啟繪圖視窗失敗: {e}")
            else:
                pass

    def search_url(self, docs):
        """
        得到題目中的對應url
        """
        sim = docs.metadata.get("simulation")
        if sim is not None:
            return sim.get("url")

    def show_outside_packages(self, subject, docs):
        """
        工具分派器
        根據科目決定要顯示「解題步驟圖」還是「模擬實驗網頁」。
        """
        if subject == 'math':
            self.check_and_show_drawing(docs)
        elif subject == 'science':
            if isinstance(docs, list):
                first_doc = docs[0]
            else:
                first_doc = docs
            url = self.search_url(first_doc)
            if url:
                self.simulator.show(url)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"目前專案基準路徑: {base_dir}")

    teaching_path_list = [
        os.path.join(base_dir, "dataset", "handouts_data", "三角形全等應用.pdf"),
        os.path.join(base_dir, "dataset", "handouts_data", "函數.pdf"),
    ]

    exercise_path = os.path.join(
        base_dir, "dataset", "raw_data", "add_id_data", "question_math_id.json"
    )

    for path in teaching_path_list:
        if not os.path.exists(path):
            print(f"警告：找不到檔案 -> {path}")

    rag_main = Rag_Main()

    # 風格傳入
    learner_style = "進階級"
    rag_main.rag_main_process(teaching_path_list, exercise_path, learner_style)
