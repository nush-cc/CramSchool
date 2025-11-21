import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import json
import argparse

load_dotenv()
key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

PROMPT_DIFFICULY = """
        請針對以下內容生成三種難度等級的解釋：

        問題：
        {problem}

        標準解法：
        {solution}

        ---
        Level 1 (基礎級) 規則：
        * 目標受眾：完全聽不懂這堂課的學生。
        * 風格：必須像聊天一樣口語化。
        * 限制：『絕對禁止』使用任何專有名詞 (例如：禁止說「一元一次方程式」、「對偶」)。
        * 產出：只能用「一步一步照著做」的SOP。

        ---
        Level 2 (標準級) 規則：
        * 目標受眾：正在準備考試的普通學生。
        * 風格：像教科書或參考書一樣嚴謹。
        * 限制：『必須』使用所有相關的專有名詞和定義。
        * 產出：標準的、最有效率的解題步驟。

        ---
        Level 3 (進階級) 規則：
        * 目標受眾：想要考高分或參加競賽的頂尖學生。
        * 風格：像大學教授在啟發學生。
        *核心要求：必須與 Level 2 產生「巨大鑑別度」。
        *限制 (必須全部做到)：
        1.『嚴格禁止』只是重複 Level 2 的解法或換句話說。
        2.『必須』深入分析「為什麼」這個解法可行（例如：背後的數學/語文原理、定理）。
        3.『必須』提供至少一種「替代解法」（例如：圖解法、速解法、逆推法）。
        4.『必須』提供「觀念延伸」，將此問題連結到「其他相關單元」或「更高年級」的知識點（例如：用高中的觀點看國中問題）。
        5.『必須』總結這個題型帶來的「易錯點」。
        """

SYSTEM_PROMPT = """
        你是一位頂尖的教學設計專家。
        你的任務是根據使用者提供的「問題」和「標準解法」，
        嚴格按照Level 1 (基礎級)、Level 2 (標準級)、Level 3 (進階級)的規則，生成三種不同難度的解釋。
        你必須確保三個等級的「用詞」、「深度」和「切入角度」有顯著的區別。
        """

class Difficulty_rating:
    def __init__(self,PROMPT_DIFFICULY,system_prompt):
        self.difficuly = PROMPT_DIFFICULY
        self.system_prompt = system_prompt

    def generate_explanation(self,problem_text,standard_solution):
        user_content = self.difficuly.format(
            problem=problem_text,
            solution=standard_solution
        )
    
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,  # 保留一點點彈性，但不要太高
                max_tokens=1024
            )
            
            return completion.choices[0].message.content
        
        except Exception as e:
            print(f"API 呼叫失敗: {e}")
            return None
        
    def post_process(self,explanation_text):
        if not explanation_text:
            return {"L1": "", "L2": "", "L3": ""}

        l1_start = re.search(r"Level 1", explanation_text, re.IGNORECASE)
        l2_start = re.search(r"Level 2", explanation_text, re.IGNORECASE)
        l3_start = re.search(r"Level 3", explanation_text, re.IGNORECASE)

        idx1 = l1_start.start() if l1_start else -1
        idx2 = l2_start.start() if l2_start else -1
        idx3 = l3_start.start() if l3_start else -1

        # 如果完全找不到標記，將全部內容當作 L2 (標準)
        if idx1 == -1 and idx2 == -1 and idx3 == -1:
            print("警告：後處理找不到 Level 標記。")
            return {"L1": "", "L2": explanation_text, "L3": ""}

        # --- 內容分割邏輯 ---
        l3_content = ""
        temp_split_l3 = []
        if idx3 != -1:
            # 從 L3 的位置切開
            temp_split_l3 = [explanation_text[:idx3], explanation_text[idx3:]]
            l3_content = temp_split_l3[1] # L3 的內容
        else:
            temp_split_l3 = [explanation_text] # 沒有 L3

        l2_content = ""
        temp_split_l2 = []
        if idx2 != -1:
            # 從 L3 之前的部分，找到 L2 並切開
            temp_split_l2 = [temp_split_l3[0][:idx2], temp_split_l3[0][idx2:]]
            l2_content = temp_split_l2[1] # L2 的內容
        else:
            temp_split_l2 = [temp_split_l3[0]] # L3 之前沒有 L2
        
        l1_content = temp_split_l2[0]

        if idx1 != -1 and idx2 == -1 and idx3 == -1:
             l1_content = explanation_text

        def clean_header(text):
            if not text: 
                return ""
            # 找到第一個換行符號或冒號，取之後的所有內容
            match = re.search(r'[:\n]', text)
            if match:
                return text[match.end():].strip()
            else:
                # 如果沒有換行或冒號，試著移除 "Level X" 本身
                return re.sub(r'^Level \d.*', '', text, 1, flags=re.IGNORECASE).strip()

        return {
            "L1": clean_header(l1_content),
            "L2": clean_header(l2_content),
            "L3": clean_header(l3_content)
        }
    
    def difficulty_main_process(self,problem_text,solution_text):
        raw_text = self.generate_explanation(problem_text, solution_text)

        levels_dict = self.post_process(raw_text)

        result = {
            "problem": problem_text,
            "solution": solution_text,
            "L1": levels_dict.get("L1", ""),
            "L2": levels_dict.get("L2", ""),
            "L3": levels_dict.get("L3", "")
        }
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="增生解題模型資料 (Pipeline Script)"
    )
    
    parser.add_argument(
        "--input-path", 
        type=str, 
        required=True, 
        help="[輸入] 增生解題模型資料路徑"
    )
    
    parser.add_argument(
        "--output-path", 
        type=str,
        required=True, 
        help="[輸出] 處理完成解題模型資料儲存路徑"
    )
    args = parser.parse_args()

    input_file = args.input_path
    output_file = args.output_path

    with open(input_file,"r",encoding="utf-8") as f:
        data = json.load(f)

    datasets = data[:3]
    difficulty_rator = Difficulty_rating(PROMPT_DIFFICULY,SYSTEM_PROMPT)
    
    result = []
    with open(output_file,"w",encoding="utf-8") as f:
        for data in tqdm(datasets,desc="生成難度回覆中"):
            problem = data["messages"][0]["content"]
            solution = data["messages"][1]["content"]
            category = data.get("category")
            
            explanations = difficulty_rator.generate_explanation(problem, solution)
            
            processed_content_dict = difficulty_rator.post_process(explanations)
            
            if explanations:
                result.append({
                    "messages": [
                        {
                            "role": "user",
                            "content": problem
                        },
                        {
                            "role": "assistant",
                            "content_L1": processed_content_dict.get("L1", ""),
                            "content_L2": processed_content_dict.get("L2", ""),
                            "content_L3": processed_content_dict.get("L3", "")
                        }
                    ],
                    "category": category
                })
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"\n處理完成！結果已儲存至 {output_file}")