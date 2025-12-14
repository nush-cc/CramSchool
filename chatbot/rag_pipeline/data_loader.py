import json

import fitz


class Data_loader:
    def __init__(self, input_path=None, input_path_list=None, output_path=None):
        self.input_path = input_path  # 單一路徑
        self.output_path = output_path  # 單一路徑
        self.input_path_list = input_path_list  # 路徑list

    def load_file(self) -> list:
        if self.input_path.lower().endswith(".pdf"):
            doc = fitz.open(self.input_path)
            data = []
            for page_num, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:
                    data.append({
                        "content": text,
                        "category": self.input_path,  # 預設分類
                        "id": page_num + 1  # 暫定 ID
                    })
            return data
        elif self.input_path.lower().endswith(".json"):
            with open(self.input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        else:
            print("不是json或pdf檔案")
            return None

    def load_file_list(self):  # 利用FITZ(PYMUPDF)讀取PDF文句
        all_data = []
        for filepath in self.input_path_list:
            if filepath.lower().endswith(".pdf"):
                doc = fitz.open(filepath)
                for page_num, page in enumerate(doc):
                    text = page.get_text("text").strip()
                    if text:
                        all_data.append({
                            "content": text,
                            "category": filepath,
                            "id": page_num + 1
                        })
            elif filepath.lower().endswith(".json"):
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
            else:
                print("不是json或pdf檔案")
                return None

        return all_data

    def get_page_content(self, data):
        all_data = []

        for item in data:
            if isinstance(item, dict) and "messages" in item:
                messages = item["messages"]
                category = item.get("category", "General")
                data_id = item.get("id", "unknown_id")

                user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
                assistant_msg = next((m["content"] for m in messages if m.get("role") == "assistant"), "")

                combined_content = f"題目：{user_msg}\n詳解：{assistant_msg}"

                doc_data = {
                    "content": combined_content,
                    "category": category,
                    "id": data_id,
                    "question": user_msg,
                    "answer": assistant_msg
                }
                if "simulation" in item:
                    doc_data["simulation"] = item["simulation"]

                all_data.append(doc_data)

            elif isinstance(item, dict) and "content" in item and "category" in item:
                # 確保所有欄位都齊全
                doc_data = {
                    "content": item["content"],
                    "category": item.get("category", "Unknown"),
                    "id": item.get("id", "Unknown")
                }

                if "simulation" in item:
                    doc_data["simulation"] = item["simulation"]

                all_data.append(doc_data)

        return all_data
