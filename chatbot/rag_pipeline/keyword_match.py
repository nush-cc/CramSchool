from keybert import KeyBERT
import jieba
import difflib
from ..config import KeyBERT_model_name

class Keyword_matching:
    def __init__(self):
        pass

    def extract_keywords(self,query): #擷取關鍵字
        kw_model = KeyBERT(model=KeyBERT_model_name)
        seg_query = " ".join(jieba.cut(query))
        keywords = kw_model.extract_keywords(seg_query, keyphrase_ngram_range=(1, 2), stop_words=None,top_n=5)
        keyword_list = [kw[0].replace(" ", "") for kw in keywords]

        common_terms = [
            "k-means", "k means", "svm", "GAI",
            "LSTM", "TCN", "RNN", "Feedforward Neural Network", 
            "decision tree", "SFT", "neural network", "Deep learning", "machine learning","前饋神經網路", "決策樹", "微調", "神經網路",
            "深度學習", "機器學習"
        ] # 補上常見的技術詞彙，如果 query 中有但沒被 KeyBERT 抓出來
        for term in common_terms: 
            if term.lower() in query.lower() and term not in keyword_list: #比較技術詞彙，有的話就加入
                keyword_list.append(term)
        return keyword_list

    def keyword_match(self,docs, keywords, threshold=0.8): #找出包含關鍵字的chunks
        matched_chunks = []
        for doc in docs:
            page_content = doc.page_content.lower() if hasattr(doc, "page_content") else str(doc).lower() # 取得純文字內容
            for keyword in keywords: # 檢查每個 keyword 是否與內容相似
                keyword = keyword.lower()  #轉小寫       
                if keyword in page_content: # 如果完全包含，直接加入
                    matched_chunks.append(doc)
                    break            
                
                for n in range(2,7):
                    for i in range(len(page_content) - n +1):
                        chunk = page_content[i:i+n]
                        if difflib.SequenceMatcher(None, chunk, keyword).ratio() > threshold:
                            matched_chunks.append(doc)
                            break
                        else:
                            continue

        return matched_chunks