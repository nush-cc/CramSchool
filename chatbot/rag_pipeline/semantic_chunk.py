from itertools import groupby, chain
from operator import itemgetter

import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from ..config import embedding_model_name


class Chunking:
    def __init__(self):
        pass

    def combine_sentences(self, sentences, window_size=4, stride=2):  # 判斷句子合併
        combined_chunks = []
        sentences.sort(key=itemgetter("category"))  # 按照來源排序
        for category_val, group in groupby(sentences, key=itemgetter("category")):
            group = list(group)  # 同一來源的句子列表
            for i in range(0, len(group) - window_size + 1, stride):
                chunk_sentences = group[i:i + window_size]
                chunk_text = " ".join(s["content"] for s in chunk_sentences)
                ids = list(sorted(set(s["id"] for s in chunk_sentences)))

                simulation_data = next((s.get("simulation") for s in chunk_sentences if s.get("simulation")), None)

                combined_chunks.append({
                    "combined_sentence": chunk_text,
                    "category": category_val,
                    "id": ids,
                    "simulation": simulation_data  # 加入 simulation
                })

            # 補上結尾不足 window 的部分
            if len(group) % stride != 0 and len(group) > 0:
                last_chunk_sentences = group[-window_size:]
                last_chunk_text = " ".join(s["content"] for s in last_chunk_sentences)
                ids = list(sorted(set(s["id"] for s in last_chunk_sentences)))

                simulation_data = next((s.get("simulation") for s in last_chunk_sentences if s.get("simulation")), None)

                if not any(c["combined_sentence"] == last_chunk_text for c in combined_chunks):
                    combined_chunks.append({
                        "combined_sentence": last_chunk_text,
                        "category": category_val,
                        "id": ids,
                        "simulation": simulation_data  # 加入 simulation
                    })

        return combined_chunks

    def semantic_chunk(self, docs):  # 文意切割
        all_content = "".join([d.get('content', '') for d in docs])
        sentences = self.combine_sentences(docs)

        model_kwargs = {"device": "cuda"}
        embedding_model = HuggingFaceBgeEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)

        texts = [x["combined_sentence"] for x in sentences]  # 遍歷COMBINED_SETENCE
        embeddings = embedding_model.embed_documents(texts)  # 對句子進行向量嵌入

        for i, sentence in enumerate(sentences):
            sentence['combined_sentence_embedding'] = embeddings[i]
        distances, sentences = self.calculate_cosine_distances(sentences)  # 計算CHUNKS間的cos距離

        if len(distances) == 0:
            print(f"[Warning] 文件內容過短，跳過語意切割，保留原始段落。")
            original_simulation = next((d.get("simulation") for d in docs if d.get("simulation")), None)

            return [{
                "text": all_content,
                "id": docs[0].get("id", "unknown"),
                "category": docs[0].get("category", "unknown"),
                "simulation": original_simulation  # 保留原始 simulation
            }]

        breakpoint_percentile_threshold = 90  # 切割點的閥值
        breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]  # 將高於閥值的切割點選出

        start_index = 0
        chunks = []
        for index in indices_above_thresh:
            end_index = index
            group = sentences[start_index:end_index + 1]  # 將介於兩切割點之間的CHUNKS組成一組
            combined_text = ''.join([d['combined_sentence'] for d in group])  # 將切割點之間的CHUNKS合併
            combined_categories = list(set([d['category'] for d in group]))
            combined_ids = list(set(chain.from_iterable(
                d["ids"] if isinstance(d.get("ids"), list) else [d.get("ids")] for d in group if "ids" in d)))

            combined_simulation = next((d.get("simulation") for d in group if d.get("simulation")), None)

            chunks.append({
                'text': combined_text,
                'category': combined_categories,
                'id': combined_ids,
                'simulation': combined_simulation  # 加入 simulation
            })
            start_index = index + 1

        if start_index < len(sentences):  # 將剩下最後一部分進行處理
            combined_text = ''.join([d['combined_sentence'] for d in sentences[start_index:]])  # 將最後的CHUNKS合併
            combined_categories = list(set([d['category'] for d in sentences[start_index:]]))
            combined_ids = list(set(chain.from_iterable(
                d["ids"] if isinstance(d.get("ids"), list) else [d.get("ids")] for d in sentences[start_index:] if
                "ids" in d)))

            combined_simulation = next((d.get("simulation") for d in group if d.get("simulation")), None)

            chunks.append({
                'text': combined_text,
                'category': combined_categories,
                'id': combined_ids,
                'simulation': combined_simulation
            })

        return chunks

    def calculate_cosine_distances(self, sentences):  # 計算CHUNKS間的cos距離
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]['combined_sentence_embedding']
            embedding_next = sentences[i + 1]['combined_sentence_embedding']
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]  # Calculate cosine similarity

            distance = 1 - similarity  # Convert to cosine distance
            distances.append(distance)  # Append cosine distance to the list
            sentences[i]['distance_to_next'] = distance  # Store distance in the dictionary
        return distances, sentences
