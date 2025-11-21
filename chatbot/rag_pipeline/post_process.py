import re
import math


class Post_process:
    def __init__(self):
        pass

    def clean_redundant_text(self, text):  # 後處理輸出格式
        lines = [
            line.strip() for line in text.split("\n") if line.strip()
        ]  # 移除重複空行
        cleaned = []
        for line in lines:  # 移除連續重複的句子（或片段）
            if not cleaned or (line != cleaned[-1] and line not in cleaned[-1]):
                cleaned.append(line)
        text = "\n".join(cleaned)

        redundant_phrases = [
            "###回答：",
            "### 回答：",
            "相關公式如下：",
            "公式如下：",
            "以下是：",
            "如下：",
            "(Provide at least 100 words in a coherent paragraph.)",
            "在連貫的段落中至少提供100個單詞。",
        ]  # 移除「###回答：」「相關公式如下」等語助詞
        for phrase in redundant_phrases:
            text = text.replace(phrase, "")
        text = re.sub(r"(\S{2,10})\1", r"\1", text)  # 抓重複詞語（2~10 字）
        return text.strip()

    """新增"""

    def split_answer(self, text) -> list:
        """
        將答案切成三段，並保護 LaTeX 公式和 Markdown 結構不被切斷
        """
        if not text:
            return []

        # === 步驟 1: 保護 LaTeX 公式和 Markdown 結構 ===
        protected_blocks = {}
        counter = 0
        protected_text = text

        # 1.1 保護區塊公式 $$...$$ (可能跨多行)
        def protect_block_dollar(match):
            nonlocal counter
            placeholder = f"___PROTECTED_BLOCK_{counter}___"
            protected_blocks[placeholder] = match.group(0)
            counter += 1
            return placeholder

        protected_text = re.sub(r'\$\$[\s\S]*?\$\$', protect_block_dollar, protected_text)

        # 1.2 保護區塊公式 \[...\] (可能跨多行)
        def protect_block_bracket(match):
            nonlocal counter
            placeholder = f"___PROTECTED_BLOCK_{counter}___"
            protected_blocks[placeholder] = match.group(0)
            counter += 1
            return placeholder

        protected_text = re.sub(r'\\\[[\s\S]*?\\\]', protect_block_bracket, protected_text)

        # 1.3 保護行內公式 $...$ (不跨行)
        def protect_inline_dollar(match):
            nonlocal counter
            placeholder = f"___PROTECTED_INLINE_{counter}___"
            protected_blocks[placeholder] = match.group(0)
            counter += 1
            return placeholder

        protected_text = re.sub(r'\$[^\$\n]+?\$', protect_inline_dollar, protected_text)

        # 1.4 保護行內公式 \(...\)
        def protect_inline_paren(match):
            nonlocal counter
            placeholder = f"___PROTECTED_INLINE_{counter}___"
            protected_blocks[placeholder] = match.group(0)
            counter += 1
            return placeholder

        protected_text = re.sub(r'\\\([^\)]+?\\\)', protect_inline_paren, protected_text)

        # 1.5 保護 Markdown 代碼塊 ```...```
        def protect_code_block(match):
            nonlocal counter
            placeholder = f"___PROTECTED_CODE_{counter}___"
            protected_blocks[placeholder] = match.group(0)
            counter += 1
            return placeholder

        protected_text = re.sub(r'```[\s\S]*?```', protect_code_block, protected_text)

        # === 步驟 2: 在保護後的文本上進行切段 ===
        # 段落中若有數字編號則以此切分
        list_pattern = r"(?:\n|^)\s*[\(]?\d+[\.\)\]]\s"
        if len(re.findall(list_pattern, protected_text)) >= 3:
            pass

        parts = [p.strip() for p in protected_text.split("\n") if p.strip()]

        if len(parts) < 3:
            split_sentences = re.split(r"([。？！；])", protected_text)
            parts = []

            for i in range(0, len(split_sentences) - 1, 2):
                parts.append(split_sentences[i] + split_sentences[i + 1])
            if len(split_sentences) % 2 != 0 and split_sentences[-1]:
                parts.append(split_sentences[-1])

            parts = [p.strip() for p in parts if p.strip()]

        if len(parts) == 0:
            return []
        if len(parts) < 3:
            # 還原保護的內容
            restored_text = self._restore_protected_blocks(text, protected_blocks)
            return [restored_text]

        # 合併成三段
        segments = self.merge_three_by_length(parts)

        # === 步驟 3: 還原所有保護的 LaTeX 公式和 Markdown 結構 ===
        restored_segments = []
        for segment in segments:
            restored_segment = self._restore_protected_blocks(segment, protected_blocks)
            restored_segments.append(restored_segment)

        return restored_segments

    def _restore_protected_blocks(self, text, protected_blocks):
        """
        還原被保護的 LaTeX 公式和 Markdown 結構
        """
        restored = text
        for placeholder, original in protected_blocks.items():
            restored = restored.replace(placeholder, original)
        return restored

    def merge_three_by_length(self, parts):
        total_length = sum(len(p) for p in parts)
        target_length = total_length / 3

        segments = []
        current_segment = []
        current_len = 0

        part_idx = 0

        # --- 組合第 1 段 ---
        while part_idx < len(parts):
            p = parts[part_idx]
            if current_len > 0 and (current_len + len(p)) > target_length * 1.1:
                break

            current_segment.append(p)
            current_len += len(p)
            part_idx += 1

            if current_len >= target_length and (len(parts) - part_idx) >= 2:
                break

        segments.append("\n".join(current_segment))

        # --- 組合第 2 段 ---
        current_segment = []
        current_len = 0
        while part_idx < len(parts):
            p = parts[part_idx]

            if current_len > 0 and (current_len + len(p)) > target_length * 1.1:
                break

            current_segment.append(p)
            current_len += len(p)
            part_idx += 1
            if current_len >= target_length and (len(parts) - part_idx) >= 1:
                break

        segments.append("\n".join(current_segment))

        # --- 組合第 3 段 (剩下的全部) ---
        if part_idx < len(parts):
            segments.append("\n".join(parts[part_idx:]))

        return [s for s in segments if s.strip()]
