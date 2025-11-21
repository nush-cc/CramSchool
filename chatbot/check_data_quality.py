"""
检查向量数据库中的数据质量
查找包含 MATH_INLINE 或 MATH_BLOCK 占位符的文档
"""

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
import os

def check_faiss_data():
    print("=" * 60)
    print("检查 FAISS 向量数据库中的数据质量")
    print("=" * 60)

    # 配置
    embedding_model_name = "BAAI/bge-small-zh-v1.5"
    model_device = "cpu"

    # 初始化 Embedding
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": model_device},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 检查教学数据库
    print("\n[1] 检查教学向量数据库...")
    if os.path.exists("faiss_index_teaching"):
        teaching_vs = FAISS.load_local(
            "faiss_index_teaching",
            embeddings,
            allow_dangerous_deserialization=True
        )

        # 获取所有文档
        total_docs = len(teaching_vs.docstore._dict)
        print(f"   总文档数: {total_docs}")

        # 检查每个文档
        problematic_docs = []
        for doc_id, doc in teaching_vs.docstore._dict.items():
            content = doc.page_content
            if 'MATH_INLINE' in content or 'MATH_BLOCK' in content:
                problematic_docs.append({
                    'id': doc_id,
                    'content_preview': content[:200],
                    'category': doc.metadata.get('category', 'Unknown')
                })

        print(f"   包含 MATH_ 占位符的文档数: {len(problematic_docs)}")

        if problematic_docs:
            print("\n   前 3 个问题文档:")
            for i, doc in enumerate(problematic_docs[:3]):
                print(f"\n   [{i+1}] ID: {doc['id']}")
                print(f"       分类: {doc['category']}")
                print(f"       内容预览: {doc['content_preview']}...")

    # 检查练习题数据库
    print("\n[2] 检查练习题向量数据库...")
    if os.path.exists("faiss_index_exercise"):
        exercise_vs = FAISS.load_local(
            "faiss_index_exercise",
            embeddings,
            allow_dangerous_deserialization=True
        )

        total_docs = len(exercise_vs.docstore._dict)
        print(f"   总文档数: {total_docs}")

        problematic_docs = []
        for doc_id, doc in exercise_vs.docstore._dict.items():
            content = doc.page_content
            if 'MATH_INLINE' in content or 'MATH_BLOCK' in content:
                problematic_docs.append({
                    'id': doc_id,
                    'content_preview': content[:200],
                    'category': doc.metadata.get('category', 'Unknown')
                })

        print(f"   包含 MATH_ 占位符的文档数: {len(problematic_docs)}")

        if problematic_docs:
            print("\n   前 3 个问题文档:")
            for i, doc in enumerate(problematic_docs[:3]):
                print(f"\n   [{i+1}] ID: {doc['id']}")
                print(f"       分类: {doc['category']}")
                print(f"       内容预览: {doc['content_preview']}...")

    print("\n" + "=" * 60)
    print("检查完成！")
    print("=" * 60)
    print("\n建议解决方案:")
    print("1. 如果数据源是 PDF，考虑使用更好的 PDF 提取工具（如 pdfplumber 或 marker）")
    print("2. 如果数据源是 Markdown，检查原始文件是否正确包含 LaTeX 公式")
    print("3. 重新处理 PDF 文件，使用支持数学公式的提取方法")
    print("4. 临时方案：前端已添加占位符过滤，会显示为 [数学式X]")

if __name__ == "__main__":
    check_faiss_data()
