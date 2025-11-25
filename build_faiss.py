import os
import shutil
import sys
import glob

# 將當前目錄加入 sys.path 以便匯入 chatbot 模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from chatbot.rag_pipeline.RAG_function import rag_process
except ImportError as e:
    print("❌ 無法匯入 rag_process，請確認此腳本位於專案根目錄 (與 manage.py 同層)")
    print(f"錯誤訊息: {e}")
    sys.exit(1)

def main():
    print("=" * 60)
    print("🚀 開始重建 FAISS 向量資料庫")
    print("=" * 60)

    # 1. 設定路徑
    base_dir = os.path.dirname(os.path.abspath(__file__))
    chatbot_dir = os.path.join(base_dir, 'chatbot')
    
    # 資料來源路徑
    teaching_data_dir = os.path.join(chatbot_dir, 'dataset', 'handouts_data')
    exercise_data_path = os.path.join(chatbot_dir, 'dataset', 'raw_data', 'add_id_data', 'question_math_id.json')
    
    # 輸出路徑 (FAISS index)
    faiss_teaching_path = os.path.join(chatbot_dir, 'faiss_index_teaching')
    faiss_exercise_path = os.path.join(chatbot_dir, 'faiss_index_exercise')

    # 2. 檢查資料是否存在
    if not os.path.exists(teaching_data_dir):
        print(f"❌ 找不到教學資料目錄: {teaching_data_dir}")
        return
    
    if not os.path.exists(exercise_data_path):
        print(f"❌ 找不到練習題 JSON: {exercise_data_path}")
        return

    # 蒐集所有 PDF 檔案路徑
    pdf_files = glob.glob(os.path.join(teaching_data_dir, "*.pdf"))
    print(f"📚 找到 {len(pdf_files)} 個教學 PDF 檔案")
    print(f"📝 練習題資料: {os.path.basename(exercise_data_path)}")

    # 3. 刪除舊的向量資料庫 (強制重建)
    print("\n🧹 清理舊的向量資料庫...")
    if os.path.exists(faiss_teaching_path):
        shutil.rmtree(faiss_teaching_path)
        print(f"   已刪除: {faiss_teaching_path}")
    
    if os.path.exists(faiss_exercise_path):
        shutil.rmtree(faiss_exercise_path)
        print(f"   已刪除: {faiss_exercise_path}")

    # 4. 初始化 RAG 處理器
    print("\n⚙️ 初始化 RAG 處理器...")
    rag = rag_process()

    # 5. 執行向量化 (切換工作目錄到 chatbot 以確保儲存路徑正確)
    original_cwd = os.getcwd()
    try:
        print("\n🔄 切換工作目錄至 chatbot 資料夾以進行儲存...")
        os.chdir(chatbot_dir)
        
        print("⚡ 開始建立索引 (這可能需要一點時間)...")
        # 注意：RAG_function 會自動讀取我們刪除後留下的空位並建立新的
        rag.vectorize_workflow(pdf_files, exercise_data_path)
        
        print("\n✅ 向量資料庫重建完成！")
        print(f"   教學庫位置: {os.path.abspath('faiss_index_teaching')}")
        print(f"   練習庫位置: {os.path.abspath('faiss_index_exercise')}")

    except Exception as e:
        print(f"\n❌ 發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 恢復工作目錄
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()