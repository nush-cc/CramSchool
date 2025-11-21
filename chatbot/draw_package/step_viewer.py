import tkinter as tk
from PIL import ImageTk
import json
import os

#匯入類別
from draw_package.drawing_engine import DrawingEngine

class StepViewer:
    def __init__(self, json_path):
        self.json_path = json_path
        self.current_step = 0
        self.root = None
        self.panel = None
        self.tk_img = None
        
        # 載入 JSON
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.layout_data = json.load(f)
            self.total_steps = len(self.layout_data.get("steps", []))
            if self.total_steps == 0:
                return
        except Exception as e:
            print(f"讀取 JSON 失敗: {e}")
            return

        self.engine = DrawingEngine()# 初始化繪圖引擎

    def show(self):
        # 修改這段檢查邏輯
        if not hasattr(self, 'layout_data'): 
            print(f"[錯誤] 無法啟動顯示器：未載入 Layout 資料 (路徑: {self.json_path})")
            return

        print(f"[系統] 正在啟動視窗，共 {self.total_steps} 步...") # 加入這行確認有跑到這裡

        self.root = tk.Tk()
        self.root.title(f"幾何步驟演示 - {os.path.basename(self.json_path)}")
        
        self.update_image()
        self.root.mainloop()

    def update_image(self):
        # 使用引擎渲染當前步驟的 PIL Image
        pil_image = self.engine.render_specific_step(self.layout_data, self.current_step)
        
        # 轉換為 Tkinter 格式
        self.tk_img = ImageTk.PhotoImage(pil_image)

        if self.panel is None:
            self.panel = tk.Label(self.root, image=self.tk_img)
            self.panel.pack(side="bottom", fill="both", expand="yes")
            # 綁定左鍵點擊事件
            self.panel.bind("<Button-1>", self.next_step)
        else:
            self.panel.configure(image=self.tk_img)
            self.panel.image = self.tk_img
            
        step_info = f"步驟: {self.current_step + 1} / {self.total_steps}"
        print(f"[Viewer] 顯示 {step_info}")
        self.root.title(f"幾何演示 - {step_info} (點擊圖片下一頁)")

    def next_step(self, event):
        if self.current_step < self.total_steps - 1:
            self.current_step += 1
            self.update_image()
        else:
            print("已是最後一步，關閉視窗。")
            self.root.destroy()