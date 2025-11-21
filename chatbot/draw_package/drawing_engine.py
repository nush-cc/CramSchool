from PIL import Image, ImageDraw, ImageFont
import os
import json

class DrawingEngine:
    def __init__(self, width=600, height=400, draw_axes=False):
        self.image = Image.new('RGBA', (width, height), (255, 255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.width = width
        self.height = height
        
        self.font = self.get_chinese_font() 
        
        self.draw.rectangle([0, 0, width-1, height-1], outline="gray", width=1)
        
        self.draw_axes = draw_axes
        if self.draw_axes:
            print("    (繪圖引擎：偵測到座標題，正在繪製座標軸...)")
            center_x, center_y = width // 2, height // 2
            self.draw.text((5,5), f"原點O({center_x},{center_y})", fill="gray", font=self.font)
            self.draw.line([(center_x, 0), (center_x, height)], fill="lightgray")  # y軸
            self.draw.line([(0, center_y), (width, center_y)], fill="lightgray")  # x軸
        else:
            print("(繪圖引擎：一般題目，建立空白畫布)")
    
    def get_chinese_font(self):
        font_paths = [
            'C:\\Windows\\Fonts\\msjh.ttc', # Windows (微軟正黑體)
            '/System/Library/Fonts/PingFang.ttc', # macOS (蘋方)
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc' # Linux (Noto Sans CJK)
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                return ImageFont.truetype(path, 15)
                    
        print("警告：找不到可用的中文字型")
        return ImageFont.load_default()

    def draw_line(self, start_point, end_point, color="black", width=2, style="solid"):
        fill_color = color
        
        if style == "dashed":
            pass 
        
        self.draw.line([tuple(start_point), tuple(end_point)], fill=fill_color, width=width)
        return f"成功：已畫線 {start_point} -> {end_point}。"

    def draw_circle(self, center, radius, color="black", width=2):
        fill_color = color
        top_left = (center[0] - radius, center[1] - radius)
        bottom_right = (center[0] + radius, center[1] + radius)
        self.draw.ellipse([top_left, bottom_right], outline=fill_color, width=width)
        return f"成功：已畫圓心 {center} 半徑 {radius} 的圓。"

    def draw_rectangle(self, top_left, bottom_right, color="black", width=2):
        x0 = top_left[0]
        y0 = top_left[1]
        x1 = bottom_right[0]
        y1 = bottom_right[1]

        true_x0 = min(x0, x1)
        true_y0 = min(y0, y1)
        true_x1 = max(x0, x1)
        true_y1 = max(y0, y1)

        corrected_top_left = (true_x0, true_y0)
        corrected_bottom_right = (true_x1, true_y1)

        self.draw.rectangle([corrected_top_left, corrected_bottom_right], outline=color, width=width)
        return f"成功：已畫長方形 {corrected_top_left} -> {corrected_bottom_right}。"

    """ 畫一個三角形 """
    def draw_triangle(self, point1=None, point2=None, point3=None, points=None, color="black", width=2):
        final_points = []
        if points:
            if isinstance(points, list) and len(points) >= 3:
                final_points = [tuple(points[0]), tuple(points[1]), tuple(points[2])]
            else:
                return f"錯誤：'points' 參數格式不正確 (需要至少 3 個點的列表)"
        elif point1 and point2 and point3:
            final_points = [tuple(point1), tuple(point2), tuple(point3)]
        else:
            return f"錯誤：draw_triangle 缺少必要的 'points' 或 ('point1', 'point2', 'point3') 參數"

        self.draw.polygon(final_points, outline=color, width=width)
        return f"成功：已畫三角形 {final_points[0]} -> {final_points[1]} -> {final_points[2]}。"

    """ 畫一個多邊形"""
    def draw_polygon(self, points, color="black", width=2):
        tuple_points = [tuple(p) for p in points]
        self.draw.polygon(tuple_points, outline=color, width=width)
        return f"成功：已畫 {len(points)} 邊形。"
        
    """ 畫一個橢圓 """
    def draw_ellipse(self, top_left, bottom_right, color="black", width=2):
        self.draw.ellipse([tuple(top_left), tuple(bottom_right)], outline=color, width=width)
        return f"成功：已畫橢圓 {top_left} -> {bottom_right}。"

    def draw_text(self, text, position, color="black", size=15):
        if isinstance(position, (int, float)):
            position = (int(position), self.height // 2)
        elif isinstance(position, list):
            if len(position) == 1:
                position = (int(position[0]), self.height // 2)
            elif len(position) >= 2:
                position = (int(position[0]), int(position[1]))
            else:
                position = (self.width // 2, self.height // 2) 
        elif not isinstance(position, tuple):
            position = (self.width // 2, self.height // 2)

        safe_text = str(text)

        self.draw.text(position, safe_text, fill=color, font=self.font)
        return f"成功：已在 {position} 寫上 '{safe_text}'。"


    def save_image(self, filepath):
        self.image.save(filepath)
        return f"成功：圖片已儲存至 {filepath}"
    

    def reset_canvas(self, width=None, height=None):
        w = width if width else self.width
        h = height if height else self.height
        self.image = Image.new('RGBA', (w, h), (255, 255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.draw.rectangle([0, 0, w-1, h-1], outline="gray", width=1)
        
        # 補上座標軸邏輯 (如果有開啟的話)
        if self.draw_axes:
            center_x, center_y = w // 2, h // 2
            self.draw.text((5,5), f"原點O({center_x},{center_y})", fill="gray", font=self.font)
            self.draw.line([(center_x, 0), (center_x, h)], fill="lightgray")
            self.draw.line([(0, center_y), (w, center_y)], fill="lightgray")
    
    def render_specific_step(self, layout_data, step_index):
        """
        渲染到指定步驟 (step_index)，包含該步驟之前的所有內容。
        回傳: PIL Image 物件
        """
        # 重置畫布
        canvas_size = tuple(layout_data.get("canvas_size", [self.width, self.height]))
        self.reset_canvas(canvas_size[0], canvas_size[1])
        
        all_steps = layout_data.get("steps", [])
        
        # 決定要畫到哪一步 (累積繪圖)
        target_index = min(step_index, len(all_steps) - 1)
        
        elements_to_draw = []
        for i in range(target_index + 1):
            elements_to_draw.extend(all_steps[i].get("elements", []))
            
        # 繪製元素
        for element in elements_to_draw:
            self.draw_single_element(element)
            
        return self.image.copy()

    def draw_single_element(self, element):
        """繪製單一元素 """
        try:
            el_type = element.get("type")
            color = element.get("color", "black")
            width = element.get("width", 2)
            
            if el_type == "triangle":
                self.draw_triangle(points=element.get("points"), color=color, width=width)
            elif el_type == "circle":
                self.draw_circle(center=element.get("center"), radius=element.get("radius"), color=color, width=width)
            elif el_type == "rectangle":
                self.draw_rectangle(top_left=element.get("top_left"), bottom_right=element.get("bottom_right"), color=color, width=width)
            elif el_type == "polygon":
                self.draw_polygon(points=element.get("points"), color=color, width=width)
            elif el_type == "line":
                self.draw_line(start_point=element.get("start"), end_point=element.get("end"), color=color, width=width)
            elif el_type == "text":
                self.draw_text(text=element.get("text"), position=element.get("pos"), color=element.get("color", "darkgreen"))
        except Exception as e:
            print(f"繪製元素失敗: {e}")

    def render_all_from_json(self, layout_json_path, output_png_path):
        try:
            with open(layout_json_path, 'r', encoding='utf-8') as f:
                layout = json.load(f)
            # 渲染最後一步即為全部
            last_step = len(layout.get("steps", [])) - 1
            self.render_specific_step(layout, last_step)
            self.save_image(output_png_path)
            return f"成功渲染預覽圖 {output_png_path}"
        except Exception as e:
            print(f"Error: {e}")
    
