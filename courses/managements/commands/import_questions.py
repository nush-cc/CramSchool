import json
import re
from django.core.management.base import BaseCommand
from courses.models import Course, Chapter, Grade, Subject, EducationLevel
from assessments.models import Question, Choice, QuestionType


class Command(BaseCommand):
    help = "匯入題目資料"

    def add_arguments(self, parser):
        parser.add_argument("json_file", type=str, help="JSON 檔案路徑")

    def handle(self, *args, **options):
        json_file = options["json_file"]

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 處理每一題
        items = data if isinstance(data, list) else [data]

        total = len(items)
        for idx, item in enumerate(items, 1):
            self.stdout.write(f"處理第 {idx}/{total} 題...")
            try:
                self.process_question(item)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"✗ 第 {idx} 題失敗: {e}"))

        self.stdout.write(self.style.SUCCESS(f"\n完成! 共處理 {total} 題"))

    def process_question(self, item):
        # 解析 category
        category = item["category"]

        # 用正則表達式解析
        # "國中 數學 第一冊（七上） 第三章_一元一次方程式 [3-3] 一元一次方程式的應用"
        pattern = r"(\S+)\s+(\S+)\s+\S+[（(](\S+)[)）]\s+(\S+)\s+(.+)"
        match = re.match(pattern, category)

        if not match:
            raise ValueError(f"無法解析 category: {category}")

        edu_level_name = match.group(1)  # 國中
        subject_name = match.group(2)  # 數學
        grade_text = match.group(3)  # 七上
        course_title = match.group(4)  # 第三章_一元一次方程式
        chapter_title = match.group(5)  # [3-3] 一元一次方程式的應用

        # 年級對應
        grade_mapping = {
            "七上": "一年級",
            "七下": "一年級",
            "八上": "二年級",
            "八下": "二年級",
            "九上": "三年級",
            "九下": "三年級",
        }
        grade_name = grade_mapping.get(grade_text, "一年級")

        # 年級排序對應
        grade_order_mapping = {
            "一年級": 1,
            "二年級": 2,
            "三年級": 3,
            "四年級": 4,
            "五年級": 5,
            "六年級": 6,
        }

        # 取得或建立學制
        edu_level, _ = EducationLevel.objects.get_or_create(
            name=edu_level_name, defaults={"order": 2}
        )

        # 取得或建立年級
        grade, _ = Grade.objects.get_or_create(
            education_level=edu_level,
            name=grade_name,
            defaults={"order": grade_order_mapping.get(grade_name, 1)},
        )

        # 取得或建立科目
        subject, _ = Subject.objects.get_or_create(name=subject_name)

        # 取得或建立課程
        course, _ = Course.objects.get_or_create(
            title=course_title,
            subject=subject,
            grade=grade,
            defaults={
                "description": f"{edu_level_name}{grade_name}{subject_name}",
                "is_active": True,
            },
        )

        # 取得或建立章節
        chapter, _ = Chapter.objects.get_or_create(
            course=course,
            title=chapter_title,
        )

        # 取得或建立題目類型
        question_type, _ = QuestionType.objects.get_or_create(name="選擇題")

        # 解析題目和選項
        messages = item["messages"]
        user_content = None
        assistant_content = None

        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]

        if not user_content or not assistant_content:
            raise ValueError("找不到題目或答案")

        # 分割題目和選項
        # 格式: "題目內容\n\n(A) 選項A\n(B) 選項B\n(C) 選項C\n(D) 選項D\n"
        question_parts = user_content.split("\n\n", 1)
        question_text = question_parts[0].strip()

        # 提取選項
        choices_text = question_parts[1] if len(question_parts) > 1 else ""
        choice_pattern = r"\(([A-D])\)\s*([^\n]+)"
        choices_matches = re.findall(choice_pattern, choices_text)

        # 提取正確答案
        # 格式: "正確答案：(A)\n解析：..."
        answer_match = re.search(r"正確答案[：:]\s*\(([A-D])\)", assistant_content)
        correct_answer = answer_match.group(1) if answer_match else None

        # 提取解析
        explanation_match = re.search(r"解析[：:]\s*(.+)", assistant_content, re.DOTALL)
        explanation = (
            explanation_match.group(1).strip()
            if explanation_match
            else assistant_content
        )

        # 建立題目
        question = Question.objects.create(
            course=course,
            chapter=chapter,
            question_type=question_type,
            content=question_text,
            explanation=explanation,
            difficulty=2,
            order=Question.objects.filter(course=course).count() + 1,
        )

        # 建立選項
        for i, (letter, choice_text) in enumerate(choices_matches, 1):
            Choice.objects.create(
                question=question,
                content=choice_text.strip(),
                is_correct=(letter == correct_answer),
                order=i,
            )

        self.stdout.write(
            self.style.SUCCESS(f"  ✓ {question_text[:30]}... (答案: {correct_answer})")
        )
