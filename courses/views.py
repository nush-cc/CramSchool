from django.http import Http404, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Q, Prefetch
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from .models import Course, Subject, Grade, Chapter
from .forms import CourseForm
from assessments.models import Question, Choice
from enrollments.models import StudentAnswer, Enrollment
from django.http import HttpResponse, Http404
from chatbot.draw_package.drawing_engine import DrawingEngine
import os
import json
from django.conf import settings
import io

LEVEL_STANDARDS = {
    (85, 100): "advanced",
    (70, 84): "standard",
    (0, 69): "basic",
}


def course_list(request):
    """課程列表 - 只顯示已選課程的科目"""

    if not request.user.is_authenticated:
        # 未登入用戶重導向到登入頁面
        return redirect("login")

    if request.user.placement_test_score is None:
        # 還沒完成預先測驗
        return render(request, "courses/course_list_no_placement_test.html", {})

    # 獲取學生已核准的選課記錄
    approved_enrollments = Enrollment.objects.filter(
        student=request.user, status="approved"
    ).select_related("course__subject", "course__grade", "course__teacher")

    # 如果學生沒有選任何課程，顯示提示頁面
    if not approved_enrollments.exists():
        return render(request, "courses/course_list_empty.html", {})

    # 直接從已核准的 enrollment 取得課程
    approved_course_ids = approved_enrollments.values_list("course_id", flat=True)
    courses = Course.objects.filter(
        id__in=approved_course_ids, is_active=True
    ).select_related("subject", "grade", "teacher")

    # 搜尋功能
    search = request.GET.get("search", "").strip()
    if search:
        courses = courses.filter(
            Q(title__icontains=search) | Q(description__icontains=search)
        )

    # 科目篩選
    subject_id = request.GET.get("subject", "").strip()
    if subject_id:
        courses = courses.filter(subject_id=subject_id)

    # 年級篩選
    grade_id = request.GET.get("grade", "").strip()
    if grade_id:
        courses = courses.filter(grade_id=grade_id)

    # 排序
    courses = courses.order_by("subject__name", "grade__order")

    # 按科目分組
    subject_groups = {}
    for course in courses:
        subject_name = course.subject.name

        if subject_name not in subject_groups:
            subject_groups[subject_name] = {
                "subject": course.subject,
                "courses": [],
            }

        subject_groups[subject_name]["courses"].append(course)

    # 轉換為有序列表
    grouped_courses = [
        {"subject_name": subject_name, "data": subject_groups[subject_name]}
        for subject_name in sorted(subject_groups.keys())
    ]

    # 獲取可篩選的科目
    subjects = Subject.objects.filter(
        id__in=courses.values_list("subject_id", flat=True)
    )
    grades = Grade.objects.all()

    # 獲取已選科目數
    enrolled_subjects = set(courses.values_list("subject_id", flat=True))
    enrolled_subjects_count = len(enrolled_subjects)

    context = {
        "grouped_courses": grouped_courses,
        "subjects": subjects,
        "grades": grades,
        "enrolled_subjects_count": enrolled_subjects_count,
        "total_courses": courses.count(),
    }

    return render(request, "courses/courses_list.html", context)


@login_required(login_url="login")
def teacher_course_list(request):
    """老師課程管理頁面 - 顯示老師自己的課程"""

    # 檢查是否為老師或管理員
    if not hasattr(request.user, "role") or request.user.role not in [
        "teacher",
        "admin",
    ]:
        messages.error(request, "只有教師和管理員可以訪問此頁面。")
        return redirect("/")

    # 獲取老師的課程
    if request.user.role == "admin":
        # 管理員可以看所有課程
        courses = Course.objects.all().select_related("subject", "grade", "teacher")
    else:
        # 老師只能看自己的課程
        courses = Course.objects.filter(teacher=request.user).select_related(
            "subject", "grade", "teacher"
        )

    # 搜尋功能
    search = request.GET.get("search", "").strip()
    if search:
        courses = courses.filter(
            Q(title__icontains=search) | Q(description__icontains=search)
        )

    # 排序
    courses = courses.order_by("-created_at")

    context = {
        "courses": courses,
        "total_courses": courses.count(),
        "is_admin": request.user.role == "admin",
    }

    return render(request, "courses/teacher_course_list.html", context)


def course_detail(request, pk):
    """課程詳情"""
    course = get_object_or_404(Course, pk=pk, is_active=True)

    # 檢查學生是否有選課權限
    if request.user.is_authenticated:
        is_enrolled = Enrollment.objects.filter(
            student=request.user, course=course, status="approved"
        ).exists()
    else:
        is_enrolled = False

    # 如果學生沒有選課，重導向到課程列表
    if not is_enrolled and not request.user.is_staff:
        return redirect("courses:course_list")

    # 排序章節 - 支持中文數字
    chapters = list(course.chapters.all())

    def extract_chapter_number(title):
        """從標題中提取章節號用於排序 - 支持 [X-Y] 格式和漢字數字"""
        import re

        chinese_to_num = {
            "零": 0,
            "一": 1,
            "二": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
        }

        try:
            # 先嘗試找 [X-Y] 格式
            match = re.search(r"\[(\d+)-(\d+)\]", title)
            if match:
                main_num = int(match.group(1))
                sub_num = int(match.group(2))
                # 返回 (主章節, 子章節) 元組用於排序
                return (main_num, sub_num)

            # 再嘗試找 第X章 格式
            start_idx = title.find("第")
            if start_idx != -1:
                i = start_idx + 1
                num_str = ""
                while i < len(title) and title[i] in chinese_to_num:
                    num_str += title[i]
                    i += 1
                if num_str and i < len(title) and title[i] == "章":
                    num = 0
                    for char in num_str:
                        if char in chinese_to_num:
                            num = num * 10 + chinese_to_num[char]
                    return (num, 0)

            return (999, 999)
        except Exception:
            return (999, 999)

    chapters.sort(key=lambda c: extract_chapter_number(c.title))

    context = {
        "course": course,
        "chapters": chapters,
    }

    return render(request, "courses/course_detail.html", context)


@login_required(login_url="login")
def course_create(request):
    """新增課程"""
    # 檢查權限
    if hasattr(request.user, "role"):
        if request.user.role not in ["teacher", "admin"]:
            messages.error(request, "只有教師和管理員可以新增課程。")
            return redirect("courses:course_list")

    # 檢查是否從課程管理頁面來
    next_page = request.GET.get("next", None)

    if request.method == "POST":
        form = CourseForm(request.POST)
        if form.is_valid():
            course = form.save(commit=False)

            # 如果沒有指派教師且用戶是教師，自動設定為當前用戶
            if (
                not course.teacher
                and hasattr(request.user, "role")
                and request.user.role == "teacher"
            ):
                course.teacher = request.user

            course.save()
            messages.success(request, "課程已成功建立！")
            # 如果有 next 參數，重定向到那裡，否則重定向到課程詳細頁面
            if next_page == "teacher_course_list":
                return redirect("courses:teacher_course_list")
            return redirect("courses:course_detail", pk=course.id)
    else:
        form = CourseForm()

    context = {
        "form": form,
        "subjects": Subject.objects.all().order_by("name"),
        "grades": Grade.objects.all().order_by("id"),
        "teachers": get_teachers(),
        "next_page": next_page,
    }

    return render(request, "courses/course_form.html", context)


@login_required(login_url="login")
def course_edit(request, pk):
    """編輯課程"""
    course = get_object_or_404(Course, pk=pk)

    # 檢查權限
    is_teacher = request.user == course.teacher
    is_admin = hasattr(request.user, "role") and request.user.role == "admin"

    if not (is_teacher or is_admin):
        messages.error(request, "只有課程教師或管理員可以編輯課程。")
        return redirect("courses:course_list")

    if request.method == "POST":
        form = CourseForm(request.POST, instance=course)
        if form.is_valid():
            form.save()
            return redirect("courses:course_detail", pk=course.id)
    else:
        form = CourseForm(instance=course)

    context = {
        "form": form,
        "subjects": Subject.objects.all().order_by("name"),
        "grades": Grade.objects.all().order_by("id"),
        "teachers": get_teachers(),
    }

    return render(request, "courses/course_form.html", context)


@login_required(login_url="login")
def course_delete(request, pk):
    """刪除課程"""
    course = get_object_or_404(Course, pk=pk)

    # 檢查權限
    is_teacher = request.user == course.teacher
    is_admin = hasattr(request.user, "role") and request.user.role == "admin"

    if not (is_teacher or is_admin):
        messages.error(request, "只有課程教師或管理員可以刪除課程。")
        return redirect("courses:course_list")

    if request.method == "POST":
        course_title = course.title
        course.delete()
        messages.success(request, f'課程 "{course_title}" 已成功刪除！')
        # 重定向回課程管理頁面
        return redirect("courses:teacher_course_list")

    context = {
        "object": course,
    }

    return render(request, "courses/course_confirm_delete.html", context)


def get_teachers():
    """獲取所有教師"""
    User = Course._meta.get_field("teacher").related_model
    if hasattr(User, "role"):
        return User.objects.filter(role__in=["teacher", "admin"]).order_by("username")
    return User.objects.all().order_by("username")


@login_required(login_url="login")
def course_qa_chat(request, pk):
    """課程 AI 問答聊天頁面"""
    course = get_object_or_404(Course, pk=pk, is_active=True)

    # 取得使用者的等級
    user_level_code = "standard"
    user_level_display = "標準級"

    if request.user.is_authenticated and request.user.level:
        user_level_code = request.user.level
        level_map = {
            "advanced": "進階級",
            "standard": "標準級",
            "basic": "基礎級",
        }
        user_level_display = level_map.get(user_level_code, "標準級")

    # === [新增] 判斷科目邏輯 ===
    rag_subject = "math"
    subj_name = course.subject.name
    if (
        "自然" in subj_name
        or "理化" in subj_name
        or "生物" in subj_name
        or "地科" in subj_name
    ):
        rag_subject = "science"
    # =========================

    context = {
        "course": course,
        "user_level_code": user_level_code,
        "user_level_display": user_level_display,
        "rag_subject": rag_subject,  # <--- 傳入 Context
    }

    return render(request, "courses/course_qa_chat.html", context)


@login_required(login_url="login")
@require_http_methods(["POST"])
def course_qa_api(request, pk):
    """
    課程 AI 問答 API - 呼叫 FastAPI RAG 服務
    接收前端問題，由後端判斷科目後轉發給 FastAPI，返回答案
    """
    import requests
    import json

    # 驗證課程存在並獲取課程資訊
    course = get_object_or_404(Course, pk=pk, is_active=True)

    try:
        # 從請求中獲取資料
        data = json.loads(request.body)
        message = data.get("message", "").strip()
        history = data.get("history", [])
        search_type = data.get("search_type", "teaching")

        # 獲取重試相關參數
        is_retry = data.get("is_retry", False)
        retry_count = data.get("retry_count", 0)
        use_alternative = data.get("use_alternative", False)

        if not message:
            return JsonResponse({"error": "問題不能為空"}, status=400)

        # 決定學習風格（優先使用前端傳來的，否則使用學生等級）
        learner_style = data.get("learner_style", None)

        if not learner_style:
            # 如果前端沒有傳 learner_style，使用學生的預設等級
            learner_style_map = {
                "advanced": "進階級",
                "standard": "標準級",
                "basic": "基礎級",
            }
            learner_style = learner_style_map.get(
                request.user.level
                if hasattr(request.user, "level") and request.user.level
                else "standard",
                "標準級",
            )

        # === [關鍵修改] 自動判斷科目 ===
        # 預設為數學
        current_subject = "math"
        subj_name = course.subject.name
        # 如果科目名稱包含自然相關關鍵字，切換為 science
        if any(
            keyword in subj_name
            for keyword in ["自然", "理化", "生物", "地科", "物理", "化學"]
        ):
            current_subject = "science"
        # ============================

        # 準備送給 FastAPI 的payload
        # 注意：請確認你的 FastAPI 服務位址正確 (預設為 8001)
        fastapi_url = "http://localhost:8001/chat_with_history"

        payload = {
            "message": message,
            "subject": current_subject,  # <--- 傳送科目給 FastAPI
            "search_type": search_type,
            "learner_style": learner_style,
            "course_id": pk,
            "course_title": course.title,
            "history": history,
            "is_retry": is_retry,
            "retry_count": retry_count,
            "use_alternative": use_alternative,
        }

        # 呼叫 FastAPI
        response = requests.post(
            fastapi_url,
            json=payload,
            timeout=60,  # 60秒 timeout
        )

        if response.status_code == 200:
            result = response.json()
            return JsonResponse(result)
        else:
            error_msg = f"FastAPI 回應錯誤: {response.status_code}"
            try:
                error_detail = response.json()
                error_msg = error_detail.get("detail", error_msg)
            except Exception:
                pass

            return JsonResponse({"error": error_msg}, status=500)

    except requests.Timeout:
        return JsonResponse({"error": "AI 服務回應超時，請稍後再試"}, status=504)
    except requests.ConnectionError:
        return JsonResponse(
            {"error": "無法連接到 AI 服務，請確認服務是否啟動"}, status=503
        )
    except json.JSONDecodeError:
        return JsonResponse({"error": "請求格式錯誤"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"發生錯誤: {str(e)}"}, status=500)


@login_required(login_url="login")
@require_http_methods(["POST"])
def course_qa_clarify(request, pk):
    """
    深入追問 API - 呼叫 FastAPI 的 clarify endpoint
    當學生點選某段文字要深入了解時使用
    """
    import requests
    import json

    # 驗證課程存在
    get_object_or_404(Course, pk=pk, is_active=True)

    try:
        data = json.loads(request.body)
        selected_text = data.get("selected_text", "").strip()
        original_query = data.get("original_query", "").strip()
        original_context = data.get("original_context", "")

        if not selected_text or not original_query:
            return JsonResponse({"error": "缺少必要參數"}, status=400)

        # 決定學習風格
        learner_style_map = {
            "A": "進階級",
            "B": "標準級",
            "C": "基礎級",
        }
        learner_style = learner_style_map.get(
            request.user.level
            if hasattr(request.user, "level") and request.user.level
            else "B",
            "標準級",
        )

        # 呼叫 FastAPI clarify endpoint
        fastapi_url = "http://localhost:8001/clarify"
        payload = {
            "selected_text": selected_text,
            "original_query": original_query,
            "learner_style": learner_style,
            "original_context": original_context,
        }

        response = requests.post(fastapi_url, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            return JsonResponse(result)
        else:
            error_msg = f"FastAPI 回應錯誤: {response.status_code}"
            try:
                error_detail = response.json()
                error_msg = error_detail.get("detail", error_msg)
            except Exception:
                pass

            return JsonResponse({"error": error_msg}, status=500)

    except requests.Timeout:
        return JsonResponse({"error": "AI 服務回應超時，請稍後再試"}, status=504)
    except requests.ConnectionError:
        return JsonResponse({"error": "無法連接到 AI 服務"}, status=503)
    except json.JSONDecodeError:
        return JsonResponse({"error": "請求格式錯誤"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"發生錯誤: {str(e)}"}, status=500)


@login_required(login_url="login")
def course_exam(request, pk):
    """課程考試"""
    course = get_object_or_404(Course, pk=pk, is_active=True)

    # 從資料庫中獲取該課程的所有題目（包含選項）
    questions = (
        Question.objects.filter(course=course)
        .prefetch_related(
            Prefetch("choices", queryset=Choice.objects.order_by("order"))
        )
        .order_by("?")
    )  # 隨機排序

    # 可以在這裡調整題數
    exam_questions = list(questions[:1])  # 取前 5 題

    # # 如果題目不足 5 題
    # if len(exam_questions) < 5:
    #     messages.warning(request, f'此課程目前只有 {len(exam_questions)} 題，無法進行完整測驗。')
    #
    # if not exam_questions:
    #     messages.error(request, '此課程尚未建立題目，無法進行測驗。')
    #     return redirect('course_detail', pk=pk)

    # 為每個題目編號
    for idx, question in enumerate(exam_questions, 1):
        question.exam_number = idx

    context = {
        "course": course,
        "questions": exam_questions,
        "total_questions": len(exam_questions),
    }

    return render(request, "courses/course_exam.html", context)


@login_required(login_url="login")
def course_exam_submit(request, pk):
    """提交考試答案"""
    if request.method != "POST":
        return redirect("courses:course_exam", pk=pk)

    course = get_object_or_404(Course, pk=pk, is_active=True)

    # 獲取提交的答案
    submitted_answers = {}
    for key, value in request.POST.items():
        if key.startswith("question_"):
            question_id = int(key.replace("question_", ""))
            choice_id = int(value)
            submitted_answers[question_id] = choice_id

    # 獲取題目和正確答案
    question_ids = list(submitted_answers.keys())
    questions = Question.objects.filter(
        id__in=question_ids, course=course
    ).prefetch_related("choices")

    # 計算成績
    results = []
    correct_count = 0
    total_questions = len(questions)

    for idx, question in enumerate(questions, 1):
        question.exam_number = idx
        user_choice_id = submitted_answers.get(question.id)
        user_choice = (
            question.choices.filter(id=user_choice_id).first()
            if user_choice_id
            else None
        )
        correct_choice = question.choices.filter(is_correct=True).first()

        is_correct = user_choice and user_choice.is_correct
        if is_correct:
            correct_count += 1

        results.append(
            {
                "question": question,
                "user_choice": user_choice,
                "correct_choice": correct_choice,
                "is_correct": is_correct,
            }
        )

        # 💾 保存作答記錄到 StudentAnswer（保留所有紀錄）
        StudentAnswer.objects.create(
            student=request.user,
            question=question,
            selected_choice=user_choice,
            is_correct=is_correct,
        )

    # 計算分數
    score = (
        round((correct_count / total_questions * 100), 2) if total_questions > 0 else 0
    )

    context = {
        "course": course,
        "results": results,
        "correct_count": correct_count,
        "total_questions": total_questions,
        "score": score,
    }

    return render(request, "courses/course_exam_result.html", context)


def get_level_by_score(score):
    """根據分數判斷等級"""
    for (min_score, max_score), level in LEVEL_STANDARDS.items():
        if min_score <= score <= max_score:
            return level
    return "basic"


def get_student_placement_course(user):
    """
    根據使用者的年級和科目取得預先測驗課程
    可以根據不同邏輯調整
    """
    # 方案 A：使用預設的預先測驗
    course = Course.objects.filter(
        course_type="placement_test", is_default_placement=True, is_active=True
    ).first()

    if not course:
        raise Http404("找不到可用的預先測驗")

    return course


@login_required
def placement_test(request):
    """預先測驗頁面 - 根據學生自動分配測驗"""

    # 獲取該學生的預先測驗課程
    course = get_student_placement_course(request.user)

    LEVEL_MAP = {
        "advanced": "進階級",
        "standard": "標準級",
        "basic": "基礎級",
    }

    # 檢查學生是否已完成測驗
    if request.user.level and request.user.placement_test_completed_at:
        return render(
            request,
            "courses/placement_test_already_done.html",
            {
                "course": course,
                "level": LEVEL_MAP[request.user.level],
                "score": request.user.placement_test_score,
            },
        )

    # 獲取該課程下所有預先測驗章節的題目
    placement_chapters = Chapter.objects.filter(
        course=course, chapter_type="placement_test"
    )

    questions = (
        Question.objects.filter(chapter__in=placement_chapters)
        .select_related("question_type")
        .prefetch_related("choices")
        .order_by("order")
    )

    # 為每個題目添加考卷號
    for index, question in enumerate(questions, 1):
        question.exam_number = f"第 {index} 題"

    total_questions = questions.count()

    if total_questions == 0:
        raise Http404("預先測驗沒有題目")

    context = {
        "course": course,
        "questions": questions,
        "total_questions": total_questions,
    }

    return render(request, "courses/placement_test.html", context)


@login_required
def placement_test_submit(request):
    """提交預先測驗 - 根據學生自動分配測驗"""
    if request.method == "POST":
        # 獲取該學生的預先測驗課程
        course = get_student_placement_course(request.user)

        placement_chapters = Chapter.objects.filter(
            course=course, chapter_type="placement_test"
        )

        questions = Question.objects.filter(chapter__in=placement_chapters)

        # 保存學生的答案
        for question in questions:
            selected_choice_id = request.POST.get(f"question_{question.id}")

            if selected_choice_id:
                try:
                    choice = question.choices.get(id=selected_choice_id)
                    is_correct = choice.is_correct

                    StudentAnswer.objects.update_or_create(
                        student=request.user,
                        question=question,
                        defaults={
                            "selected_choice": choice,
                            "is_correct": is_correct,
                        },
                    )
                except:  # noqa: E722
                    pass

        # 計算成績
        total_questions = questions.count()
        correct_answers = StudentAnswer.objects.filter(
            student=request.user, question__in=questions, is_correct=True
        ).count()

        score = (correct_answers / total_questions * 100) if total_questions > 0 else 0

        # 根據分數判斷等級
        level = get_level_by_score(score)

        # 更新使用者的等級和成績
        user = request.user
        user.level = level
        user.placement_test_score = score
        user.placement_test_completed_at = timezone.now()
        user.save()

        context = {
            "course": course,
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "score": score,
            "level": level,
            "level_display": user.get_level_display(),
        }

        return render(request, "courses/placement_test_result.html", context)

    return redirect("courses:placement_test")


def get_drawing_step_image(request, drawing_id, step):
    """
    API: /courses/api/drawing/<drawing_id>/<step>/
    功能: 讀取 {drawing_id}_layout.json，即時繪製第 step 步的圖片
    """

    # 1. 定義路徑
    BASE_DIR = settings.BASE_DIR
    # 指向存放 json 的資料夾
    DRAWING_DIR = os.path.join(BASE_DIR, "chatbot", "dataset", "llama_drawing_steps")

    # 2. 尋找 layout 檔案 (對應你的截圖檔名格式)
    json_filename = f"{drawing_id}_layout.json"  # 例如 2907_layout.json
    json_path = os.path.join(DRAWING_DIR, json_filename)

    if not os.path.exists(json_path):
        # 如果找不到，嘗試找沒有 _layout 後綴的 (以防萬一)
        json_path = os.path.join(DRAWING_DIR, f"{drawing_id}.json")
        if not os.path.exists(json_path):
            raise Http404(f"找不到繪圖資料: {drawing_id}")

    try:
        # 3. 讀取 JSON
        with open(json_path, "r", encoding="utf-8") as f:
            layout_data = json.load(f)

        # 4. 初始化繪圖引擎
        # 注意：這裡可以傳入 canvas_size，如果 json 裡有寫，就用 json 的，否則預設
        width = 600
        height = 400
        if "canvas_size" in layout_data:
            width, height = layout_data["canvas_size"]

        engine = DrawingEngine(width=width, height=height)

        # 5. 計算步驟 (前端傳 1-based，轉為 0-based)
        try:
            step_index = int(step) - 1
        except ValueError:
            step_index = 0

        if step_index < 0:
            step_index = 0

        # 確保不超過總步數
        total_steps = len(layout_data.get("steps", []))
        if step_index >= total_steps:
            step_index = total_steps - 1

        # 6. 渲染圖片 (render_specific_step 會畫出 0 到 step_index 的所有內容)
        pil_image = engine.render_specific_step(layout_data, step_index)

        # 7. 將圖片轉為 Bytes 回傳 (不存檔)
        img_io = io.BytesIO()
        pil_image.save(img_io, format="PNG")
        img_io.seek(0)

        return HttpResponse(img_io, content_type="image/png")

    except Exception as e:
        print(f"繪圖引擎錯誤: {e}")
        # 在開發模式下，可以考慮回傳錯誤訊息圖片，這裡先回傳 404
        raise Http404("Error generating image")
