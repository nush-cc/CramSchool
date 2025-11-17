from django.http import Http404
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Q, Prefetch
from django.utils import timezone
from .models import Course, Subject, Grade, Chapter
from .forms import CourseForm
from assessments.models import Question, Choice
from enrollments.models import StudentAnswer, Enrollment

LEVEL_STANDARDS = {
    (85, 100): "A",
    (70, 84): "B",
    (0, 69): "C",
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
    ).select_related("course__subject", "course__grade")

    # 如果學生沒有選任何課程，顯示提示頁面
    if not approved_enrollments.exists():
        return render(request, "courses/course_list_empty.html", {})

    # 獲取學生已選的科目列表
    enrolled_subjects = set(
        enrollment.course.subject_id for enrollment in approved_enrollments
    )

    # 只顯示學生已選科目下的課程（排除預先測驗）
    courses = Course.objects.filter(
        course_type="regular", is_active=True, subject_id__in=enrolled_subjects
    ).select_related("subject", "grade", "teacher")

    # 搜尋功能
    search = request.GET.get("search", "").strip()
    if search:
        courses = courses.filter(
            Q(title__icontains=search) | Q(description__icontains=search)
        )

    # 科目篩選（只能篩選已選的科目）
    subject_id = request.GET.get("subject", "").strip()
    if subject_id:
        if int(subject_id) in enrolled_subjects:
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

    # 獲取可篩選的科目（只有已選的科目）
    subjects = Subject.objects.filter(id__in=enrolled_subjects)
    grades = Grade.objects.all()

    # 獲取已選科目數
    enrolled_subjects_count = len(enrolled_subjects)

    context = {
        "grouped_courses": grouped_courses,
        "subjects": subjects,
        "grades": grades,
        "enrolled_subjects_count": enrolled_subjects_count,
        "total_courses": courses.count(),
    }

    return render(request, "courses/courses_list.html", context)


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

    context = {
        "course": course,
    }

    return render(request, "courses/course_detail.html", context)


@login_required(login_url="login")
def course_create(request):
    """新增課程"""
    # 檢查權限
    if hasattr(request.user, "role"):
        if request.user.role not in ["teacher", "admin"]:
            messages.error(request, "只有教師和管理員可以新增課程。")
            return redirect("course_list")

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
            return redirect("course_detail", pk=course.id)
    else:
        form = CourseForm()

    context = {
        "form": form,
        "subjects": Subject.objects.all().order_by("name"),
        "grades": Grade.objects.all().order_by("id"),
        "teachers": get_teachers(),
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
        return redirect("course_list")

    if request.method == "POST":
        form = CourseForm(request.POST, instance=course)
        if form.is_valid():
            form.save()
            messages.success(request, "課程已成功更新！")
            return redirect("course_detail", pk=course.id)
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
        return redirect("course_list")

    if request.method == "POST":
        course_title = course.title
        course.delete()
        messages.success(request, f'課程 "{course_title}" 已成功刪除！')
        return redirect("course_list")

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
    """課程 AI 問答聊天"""
    course = get_object_or_404(Course, pk=pk, is_active=True)

    context = {
        "course": course,
    }

    return render(request, "courses/course_qa_chat.html", context)


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
        return redirect("course_exam", pk=pk)

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
    return "C"


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

    # 檢查學生是否已完成測驗
    if request.user.level and request.user.placement_test_completed_at:
        return render(
            request,
            "courses/placement_test_already_done.html",
            {
                "course": course,
                "level": request.user.level,
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

    return redirect("placement_test")
