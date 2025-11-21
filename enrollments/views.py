# enrollments/views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from courses.models import Course
from enrollments.models import Enrollment


def extract_chapter_number(title):
    """從標題中提取章節號用於排序 - 支持漢字數字"""
    # 漢字數字對應表
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
        # 找 '第' 的位置
        start_idx = title.find("第")
        if start_idx == -1:
            return 999

        # 從 '第' 後面開始取數字（支持漢字或阿拉伯數字）
        i = start_idx + 1
        num_str = ""
        while i < len(title) and title[i] in chinese_to_num:
            num_str += title[i]
            i += 1

        # 檢查後面是否有 '章'
        if num_str and i < len(title) and title[i] == "章":
            # 將漢字數字轉換為阿拉伯數字
            num = 0
            for char in num_str:
                if char in chinese_to_num:
                    num = num * 10 + chinese_to_num[char]
            return num
        else:
            return 999

    except Exception:
        return 999


@login_required
def enrollment_list(request):
    """選課系統首頁 - 按科目+年級分組"""

    if request.user.placement_test_score is None:
        # 還沒完成預先測驗
        return render(request, "courses/course_list_no_placement_test.html", {})

    # 只顯示一般課程，排除預先測驗
    # 不使用 order_by，讓 Python 層面來排序
    courses = Course.objects.filter(
        course_type="regular", is_active=True
    ).select_related("subject", "grade", "teacher")

    # 獲取當前學生已選的課程（包含待審核和已核准）
    if request.user.is_authenticated:
        student_enrollments = Enrollment.objects.filter(
            student=request.user
        ).values_list("course_id", flat=True)
    else:
        student_enrollments = []

    # 按 科目 + 年級 分組
    subject_groups = {}
    for course in courses:
        group_key = f"{course.grade} - {course.subject.name}"

        if group_key not in subject_groups:
            subject_groups[group_key] = {
                "grade": course.grade,
                "subject": course.subject,
                "courses": [],
                "is_enrolled": False,  # 是否已選該科目下的課程
                "pending_count": 0,  # 該科目下待審核課程數
                "approved_count": 0,  # 該科目下已核准課程數
            }

        # 檢查該課程是否已選
        is_enrolled = course.id in student_enrollments
        if is_enrolled:
            # 獲取該課程的選課狀態
            enrollment = Enrollment.objects.get(student=request.user, course=course)
            course.enrollment_status = enrollment.status

            if enrollment.is_pending:
                subject_groups[group_key]["pending_count"] += 1
            elif enrollment.is_approved:
                subject_groups[group_key]["approved_count"] += 1
                subject_groups[group_key]["is_enrolled"] = True
        else:
            course.enrollment_status = None

        subject_groups[group_key]["courses"].append(course)

    # 將分組轉換為列表並排序 - 按年級order排序，再按科目名稱排序
    grouped_courses = []
    for group_key in sorted(
        subject_groups.keys(),
        key=lambda x: (
            subject_groups[x]["grade"].order,
            subject_groups[x]["subject"].name,
        ),
    ):
        # 對每個分組內的課程按章節號排序
        courses_list = subject_groups[group_key]["courses"]
        courses_list.sort(key=lambda c: extract_chapter_number(c.title))
        subject_groups[group_key]["courses"] = courses_list

        grouped_courses.append({"key": group_key, "data": subject_groups[group_key]})

    context = {
        "grouped_courses": grouped_courses,
    }

    return render(request, "enrollments/enrollment_list.html", context)


@login_required
def my_courses(request):
    """我的課程 - 顯示已選的課程"""

    # 獲取已核准的選課
    approved_enrollments = Enrollment.objects.filter(
        student=request.user, status="approved"
    ).select_related("course__subject", "course__grade", "course__teacher")

    # 獲取待審核的選課
    pending_enrollments = Enrollment.objects.filter(
        student=request.user, status="pending"
    ).select_related("course__subject", "course__grade", "course__teacher")

    # 按科目+年級分組已選課程
    approved_groups = {}
    for enrollment in approved_enrollments:
        course = enrollment.course
        group_key = f"{course.grade} - {course.subject.name}"

        if group_key not in approved_groups:
            approved_groups[group_key] = []

        approved_groups[group_key].append(enrollment)

    approved_grouped = [
        {"key": key, "enrollments": approved_groups[key]}
        for key in sorted(approved_groups.keys())
    ]

    context = {
        "approved_grouped": approved_grouped,
        "pending_enrollments": pending_enrollments,
    }

    return render(request, "enrollments/my_courses.html", context)


@login_required
def enroll_course(request, course_id):
    """申請選課 - 一次申請該年級同科目的所有課程"""

    course = get_object_or_404(Course, pk=course_id, is_active=True)

    # 獲取該課程的年級和科目
    grade = course.grade
    subject = course.subject

    # 找出該年級同科目的所有課程
    courses_to_enroll = Course.objects.filter(
        grade=grade,
        subject=subject,
        course_type="regular",
        is_active=True,
    )

    if not courses_to_enroll.exists():
        return render(
            request,
            "enrollments/enrollment_error.html",
            {
                "message": "找不到可申請的課程",
                "course": course,
            },
        )

    # 檢查是否已經完全選過這些課程
    existing_enrollments = Enrollment.objects.filter(
        student=request.user, course__in=courses_to_enroll
    )

    all_approved = (
        existing_enrollments.filter(status="approved").count()
        == courses_to_enroll.count()
    )

    if all_approved:
        # 已全部核准，不能重複申請
        return render(
            request,
            "enrollments/enrollment_error.html",
            {
                "message": f"您已選過 {grade} {subject.name} 的所有課程",
                "course": course,
            },
        )

    # 刪除已拒絕的舊申請
    Enrollment.objects.filter(
        student=request.user, course__in=courses_to_enroll, status="rejected"
    ).delete()

    # 申請該年級同科目的所有課程
    enrolled_count = 0
    pending_count = 0

    for course_to_enroll in courses_to_enroll:
        existing = Enrollment.objects.filter(
            student=request.user, course=course_to_enroll
        ).first()

        if existing:
            if existing.is_approved:
                # 已核准，跳過
                enrolled_count += 1
            elif existing.is_pending:
                # 待審核，跳過
                pending_count += 1
            elif existing.is_rejected:
                # 已拒絕，重新申請
                existing.delete()
                Enrollment.objects.create(
                    student=request.user, course=course_to_enroll, status="pending"
                )
                pending_count += 1
        else:
            # 新申請
            Enrollment.objects.create(
                student=request.user, course=course_to_enroll, status="pending"
            )
            pending_count += 1

    return render(
        request,
        "enrollments/enrollment_success.html",
        {
            "course": course,
            "grade": grade,
            "subject": subject,
            "total_courses": courses_to_enroll.count(),
            "enrolled_count": enrolled_count,
            "pending_count": pending_count,
            "message": f"已申請 {grade} {subject.name} 的全部課程，請等候審核",
        },
    )


@login_required
def approve_enrollment(request, enrollment_id):
    """核准或拒絕選課（管理員功能）- 會連同該分組的所有課程"""

    if request.user.role not in ["admin", "teacher"]:
        return redirect("admin:enrollments_enrollment_changelist")

    enrollment = get_object_or_404(Enrollment, pk=enrollment_id)

    # 找出同一分組的所有 enrollment（同學生、同年級、同科目）
    group_enrollments = Enrollment.objects.filter(
        student=enrollment.student,
        course__grade=enrollment.course.grade,
        course__subject=enrollment.course.subject,
        status="pending",
    )

    # 支持 GET 和 POST - GET 用於 admin 按鈕，POST 用於 form
    action = request.GET.get("action") or request.POST.get("action")

    if request.method == "POST" or request.method == "GET":
        if action == "reject":
            # 拒絕該分組的所有課程
            group_enrollments.update(status="rejected")
        else:
            # 核准該分組的所有課程
            now = timezone.now()
            for e in group_enrollments:
                e.status = "approved"
                e.approved_at = now
                e.save()

    return redirect("admin:enrollments_enrollment_changelist")


@login_required
def cancel_enrollment(request, enrollment_id):
    """取消選課申請"""

    enrollment = get_object_or_404(Enrollment, pk=enrollment_id)

    # 只有學生本人或管理員可以取消
    if request.user != enrollment.student and request.user.role != "admin":
        return redirect("my_courses")

    # 只有待審核的才能取消
    if enrollment.is_pending:
        enrollment.delete()

    return redirect("my_courses")


@login_required
def drop_course(request, enrollment_id):
    """退選課程"""

    enrollment = get_object_or_404(Enrollment, pk=enrollment_id)

    # 只有學生本人或管理員可以退選
    if request.user != enrollment.student and request.user.role != "admin":
        return redirect("my_courses")

    # 只有已核准的才能退選
    if enrollment.is_approved:
        enrollment.delete()

    return redirect("my_courses")


@login_required
def teacher_approval_list(request):
    """老師核准管理頁面 - 顯示待審核的申請按分組"""

    # 檢查是否為老師或管理員
    user_role = getattr(request.user, "role", None)
    if user_role not in ["teacher", "admin"]:
        return redirect("/")

    # 獲取待審核的申請
    if user_role == "teacher":
        # 老師只看自己課程的申請
        pending_enrollments = Enrollment.objects.filter(
            status="pending", course__teacher=request.user
        ).select_related(
            "student", "course__grade", "course__subject", "course__teacher"
        )
    else:
        # 管理員看全部
        pending_enrollments = Enrollment.objects.filter(
            status="pending"
        ).select_related(
            "student", "course__grade", "course__subject", "course__teacher"
        )

    # 按 (student_id, grade_id, subject_id) 分組
    grouped = {}
    for enrollment in pending_enrollments:
        key = (
            enrollment.student.id,
            enrollment.course.grade.id,
            enrollment.course.subject.id,
        )
        if key not in grouped:
            grouped[key] = {
                "student": enrollment.student,
                "grade": enrollment.course.grade,
                "subject": enrollment.course.subject,
                "enrollments": [],
            }
        grouped[key]["enrollments"].append(enrollment)

    pending_groups = list(grouped.values())

    context = {
        "pending_groups": pending_groups,
        "pending_count": sum(len(g["enrollments"]) for g in pending_groups),
    }

    return render(request, "enrollments/teacher_approval_list.html", context)


@login_required
def teacher_approve_group(request):
    """老師核准一個申請分組"""

    user_role = getattr(request.user, "role", None)
    if user_role not in ["teacher", "admin"]:
        return redirect("/")

    if request.method == "POST":
        student_id = request.POST.get("student_id")
        grade_id = request.POST.get("grade_id")
        subject_id = request.POST.get("subject_id")

        if not all([student_id, grade_id, subject_id]):
            return redirect("enrollments:teacher_approval_list")

        # 查詢該分組下的所有待審核申請
        if user_role == "teacher":
            enrollments = Enrollment.objects.filter(
                status="pending",
                student_id=student_id,
                course__grade_id=grade_id,
                course__subject_id=subject_id,
                course__teacher=request.user,
            )
        else:
            enrollments = Enrollment.objects.filter(
                status="pending",
                student_id=student_id,
                course__grade_id=grade_id,
                course__subject_id=subject_id,
            )

        # 核准所有申請
        approved_count = 0
        for enrollment in enrollments:
            enrollment.status = "approved"
            enrollment.approved_at = timezone.now()
            enrollment.save()
            approved_count += 1

        from django.contrib import messages

        messages.success(request, f"已核准 {approved_count} 筆申請")

    return redirect("enrollments:teacher_approval_list")


@login_required
def teacher_reject_group(request):
    """老師拒絕一個申請分組"""

    user_role = getattr(request.user, "role", None)
    if user_role not in ["teacher", "admin"]:
        return redirect("/")

    if request.method == "POST":
        student_id = request.POST.get("student_id")
        grade_id = request.POST.get("grade_id")
        subject_id = request.POST.get("subject_id")

        if not all([student_id, grade_id, subject_id]):
            return redirect("enrollments:teacher_approval_list")

        # 查詢該分組下的所有待審核申請
        if user_role == "teacher":
            enrollments = Enrollment.objects.filter(
                status="pending",
                student_id=student_id,
                course__grade_id=grade_id,
                course__subject_id=subject_id,
                course__teacher=request.user,
            )
        else:
            enrollments = Enrollment.objects.filter(
                status="pending",
                student_id=student_id,
                course__grade_id=grade_id,
                course__subject_id=subject_id,
            )

        # 拒絕所有申請
        rejected_count = 0
        for enrollment in enrollments:
            enrollment.status = "rejected"
            enrollment.save()
            rejected_count += 1

        from django.contrib import messages

        messages.success(request, f"已拒絕 {rejected_count} 筆申請")

    return redirect("enrollments:teacher_approval_list")
