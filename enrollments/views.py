# enrollments/views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.db.models import Q, Count
from courses.models import Course, Subject, Grade
from enrollments.models import Enrollment


@login_required
def enrollment_list(request):
    """選課系統首頁 - 按科目+年級分組"""

    if request.user.placement_test_score is None:
        # 還沒完成預先測驗
        return render(request, "courses/course_list_no_placement_test.html", {})

    # 只顯示一般課程，排除預先測驗
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

    # 將分組轉換為列表並排序
    grouped_courses = []
    for group_key in sorted(subject_groups.keys()):
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
    """申請選課"""

    course = get_object_or_404(Course, pk=course_id, is_active=True)

    # 檢查是否已經選過這門課
    existing_enrollment = Enrollment.objects.filter(
        student=request.user, course=course
    ).first()

    if existing_enrollment:
        if existing_enrollment.is_approved:
            # 已核准，不能重複申請
            return render(
                request,
                "enrollments/enrollment_error.html",
                {
                    "message": "您已選過此課程",
                    "course": course,
                },
            )
        elif existing_enrollment.is_pending:
            # 待審核，不能重複申請
            return render(
                request,
                "enrollments/enrollment_error.html",
                {
                    "message": "您已申請此課程，請等候審核",
                    "course": course,
                },
            )
        elif existing_enrollment.is_rejected:
            # 已拒絕，可以重新申請
            existing_enrollment.delete()

    # 新增選課申請
    enrollment = Enrollment.objects.create(
        student=request.user, course=course, status="pending"
    )

    return render(
        request,
        "enrollments/enrollment_success.html",
        {
            "course": course,
            "message": "申請已送出，請等候審核",
        },
    )


@login_required
def approve_enrollment(request, enrollment_id):
    """核准選課（管理員功能）"""

    if request.user.role not in ["admin", "teacher"]:
        return redirect("enrollment_list")

    enrollment = get_object_or_404(Enrollment, pk=enrollment_id)

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "approve":
            enrollment.status = "approved"
            enrollment.approved_at = timezone.now()
            enrollment.save()
        elif action == "reject":
            enrollment.status = "rejected"
            enrollment.save()

    return redirect("enrollment_list")


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
