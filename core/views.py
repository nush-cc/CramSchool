from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from courses.models import Course
from enrollments.models import Enrollment
from assessments.models import Question

User = get_user_model()


# @login_required
def home(request):
    context = {}

    if request.user.is_authenticated:
        # 課程數量 - 只顯示啟用的課程
        context["course_count"] = Course.objects.filter(is_active=True).count()

        # 考卷/題目數量
        context["assessment_count"] = Question.objects.count()

        # 報名數量 - 當前用戶的選課記錄
        context["enrollment_count"] = Enrollment.objects.filter(
            student=request.user
        ).count()

        # 使用者總數 - 只有教師和管理員才能看
        if request.user.role in ["teacher", "admin"]:
            context["user_count"] = User.objects.count()
        else:
            context["user_count"] = 0

    return render(request, "core/home.html", context)
