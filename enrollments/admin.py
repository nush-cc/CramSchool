# enrollments/admin.py

from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.db.models import Q
from enrollments.models import Enrollment, StudentAnswer


@admin.register(Enrollment)
class EnrollmentAdmin(admin.ModelAdmin):
    """選課記錄管理"""

    list_display = [
        "student_name",
        "course_title",
        "status_badge",
        "enrolled_at",
        "approved_at",
        "action_buttons",
    ]

    list_filter = ["status", "course__subject", "course__grade", "enrolled_at"]

    search_fields = ["student__username", "student__email", "course__title"]

    readonly_fields = ["student", "course", "enrolled_at", "approved_at", "status_info"]

    fieldsets = (
        ("選課資訊", {"fields": ("student", "course", "enrolled_at", "status_info")}),
        ("審核資訊", {"fields": ("status", "approved_at")}),
    )

    def student_name(self, obj):
        """顯示學生名稱"""
        return f"{obj.student.username} ({obj.student.email})"

    student_name.short_description = "學生"

    def course_title(self, obj):
        """顯示課程標題"""
        return f"{obj.course.grade} - {obj.course.subject.name} - {obj.course.title}"

    course_title.short_description = "課程"

    def status_badge(self, obj):
        """顯示狀態徽章"""
        if obj.status == "approved":
            color = "green"
            label = "✓ 已核准"
        elif obj.status == "pending":
            color = "orange"
            label = "⏳ 待審核"
        elif obj.status == "rejected":
            color = "red"
            label = "✗ 已拒絕"
        else:
            color = "gray"
            label = "未知"

        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>', color, label
        )

    status_badge.short_description = "狀態"

    def status_info(self, obj):
        """顯示詳細的狀態資訊"""
        info = f"目前狀態: <strong>{obj.get_status_display()}</strong>"
        if obj.status == "approved" and obj.approved_at:
            info += f"<br>核准時間: {obj.approved_at.strftime('%Y-%m-%d %H:%M:%S')}"
        return format_html(info)

    status_info.short_description = "狀態詳情"

    def action_buttons(self, obj):
        """顯示操作按鈕"""
        if obj.status == "pending":
            approve_url = reverse("admin:enrollments_enrollment_change", args=[obj.id])
            return format_html('<a class="button" href="{}">審核</a>', approve_url)
        return "-"

    action_buttons.short_description = "操作"

    def get_queryset(self, request):
        """只顯示待審核和已核准的"""
        qs = super().get_queryset(request)
        # 管理員看全部，教師只看自己課程的
        if request.user.role == "teacher":
            qs = qs.filter(course__teacher=request.user)
        return qs

    # 自訂 change_list 顯示待審核的數量
    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        pending_count = Enrollment.objects.filter(status="pending").count()
        extra_context["pending_count"] = pending_count
        return super().changelist_view(request, extra_context=extra_context)


@admin.register(StudentAnswer)
class StudentAnswerAdmin(admin.ModelAdmin):
    """作答記錄管理"""

    list_display = [
        "student",
        "question",
        "selected_choice",
        "is_correct",
        "answered_at",
    ]

    list_filter = ["is_correct", "answered_at", "question__course"]

    search_fields = ["student__username", "question__content"]

    readonly_fields = [
        "student",
        "question",
        "selected_choice",
        "answer_text",
        "is_correct",
        "answered_at",
    ]
