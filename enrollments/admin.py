# enrollments/admin.py

from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils import timezone
from enrollments.models import Enrollment, StudentAnswer


@admin.register(Enrollment)
class EnrollmentAdmin(admin.ModelAdmin):
    """選課記錄管理"""

    list_display = [
        "student_name",
        "enrollment_group",
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

    # 把 actions 選單移到上面並隱藏下面的
    actions_on_top = True
    actions_on_bottom = False

    def student_name(self, obj):
        """顯示學生名稱"""
        return f"{obj.student.username} ({obj.student.email})"

    student_name.short_description = "學生"

    def enrollment_group(self, obj):
        """顯示申請分組 - 年級+科目"""
        return f"{obj.course.grade} - {obj.course.subject.name}"

    enrollment_group.short_description = "申請分組"

    def course_title(self, obj):
        """顯示課程標題"""
        return obj.course.title

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
        """顯示操作按鈕 - 快速核准/拒絕"""
        if obj.status == "pending":
            # 核准按鈕
            approve_url = reverse("enrollments:approve_enrollment", args=[obj.id])
            # 拒絕按鈕 - 使用同一個 view，但用 query parameter
            reject_url = f"{reverse('enrollments:approve_enrollment', args=[obj.id])}?action=reject"

            buttons_html = f'''
                <a href="{approve_url}" class="button" style="background-color: #28a745; color: white; padding: 5px 10px; text-decoration: none; border-radius: 3px;">核准</a>
                <a href="{reject_url}" class="button" style="background-color: #dc3545; color: white; padding: 5px 10px; text-decoration: none; border-radius: 3px; margin-left: 5px;">拒絕</a>
            '''
            return format_html(buttons_html)
        elif obj.status == "approved":
            return format_html(
                '<span style="color: green; font-weight: bold;">✓ 已核准</span>'
            )
        elif obj.status == "rejected":
            return format_html(
                '<span style="color: red; font-weight: bold;">✗ 已拒絕</span>'
            )
        return "-"

    action_buttons.short_description = "操作"

    actions = ["approve_group_enrollments", "reject_group_enrollments"]

    def approve_group_enrollments(self, request, queryset):
        """批量核准同一申請分組的所有課程"""
        # 按照 (student, grade, subject) 分組
        groups = {}
        for enrollment in queryset.filter(status="pending"):
            key = (
                enrollment.student.id,
                enrollment.course.grade.id,
                enrollment.course.subject.id,
            )
            if key not in groups:
                groups[key] = []
            groups[key].append(enrollment)

        # 對每個分組進行核准
        total_approved = 0
        for enrollments in groups.values():
            for enrollment in enrollments:
                if enrollment.status == "pending":
                    enrollment.status = "approved"
                    enrollment.approved_at = timezone.now()
                    enrollment.save()
                    total_approved += 1

    approve_group_enrollments.short_description = "批量核准 - 同申請分組"

    def reject_group_enrollments(self, request, queryset):
        """批量拒絕同一申請分組的所有課程"""
        # 按照 (student, grade, subject) 分組
        groups = {}
        for enrollment in queryset.filter(status="pending"):
            key = (
                enrollment.student.id,
                enrollment.course.grade.id,
                enrollment.course.subject.id,
            )
            if key not in groups:
                groups[key] = []
            groups[key].append(enrollment)

        # 對每個分組進行拒絕
        total_rejected = 0
        for enrollments in groups.values():
            for enrollment in enrollments:
                if enrollment.status == "pending":
                    enrollment.status = "rejected"
                    enrollment.save()
                    total_rejected += 1

        self.message_user(request, f"已拒絕 {total_rejected} 筆申請")

    reject_group_enrollments.short_description = "批量拒絕 - 同申請分組"

    def has_module_permission(self, request):
        """允許老師和管理員訪問此模型"""
        user_role = getattr(request.user, "role", None)
        return user_role in ["teacher", "admin"]

    def has_view_permission(self, request, obj=None):
        """允許老師和管理員查看"""
        user_role = getattr(request.user, "role", None)
        return user_role in ["teacher", "admin"]

    def has_change_permission(self, request, obj=None):
        """允許老師和管理員修改"""
        user_role = getattr(request.user, "role", None)
        return user_role in ["teacher", "admin"]

    def has_delete_permission(self, request, obj=None):
        """允許老師和管理員刪除"""
        user_role = getattr(request.user, "role", None)
        return user_role in ["teacher", "admin"]

    def get_queryset(self, request):
        """只顯示待審核和已核准的"""
        qs = super().get_queryset(request)
        # 管理員看全部，教師只看自己課程的
        user_role = getattr(request.user, "role", None)
        if user_role == "teacher":
            qs = qs.filter(course__teacher=request.user)
        return qs

    # 自訂 change_list 顯示待審核的數量
    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}

        # 獲取待審核的分組（按 student + grade + subject 分組）
        pending_groups = []

        # 根據角色篩選待審核的申請
        user_role = getattr(request.user, "role", None)
        if user_role == "teacher":
            # 老師只看自己課程的待審核申請
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
                    "sample_enrollment": enrollment,  # 取第一個作為代表
                }
            grouped[key]["enrollments"].append(enrollment)

        for group_data in grouped.values():
            pending_groups.append(group_data)

        extra_context["pending_groups"] = pending_groups
        extra_context["pending_count"] = sum(
            len(g["enrollments"]) for g in pending_groups
        )

        # 改用自訂 template
        self.change_list_template = (
            "admin/enrollments/enrollment_grouped_changelist.html"
        )

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
