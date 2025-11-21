from django.contrib import admin
from .models import EducationLevel, Grade, Subject, Course, Chapter
from users.models import User


# 隱藏不需要單獨管理的模型
admin.site.register(EducationLevel)
admin.site.register(Grade)
admin.site.register(Subject)


@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    """第一部分：國中 | 一年級 | 數學"""

    list_display = ["title", "education_level", "grade_name", "teacher_name"]
    list_filter = ["grade__education_level", "grade", "title"]
    search_fields = ["title"]
    ordering = [
        "grade__education_level__order",
        "grade__order",
        "subject__name",
        "title",
    ]

    def education_level(self, obj):
        return obj.grade.education_level.name

    education_level.short_description = "學制"

    def grade_name(self, obj):
        return obj.grade.name

    grade_name.short_description = "年級"

    def subject_name(self, obj):
        return obj.subject.name

    subject_name.short_description = "科目"

    def teacher_name(self, obj):
        return obj.teacher.username if obj.teacher else "未指派"

    teacher_name.short_description = "授課教師"

    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        """過濾教師下拉菜單，只顯示角色為教師的使用者"""
        if db_field.name == "teacher":
            kwargs["queryset"] = User.objects.filter(role="teacher")
        return super().formfield_for_foreignkey(db_field, request, **kwargs)

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
        """只有管理員可以刪除課程"""
        user_role = getattr(request.user, "role", None)
        if user_role == "admin":
            return True
        if obj and user_role == "teacher":
            return obj.teacher == request.user
        return False

    def get_queryset(self, request):
        """老師只能看到自己的課程，管理員看全部"""
        qs = super().get_queryset(request)
        user_role = getattr(request.user, "role", None)
        if user_role == "teacher":
            return qs.filter(teacher=request.user)
        return qs

    def has_add_permission(self, request):
        """只有管理員可以新增課程"""
        user_role = getattr(request.user, "role", None)
        return user_role == "admin"


@admin.register(Chapter)
class ChapterAdmin(admin.ModelAdmin):
    list_display = ["course_title", "title"]  # 把 'title' 移到第二個
    list_filter = ["course__grade__education_level", "course__grade", "course__subject"]
    search_fields = ["title", "course__title"]
    ordering = [
        "course__grade__education_level__order",
        "course__grade__order",
        "course__subject__name",
        "course__title",
        "title",
    ]

    def course_title(self, obj):
        return obj.course.title

    course_title.short_description = "課程"

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
        """只有管理員可以刪除章節，老師只能刪除自己課程的"""
        user_role = getattr(request.user, "role", None)
        if user_role == "admin":
            return True
        if obj and user_role == "teacher":
            return obj.course.teacher == request.user
        return False

    def get_queryset(self, request):
        """老師只能看到自己的課程章節，管理員看全部"""
        qs = super().get_queryset(request)
        user_role = getattr(request.user, "role", None)
        if user_role == "teacher":
            return qs.filter(course__teacher=request.user)
        return qs

    def has_add_permission(self, request):
        """只有管理員可以新增章節"""
        user_role = getattr(request.user, "role", None)
        return user_role == "admin"
