from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html
from .models import EducationLevel, Grade, Subject, Course, Chapter


# 隱藏不需要單獨管理的模型
admin.site.register(EducationLevel)
admin.site.register(Grade)
admin.site.register(Subject)


@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    """第一部分：國中 | 一年級 | 數學"""

    list_display = ['title', 'education_level', 'grade_name']
    list_filter = ['grade__education_level', 'grade', 'title']
    search_fields = ['title']

    def education_level(self, obj):
        return obj.grade.education_level.name
    education_level.short_description = '學制'

    def grade_name(self, obj):
        return obj.grade.name
    grade_name.short_description = '年級'

    def subject_name(self, obj):
        return obj.subject.name
    subject_name.short_description = '科目'


@admin.register(Chapter)
class ChapterAdmin(admin.ModelAdmin):
    list_display = ['course_title', 'title']  # 把 'title' 移到第二個
    list_filter = ['course__grade__education_level', 'course__grade', 'course__subject']
    search_fields = ['title', 'course__title']

    def course_title(self, obj):
        return obj.course.title
    course_title.short_description = '課程'