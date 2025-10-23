from django.contrib import admin
from .models import Enrollment, StudentAnswer


@admin.register(Enrollment)
class EnrollmentAdmin(admin.ModelAdmin):
    list_display = ["student", "course", "enrolled_at"]
    list_filter = ["course", "enrolled_at"]
    search_fields = ["student__username", "course__title"]


@admin.register(StudentAnswer)
class StudentAnswerAdmin(admin.ModelAdmin):
    list_display = ["student", "question", "is_correct", "answered_at"]
    list_filter = ["is_correct", "question__course"]
    search_fields = ["student__username"]
