from django.contrib import admin
from .models import QuestionType, Question, Choice


@admin.register(QuestionType)
class QuestionTypeAdmin(admin.ModelAdmin):
    list_display = ["name"]


class ChoiceInline(admin.TabularInline):
    model = Choice
    extra = 0
    fields = ["order", "content", "is_correct"]
    ordering = ["order"]


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = [
        "content_preview",
        "course_display",
        "chapter",
        "question_type",
        "difficulty",
        "order",
    ]
    list_filter = ["question_type", "difficulty", "course"]
    search_fields = ["content"]
    inlines = [ChoiceInline]

    def content_preview(self, obj):
        return obj.content[:50] + "..." if len(obj.content) > 50 else obj.content

    content_preview.short_description = "題目內容"

    def course_display(self, obj):
        return obj.course.title

    course_display.short_description = "課程"
