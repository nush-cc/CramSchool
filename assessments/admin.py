from django.contrib import admin
from django.core.management import call_command
from django.utils.html import format_html
from django.urls import reverse, path
from django.template.response import TemplateResponse
from django.db.models import Count, Q
from .models import QuestionType, Question, Choice
from enrollments.models import StudentAnswer


@admin.register(QuestionType)
class QuestionTypeAdmin(admin.ModelAdmin):
    list_display = ["name"]


class ChoiceInline(admin.TabularInline):
    model = Choice
    extra = 0
    fields = ["order", "content", "is_correct"]
    ordering = ["order"]


def update_question_stats(modeladmin, request, queryset):
    """Admin action: 執行統計更新"""
    call_command("update_question_stats")
    modeladmin.message_user(request, "✓ 題目統計已更新完成！")


update_question_stats.short_description = "📊 執行題目統計與難度調整"


def update_question_stats_with_adjust(modeladmin, request, queryset):
    """Admin action: 執行統計與自動調整難度"""
    call_command("update_question_stats", "--auto-adjust")
    modeladmin.message_user(request, "✓ 題目統計已更新，難度已自動調整！")


update_question_stats_with_adjust.short_description = "📊 執行統計 + 自動調整難度"


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = [
        "content_preview",
        "course_display",
        "chapter",
        "question_type",
        "difficulty",
        "stats_display",
        "order",
    ]
    list_filter = ["question_type", "difficulty", "course"]
    search_fields = ["content"]
    inlines = [ChoiceInline]
    actions = [update_question_stats, update_question_stats_with_adjust]
    readonly_fields = [
        "correct_count",
        "total_attempts",
        "correct_rate",
        "last_stat_updated",
    ]
    change_list_template = "admin/assessments/question_changelist.html"

    def content_preview(self, obj):
        return obj.content[:50] + "..." if len(obj.content) > 50 else obj.content

    content_preview.short_description = "題目內容"

    def course_display(self, obj):
        return obj.course.title

    course_display.short_description = "課程"

    def stats_display(self, obj):
        """顯示統計資訊"""
        if obj.total_attempts == 0:
            return format_html("<span style='color: gray;'>尚無作答</span>")
        
        # 根據正確率設定顏色
        if obj.correct_rate >= 80:
            color = "green"
            emoji = "✓"
        elif obj.correct_rate < 30:
            color = "red"
            emoji = "✗"
        else:
            color = "orange"
            emoji = "≈"
        
        # 先在 Python 層進行格式化
        rate_str = f"{obj.correct_rate:.1f}"
        
        return format_html(
            "<span style='color: {}; font-weight: bold;'>{} {}% "
            "({}/{})</span>",
            color,
            emoji,
            rate_str,
            obj.correct_count,
            obj.total_attempts,
        )

    stats_display.short_description = "正確率"
    stats_display.admin_order_field = '-correct_rate'

    def changelist_view(self, request, extra_context=None):
        """覆蓋 changelist_view 來新增統計摘要"""
        extra_context = extra_context or {}
        
        # 計算難度統計
        easy_count = Question.objects.filter(difficulty=1).count()
        medium_count = Question.objects.filter(difficulty=2).count()
        hard_count = Question.objects.filter(difficulty=3).count()
        total_count = Question.objects.count()
        
        # 計算正確率統計
        no_attempt = Question.objects.filter(total_attempts=0).count()
        high_correct = Question.objects.filter(correct_rate__gte=80).count()
        low_correct = Question.objects.filter(correct_rate__lt=30, total_attempts__gt=0).count()
        
        extra_context["difficulty_stats"] = {
            "easy": easy_count,
            "medium": medium_count,
            "hard": hard_count,
            "total": total_count,
        }
        
        extra_context["performance_stats"] = {
            "no_attempt": no_attempt,
            "high_correct": high_correct,
            "low_correct": low_correct,
        }
        
        return super().changelist_view(request, extra_context)

    fieldsets = (
        ("基本資訊", {
            "fields": ("course", "chapter", "question_type", "content", "explanation")
        }),
        ("難度設定", {
            "fields": ("difficulty", "order")
        }),
        ("統計資訊", {
            "fields": (
                "correct_count",
                "total_attempts",
                "correct_rate",
                "last_stat_updated",
            ),
            "classes": ("collapse",),
        }),
    )
