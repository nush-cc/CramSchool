from django.db import models
from django.conf import settings


class QuestionType(models.Model):
    """題目類型"""

    name = models.CharField(max_length=50, verbose_name="類型名稱")

    class Meta:
        verbose_name = "題目類型"
        verbose_name_plural = "題目類型"

    def __str__(self):
        return self.name


class Question(models.Model):
    """題目"""

    DIFFICULTY_CHOICES = [
        (1, "簡單"),
        (2, "中等"),
        (3, "困難"),
    ]

    course = models.ForeignKey(
        "courses.Course", on_delete=models.CASCADE, verbose_name="所屬課程"
    )
    chapter = models.ForeignKey(
        "courses.Chapter",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        verbose_name="所屬章節",
    )
    question_type = models.ForeignKey(
        QuestionType, on_delete=models.CASCADE, verbose_name="題目類型"
    )
    content = models.TextField(verbose_name="題目內容")
    explanation = models.TextField(blank=True, verbose_name="詳解")
    difficulty = models.IntegerField(
        choices=DIFFICULTY_CHOICES, default=2, verbose_name="難度"
    )
    order = models.IntegerField(verbose_name="題目順序")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="建立時間")
    # 統計相關欄位
    correct_count = models.IntegerField(default=0, verbose_name="正確作答人數")
    total_attempts = models.IntegerField(default=0, verbose_name="總作答次數")
    correct_rate = models.FloatField(default=0.0, verbose_name="正確率 (%)")
    last_stat_updated = models.DateTimeField(
        auto_now=True, verbose_name="最後統計更新時間"
    )

    class Meta:
        verbose_name = "題目"
        verbose_name_plural = "題目"
        ordering = ["order"]

    def __str__(self):
        if self.chapter:
            return f"{self.chapter} - Q{self.order}"
        return f"{self.course} - Q{self.order}"


class Choice(models.Model):
    """選項"""

    question = models.ForeignKey(
        Question,
        on_delete=models.CASCADE,
        related_name="choices",
        verbose_name="所屬題目",
    )
    content = models.CharField(max_length=500, verbose_name="選項內容")
    is_correct = models.BooleanField(default=False, verbose_name="是否正確")
    order = models.IntegerField(verbose_name="順序")

    class Meta:
        verbose_name = "選項"
        verbose_name_plural = "選項"
        ordering = ["order"]

    def __str__(self):
        letter = chr(64 + self.order)  # 1->A, 2->B, 3->C, 4->D
        return f"{self.question} - 選項{letter}"
