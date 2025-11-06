# enrollments/models.py - 更新版本

from django.db import models
from django.conf import settings


class Enrollment(models.Model):
    """選課記錄"""

    STATUS_CHOICES = [
        ("pending", "待審核"),
        ("approved", "已核准"),
        ("rejected", "已拒絕"),
    ]

    student = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, verbose_name="學生"
    )
    course = models.ForeignKey(
        "courses.Course", on_delete=models.CASCADE, verbose_name="課程"
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="pending",
        verbose_name="審核狀態",
    )
    enrolled_at = models.DateTimeField(auto_now_add=True, verbose_name="申請時間")
    approved_at = models.DateTimeField(null=True, blank=True, verbose_name="核准時間")

    class Meta:
        verbose_name = "選課記錄"
        verbose_name_plural = "選課記錄"
        unique_together = ["student", "course"]

    def __str__(self):
        return f"{self.student.username} - {self.course.title} ({self.get_status_display()})"

    @property
    def is_approved(self):
        """是否已核准"""
        return self.status == "approved"

    @property
    def is_pending(self):
        """是否待審核"""
        return self.status == "pending"

    @property
    def is_rejected(self):
        """是否已拒絕"""
        return self.status == "rejected"


class StudentAnswer(models.Model):
    """作答記錄"""

    student = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, verbose_name="學生"
    )
    question = models.ForeignKey(
        "assessments.Question", on_delete=models.CASCADE, verbose_name="題目"
    )
    selected_choice = models.ForeignKey(
        "assessments.Choice",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        verbose_name="選擇的選項",
    )
    answer_text = models.TextField(blank=True, verbose_name="文字答案")
    is_correct = models.BooleanField(null=True, blank=True, verbose_name="是否正確")
    answered_at = models.DateTimeField(auto_now_add=True, verbose_name="作答時間")

    class Meta:
        verbose_name = "作答記錄"
        verbose_name_plural = "作答記錄"
        unique_together = ["student", "question"]

    def __str__(self):
        return f"{self.student.username} - {self.question}"
