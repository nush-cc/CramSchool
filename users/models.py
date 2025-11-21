from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """自訂使用者模型"""

    ROLE_CHOICES = [
        ("student", "學生"),
        ("teacher", "教師"),
        ("admin", "管理員"),
    ]

    LEVEL_CHOICES = [
        ("advanced", "進階級"),
        ("standard", "標準級"),
        ("basic", "基礎級"),
    ]

    phone = models.CharField(max_length=20, blank=True, verbose_name="電話號碼")
    role = models.CharField(
        max_length=10, choices=ROLE_CHOICES, default="student", verbose_name="角色"
    )
    grade = models.ForeignKey(
        "courses.Grade",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        verbose_name="年級",
    )
    level = models.CharField(
        max_length=20,
        choices=LEVEL_CHOICES,
        null=True,
        blank=True,
        verbose_name="學生等級",
    )
    placement_test_score = models.FloatField(
        null=True, blank=True, verbose_name="預先測驗成績"
    )
    placement_test_completed_at = models.DateTimeField(
        null=True, blank=True, verbose_name="完成預先測驗時間"
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="建立時間")

    class Meta:
        verbose_name = "使用者"
        verbose_name_plural = "使用者"

    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"

    def is_student(self):
        return self.role == "student"

    def is_teacher(self):
        return self.role == "teacher"

    def is_admin(self):
        return self.role == "admin"

    def save(self, *args, **kwargs):
        """保存時自動根據角色設定 is_staff"""
        if self.role in ["teacher", "admin"]:
            self.is_staff = True
        else:
            self.is_staff = False
        super().save(*args, **kwargs)
