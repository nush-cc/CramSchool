from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """自訂使用者模型"""

    ROLE_CHOICES = [
        ("student", "學生"),
        ("teacher", "教師"),
        ("admin", "管理員"),
    ]

    phone = models.CharField(max_length=20, blank=True, verbose_name="電話號碼")

    role = models.CharField(
        max_length=10, choices=ROLE_CHOICES, default="student", verbose_name="角色"
    )

    # grade = models.ForeignKey(
    #     'courses.Grade',
    #     on_delete=models.SET_NULL,
    #     null=True,
    #     blank=True,
    #     verbose_name='所屬年級',
    #     help_text='學生的年級,教師和管理員為空'
    # )

    created_at = models.DateTimeField(auto_now_add=True, verbose_name="建立時間")

    class Meta:
        verbose_name = "使用者"
        verbose_name_plural = "使用者"

    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"

    def is_student(self):
        """是否為學生"""
        return self.role == "student"

    def is_teacher(self):
        """是否為教師"""
        return self.role == "teacher"

    def is_admin(self):
        """是否為管理員"""
        return self.role == "admin"
