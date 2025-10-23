from django.db import models
from django.conf import settings


class EducationLevel(models.Model):
    """學制"""

    name = models.CharField(max_length=50, verbose_name="學制名稱")
    order = models.IntegerField(verbose_name="排序")

    class Meta:
        verbose_name = "學制"
        verbose_name_plural = "學制"
        ordering = ["order"]

    def __str__(self):
        return self.name


class Grade(models.Model):
    """年級"""

    education_level = models.ForeignKey(
        EducationLevel, on_delete=models.CASCADE, verbose_name="所屬學制"
    )
    name = models.CharField(max_length=50, verbose_name="年級名稱")
    order = models.IntegerField(verbose_name="排序")

    class Meta:
        verbose_name = "年級"
        verbose_name_plural = "年級"
        ordering = ["education_level__order", "order"]

    def __str__(self):
        return f"{self.education_level.name}{self.name}"


class Subject(models.Model):
    """科目"""

    name = models.CharField(max_length=50, verbose_name="科目名稱")

    class Meta:
        verbose_name = "科目"
        verbose_name_plural = "科目"

    def __str__(self):
        return self.name


class Course(models.Model):
    """課程"""

    title = models.CharField(max_length=200, verbose_name="課程標題")
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, verbose_name="科目")
    grade = models.ForeignKey(Grade, on_delete=models.CASCADE, verbose_name="年級")
    teacher = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        verbose_name="授課教師",
    )
    description = models.TextField(verbose_name="課程描述", null=True, blank=True)
    is_active = models.BooleanField(default=True, verbose_name="是否啟用")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="建立時間")

    class Meta:
        verbose_name = "課程"
        verbose_name_plural = "課程"
        ordering = ["grade", "subject"]

    def __str__(self):
        return f"{self.grade} - {self.subject.name} - {self.title}"


class Chapter(models.Model):
    """章節"""

    course = models.ForeignKey(
        Course,
        on_delete=models.CASCADE,
        related_name="chapters",
        verbose_name="所屬課程",
    )
    title = models.CharField(max_length=200, verbose_name="章節標題")
    # order = models.IntegerField(verbose_name='排序')

    class Meta:
        verbose_name = "章節"
        verbose_name_plural = "章節"

    def __str__(self):
        return self.title
