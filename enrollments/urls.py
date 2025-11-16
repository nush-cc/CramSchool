# enrollments/urls.py

from django.urls import path
from . import views

app_name = "enrollments"

urlpatterns = [
    # 選課系統首頁 - 按科目分組顯示課程
    path("", views.enrollment_list, name="enrollment_list"),
    # 申請選課
    path("enroll/<int:course_id>/", views.enroll_course, name="enroll_course"),
    # 核准選課（管理員）
    path(
        "approve/<int:enrollment_id>/",
        views.approve_enrollment,
        name="approve_enrollment",
    ),
    # 取消申請
    path(
        "cancel/<int:enrollment_id>/", views.cancel_enrollment, name="cancel_enrollment"
    ),
    # 退選課程
    path("drop/<int:enrollment_id>/", views.drop_course, name="drop_course"),
]
