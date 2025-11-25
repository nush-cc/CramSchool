from django.urls import path
from . import views

app_name = "courses"

urlpatterns = [
    path("teacher/courses/", views.teacher_course_list, name="teacher_course_list"),
    path("courselist/", views.course_list, name="course_list"),
    path("course/detail/<int:pk>/", views.course_detail, name="course_detail"),
    path("create/create", views.course_create, name="course_create"),
    path("course/edit/<int:pk>/", views.course_edit, name="course_edit"),
    path("course/delete/<int:pk>/", views.course_delete, name="course_delete"),
    path("course/<int:pk>/qa", views.course_qa_chat, name="course_qa_chat"),
    path("course/<int:pk>/qa/api", views.course_qa_api, name="course_qa_api"),
    path("course/<int:pk>/qa/clarify", views.course_qa_clarify, name="course_qa_clarify"),
    path("courses/<int:pk>/exam/", views.course_exam, name="course_exam"),
    path(
        "courses/<int:pk>/exam/submit/",
        views.course_exam_submit,
        name="course_exam_submit",
    ),
    path("placement-test/", views.placement_test, name="placement_test"),
    path(
        "placement-test/submit/",
        views.placement_test_submit,
        name="placement_test_submit",
    ),
    path("api/drawing/<str:drawing_id>/<int:step>/", views.get_drawing_step_image, name="drawing_step_image"),
]
