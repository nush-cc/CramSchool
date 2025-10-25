from django.urls import path
from . import views

urlpatterns = [
    path('courselist/', views.course_list, name='course_list'),
    path('course/detail/<int:pk>/', views.course_detail, name='course_detail'),
    path('create/create', views.course_create, name='course_create'),
    path('course/edit/<int:pk>/', views.course_edit, name='course_edit'),
    path('course/delete/<int:pk>/', views.course_delete, name='course_delete'),
    path('course/<int:pk>/qa', views.course_qa_chat, name='course_qa_chat'),
    path('courses/<int:pk>/exam/', views.course_exam, name='course_exam'),
    path('courses/<int:pk>/exam/submit/', views.course_exam_submit, name='course_exam_submit'),
]