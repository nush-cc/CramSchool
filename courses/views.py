from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q, Prefetch

from .models import Course, Subject, Grade
from .forms import CourseForm
from assessments.models import Question, Choice

import random


def course_list(request):
    """課程列表"""
    queryset = Course.objects.filter(is_active=True).select_related(
        'subject', 'grade', 'teacher'
    )

    # 搜尋
    search = request.GET.get('search', '').strip()
    if search:
        queryset = queryset.filter(
            Q(title__icontains=search) |
            Q(description__icontains=search)
        )

    # 篩選科目
    subject_id = request.GET.get('subject', '').strip()
    if subject_id:
        queryset = queryset.filter(subject_id=subject_id)

    # 篩選年級
    grade_id = request.GET.get('grade', '').strip()
    if grade_id:
        queryset = queryset.filter(grade_id=grade_id)

    # 排序
    queryset = queryset.order_by('-created_at')

    # 分頁
    paginator = Paginator(queryset, 12)
    page_number = request.GET.get('page')
    courses = paginator.get_page(page_number)

    context = {
        'courses': courses,
        'subjects': Subject.objects.all().order_by('name'),
        'grades': Grade.objects.all().order_by('id'),
    }

    return render(request, 'courses/courses_list.html', context)


def course_detail(request, pk):
    """課程詳情"""
    course = get_object_or_404(Course, pk=pk, is_active=True)

    context = {
        'course': course,
    }

    return render(request, 'courses/course_detail.html', context)


@login_required(login_url='login')
def course_create(request):
    """新增課程"""
    # 檢查權限
    if hasattr(request.user, 'role'):
        if request.user.role not in ['teacher', 'admin']:
            messages.error(request, '只有教師和管理員可以新增課程。')
            return redirect('course_list')

    if request.method == 'POST':
        form = CourseForm(request.POST)
        if form.is_valid():
            course = form.save(commit=False)

            # 如果沒有指派教師且用戶是教師，自動設定為當前用戶
            if not course.teacher and hasattr(request.user, 'role') and request.user.role == 'teacher':
                course.teacher = request.user

            course.save()
            messages.success(request, '課程已成功建立！')
            return redirect('course_detail', pk=course.id)
    else:
        form = CourseForm()

    context = {
        'form': form,
        'subjects': Subject.objects.all().order_by('name'),
        'grades': Grade.objects.all().order_by('id'),
        'teachers': get_teachers(),
    }

    return render(request, 'courses/course_form.html', context)


@login_required(login_url='login')
def course_edit(request, pk):
    """編輯課程"""
    course = get_object_or_404(Course, pk=pk)

    # 檢查權限
    is_teacher = request.user == course.teacher
    is_admin = hasattr(request.user, 'role') and request.user.role == 'admin'

    if not (is_teacher or is_admin):
        messages.error(request, '只有課程教師或管理員可以編輯課程。')
        return redirect('course_list')

    if request.method == 'POST':
        form = CourseForm(request.POST, instance=course)
        if form.is_valid():
            form.save()
            messages.success(request, '課程已成功更新！')
            return redirect('course_detail', pk=course.id)
    else:
        form = CourseForm(instance=course)

    context = {
        'form': form,
        'subjects': Subject.objects.all().order_by('name'),
        'grades': Grade.objects.all().order_by('id'),
        'teachers': get_teachers(),
    }

    return render(request, 'courses/course_form.html', context)


@login_required(login_url='login')
def course_delete(request, pk):
    """刪除課程"""
    course = get_object_or_404(Course, pk=pk)

    # 檢查權限
    is_teacher = request.user == course.teacher
    is_admin = hasattr(request.user, 'role') and request.user.role == 'admin'

    if not (is_teacher or is_admin):
        messages.error(request, '只有課程教師或管理員可以刪除課程。')
        return redirect('course_list')

    if request.method == 'POST':
        course_title = course.title
        course.delete()
        messages.success(request, f'課程 "{course_title}" 已成功刪除！')
        return redirect('course_list')

    context = {
        'object': course,
    }

    return render(request, 'courses/course_confirm_delete.html', context)


def get_teachers():
    """獲取所有教師"""
    User = Course._meta.get_field('teacher').related_model
    if hasattr(User, 'role'):
        return User.objects.filter(role__in=['teacher', 'admin']).order_by('username')
    return User.objects.all().order_by('username')

@login_required(login_url='login')
def course_qa_chat(request, pk):
    """課程 AI 問答聊天"""
    course = get_object_or_404(Course, pk=pk, is_active=True)

    context = {
        'course': course,
    }

    return render(request, 'courses/course_qa_chat.html', context)

@login_required(login_url='login')
def course_exam(request, pk):
    """課程考試"""
    course = get_object_or_404(Course, pk=pk, is_active=True)

    # 從資料庫中獲取該課程的所有題目（包含選項）
    questions = Question.objects.filter(
        course=course
    ).prefetch_related(
        Prefetch('choices', queryset=Choice.objects.order_by('order'))
    ).order_by('?')  # 隨機排序

    # 可以在這裡調整題數
    exam_questions = list(questions[:1])  # 取前 5 題

    # # 如果題目不足 5 題
    # if len(exam_questions) < 5:
    #     messages.warning(request, f'此課程目前只有 {len(exam_questions)} 題，無法進行完整測驗。')
    #
    # if not exam_questions:
    #     messages.error(request, '此課程尚未建立題目，無法進行測驗。')
    #     return redirect('course_detail', pk=pk)

    # 為每個題目編號
    for idx, question in enumerate(exam_questions, 1):
        question.exam_number = idx

    context = {
        'course': course,
        'questions': exam_questions,
        'total_questions': len(exam_questions),
    }

    return render(request, 'courses/course_exam.html', context)


@login_required(login_url='login')
def course_exam_submit(request, pk):
    """提交考試答案"""
    if request.method != 'POST':
        return redirect('course_exam', pk=pk)

    course = get_object_or_404(Course, pk=pk, is_active=True)

    # 獲取提交的答案
    submitted_answers = {}
    for key, value in request.POST.items():
        if key.startswith('question_'):
            question_id = int(key.replace('question_', ''))
            choice_id = int(value)
            submitted_answers[question_id] = choice_id

    # 獲取題目和正確答案
    question_ids = list(submitted_answers.keys())
    questions = Question.objects.filter(
        id__in=question_ids,
        course=course
    ).prefetch_related('choices')

    # 計算成績
    results = []
    correct_count = 0
    total_questions = len(questions)

    for idx, question in enumerate(questions, 1):
        question.exam_number = idx
        user_choice_id = submitted_answers.get(question.id)
        user_choice = question.choices.filter(id=user_choice_id).first() if user_choice_id else None
        correct_choice = question.choices.filter(is_correct=True).first()

        is_correct = user_choice and user_choice.is_correct
        if is_correct:
            correct_count += 1

        results.append({
            'question': question,
            'user_choice': user_choice,
            'correct_choice': correct_choice,
            'is_correct': is_correct,
        })

    # 計算分數
    score = round((correct_count / total_questions * 100), 2) if total_questions > 0 else 0

    context = {
        'course': course,
        'results': results,
        'correct_count': correct_count,
        'total_questions': total_questions,
        'score': score,
    }

    return render(request, 'courses/course_exam_result.html', context)