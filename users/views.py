from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import StudentRegisterForm, LoginForm
from django.contrib.auth import logout, login, authenticate
from django.contrib.auth.decorators import login_required


def register(request):
    if request.method == 'POST':
        form = StudentRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = StudentRegisterForm()

    return render(request, 'users/register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('/')
    else:
        form = LoginForm()

    return render(request, 'users/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('/')

@login_required
def account_settings(request):
    """帳號設定/個人資料頁面"""
    return render(request, 'users/account_settings.html', {
        'user': request.user
    })