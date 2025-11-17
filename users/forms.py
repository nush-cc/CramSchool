from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import User
from courses.models import Grade, EducationLevel


class StudentRegisterForm(UserCreationForm):
    email = forms.EmailField(required=True, label="電子郵件")
    # grade = forms.ModelChoiceField(
    #     queryset=Grade.objects.all(),
    #     required=True,
    #     label="年級",
    # )
    education_level = forms.ModelChoiceField(
        queryset=EducationLevel.objects.all(),
        required=True,
        label="年級",
    )

    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2", "education_level"]
        labels = {
            "username": "使用者名稱",
            "password1": "密碼",
            "password2": "確認密碼",
        }

    def save(self, commit=True):
        user = super().save(commit=False)
        user.role = "student"
        if commit:
            user.save()
        return user


class LoginForm(AuthenticationForm):
    username = forms.CharField(
        label="使用者名稱",
        widget=forms.TextInput(attrs={"placeholder": "請輸入使用者名稱"}),
    )
    password = forms.CharField(
        label="密碼", widget=forms.PasswordInput(attrs={"placeholder": "請輸入密碼"})
    )
