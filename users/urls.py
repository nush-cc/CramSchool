from django.urls import path
from . import views

urlpatterns = [
    path("accounts/register/", views.register, name="register"),
    path("accounts/logout/", views.logout_view, name="logout"),
    path("accounts/login/", views.login_view, name="login"),
    path("accounts/profile/", views.account_settings, name="account_settings"),
]
