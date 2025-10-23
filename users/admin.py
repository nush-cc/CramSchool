from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    """自訂使用者管理介面"""

    # 列表頁顯示的欄位
    list_display = ["username", "email", "role", "is_active", "created_at"]

    # 可篩選的欄位
    list_filter = ["role", "is_active", "created_at"]

    # 可搜尋的欄位
    search_fields = ["username", "email", "phone"]

    # 編輯頁面的欄位分組
    fieldsets = BaseUserAdmin.fieldsets + (("額外資訊", {"fields": ("phone", "role")}),)

    # 新增使用者時的欄位分組
    add_fieldsets = BaseUserAdmin.add_fieldsets + (
        ("額外資訊", {"fields": ("email", "phone", "role", "grade")}),
    )

    # 唯讀欄位
    readonly_fields = ["created_at"]
