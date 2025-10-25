from django import forms
from .models import Course


class CourseForm(forms.ModelForm):
    """課程表單"""

    class Meta:
        model = Course
        fields = ['title', 'subject', 'grade', 'teacher', 'description', 'is_active']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '輸入課程標題',
                'maxlength': '200'
            }),
            'subject': forms.Select(attrs={
                'class': 'form-select'
            }),
            'grade': forms.Select(attrs={
                'class': 'form-select'
            }),
            'teacher': forms.Select(attrs={
                'class': 'form-select'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 6,
                'placeholder': '輸入課程詳細描述...'
            }),
            'is_active': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
        }
        labels = {
            'title': '課程標題',
            'subject': '科目',
            'grade': '年級',
            'teacher': '授課教師',
            'description': '課程描述',
            'is_active': '啟用此課程',
        }
        help_texts = {
            'title': '最多 200 個字元',
            'description': '可使用 Markdown 格式',
            'teacher': '可不指派，稍後再修改',
            'is_active': '未啟用的課程不會在課程列表中顯示',
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 使科目和年級為必填
        self.fields['subject'].required = True
        self.fields['grade'].required = True
        self.fields['teacher'].required = False
        self.fields['description'].required = False