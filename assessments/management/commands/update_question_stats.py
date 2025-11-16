"""
更新題目統計資訊（正確率、難度調整）
統計方式：每位學生只計最新的一次作答
"""

from django.core.management.base import BaseCommand
from django.db.models import Count, Q
from assessments.models import Question
from enrollments.models import StudentAnswer
from django.contrib.auth import get_user_model

User = get_user_model()


class Command(BaseCommand):
    help = "更新所有題目的統計資訊（正確率）並自動調整難度"

    def add_arguments(self, parser):
        parser.add_argument(
            "--auto-adjust",
            action="store_true",
            help="是否自動調整難度",
        )

    def handle(self, *args, **options):
        auto_adjust = options.get("auto_adjust", False)
        
        self.stdout.write(self.style.SUCCESS("開始更新題目統計..."))
        
        questions = Question.objects.all()
        updated_count = 0
        adjusted_count = 0
        
        for question in questions:
            # 計算該題的作答統計
            # 方式：計算所有作答記錄（不去重，包括重複作答）
            all_answers = StudentAnswer.objects.filter(question=question)
            
            total = all_answers.count()
            correct = all_answers.filter(is_correct=True).count()
            
            # 計算正確率（百分比）
            correct_rate = (correct / total * 100) if total > 0 else 0.0
            
            # 更新題目統計欄位
            question.correct_count = correct
            question.total_attempts = total
            question.correct_rate = round(correct_rate, 2)
            question.save(update_fields=[
                "correct_count", "total_attempts", "correct_rate"
            ])
            
            updated_count += 1
            
            # 自動調整難度
            if auto_adjust and total >= 5:  # 至少要有5次作答才調整
                old_difficulty = question.difficulty
                
                if correct_rate < 30 and question.difficulty < 3:
                    # 正確率低於30% → 難度提升
                    question.difficulty += 1
                    adjusted_count += 1
                    self.stdout.write(
                        f"  題目 #{question.id} 難度提升 "
                        f"({old_difficulty} → {question.difficulty}), "
                        f"正確率: {correct_rate:.1f}% ({correct}/{total})"
                    )
                elif correct_rate > 80 and question.difficulty > 1:
                    # 正確率高於80% → 難度降低
                    question.difficulty -= 1
                    adjusted_count += 1
                    self.stdout.write(
                        f"  題目 #{question.id} 難度降低 "
                        f"({old_difficulty} → {question.difficulty}), "
                        f"正確率: {correct_rate:.1f}% ({correct}/{total})"
                    )
                else:
                    self.stdout.write(
                        f"  題目 #{question.id} 難度保持 "
                        f"(難度: {question.difficulty}), "
                        f"正確率: {correct_rate:.1f}% ({correct}/{total})"
                    )
                
                question.save()
        
        
        self.stdout.write(self.style.SUCCESS(f"\n✓ 統計完成！"))
        self.stdout.write(f"  已更新 {updated_count} 題")
        if auto_adjust:
            self.stdout.write(f"  已調整難度 {adjusted_count} 題")
