import random
from textwrap import dedent
import os

class ODEProblemGenerator:
    def __init__(self, num_variants=30):
        self.num_variants = num_variants
        self.themes = {
            "Дифференциальные уравнения с разделяющимися переменными": [
                self._sep_vars_1,
                self._sep_vars_2,
                self._sep_vars_3,
                self._sep_vars_4
            ],
            "Линейные дифференциальные уравнения первого порядка": [
                self._linear_first_order_1,
                self._linear_first_order_2,
                self._linear_first_order_3,
                self._linear_first_order_4
            ],
            "Однородные дифференциальные уравнения": [
                self._homogeneous_1,
                self._homogeneous_2,
                self._homogeneous_3,
                self._homogeneous_4
            ],
            "Линейные дифференциальные уравнения второго порядка с постоянными коэффициентами": [
                self._linear_second_order_const_1,
                self._linear_second_order_const_2,
                self._linear_second_order_const_3,
                self._linear_second_order_const_4
            ],
            "Уравнения в полных дифференциалах": [
                self._exact_equation_1,
                self._exact_equation_2,
                self._exact_equation_3,
                self._exact_equation_4
            ]
        }
    
    def generate_variants(self, filename="control_work.tex"):
        with open(filename, 'w', encoding='utf-8') as f:
            self._write_preamble(f)
            
            for variant in range(1, self.num_variants + 1):
                self._write_variant(f, variant)
                
            self._write_end(f)
    
    def _write_preamble(self, f):
        preamble = dedent(r"""
        \documentclass[12pt]{article}
        \usepackage{fontspec}
        \setmainfont{Liberation Serif} % Или другой шрифт, поддерживающий кириллицу
        \usepackage{polyglossia}
        \setdefaultlanguage{russian}
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{geometry}
        \usepackage{enumitem}
        \geometry{a4paper, top=20mm, bottom=20mm, left=20mm, right=20mm}

        \newcommand{\variant}[1]{
            \begin{center}
                \textbf{Вариант #1}
            \end{center}
        }

        \begin{document}
        \begin{center}
            \Large\textbf{Контрольная работа по обыкновенным дифференциальным уравнениям} \\
            \normalsize
        \end{center}
        """)
        f.write(preamble.strip() + '\n')
    
    def _write_variant(self, f, variant_num):
        f.write(f"\\variant{{{variant_num}}}\n")
        f.write("\\begin{enumerate}\n")
        
        for theme, problems in self.themes.items():
            f.write(f"\\item {theme}\n")
            f.write("\\begin{enumerate}[label={}]\n")
            
            # Выбираем случайную задачу из каждой темы
            selected_problems = random.sample(problems, k=1)
            for i, problem in enumerate(selected_problems, 1):
                problem_text = problem()
                f.write(f"\\item {problem_text}\n")
            
            f.write("\\end{enumerate}\n")
        
        f.write("\\end{enumerate}\n")
        f.write("\\newpage\n")
    
    def _write_end(self, f):
        f.write(dedent(r"""
        \end{document}
        """))
    
    # Методы для генерации задач по темам (остаются без изменений)
    def _sep_vars_1(self):
        a = random.randint(1, 5)
        b = random.randint(1, 5)
        return f"Решить дифференциальное уравнение с разделяющимися переменными: $y' = {a}x^{b}y$"
    
    def _sep_vars_2(self):
        a = random.randint(2, 5)
        return f"Найти общее решение уравнения: $(1 + x^2)y' - {a}xy = 0$"
    
    def _sep_vars_3(self):
        a = random.randint(1, 5)
        b = random.randint(1, 5)
        return f"Решить уравнение: $y'\\sin x = y\\ln y$, $y(\\pi/{a}) = e^{b}$"
    
    def _sep_vars_4(self):
        a = random.randint(1, 5)
        return f"Найти частное решение: $y' = {a}xy^2$, $y(0) = 1$"
    
    def _linear_first_order_1(self):
        a = random.randint(1, 5)
        b = random.randint(1, 5)
        return f"Решить линейное уравнение: $y' + {a}y = e^{b}x$"
    
    def _linear_first_order_2(self):
        a = random.randint(1, 5)
        return f"Найти общее решение: $y' - \\frac{{y}}{{x}} = x^{a}\\cos x$"
    
    def _linear_first_order_3(self):
        a = random.randint(1, 5)
        b = random.randint(1, 5)
        return f"Решить задачу Коши: $y' + \\frac{{{a}y}}{{x}} = x^{b}$, $y(1) = 1$"
    
    def _linear_first_order_4(self):
        a = random.randint(1, 5)
        return f"Решить уравнение: $(x + 1)y' - {a}y = (x + 1)^{a + 1}$"
    
    def _homogeneous_1(self):
        a = random.randint(1, 5)
        return f"Решить однородное уравнение: $y' = \\frac{{{a}y}}{{x}} + \\cos\\left(\\frac{{y}}{{x}}\\right)$"
    
    def _homogeneous_2(self):
        a = random.randint(1, 5)
        b = random.randint(1, 5)
        return f"Найти общее решение: $xy' = y + \\sqrt{{{a}x^2 + {b}y^2}}$"
    
    def _homogeneous_3(self):
        a = random.randint(1, 5)
        return f"Решить уравнение: $y' = \\frac{{x + {a}y}}{{y - {a}x}}$"
    
    def _homogeneous_4(self):
        return "Решить однородное уравнение: $(x^2 + y^2)dx - 2xydy = 0$"
    
    def _linear_second_order_const_1(self):
        a = random.randint(1, 5)
        b = random.randint(1, 5)
        return f"Решить уравнение: $y'' + {a}y' + {b}y = 0$"
    
    def _linear_second_order_const_2(self):
        a = random.randint(1, 5)
        return f"Найти общее решение: $y'' - {a}^2y = e^{a}x$"
    
    def _linear_second_order_const_3(self):
        a = random.randint(1, 5)
        b = random.randint(1, 5)
        return f"Решить задачу Коши: $y'' + {a}y = 0$, $y(0) = 1$, $y'(0) = {b}$"
    
    def _linear_second_order_const_4(self):
        a = random.randint(1, 5)
        return f"Решить уравнение: $y'' + {a}y' = x^2$"
    
    def _exact_equation_1(self):
        a = random.randint(1, 5)
        return f"Решить уравнение в полных дифференциалах: $(x + {a}y)dx + ({a}x + y)dy = 0$"
    
    def _exact_equation_2(self):
        return "Проверить, что уравнение является уравнением в полных дифференциалах и решить его: $(2x + 3y)dx + (3x + 2y)dy = 0$"
    
    def _exact_equation_3(self):
        a = random.randint(1, 5)
        return f"Решить уравнение: $(x^2 + y^2 + {a})dx + 2xydy = 0$"
    
    def _exact_equation_4(self):
        return "Найти общий интеграл уравнения: $(e^x + y)dx + (x + y)dy = 0$"


if __name__ == "__main__":
    num_variants = int(input("Введите количество вариантов: "))
    generator = ODEProblemGenerator(num_variants)
    generator.generate_variants()
    print(f"Сгенерировано {num_variants} вариантов в файле control_work.tex")
    os.system("xelatex ./control_work.tex")
    print("Для компиляции используется команда: xelatex control_work.tex")