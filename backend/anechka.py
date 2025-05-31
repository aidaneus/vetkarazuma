import sympy
from sympy import symbols, diff, latex, Function, Eq
from IPython.display import display, Math
import random

# блок 1 - технический

random.seed(14) # число для стабильности теста, при запуске быть не должно
x = symbols('x')
y = Function('y')(x)  # y как функция от x для линейных ДУ
x_alt, y_alt = symbols('x y')  # Для остальных типов уравнений

def generate_coeff():
    return random.choice([i for i in range(-7, 8) if i != 0])

def generate_trig_arg(var):
    return generate_coeff() * var**random.randint(1, 3)

def generate_expression():
    functions = [
        lambda: (generate_coeff()*x_alt**random.randint(1,3)*y_alt**random.randint(1,3) +
                generate_coeff()*y_alt**random.randint(1,3)*x_alt**random.randint(1,3)) * generate_coeff(),
        lambda: (generate_coeff()*x_alt**random.randint(1,3) + generate_coeff()*y_alt**random.randint(1,3)) +
                generate_coeff()*x_alt**random.randint(1,3)*y_alt**random.randint(1,3),
        lambda: generate_coeff()*x_alt**random.randint(1,3) + generate_coeff()*x_alt**random.randint(1,3) +
                generate_coeff()*y_alt**random.randint(1,3) + generate_coeff()*y_alt**random.randint(1,3),
        lambda: (generate_coeff()*x_alt**random.randint(1,3) + generate_coeff()*y_alt**random.randint(1,3))**random.randint(1,3),


        lambda: (generate_coeff()*sympy.sin(generate_trig_arg(x_alt))) + (generate_coeff()*y_alt**random.randint(1,3)) * generate_coeff(),
        lambda: (generate_coeff()*x_alt**random.randint(1,3) + generate_coeff()*sympy.sin(generate_trig_arg(y_alt))) * generate_coeff(),
        lambda: (generate_coeff()*sympy.cos(generate_trig_arg(x_alt))) + (generate_coeff()*y_alt**random.randint(1,3)) * generate_coeff(),
        lambda: (generate_coeff()*x_alt**random.randint(1,3) + generate_coeff()*sympy.cos(generate_trig_arg(y_alt))) * generate_coeff(),
    ]
    return random.choice(functions)()

# блок 2 - генерация самих ДУ и взаимодействие с пользователем

# Уравнение, разрешенное относительно производной
def generate_explicit():
    F = generate_expression()
    # dy/dx = F(x,y)
    return F

# Линейное уравнение с постоянными коэффициентами
def generate_linear_ode(order):
    coeffs = [generate_coeff() for _ in range(order + 1)]
    free_term = generate_coeff()

    lhs = sum(coeff * y.diff(x, i) if i > 0 else coeff * y
             for i, coeff in enumerate(coeffs))
    # aₙy⁽ⁿ⁾ + ... + a₁y' + a₀y = b
    return Eq(lhs, free_term)

# Уравнение с разделенными переменными
def generate_separated():
    while True:
        f_x = generate_expression().subs(y_alt, 0)
        g_y = generate_expression().subs(x_alt, 0)
        if f_x != 0 and g_y != 0:
            # f(x)dx + g(y)dy = 0
            return f_x, g_y

# Уравнение с разделяющимися переменными
def generate_separable():
    while True:
        f_x = generate_expression().subs(y_alt, 0)
        g_y = generate_expression().subs(x_alt, 0)
        h_x = generate_expression().subs(y_alt, 0)
        k_y = generate_expression().subs(x_alt, 0)
        fg = f_x * g_y
        hk = h_x * k_y
        if fg != 0 and hk != 0:
            # f(x)g(y)dx + h(x)k(y)dy = 0
            return fg, hk

# Однородное уравнение
def generate_homogeneous():
    v = y_alt / x_alt  # Замена переменных: v = y/x
    expr_types = [
        # Степенная зависимость от v
        lambda: generate_coeff() * v**random.randint(1, 3),
        lambda: generate_coeff() * v**random.randint(1, 2) + generate_coeff() * v**random.randint(1, 2),

        # Тригонометрическая зависимость от v
        lambda: generate_coeff() * sympy.sin(generate_trig_arg(v)),
        lambda: generate_coeff() * sympy.cos(generate_trig_arg(v)),

        # Комбинации
        lambda: (generate_coeff() * v**2 + generate_coeff() * sympy.sin(v)),
        lambda: (generate_coeff() * sympy.cos(v) + generate_coeff() * v**3),
    ]

    f_v = random.choice(expr_types)()
    # dy/dx = f(y/x)
    return f_v.subs(v, y_alt/x_alt)

# Уравнение, приводимое к однородному
def generate_to_homogeneous():
    a1, b1, c1 = generate_coeff(), generate_coeff(), generate_coeff()
    a2, b2, c2 = generate_coeff(), generate_coeff(), generate_coeff()
    numerator = a1*x_alt + b1*y_alt + c1
    denominator = a2*x_alt + b2*y_alt + c2
    # dy/dx = (a₁x + b₁y + c₁)/(a₂x + b₂y + c₂)
    return numerator, denominator

# Точное уравнение в полных дифференциалах
def generate_exact():
    while True:
        M = generate_expression()
        N = generate_expression()
        dM_dy = diff(M, y_alt)
        dN_dx = diff(N, x_alt)
        if dM_dy == dN_dx:
            # M(x,y)dx + N(x,y)dy = 0, где ∂M/∂y = ∂N/∂x
            return M, N

# Неточное уравнение в полных дифференциалах
def generate_inexact():
    while True:
        M = generate_expression()
        N = generate_expression()
        dM_dy = diff(M, y_alt)
        dN_dx = diff(N, x_alt)
        if dM_dy != dN_dx:
            # M(x,y)dx + N(x,y)dy = 0, где ∂M/∂y ≠ ∂N/∂x
            return M, N


# блок 3

if __name__ == "__main__":
    print("\nВыберите тип уравнения:")
    print("1) Уравнение, разрешенное относительно производной")
    print("2) Линейное уравнение с постоянными коэффициентами")
    print("3) Уравнение с разделенными переменными")
    print("4) Уравнение с разделяющимися переменными")
    print("5) Однородное уравнение")
    print("6) Уравнение, приводимое к однородному")
    print("7) Точное уравнение в полных дифференциалах")
    print("8) Неточное уравнение в полных дифференциалах")
    print("9) Все варианты уравнений")

    choice = input("Введите номер: ")

    if choice == "1":
        F = generate_explicit()
        print("\nУравнение, разрешенное относительно производной:")
        display(Math(f"\\frac{{d{y_alt}}}{{d{x_alt}}} = {latex(F)}"))

    elif choice == "2":
        order = int(input("Введите порядок линейного уравнения (1-5): "))
        equation = generate_linear_ode(order)
        print(f"\nЛинейное уравнение {order}-го порядка с постоянными коэффициентами:")
        display(Math(latex(equation)))

    elif choice == "3":
        f_x, g_y = generate_separated()
        print("\nУравнение с разделенными переменными:")
        display(Math(f"({latex(f_x)}) \\, d{x_alt} + ({latex(g_y)}) \\, d{y_alt} = 0"))

    elif choice == "4":
        fg, hk = generate_separable()
        print("\nУравнение с разделяемыми переменными:")
        display(Math(f"({latex(fg)}) \\, d{x_alt} + ({latex(hk)}) \\, d{y_alt} = 0"))

    elif choice == "5":
        f = generate_homogeneous()
        print("\nОднородное уравнение:")
        display(Math(f"\\frac{{d{y_alt}}}{{d{x_alt}}} = {latex(f)}"))

    elif choice == "6":
        numerator, denominator = generate_to_homogeneous()
        print("\nУравнение, приводимое к однородному:")
        display(Math(f"\\frac{{d{y_alt}}}{{d{x_alt}}} = \\frac{{{latex(numerator)}}}{{{latex(denominator)}}}"))

    elif choice == "7":
        M, N = generate_exact()
        print("\nТочное уравнение в полных дифференциалах:")
        display(Math(f"({latex(M)}) \\, d{x_alt} + ({latex(N)}) \\, d{y_alt} = 0"))

    elif choice == "8":
        M, N = generate_inexact()
        print("\nНеточное уравнение в полных дифференциалах:")
        display(Math(f"({latex(M)}) \\, d{x_alt} + ({latex(N)}) \\, d{y_alt} = 0"))

    elif choice == "9":
        # Генерация всех типов уравнений
        F = generate_explicit()
        order = random.randint(1, 3)
        linear_ode = generate_linear_ode(order)
        f_x, g_y = generate_separated()
        fg, hk = generate_separable()
        f_homogeneous = generate_homogeneous()
        numerator, denominator = generate_to_homogeneous()
        M_exact, N_exact = generate_exact()
        M_inexact, N_inexact = generate_inexact()

        print("\nУравнение, разрешенное относительно производной:")
        display(Math(f"\\frac{{d{y_alt}}}{{d{x_alt}}} = {latex(F)}"))

        print(f"\nЛинейное уравнение {order}-го порядка с постоянными коэффициентами:")
        display(Math(latex(linear_ode)))

        print("\nУравнение с разделенными переменными:")
        display(Math(f"({latex(f_x)}) \\, d{x_alt} + ({latex(g_y)}) \\, d{y_alt} = 0"))

        print("\nУравнение с разделяемыми переменными:")
        display(Math(f"({latex(fg)}) \\, d{x_alt} + ({latex(hk)}) \\, d{y_alt} = 0"))

        print("\nОднородное уравнение:")
        display(Math(f"\\frac{{d{y_alt}}}{{d{x_alt}}} = {latex(f_homogeneous)}"))

        print("\nУравнение, приводимое к однородному:")
        display(Math(f"\\frac{{d{y_alt}}}{{d{x_alt}}} = \\frac{{{latex(numerator)}}}{{{latex(denominator)}}}"))

        print("\nТочное уравнение в полных дифференциалах:")
        display(Math(f"({latex(M_exact)}) \\, d{x_alt} + ({latex(N_exact)}) \\, d{y_alt} = 0"))

        print("\nНеточное уравнение в полных дифференциалах:")
        display(Math(f"({latex(M_inexact)}) \\, d{x_alt} + ({latex(N_inexact)}) \\, d{y_alt} = 0"))
