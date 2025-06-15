import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, lambdify


# Метод Эйлера
def euler_method(f, x0, y0, h, n):
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0] = x0
    y[0] = y0
    for i in range(n):
        y[i+1] = y[i] + h * f(x[i], y[i])
        x[i+1] = x[i] + h
    return x, y

# Метод Рунге-Кутты 4-го порядка
def runge_kutta_method(f, x0, y0, h, n):
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0] = x0
    y[0] = y0
    for i in range(n):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h/2, y[i] + k1/2)
        k3 = h * f(x[i] + h/2, y[i] + k2/2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        x[i+1] = x[i] + h
    return x, y

# # Метод изоклин (упрощенная реализация)
# def isocline_method(f, x0, y0, h, n):
#     x = np.linspace(x0 - 2, x0 + 2, 20)
#     y = np.linspace(y0 - 2, y0 + 2, 20)
#     X, Y = np.meshgrid(x, y)
#     slopes = f(X, Y)
    
#     # Решение методом Эйлера для траектории
#     x_traj, y_traj = euler_method(f, x0, y0, h, n)
    
#     return x_traj, y_traj, (X, Y, slopes)

# Функция для преобразования строки уравнения в функцию
def parse_equation(equation_str):
    x_sym, y_sym = symbols('x y')
    try:
        # Пробуем распарсить как уравнение y' = f(x,y)
        if '=' in equation_str:
            parts = equation_str.split('=')
            f_expr = sympify(parts[1].strip())
        else:
            f_expr = sympify(equation_str)
            
        f_lambdified = lambdify((x_sym, y_sym), f_expr, 'numpy')
        return f_lambdified
    except Exception as e:
        raise ValueError(f"Ошибка парсинга уравнения: {str(e)}")
# Метод Эйлера
def euler_method(f, x0, y0, h, n):
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0] = x0
    y[0] = y0
    for i in range(n):
        y[i+1] = y[i] + h * f(x[i], y[i])  # y_{i+1} = y_i + h * f(x_i, y_i)
        x[i+1] = x[i] + h  # x_{i+1} = x_i + h
    return x, y

# Метод Рунге-Кутты 4-го порядка
def runge_kutta_method(f, x0, y0, h, n):
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0] = x0
    y[0] = y0
    for i in range(n):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h/2, y[i] + k1/2)
        k3 = h * f(x[i] + h/2, y[i] + k2/2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        x[i+1] = x[i] + h
    return x, y


def isocline_method(f, x0, y0, h, n):
    # Решение методом Эйлера для траектории
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0] = x0
    y[0] = y0
    
    for i in range(n):
        slope = f(x[i], y[i])
        y[i+1] = y[i] + h * slope
        x[i+1] = x[i] + h
    
    # Создаем сетку для изоклин
    x_grid = np.linspace(x0 - 2, x0 + 2, 20)
    y_grid = np.linspace(y0 - 2, y0 + 2, 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    slopes = f(X, Y)
    
    return x, y, (X, Y, slopes)

#ввод
def input_equation():
    x, y = symbols('x y')
    equation_str = input("Введите ОДУ в виде y' = f(x, y): ")  #Например, 4*x + 4*y
    f_expr = sympify(equation_str)  #строку в символьное выр
    f_lambdified = lambda x_val, y_val: float(f_expr.subs({'x': x_val, 'y': y_val}))  #Лямбда-функция
    return f_lambdified

def input_initial_conditions():
    x0 = float(input("Введите начальное значение x0: "))
    y0 = float(input("Введите начальное значение y0: "))
    return x0, y0

def input_step_and_steps():
    h = float(input("Введите шаг h: "))
    n = int(input("Введите количество шагов n: "))
    return h, n

#обр
if __name__ == "__main__":
    #ОДУ
    print("Пример ввода ОДУ: 4*x + 4*y")
    f = input_equation()

    #нач условия
    x0, y0 = input_initial_conditions()

    #шаг и кол-во шагов
    h, n = input_step_and_steps()

    #Эйлер
    x_euler, y_euler = euler_method(f, x0, y0, h, n)

    #Рунге-Кутты
    x_rk, y_rk = runge_kutta_method(f, x0, y0, h, n)

    #изоклин
    x_iso, y_iso = isocline_method(f, x0, y0, h, n)


    plt.figure(figsize=(15, 5))


    plt.subplot(1, 3, 1)
    plt.plot(x_euler, y_euler, label='Метод Эйлера', color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Метод Эйлера')
    plt.grid(True)
    plt.legend()


    plt.subplot(1, 3, 2)
    plt.plot(x_rk, y_rk, label='Метод Рунге-Кутты', color='green')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Метод Рунге-Кутты')
    plt.grid(True)
    plt.legend()


    plt.subplot(1, 3, 3)
    plt.plot(x_iso, y_iso, label='Метод изоклин', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Метод изоклин')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


    import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify

#Эйлер
def euler_method(f, x0, y0, h, n):
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0] = x0
    y[0] = y0
    for i in range(n):
        y[i+1] = y[i] + h * f(x[i], y[i])
        x[i+1] = x[i] + h
    return x, y

#Метод Рунге-Кутты 4-го порядка
def runge_kutta_method(f, x0, y0, h, n):
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0] = x0
    y[0] = y0
    for i in range(n):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h/2, y[i] + k1/2)
        k3 = h * f(x[i] + h/2, y[i] + k2/2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
        x[i+1] = x[i] + h
    return x, y

# # Изоклин (упрощённый)
# def isocline_method(f, x0, y0, h, n):
#     x = np.zeros(n+1)
#     y = np.zeros(n+1)
#     x[0] = x0
#     y[0] = y0
#     for i in range(n):
#         slope = f(x[i], y[i])
#         y[i+1] = y[i] + h * slope
#         x[i+1] = x[i] + h
#     return x, y

#ФитцХью — Нагумо
def fitzhugh_nagumo_model(v, w, I_ext=0.5, epsilon=0.08, a=0.7, b=0.8):
    dvdt = v - v**3 / 3 - w + I_ext
    dwdt = epsilon * (v + a - b * w)
    return dvdt, dwdt

#хищник-жертва
def predatormodel(x, y, alpha=1.1, beta=0.4, delta=0.1, gamma=0.4):
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return dxdt, dydt

#Ввод с клавы
def input_equation():
    x, y = symbols('x y')
    equation_str = input("Введите ОДУ в виде y' = f(x, y): ")
    f_expr = sympify(equation_str)
    f_lambdified = lambda x_val, y_val: float(f_expr.subs({'x': x_val, 'y': y_val}))
    return f_lambdified

def input_initial_conditions():
    x0 = float(input("Введите начальное значение x0: "))
    y0 = float(input("Введите начальное значение y0: "))
    return x0, y0

def input_step_and_steps():
    h = float(input("Введите шаг h: "))
    n = int(input("Введите количество шагов n: "))
    return h, n

#тело
if __name__ == "__main__": #мб поменяю
    # ОДУ
    print("Пример ввода ОДУ: 4*x + 4*y")
    f = input_equation()

    # Начальные условия
    x0, y0 = input_initial_conditions()

    # Шаг и количество шагов
    h, n = input_step_and_steps()

    # Эйлер
    x_euler, y_euler = euler_method(f, x0, y0, h, n)

    # Рунге-Кутты
    x_rk, y_rk = runge_kutta_method(f, x0, y0, h, n)

    # Изоклин
    x_iso, y_iso = isocline_method(f, x0, y0, h, n)

# Модель ФитцХью-Нагумо
    v = np.zeros(n+1)
    w = np.zeros(n+1)
    v[0], w[0] = x0, y0
    for i in range(n):
        dvdt, dwdt = fitzhugh_nagumo_model(v[i], w[i])
        v[i+1] = v[i] + h * dvdt
        w[i+1] = w[i] + h * dwdt
    
    # Модель хищник-жертва
    x_pp = np.zeros(n+1)
    y_pp = np.zeros(n+1)
    x_pp[0], y_pp[0] = x0, y0
    for i in range(n):
        dxdt, dydt = predatormodel(x_pp[i], y_pp[i])
        x_pp[i+1] = x_pp[i] + h * dxdt
        y_pp[i+1] = y_pp[i] + h * dydt

# Уравнение теплопроводности
def heat_equation_solution(L=1, T=1, alpha=0.01, nx=20, nt=100):
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / dx**2
    
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    u = np.zeros((nt, nx))
    
    u[0, :] = np.sin(np.pi * x / L)
    u[:, 0] = 0
    u[:, -1] = 0
    
    for n in range(0, nt-1):
        for i in range(1, nx-1):
            u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
    
    return x, t, u

# Метод изоклин (улучшенная реализация)
def isocline_method(f, x0, y0, h, n):
    # Решение методом Эйлера для траектории
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0] = x0
    y[0] = y0
    
    for i in range(n):
        slope = f(x[i], y[i])
        y[i+1] = y[i] + h * slope
        x[i+1] = x[i] + h
    
    # Создаем сетку для изоклин
    x_grid = np.linspace(x0 - 2, x0 + 2, 20)
    y_grid = np.linspace(y0 - 2, y0 + 2, 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    slopes = f(X, Y)
    
    return x, y, (X, Y, slopes)

    # # Метод Эйлера
    # plt.subplot(2, 3, 1)
    # plt.plot(x_euler, y_euler, label='Метод Эйлера', color='blue')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Метод Эйлера')
    # plt.grid(True)
    # plt.legend()

    # # Метод Рунге-Кутты
    # plt.subplot(2, 3, 2)
    # plt.plot(x_rk, y_rk, label='Метод Рунге-Кутты', color='green')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Метод Рунге-Кутты')
    # plt.grid(True)
    # plt.legend()

    # # Метод изоклин
    # plt.subplot(2, 3, 3)
    # plt.plot(x_iso, y_iso, label='Метод изоклин', color='red')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Метод изоклин')
    # plt.grid(True)
    # plt.legend()

    # # Модель ФитцХью — Нагумо
    # plt.subplot(2, 3, 4)
    # plt.plot(v, w, label='Модель ФитцХью — Нагумо', color='purple')
    # plt.xlabel('v (мембранный потенциал)')
    # plt.ylabel('w (восстановительная переменная)')
    # plt.title('Модель ФитцХью — Нагумо')
    # plt.grid(True)
    # plt.legend()

    # # Модель хищник-жертва
    # plt.subplot(2, 3, 5)
    # plt.plot(x_pp, y_pp, label='Модель хищник-жертва', color='orange')
    # plt.xlabel('Жертвы (x)')
    # plt.ylabel('Хищники (y)')
    # plt.title('Модель хищник-жертва')
    # plt.grid(True)
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

#явная схема
def heat_equation_solution(L=1, T=1, alpha=0.01, nx=20, nt=100):

    dx = L / (nx - 1)
    dt = T / nt

    #Коэффициент устойчивости
    r = alpha * dt / dx**2
    if r > 0.5:
        print(f" r = {r:.2f} > 0.5- неуст.")

    #Инициализация сетки
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    u = np.zeros((nt, nx))

    #Начальное условие
    u[0, :] = np.sin(np.pi * x / L)

    #Граничные
    u[:, 0] = 0
    u[:, -1] = 0

    for n in range(0, nt-1):
        for i in range(1, nx-1):
            u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])

    return x, t, u


if __name__ == "__main__":

    x_heat, t_heat, u_heat = heat_equation_solution(L=1, T=0.1, alpha=0.01, nx=50, nt=200)

    #3D график
    fig = plt.figure(figsize=(20, 10))

    ax = fig.add_subplot(2, 3, 6, projection='3d')
    T, X = np.meshgrid(t_heat, x_heat)
    ax.plot_surface(X, T, u_heat.T, cmap='hot')
    ax.set_xlabel('Пространство (x)')
    ax.set_ylabel('Время (t)')
    ax.set_zlabel('Температура (u)')
    ax.set_title('Уравнение теплопроводности')


    plt.tight_layout()

    #Анимация
    fig_anim, ax_anim = plt.subplots(figsize=(8, 6))
    ax_anim.set_xlim(0, 1)
    ax_anim.set_ylim(0, 1)
    ax_anim.set_xlabel('Пространство (x)')
    ax_anim.set_ylabel('Температура (u)')
    ax_anim.set_title('Распределение температуры по времени')
    line, = ax_anim.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        line.set_data(x_heat, u_heat[i, :])
        ax_anim.set_title(f'Распределение температуры (t={t_heat[i]:.3f})')
        return line,

    anim = FuncAnimation(fig_anim, animate, frames=len(t_heat),
                        init_func=init, blit=True, interval=50)

    plt.tight_layout()
    plt.show()