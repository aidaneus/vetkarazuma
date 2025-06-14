from flask import Flask, request, jsonify, render_template
import sympy
from sympy import symbols, diff, latex, Function, Eq
import random
from flask_cors import CORS
from backend.anechka import *
import html
from jinja2 import Template
import jinja2
app = Flask(__name__)
CORS(app)  # Разрешаем CORS для всех доменов


# Инициализация символов
x = symbols('x')
y = Function('y')(x)
x_alt, y_alt = symbols('x y')

# ... [Весь код из anechka.py, начиная с generate_coeff() до конца функций генерации] ...

@app.route('/')
def index():
    return render_template('new.html')
@app.route('/diff_eq')
def diff_eq():
    return render_template('diff_eq.html')
@app.route('/linear_algebra')
def linear_algebra():
    return render_template('linear_algebra.html')
@app.route('/matrix_calculator')
def matrix_calculator():
    return render_template('matrix_calculator.html')
@app.route('/all_tasks')
def all_tasks():
    return render_template('all_tasks.html')
@app.route('/solve_ode', methods=['POST'])
def solve_ode():
    from backend.milya import euler_method, runge_kutta_method, isocline_method, parse_equation
    from backend.milya import fitzhugh_nagumo_model, predatormodel, heat_equation_solution
    
    data = request.json
    equation_str = data.get('equation', '')
    x0 = float(data.get('x0'))
    y0 = float(data.get('y0'))
    h = float(data.get('h'))
    n = int(data.get('n'))
    method = data.get('method', 'euler')
    
    try:
        if method == 'fitzhugh-nagumo':
            I_ext = float(data.get('fitzhughIext', 0.5))
            epsilon = float(data.get('fitzhughEpsilon', 0.08))
            a = float(data.get('fitzhughA', 0.7))
            b = float(data.get('fitzhughB', 0.8))
            
            v = np.zeros(n+1)
            w = np.zeros(n+1)
            v[0], w[0] = x0, y0
            for i in range(n):
                dvdt, dwdt = fitzhugh_nagumo_model(v[i], w[i], I_ext, epsilon, a, b)
                v[i+1] = v[i] + h * dvdt
                w[i+1] = w[i] + h * dwdt
            
            return jsonify({
                'method': 'Модель ФитцХью-Нагумо',
                'x': v.tolist(),
                'y': w.tolist()
            })
            
        elif method == 'predator-prey':
            alpha = float(data.get('predatorAlpha', 1.1))
            beta = float(data.get('predatorBeta', 0.4))
            delta = float(data.get('predatorDelta', 0.1))
            gamma = float(data.get('predatorGamma', 0.4))
            
            x_pp = np.zeros(n+1)
            y_pp = np.zeros(n+1)
            x_pp[0], y_pp[0] = x0, y0
            for i in range(n):
                dxdt, dydt = predatormodel(x_pp[i], y_pp[i], alpha, beta, delta, gamma)
                x_pp[i+1] = x_pp[i] + h * dxdt
                y_pp[i+1] = y_pp[i] + h * dydt
            
            return jsonify({
                'method': 'Модель хищник-жертва',
                'x': x_pp.tolist(),
                'y': y_pp.tolist()
            })
            
        elif method == 'heat-equation':
            L = float(data.get('heatL', 1))
            T = float(data.get('heatT', 0.1))
            alpha_val = float(data.get('heatAlpha', 0.01))
            nx = int(data.get('heatNx', 50))
            
            x, t, u = heat_equation_solution(L=L, T=T, alpha=alpha_val, nx=nx, nt=n)
            return jsonify({
                'method': 'Уравнение теплопроводности',
                'x': x.tolist(),
                't': t.tolist(),
                'u': u.tolist()
            })
            
        elif method == 'predator-prey':
            # Параметры модели
            alpha = float(data.get('predatorAlpha', 1.1))
            beta = float(data.get('predatorBeta', 0.4))
            delta = float(data.get('predatorDelta', 0.1))
            gamma = float(data.get('predatorGamma', 0.4))
            
            # Решаем систему уравнений
            x_pp = np.zeros(n+1)
            y_pp = np.zeros(n+1)
            x_pp[0], y_pp[0] = x0, y0
            for i in range(n):
                dxdt, dydt = predatormodel(x_pp[i], y_pp[i], alpha, beta, delta, gamma)
                x_pp[i+1] = x_pp[i] + h * dxdt
                y_pp[i+1] = y_pp[i] + h * dydt
            
            return jsonify({
                'method': 'Модель хищник-жертва',
                'x': x_pp.tolist(),
                'y': y_pp.tolist()
            })
            
        elif method == 'heat-equation':
            # Обработка уравнения теплопроводности
            return solve_heat_equation(data)
            
        # Обычные методы решения ОДУ
        f = parse_equation(equation_str)
        
        if method == 'euler':
            x_vals, y_vals = euler_method(f, x0, y0, h, n)
            method_name = "Метод Эйлера"
        elif method == 'runge-kutta':
            x_vals, y_vals = runge_kutta_method(f, x0, y0, h, n)
            method_name = "Метод Рунге-Кутты 4-го порядка"
        elif method == 'isocline':
            x_vals, y_vals, _ = isocline_method(f, x0, y0, h, n)
            method_name = "Метод изоклин"
        else:
            return jsonify({'error': 'Неизвестный метод решения'}), 400
        
        return jsonify({
            'method': method_name,
            'x': x_vals.tolist(),
            'y': y_vals.tolist()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def solve_heat_equation(data):
    # Реализация решения уравнения теплопроводности
    L = float(data.get('heatL', 1))
    T = float(data.get('heatT', 0.1))
    alpha = float(data.get('heatAlpha', 0.01))
    nx = int(data.get('heatNx', 50))
    nt = int(data.get('steps', 100))
    
    dx = L / (nx - 1)
    dt = T / nt
    
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    u = np.zeros((nt, nx))
    
    # Начальное условие
    u[0, :] = np.sin(np.pi * x / L)
    
    # Граничные условия
    u[:, 0] = 0
    u[:, -1] = 0
    
    for n in range(0, nt-1):
        for i in range(1, nx-1):
            u[n+1, i] = u[n, i] + alpha * dt / dx**2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
    
    return jsonify({
        'method': 'Уравнение теплопроводности',
        'x': x.tolist(),
        't': t.tolist(),
        'u': u.tolist()
    })
    
@app.route('/matrix_operation', methods=['POST'])
def matrix_operation():
    from backend.yana import (
        matrix_sum, matrix_mult, matrix_transpose,
        matrix_eigen, matrix_gauss_jordan_inversion,
        matrix_exponential, sinm, cosm
    )
    import numpy as np
    
    data = request.json
    operation = data.get('operation')
    matrix1 = np.array(data.get('matrix1'))
    
    try:
        if operation == 'sum':
            matrix2 = np.array(data.get('matrix2'))
            result, steps = matrix_sum(matrix1, matrix2)
            return jsonify({
                'result': result.tolist(),
                'steps': steps
            })
            
        elif operation == 'mult':
            matrix2 = np.array(data.get('matrix2'))
            result, steps = matrix_mult(matrix1, matrix2)
            return jsonify({
                'result': result.tolist(),
                'steps': steps
            })
            
        elif operation == 'transpose':
            result, steps = matrix_transpose(matrix1)
            return jsonify({
                'result': result.tolist(),
                'steps': steps
            })
            
        elif operation == 'eigen':
            eigenvals, eigenvectors, steps = matrix_eigen(matrix1)
            # Convert sympy values to floats if possible
            eigenvals_float = [float(val.evalf()) if hasattr(val, 'evalf') else float(val) for val in eigenvals]
            return jsonify({
                'result': {
                    'eigenvalues': eigenvals_float,
                    'eigenvectors': [vec.tolist() for vec in eigenvectors]
                },
                'steps': steps
            })
            
        elif operation == 'inverse':
            result, steps = matrix_gauss_jordan_inversion(matrix1)
            return jsonify({
                'result': result.tolist(),
                'steps': steps
            })
            
        elif operation == 'exp':
            result, steps = matrix_exponential(matrix1)
            return jsonify({
                'result': result.tolist(),
                'steps': steps
            })
            
        elif operation == 'sin':
            result, steps = sinm(matrix1)
            return jsonify({
                'result': result.tolist(),
                'steps': steps
            })
            
        elif operation == 'cos':
            result, steps = cosm(matrix1)
            return jsonify({
                'result': result.tolist(),
                'steps': steps
            })
            
        else:
            return jsonify({'error': 'Unknown operation'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/solve_linear_system', methods=['POST'])
def solve_linear_system():
    from backend.artem import cramer_rule, gauss_rule, symbolic_sol
    from sympy import Matrix, latex
    
    data = request.json
    method = data.get('method')
    matrix_data = data.get('matrix')
    vector_data = data.get('vector')
    
    try:
        # Convert to SymPy matrices
        A = Matrix(matrix_data)
        B = Matrix(vector_data)
        
        if method == 'cramer':
            solutions = cramer_rule(A, B)
            method_name = "Метод Крамера"
            steps = ""
        elif method == 'gauss':
            solutions = gauss_rule(A, B)
            method_name = "Метод Гаусса"
            # Получаем ступенчатую матрицу в виде LaTeX
            augmented_matrix = A.col_insert(A.shape[0], B)
            row_reduced_matrix = augmented_matrix.rref()[0]
            steps = latex(row_reduced_matrix)
        else:
            return jsonify({'error': 'Неизвестный метод решения'}), 400
        
        # Преобразуем решения в JSON-сериализуемый формат
        def convert_solution(sol):
            if isinstance(sol, (list, tuple)):
                return [float(val) if val.is_number else str(val) for val in sol]
            elif hasattr(sol, 'is_number') and sol.is_number:
                return float(sol)
            else:
                return str(sol)
        
        # Форматируем ответ
        if solutions == 0:
            response = {
                'method': method_name,
                'solution': "Система не имеет решений",
                'steps': steps if 'steps' in locals() else ""
            }
        else:
            response = {
                'method': method_name,
                'solution': convert_solution(solutions),
                'steps': steps if 'steps' in locals() else ""
            }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_ode', methods=['POST'])
def generate_ode():
    data = request.json
    choice = data.get('type')
    
    try:
        if choice == "1":
            F = generate_explicit()
            return jsonify({
                'type': 'Уравнение, разрешенное относительно производной',
                'equation': f"\\frac{{d{y_alt}}}{{d{x_alt}}} = {latex(F)}"
            })
        
        elif choice == "2":
            order = int(data.get('order', 2))
            equation = generate_linear_ode(order)
            return jsonify({
                'type': f'Линейное уравнение {order}-го порядка с постоянными коэффициентами',
                'equation': latex(equation)
            })
        
        elif choice == "3":
            f_x, g_y = generate_separated()
            return jsonify({
                'type': 'Уравнение с разделенными переменными',
                'equation': f"({latex(f_x)}) \\, d{x_alt} + ({latex(g_y)}) \\, d{y_alt} = 0"
            })
        
        elif choice == "4":
            fg, hk = generate_separable()
            return jsonify({
                'type': 'Уравнение с разделяемыми переменными',
                'equation': f"({latex(fg)}) \\, d{x_alt} + ({latex(hk)}) \\, d{y_alt} = 0"
            })
        
        elif choice == "5":
            f = generate_homogeneous()
            return jsonify({
                'type': 'Однородное уравнение',
                'equation': f"\\frac{{d{y_alt}}}{{d{x_alt}}} = {latex(f)}"
            })
        
        elif choice == "6":
            numerator, denominator = generate_to_homogeneous()
            return jsonify({
                'type': 'Уравнение, приводимое к однородному',
                'equation': f"\\frac{{d{y_alt}}}{{d{x_alt}}} = \\frac{{{latex(numerator)}}}{{{latex(denominator)}}}"
            })
        
        elif choice == "7":
            M, N = generate_exact()
            return jsonify({
                'type': 'Точное уравнение в полных дифференциалах',
                'equation': f"({latex(M)}) \\, d{x_alt} + ({latex(N)}) \\, d{y_alt} = 0"
            })
        
        elif choice == "8":
            M, N = generate_inexact()
            return jsonify({
                'type': 'Неточное уравнение в полных дифференциалах',
                'equation': f"({latex(M)}) \\, d{x_alt} + ({latex(N)}) \\, d{y_alt} = 0"
            })
        
        elif choice == "9":
            # Генерация всех типов уравнений
            equations = []
            
            # Уравнение, разрешенное относительно производной
            F = generate_explicit()
            equations.append({
                'type': 'Уравнение, разрешенное относительно производной',
                'equation': f"\\frac{{d{y_alt}}}{{d{x_alt}}} = {latex(F)}"
            })
            
            # Линейное уравнение
            order = random.randint(1, 3)
            linear_ode = generate_linear_ode(order)
            equations.append({
                'type': f'Линейное уравнение {order}-го порядка с постоянными коэффициентами',
                'equation': latex(linear_ode)
            })
            
            # Уравнение с разделенными переменными
            f_x, g_y = generate_separated()
            equations.append({
                'type': 'Уравнение с разделенными переменными',
                'equation': f"({latex(f_x)}) \\, d{x_alt} + ({latex(g_y)}) \\, d{y_alt} = 0"
            })
            
            # Уравнение с разделяющимися переменными
            fg, hk = generate_separable()
            equations.append({
                'type': 'Уравнение с разделяемыми переменными',
                'equation': f"({latex(fg)}) \\, d{x_alt} + ({latex(hk)}) \\, d{y_alt} = 0"
            })
            
            # Однородное уравнение
            f_homogeneous = generate_homogeneous()
            equations.append({
                'type': 'Однородное уравнение',
                'equation': f"\\frac{{d{y_alt}}}{{d{x_alt}}} = {latex(f_homogeneous)}"
            })
            
            # Уравнение, приводимое к однородному
            numerator, denominator = generate_to_homogeneous()
            equations.append({
                'type': 'Уравнение, приводимое к однородному',
                'equation': f"\\frac{{d{y_alt}}}{{d{x_alt}}} = \\frac{{{latex(numerator)}}}{{{latex(denominator)}}}"
            })
            
            # Точное уравнение
            M_exact, N_exact = generate_exact()
            equations.append({
                'type': 'Точное уравнение в полных дифференциалах',
                'equation': f"({latex(M_exact)}) \\, d{x_alt} + ({latex(N_exact)}) \\, d{y_alt} = 0"
            })
            
            # Неточное уравнение
            M_inexact, N_inexact = generate_inexact()
            equations.append({
                'type': 'Неточное уравнение в полных дифференциалах',
                'equation': f"({latex(M_inexact)}) \\, d{x_alt} + ({latex(N_inexact)}) \\, d{y_alt} = 0"
            })
            
            return jsonify(equations)
        
        else:
            return jsonify({'error': 'Неверный тип уравнения'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)