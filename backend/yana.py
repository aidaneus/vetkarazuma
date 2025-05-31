import numpy as np
import sympy as sp
from scipy.linalg import expm, sinm, cosm

class Solution:
    def __init__(self):
        self.messages = []

    def add_message(self, msg=''):
        self.messages.append(str(msg))

    def get_message(self):
        return "\n".join(self.messages)
    
    def add_matrix_to_solution(self, mat):
        mat = np.array(mat)  # убеждаемся, что это массив
        self.add_message('```')
        for row in mat:
            self.add_message("[ " + "  ".join(f"{float(val):8.4f}" if isinstance(val, (int, float, np.number)) else f"{str(val):>8}" for val in row) + " ]")
        self.add_message('```')

def matrix_sum(matrix1, matrix2):  # сложение с пошаговым выводом в виде матрицы
    solution = Solution()
    solution.add_message('## Сложение матриц')
    solution.add_message("### Матрица 1:")
    solution.add_matrix_to_solution(matrix1)

    solution.add_message("### Матрица 2:")
    solution.add_matrix_to_solution(matrix2)
    
    solution.add_message("### Пошаговое сложение (в виде матрицы):")
    rows, cols = matrix1.shape
    step_matrix = [[f"{matrix1[i, j]}+{matrix2[i, j]}" for j in range(cols)] for i in range(rows)]
    solution.add_matrix_to_solution(step_matrix)

    result = matrix1 + matrix2
    solution.add_message("### Результат сложения:")
    solution.add_matrix_to_solution(result)

    return result, solution.get_message()


def matrix_mult(matrix1, matrix2):
    solution = Solution()
    solution.add_message("## Умножение матриц")
    solution.add_message("### Матрица 1:")
    solution.add_matrix_to_solution(matrix1)

    solution.add_message("### Матрица 2:")
    solution.add_matrix_to_solution(matrix2)

    solution.add_message("### Пошаговое умножение (вычисление каждого элемента):")
    rows, cols = matrix1.shape[0], matrix2.shape[1]
    result = np.zeros((rows, cols), dtype=int)

    for i in range(rows):
        for j in range(cols):
            terms = [f"{matrix1[i, k]}\*{matrix2[k, j]}" for k in range(matrix1.shape[1])]
            products = [matrix1[i, k] * matrix2[k, j] for k in range(matrix1.shape[1])]
            result[i, j] = sum(products)
            expression = " + ".join(terms)
            solution.add_message()
            solution.add_message(f"**C[{i},{j}] = {expression} = {result[i, j]}**")

    solution.add_message("\n### Результат умножения:")
    solution.add_matrix_to_solution(result)
    return result, solution.get_message()



def matrix_transpose(matrix1):
    solution = Solution()
    solution.add_message("## Транспонирование матрицы")
    solution.add_message("### Исходная матрица:")
    solution.add_matrix_to_solution(matrix1)

    solution.add_message("### Пошаговое транспонирование:")
    solution.add_message("Меняем местами строки и столбцы: элемент [i, j] → [j, i]")
    rows, cols = matrix1.shape
    for i in range(rows):
        for j in range(cols):
            solution.add_message()
            solution.add_message(f"**({i}, {j}) = {matrix1[i, j]} → позиция ({j}, {i})**")

    result = np.transpose(matrix1)
    solution.add_message("### Результат транспонирования:")
    solution.add_matrix_to_solution(result)

    return result, solution.get_message()



def matrix_eigen(matrix):
    solution = Solution()
    solution.add_message("## Собственные значения и векторы")
    solution.add_message("### Матрица A:")
    solution.add_matrix_to_solution(matrix)
    solution.add_message()

    lam = sp.symbols('λ')
    A_sym = sp.Matrix(matrix)

    solution.add_message("### Матрица (A - λI):")
    char_matrix = A_sym - lam * sp.eye(matrix.shape[0])
    solution.add_message(char_matrix)
    solution.add_message()

    solution.add_message("### Характеристический многочлен:")
    char_poly = char_matrix.det()
    solution.add_message(f"det(A - λI) = {char_poly}".replace('*','\*'))
    solution.add_message()

    solution.add_message("### Решение уравнения det(A - λI) = 0:")
    eigenvals = sp.solve(char_poly, lam)
    for i, val in enumerate(eigenvals, 1):
        solution.add_message(f"- **λ_{i} = {val.evalf():.2f}**")
    solution.add_message()

    eigenvectors = []
    for i, val in enumerate(eigenvals, 1):
        solution.add_message(f"---\n### Собственный вектор для λ_{i} = {val.evalf():.2f}")
        mat = A_sym - val * sp.eye(matrix.shape[0])
        solution.add_message("Матрица (A - λI):")
        solution.add_matrix_to_solution(mat)
        solution.add_message()

        solution.add_message("Решаем систему (A - λI) * x = 0:")
        nullspace = mat.nullspace()
        if nullspace:
            vec = nullspace[0]
            solution.add_message("Базис ядра (собственный вектор):")
            solution.add_matrix_to_solution(vec)

            if matrix.shape[0] <= 3:
                solution.add_message("#### Расписываем систему уравнений:")
                rows = mat.tolist()
                vars = sp.symbols(f'x1:{matrix.shape[0]+1}')
                for r, row in enumerate(rows):
                    eq_lhs = sum(c*v for c,v in zip(row, vars))
                    solution.add_message(f"Уравнение {r+1}: {eq_lhs} = 0")
            eigenvectors.append(np.array(vec).astype(np.float64).flatten())
        else:
            solution.add_message("Нет нетривиального решения.")
        solution.add_message()

    return eigenvals, eigenvectors, solution.get_message()



def matrix_gauss_jordan_inversion(matrix1):
    solution = Solution()
    solution.add_message("## Обратная матрица методом Гаусса-Жордана")

    n = matrix1.shape[0]
    I = np.eye(n)
    augmented_matrix = np.hstack((matrix1.astype(float), I.astype(float)))

    solution.add_message("### Исходная матрица (A | I):")
    solution.add_message('```')
    for row in augmented_matrix:
        left = " ".join(f"{val:6.1f}" for val in row[:n])
        right = " ".join(f"{val:6.1f}" for val in row[n:])
        solution.add_message(f"[ {left} | {right} ]")
    solution.add_message('```')

    for i in range(n):
        if augmented_matrix[i, i] == 0:
            for j in range(i + 1, n):
                if augmented_matrix[j, i] != 0:
                    augmented_matrix[[i, j]] = augmented_matrix[[j, i]]
                    solution.add_message(f"\n**Поменяли строки {i+1} и {j+1} из-за нулевого ведущего элемента.**")
                    break

        pivot = augmented_matrix[i, i]
        augmented_matrix[i] = augmented_matrix[i] / pivot
        solution.add_message(f"\n**Нормализуем строку {i+1} (делим на {pivot:.2f}):**")
        solution.add_message('```')
        for row in augmented_matrix:
            left = " ".join(f"{val:8.3f}" for val in row[:n])
            right = " ".join(f"{val:8.3f}" for val in row[n:])
            solution.add_message(f"[ {left} | {right} ]")
        solution.add_message('```')

        for j in range(n):
            if j != i:
                factor = augmented_matrix[j, i]
                augmented_matrix[j] -= factor * augmented_matrix[i]
                solution.add_message(f"\n**Обнуляем элемент в строке {j+1}: вычитаем {factor:.3f} * строку {i+1}**")
                solution.add_message('```')
                for row in augmented_matrix:
                    left = " ".join(f"{val:8.3f}" for val in row[:n])
                    right = " ".join(f"{val:8.3f}" for val in row[n:])
                    solution.add_message(f"[ {left} | {right} ]")
                solution.add_message('```')

    solution.add_message("\n### Обратная матрица (правая часть):")
    inverse = augmented_matrix[:, n:]
    solution.add_matrix_to_solution(inverse)

    return inverse, solution.get_message()


def matrix_exponential(matrix1, terms=10):
    solution = Solution()
    solution.add_message("## Матричная экспонента $e^A$")
    solution.add_message("### Формула разложения:")
    solution.add_message("$$e^A = I + A + \\frac{A^2}{2!} + \\frac{A^3}{3!} + \\dots$$\n")

    n = matrix1.shape[0]
    result = np.eye(n)
    current_term = np.eye(n)

    solution.add_message("### Шаг 0: I (единичная матрица)")
    solution.add_matrix_to_solution(result)

    for k in range(1, terms):
        current_term = current_term @ matrix1
        factorial = np.math.factorial(k)
        term = current_term / factorial
        result += term
        solution.add_message(f"### Шаг {k}: $A^{k} / {k}!$")
        solution.add_matrix_to_solution(term)

    solution.add_message("\n### Приближённое значение $e^A$:")
    solution.add_matrix_to_solution(result)
    return result, solution.get_message()


def sinm(matrix1, terms=10):
    """
    Вычисляет синус матрицы sin(A) через разложение в ряд Тейлора.
    terms — количество членов ряда (по умолчанию 10).
    """
    solution = Solution()
    solution.add_message('## Синус матрицы')
    n = matrix1.shape[0]
    result = np.zeros((n, n))
    current_power = matrix1.copy()

    solution.add_message("### Вычисление синуса матрицы sin(A) с помощью ряда Тейлора:")
    solution.add_message("Формула: sin(A) = A - A^3/3! + A^5/5! - A^7/7! + ...\n")

    for k in range(terms):
        power = 2 * k + 1
        factorial = np.math.factorial(power)
        if k > 0:
            current_power = current_power @ matrix1 @ matrix1  # A^{2k+1}
        sign = (-1) ** k
        term = sign * current_power / factorial
        result += term

        solution.add_message(f"**Шаг {k + 1}: {'-' if sign < 0 else '+'} A^{power} / {power}! =**")
        solution.add_matrix_to_solution(term)
        solution.add_message()

    solution.add_message("### Результат — приближённое значение sin(A):")
    solution.add_matrix_to_solution(result)

    return result, solution.get_message()

def cosm(matrix1, terms=10):
    """
    Вычисляет косинус матрицы cos(A) через разложение в ряд Тейлора.
    terms — количество членов ряда (по умолчанию 10).
    """
    solution = Solution()
    solution.add_message('## Косинус матрицы')
    n = matrix1.shape[0]
    result = np.eye(n)
    current_power = np.eye(n)

    solution.add_message("### Вычисление косинуса матрицы cos(A) с помощью ряда Тейлора:")
    solution.add_message("Формула: cos(A) = I - A^2/2! + A^4/4! - A^6/6! + ...\n")

    for k in range(1, terms):
        power = 2 * k
        factorial = np.math.factorial(power)
        current_power = current_power @ matrix1 @ matrix1  # A^{2k}
        sign = (-1) ** k
        term = sign * current_power / factorial
        result += term

        solution.add_message(f"**Шаг {k}: {'-' if sign < 0 else '+'} A^{power} / {power}! =**")
        solution.add_matrix_to_solution(term)
        solution.add_message()

    solution.add_message("### Результат — приближённое значение cos(A):")
    solution.add_matrix_to_solution(result)

    return result, solution.get_message()

# def matrix_cos(matrix1):  # косинус матрицы
#     return cosm(matrix1)

def example(): # пример
    matrix1 = np.array([[5, 4, 2, 1], [0, 1, -1, -1], [-1, -1, 3, 0], [1, 1, -1, 2]])
    matrix2 = np.array([[5, 3, 5, 1], [3, 6, 0, -1], [-1, 3, 3, 2], [1, 3, -1, 2]])
    _, message = matrix_sum(matrix1, matrix2)
    print(message)
    _, message = matrix_mult(matrix1, matrix2)
    print(message)
    _, message = matrix_transpose(matrix1)
    print(message)
    _, _, message = matrix_eigen(matrix1)
    print(message)
    _, message = matrix_gauss_jordan_inversion(matrix1)
    print(message)
    _, message = matrix_exponential(matrix1)
    print(message)
    _, message = sinm(matrix1)
    print(message)
    _, message = cosm(matrix1)
    print(message)

if __name__ == '__main__':
    example()