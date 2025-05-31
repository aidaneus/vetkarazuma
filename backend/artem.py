from sympy import symbols, Eq, solve, Matrix, pprint
from IPython.display import display, Math

def cramer_rule(A, B):
    if A.shape[0]!=A.shape[1]:
      solutions = symbolic_sol(A,B)
      return solutions
    # Вычисление определителя главной матрицы
    det_A = A.det()
    solutions = []
    # Проверка на случай, если определитель главной матрицы равен нулю
    if det_A == 0:
        print("Определитель матрицы коэффициентов равен нулю, метод Крамера не применим.")
        solutions = gauss_rule(A,B)
        return solutions

    else:
    # Проходим по каждому столбцу матрицы и вычисляем определитель со заменой столбца на вектор значений
      for i in range(A.shape[0]):
          Ai = A.copy()
          Ai[:, i] = B
          solutions.append(Ai.det() / det_A)

    return solutions

def print_row_reduced_matrix(matrix):
    print("Ступенчатая матрица:")
    pprint(matrix)

def gauss_rule(A, B):
    # Определение расширенной матрицы [A|B]
    augmented_matrix = A.col_insert(A.shape[0], B)

    # Выполнение приведения матрицы к ступенчатому виду
    row_reduced_matrix, a = augmented_matrix.rref()
    print_row_reduced_matrix(row_reduced_matrix)


    # print(a)
    # print(row_reduced_matrix)
    for i in range(0, A.shape[0]):
      counter = 0
      for j in row_reduced_matrix.row(i):
        if j==0:
          counter+=1
      if counter==A.shape[0] and row_reduced_matrix.col(-1)[i]!=0:
        print("Решений нет, есть нулевые вектор(ы) с ненулевым ответом.")

        return 0


    # Вызов функции для вывода ступенчатой матрицы

    solutions = []
    for i in range(A.shape[0]):
        solutions.append(row_reduced_matrix.col(-1)[i])


    return solutions

def symbolic_sol(A,B):
  res_vars = ''
  rng = max(A.shape[0], A.shape[1]) + 1
  for i in range(1,rng):
    res_vars = res_vars + " x%s" %i

  my_var = symbols(res_vars)

  x = Matrix(my_var)
  A_x = []
  for i in range(0,A.shape[0]):
    A_x.append(Eq((A*x)[i], B[i]))
  symbolic_solution_1 = solve(A_x, list(x))
  # Ищем переменные, через которые выражены другие значения
  my_subs = list(x)
  my_x = list(symbolic_solution_1.keys())
  # Перебираем эти значения и подставляем в них конкретные значения, чтобы найти точки
  for i in my_x:
    my_subs.remove(i)
  # Финалный список списков с конкретными точками
  num_of_points = 6
  final_jazz = []
  for i in my_x:
    res_sol = symbolic_solution_1.get(i)
    res_x = []
    for n in range(0,num_of_points):
      xl = res_sol
      for j in my_subs:
        xl = xl.subs(j,n)
      res_x.append(xl)
    final_jazz.append(res_x)
  final_jazz
  print("Нет единственного решения:", symbolic_solution_1)
  print("Один из вариантов, при подстановке случайных чисел:\n")

  latex_code_final = r"\vec{x} = x + "
  for i in range(0, len(final_jazz)):
    vector = final_jazz[i]
    if i == 0:
      latex_code = r"\vec{t1} \begin{pmatrix} "  + r" \\ ".join(map(str, vector)) + r" \end{pmatrix}"
    else:
      latex_code = r"+\vec{t%s} \begin{pmatrix} " % (i+1) + r" \\ ".join(map(str, vector)) + r" \end{pmatrix}"
    latex_code_final = latex_code_final + latex_code
  display(Math(latex_code_final))
  return 0

# A = Matrix([
#     [4, 2, -3],
#     [-8, -7, 1],
#     [4, 2, -3]
# ])

# # Вектор значений
# B = Matrix([0, 1, 0])
# print(cramer_rule(A, B))