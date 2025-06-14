## Сложение матриц
### Матрица 1:
```
[   5.0000    4.0000    2.0000    1.0000 ]
[   0.0000    1.0000   -1.0000   -1.0000 ]
[  -1.0000   -1.0000    3.0000    0.0000 ]
[   1.0000    1.0000   -1.0000    2.0000 ]
```
### Матрица 2:
```
[   5.0000    3.0000    5.0000    1.0000 ]
[   3.0000    6.0000    0.0000   -1.0000 ]
[  -1.0000    3.0000    3.0000    2.0000 ]
[   1.0000    3.0000   -1.0000    2.0000 ]
```
### Пошаговое сложение (в виде матрицы):
```
[      5+5       4+3       2+5       1+1 ]
[      0+3       1+6      -1+0     -1+-1 ]
[    -1+-1      -1+3       3+3       0+2 ]
[      1+1       1+3     -1+-1       2+2 ]
```
### Результат сложения:
```
[  10.0000    7.0000    7.0000    2.0000 ]
[   3.0000    7.0000   -1.0000   -2.0000 ]
[  -2.0000    2.0000    6.0000    2.0000 ]
[   2.0000    4.0000   -2.0000    4.0000 ]
```
## Умножение матриц
### Матрица 1:
```
[   5.0000    4.0000    2.0000    1.0000 ]
[   0.0000    1.0000   -1.0000   -1.0000 ]
[  -1.0000   -1.0000    3.0000    0.0000 ]
[   1.0000    1.0000   -1.0000    2.0000 ]
```
### Матрица 2:
```
[   5.0000    3.0000    5.0000    1.0000 ]
[   3.0000    6.0000    0.0000   -1.0000 ]
[  -1.0000    3.0000    3.0000    2.0000 ]
[   1.0000    3.0000   -1.0000    2.0000 ]
```
### Пошаговое умножение (вычисление каждого элемента):

**C[0,0] = 5\*5 + 4\*3 + 2\*-1 + 1\*1 = 36**

**C[0,1] = 5\*3 + 4\*6 + 2\*3 + 1\*3 = 48**

**C[0,2] = 5\*5 + 4\*0 + 2\*3 + 1\*-1 = 30**

**C[0,3] = 5\*1 + 4\*-1 + 2\*2 + 1\*2 = 7**

**C[1,0] = 0\*5 + 1\*3 + -1\*-1 + -1\*1 = 3**

**C[1,1] = 0\*3 + 1\*6 + -1\*3 + -1\*3 = 0**

**C[1,2] = 0\*5 + 1\*0 + -1\*3 + -1\*-1 = -2**

**C[1,3] = 0\*1 + 1\*-1 + -1\*2 + -1\*2 = -5**

**C[2,0] = -1\*5 + -1\*3 + 3\*-1 + 0\*1 = -11**

**C[2,1] = -1\*3 + -1\*6 + 3\*3 + 0\*3 = 0**

**C[2,2] = -1\*5 + -1\*0 + 3\*3 + 0\*-1 = 4**

**C[2,3] = -1\*1 + -1\*-1 + 3\*2 + 0\*2 = 6**

**C[3,0] = 1\*5 + 1\*3 + -1\*-1 + 2\*1 = 11**

**C[3,1] = 1\*3 + 1\*6 + -1\*3 + 2\*3 = 12**

**C[3,2] = 1\*5 + 1\*0 + -1\*3 + 2\*-1 = 0**

**C[3,3] = 1\*1 + 1\*-1 + -1\*2 + 2\*2 = 2**

### Результат умножения:
```
[  36.0000   48.0000   30.0000    7.0000 ]
[   3.0000    0.0000   -2.0000   -5.0000 ]
[ -11.0000    0.0000    4.0000    6.0000 ]
[  11.0000   12.0000    0.0000    2.0000 ]
```
## Транспонирование матрицы
### Исходная матрица:
```
[   5.0000    4.0000    2.0000    1.0000 ]
[   0.0000    1.0000   -1.0000   -1.0000 ]
[  -1.0000   -1.0000    3.0000    0.0000 ]
[   1.0000    1.0000   -1.0000    2.0000 ]
```
### Пошаговое транспонирование:
Меняем местами строки и столбцы: элемент [i, j] → [j, i]

**(0, 0) = 5 → позиция (0, 0)**

**(0, 1) = 4 → позиция (1, 0)**

**(0, 2) = 2 → позиция (2, 0)**

**(0, 3) = 1 → позиция (3, 0)**

**(1, 0) = 0 → позиция (0, 1)**

**(1, 1) = 1 → позиция (1, 1)**

**(1, 2) = -1 → позиция (2, 1)**

**(1, 3) = -1 → позиция (3, 1)**

**(2, 0) = -1 → позиция (0, 2)**

**(2, 1) = -1 → позиция (1, 2)**

**(2, 2) = 3 → позиция (2, 2)**

**(2, 3) = 0 → позиция (3, 2)**

**(3, 0) = 1 → позиция (0, 3)**

**(3, 1) = 1 → позиция (1, 3)**

**(3, 2) = -1 → позиция (2, 3)**

**(3, 3) = 2 → позиция (3, 3)**
### Результат транспонирования:
```
[   5.0000    0.0000   -1.0000    1.0000 ]
[   4.0000    1.0000   -1.0000    1.0000 ]
[   2.0000   -1.0000    3.0000   -1.0000 ]
[   1.0000   -1.0000    0.0000    2.0000 ]
```
## Собственные значения и векторы
### Матрица A:
```
[   5.0000    4.0000    2.0000    1.0000 ]
[   0.0000    1.0000   -1.0000   -1.0000 ]
[  -1.0000   -1.0000    3.0000    0.0000 ]
[   1.0000    1.0000   -1.0000    2.0000 ]
```

### Матрица (A - λI):
Matrix([[5 - λ, 4, 2, 1], [0, 1 - λ, -1, -1], [-1, -1, 3 - λ, 0], [1, 1, -1, 2 - λ]])

### Характеристический многочлен:
det(A - λI) = λ\*\*4 - 11\*λ\*\*3 + 42\*λ\*\*2 - 64\*λ + 32

### Решение уравнения det(A - λI) = 0:
- **λ_1 = 1.00**
- **λ_2 = 2.00**
- **λ_3 = 4.00**

---
### Собственный вектор для λ_1 = 1.00
Матрица (A - λI):
```
[        4         4         2         1 ]
[        0         0        -1        -1 ]
[       -1        -1         2         0 ]
[        1         1        -1         1 ]
```

Решаем систему (A - λI) * x = 0:
Базис ядра (собственный вектор):
```
[       -1 ]
[        1 ]
[        0 ]
[        0 ]
```

---
### Собственный вектор для λ_2 = 2.00
Матрица (A - λI):
```
[        3         4         2         1 ]
[        0        -1        -1        -1 ]
[       -1        -1         1         0 ]
[        1         1        -1         0 ]
```

Решаем систему (A - λI) * x = 0:
Базис ядра (собственный вектор):
```
[        1 ]
[       -1 ]
[        0 ]
[        1 ]
```

---
### Собственный вектор для λ_3 = 4.00
Матрица (A - λI):
```
[        1         4         2         1 ]
[        0        -3        -1        -1 ]
[       -1        -1        -1         0 ]
[        1         1        -1        -2 ]
```

Решаем систему (A - λI) * x = 0:
Базис ядра (собственный вектор):
```
[        1 ]
[        0 ]
[       -1 ]
[        1 ]
```

## Обратная матрица методом Гаусса-Жордана
### Исходная матрица (A | I):
```
[    5.0    4.0    2.0    1.0 |    1.0    0.0    0.0    0.0 ]
[    0.0    1.0   -1.0   -1.0 |    0.0    1.0    0.0    0.0 ]
[   -1.0   -1.0    3.0    0.0 |    0.0    0.0    1.0    0.0 ]
[    1.0    1.0   -1.0    2.0 |    0.0    0.0    0.0    1.0 ]
```

**Нормализуем строку 1 (делим на 5.00):**
```
[    1.000    0.800    0.400    0.200 |    0.200    0.000    0.000    0.000 ]
[    0.000    1.000   -1.000   -1.000 |    0.000    1.000    0.000    0.000 ]
[   -1.000   -1.000    3.000    0.000 |    0.000    0.000    1.000    0.000 ]
[    1.000    1.000   -1.000    2.000 |    0.000    0.000    0.000    1.000 ]
```

**Обнуляем элемент в строке 2: вычитаем 0.000 * строку 1**
```
[    1.000    0.800    0.400    0.200 |    0.200    0.000    0.000    0.000 ]
[    0.000    1.000   -1.000   -1.000 |    0.000    1.000    0.000    0.000 ]
[   -1.000   -1.000    3.000    0.000 |    0.000    0.000    1.000    0.000 ]
[    1.000    1.000   -1.000    2.000 |    0.000    0.000    0.000    1.000 ]
```

**Обнуляем элемент в строке 3: вычитаем -1.000 * строку 1**
```
[    1.000    0.800    0.400    0.200 |    0.200    0.000    0.000    0.000 ]
[    0.000    1.000   -1.000   -1.000 |    0.000    1.000    0.000    0.000 ]
[    0.000   -0.200    3.400    0.200 |    0.200    0.000    1.000    0.000 ]
[    1.000    1.000   -1.000    2.000 |    0.000    0.000    0.000    1.000 ]
```

**Обнуляем элемент в строке 4: вычитаем 1.000 * строку 1**
```
[    1.000    0.800    0.400    0.200 |    0.200    0.000    0.000    0.000 ]
[    0.000    1.000   -1.000   -1.000 |    0.000    1.000    0.000    0.000 ]
[    0.000   -0.200    3.400    0.200 |    0.200    0.000    1.000    0.000 ]
[    0.000    0.200   -1.400    1.800 |   -0.200    0.000    0.000    1.000 ]
```

**Нормализуем строку 2 (делим на 1.00):**
```
[    1.000    0.800    0.400    0.200 |    0.200    0.000    0.000    0.000 ]
[    0.000    1.000   -1.000   -1.000 |    0.000    1.000    0.000    0.000 ]
[    0.000   -0.200    3.400    0.200 |    0.200    0.000    1.000    0.000 ]
[    0.000    0.200   -1.400    1.800 |   -0.200    0.000    0.000    1.000 ]
```

**Обнуляем элемент в строке 1: вычитаем 0.800 * строку 2**
```
[    1.000    0.000    1.200    1.000 |    0.200   -0.800    0.000    0.000 ]
[    0.000    1.000   -1.000   -1.000 |    0.000    1.000    0.000    0.000 ]
[    0.000   -0.200    3.400    0.200 |    0.200    0.000    1.000    0.000 ]
[    0.000    0.200   -1.400    1.800 |   -0.200    0.000    0.000    1.000 ]
```

**Обнуляем элемент в строке 3: вычитаем -0.200 * строку 2**
```
[    1.000    0.000    1.200    1.000 |    0.200   -0.800    0.000    0.000 ]
[    0.000    1.000   -1.000   -1.000 |    0.000    1.000    0.000    0.000 ]
[    0.000    0.000    3.200    0.000 |    0.200    0.200    1.000    0.000 ]
[    0.000    0.200   -1.400    1.800 |   -0.200    0.000    0.000    1.000 ]
```

**Обнуляем элемент в строке 4: вычитаем 0.200 * строку 2**
```
[    1.000    0.000    1.200    1.000 |    0.200   -0.800    0.000    0.000 ]
[    0.000    1.000   -1.000   -1.000 |    0.000    1.000    0.000    0.000 ]
[    0.000    0.000    3.200    0.000 |    0.200    0.200    1.000    0.000 ]
[    0.000    0.000   -1.200    2.000 |   -0.200   -0.200    0.000    1.000 ]
```

**Нормализуем строку 3 (делим на 3.20):**
```
[    1.000    0.000    1.200    1.000 |    0.200   -0.800    0.000    0.000 ]
[    0.000    1.000   -1.000   -1.000 |    0.000    1.000    0.000    0.000 ]
[    0.000    0.000    1.000    0.000 |    0.062    0.062    0.312    0.000 ]
[    0.000    0.000   -1.200    2.000 |   -0.200   -0.200    0.000    1.000 ]
```

**Обнуляем элемент в строке 1: вычитаем 1.200 * строку 3**
```
[    1.000    0.000    0.000    1.000 |    0.125   -0.875   -0.375    0.000 ]
[    0.000    1.000   -1.000   -1.000 |    0.000    1.000    0.000    0.000 ]
[    0.000    0.000    1.000    0.000 |    0.062    0.062    0.312    0.000 ]
[    0.000    0.000   -1.200    2.000 |   -0.200   -0.200    0.000    1.000 ]
```

**Обнуляем элемент в строке 2: вычитаем -1.000 * строку 3**
```
[    1.000    0.000    0.000    1.000 |    0.125   -0.875   -0.375    0.000 ]
[    0.000    1.000    0.000   -1.000 |    0.062    1.062    0.312    0.000 ]
[    0.000    0.000    1.000    0.000 |    0.062    0.062    0.312    0.000 ]
[    0.000    0.000   -1.200    2.000 |   -0.200   -0.200    0.000    1.000 ]
```

**Обнуляем элемент в строке 4: вычитаем -1.200 * строку 3**
```
[    1.000    0.000    0.000    1.000 |    0.125   -0.875   -0.375    0.000 ]
[    0.000    1.000    0.000   -1.000 |    0.062    1.062    0.312    0.000 ]
[    0.000    0.000    1.000    0.000 |    0.062    0.062    0.312    0.000 ]
[    0.000    0.000    0.000    2.000 |   -0.125   -0.125    0.375    1.000 ]
```

**Нормализуем строку 4 (делим на 2.00):**
```
[    1.000    0.000    0.000    1.000 |    0.125   -0.875   -0.375    0.000 ]
[    0.000    1.000    0.000   -1.000 |    0.062    1.062    0.312    0.000 ]
[    0.000    0.000    1.000    0.000 |    0.062    0.062    0.312    0.000 ]
[    0.000    0.000    0.000    1.000 |   -0.062   -0.062    0.188    0.500 ]
```

**Обнуляем элемент в строке 1: вычитаем 1.000 * строку 4**
```
[    1.000    0.000    0.000    0.000 |    0.188   -0.812   -0.562   -0.500 ]
[    0.000    1.000    0.000   -1.000 |    0.062    1.062    0.312    0.000 ]
[    0.000    0.000    1.000    0.000 |    0.062    0.062    0.312    0.000 ]
[    0.000    0.000    0.000    1.000 |   -0.062   -0.062    0.188    0.500 ]
```

**Обнуляем элемент в строке 2: вычитаем -1.000 * строку 4**
```
[    1.000    0.000    0.000    0.000 |    0.188   -0.812   -0.562   -0.500 ]
[    0.000    1.000    0.000    0.000 |    0.000    1.000    0.500    0.500 ]
[    0.000    0.000    1.000    0.000 |    0.062    0.062    0.312    0.000 ]
[    0.000    0.000    0.000    1.000 |   -0.062   -0.062    0.188    0.500 ]
```

**Обнуляем элемент в строке 3: вычитаем 0.000 * строку 4**
```
[    1.000    0.000    0.000    0.000 |    0.188   -0.812   -0.562   -0.500 ]
[    0.000    1.000    0.000    0.000 |    0.000    1.000    0.500    0.500 ]
[    0.000    0.000    1.000    0.000 |    0.062    0.062    0.312   -0.000 ]
[    0.000    0.000    0.000    1.000 |   -0.062   -0.062    0.188    0.500 ]
```

### Обратная матрица (правая часть):
```
[   0.1875   -0.8125   -0.5625   -0.5000 ]
[   0.0000    1.0000    0.5000    0.5000 ]
[   0.0625    0.0625    0.3125   -0.0000 ]
[  -0.0625   -0.0625    0.1875    0.5000 ]
```
## Матричная экспонента $e^A$
### Формула разложения:
$$e^A = I + A + \frac{A^2}{2!} + \frac{A^3}{3!} + \dots$$

### Шаг 0: I (единичная матрица)
```
[   1.0000    0.0000    0.0000    0.0000 ]
[   0.0000    1.0000    0.0000    0.0000 ]
[   0.0000    0.0000    1.0000    0.0000 ]
[   0.0000    0.0000    0.0000    1.0000 ]
```
### Шаг 1: $A^1 / 1!$
```
[   5.0000    4.0000    2.0000    1.0000 ]
[   0.0000    1.0000   -1.0000   -1.0000 ]
[  -1.0000   -1.0000    3.0000    0.0000 ]
[   1.0000    1.0000   -1.0000    2.0000 ]
```
### Шаг 2: $A^2 / 2!$
```
[  12.0000   11.5000    5.5000    1.5000 ]
[   0.0000    0.5000   -1.5000   -1.5000 ]
[  -4.0000   -4.0000    4.0000    0.0000 ]
[   4.0000    4.0000   -2.0000    2.0000 ]
```
### Шаг 3: $A^3 / 3!$
```
[  18.6667   18.5000    9.1667    1.1667 ]
[   0.0000    0.1667   -1.1667   -1.1667 ]
[  -8.0000   -8.0000    2.6667    0.0000 ]
[   8.0000    8.0000   -1.3333    1.3333 ]
```
### Шаг 4: $A^4 / 4!$
```
[  21.3333   21.2917   11.2917    0.6250 ]
[   0.0000    0.0417   -0.6250   -0.6250 ]
[ -10.6667  -10.6667    0.0000    0.0000 ]
[  10.6667   10.6667    0.6667    0.6667 ]
```
### Шаг 5: $A^5 / 5!$
```
[  19.2000   19.1917   10.9250    0.2583 ]
[   0.0000    0.0083   -0.2583   -0.2583 ]
[ -10.6667  -10.6667   -2.1333    0.0000 ]
[  10.6667   10.6667    2.4000    0.2667 ]
```
### Шаг 6: $A^6 / 6!$
```
[  14.2222   14.2208    8.6208    0.0875 ]
[   0.0000    0.0014   -0.0875   -0.0875 ]
[  -8.5333   -8.5333   -2.8444    0.0000 ]
[   8.5333    8.5333    2.9333    0.0889 ]
```
### Шаг 7: $A^7 / 7!$
```
[   8.9397    8.9395    5.7141    0.0252 ]
[   0.0000    0.0002   -0.0252   -0.0252 ]
[  -5.6889   -5.6889   -2.4381    0.0000 ]
[   5.6889    5.6889    2.4635    0.0254 ]
```
### Шаг 8: $A^8 / 8!$
```
[   4.8762    4.8762    3.2571    0.0063 ]
[   0.0000    0.0000   -0.0063   -0.0063 ]
[  -3.2508   -3.2508   -1.6254    0.0000 ]
[   3.2508    3.2508    1.6317    0.0063 ]
```
### Шаг 9: $A^9 / 9!$
```
[   2.3478    2.3478    1.6268    0.0014 ]
[   0.0000    0.0000   -0.0014   -0.0014 ]
[  -1.6254   -1.6254   -0.9030    0.0000 ]
[   1.6254    1.6254    0.9044    0.0014 ]
```

### Приближённое значение $e^A$:
```
[ 107.5859  104.8676   58.1022    4.6704 ]
[   0.0000    2.7183   -4.6704   -4.6704 ]
[ -53.4317  -53.4317    0.7224    0.0000 ]
[  53.4317   53.4317    6.6663    7.3887 ]
```
## Синус матрицы
### Вычисление синуса матрицы sin(A) с помощью ряда Тейлора:
Формула: sin(A) = A - A^3/3! + A^5/5! - A^7/7! + ...

**Шаг 1: + A^1 / 1! =**
```
[   5.0000    4.0000    2.0000    1.0000 ]
[   0.0000    1.0000   -1.0000   -1.0000 ]
[  -1.0000   -1.0000    3.0000    0.0000 ]
[   1.0000    1.0000   -1.0000    2.0000 ]
```

**Шаг 2: - A^3 / 3! =**
```
[ -18.6667  -18.5000   -9.1667   -1.1667 ]
[   0.0000   -0.1667    1.1667    1.1667 ]
[   8.0000    8.0000   -2.6667    0.0000 ]
[  -8.0000   -8.0000    1.3333   -1.3333 ]
```

**Шаг 3: + A^5 / 5! =**
```
[  19.2000   19.1917   10.9250    0.2583 ]
[   0.0000    0.0083   -0.2583   -0.2583 ]
[ -10.6667  -10.6667   -2.1333    0.0000 ]
[  10.6667   10.6667    2.4000    0.2667 ]
```

**Шаг 4: - A^7 / 7! =**
```
[  -8.9397   -8.9395   -5.7141   -0.0252 ]
[   0.0000   -0.0002    0.0252    0.0252 ]
[   5.6889    5.6889    2.4381    0.0000 ]
[  -5.6889   -5.6889   -2.4635   -0.0254 ]
```

**Шаг 5: + A^9 / 9! =**
```
[   2.3478    2.3478    1.6268    0.0014 ]
[   0.0000    0.0000   -0.0014   -0.0014 ]
[  -1.6254   -1.6254   -0.9030    0.0000 ]
[   1.6254    1.6254    0.9044    0.0014 ]
```

**Шаг 6: - A^11 / 11! =**
```
[  -0.3940   -0.3940   -0.2890   -0.0001 ]
[   0.0000   -0.0000    0.0001    0.0001 ]
[   0.2890    0.2890    0.1839    0.0000 ]
[  -0.2890   -0.2890   -0.1839   -0.0001 ]
```

**Шаг 7: + A^13 / 13! =**
```
[   0.0458    0.0458    0.0350    0.0000 ]
[   0.0000    0.0000   -0.0000   -0.0000 ]
[  -0.0350   -0.0350   -0.0242    0.0000 ]
[   0.0350    0.0350    0.0242    0.0000 ]
```

**Шаг 8: - A^15 / 15! =**
```
[  -0.0006   -0.0006    0.0002   -0.0000 ]
[   0.0000   -0.0000    0.0000    0.0000 ]
[  -0.0002   -0.0002   -0.0010    0.0000 ]
[   0.0002    0.0002    0.0010   -0.0000 ]
```

**Шаг 9: + A^17 / 17! =**
```
[   0.0000   -0.0000    0.0000    0.0000 ]
[   0.0000    0.0000   -0.0000   -0.0000 ]
[   0.0000    0.0000    0.0000    0.0000 ]
[   0.0000    0.0000    0.0000    0.0000 ]
```

**Шаг 10: - A^19 / 19! =**
```
[   0.0000    0.0000   -0.0000   -0.0000 ]
[   0.0000   -0.0000    0.0000    0.0000 ]
[   0.0000    0.0000    0.0000    0.0000 ]
[   0.0000    0.0000   -0.0000   -0.0000 ]
```

### Результат — приближённое значение sin(A):
```
[  -1.4074   -2.2489   -0.5827    0.0678 ]
[   0.0000    0.8415   -0.0678   -0.0678 ]
[   0.6506    0.6506   -0.1063    0.0000 ]
[  -0.6506   -0.6506    1.0156    0.9093 ]
```
## Косинус матрицы
### Вычисление косинуса матрицы cos(A) с помощью ряда Тейлора:
Формула: cos(A) = I - A^2/2! + A^4/4! - A^6/6! + ...

**Шаг 1: - A^2 / 2! =**
```
[ -12.0000  -11.5000   -5.5000   -1.5000 ]
[  -0.0000   -0.5000    1.5000    1.5000 ]
[   4.0000    4.0000   -4.0000   -0.0000 ]
[  -4.0000   -4.0000    2.0000   -2.0000 ]
```

**Шаг 2: + A^4 / 4! =**
```
[  21.3333   21.2917   11.2917    0.6250 ]
[   0.0000    0.0417   -0.6250   -0.6250 ]
[ -10.6667  -10.6667    0.0000    0.0000 ]
[  10.6667   10.6667    0.6667    0.6667 ]
```

**Шаг 3: - A^6 / 6! =**
```
[ -14.2222  -14.2208   -8.6208   -0.0875 ]
[  -0.0000   -0.0014    0.0875    0.0875 ]
[   8.5333    8.5333    2.8444   -0.0000 ]
[  -8.5333   -8.5333   -2.9333   -0.0889 ]
```

**Шаг 4: + A^8 / 8! =**
```
[   4.8762    4.8762    3.2571    0.0063 ]
[   0.0000    0.0000   -0.0063   -0.0063 ]
[  -3.2508   -3.2508   -1.6254    0.0000 ]
[   3.2508    3.2508    1.6317    0.0063 ]
```

**Шаг 5: - A^10 / 10! =**
```
[  -1.0114   -1.0114   -0.7227   -0.0003 ]
[  -0.0000   -0.0000    0.0003    0.0003 ]
[   0.7224    0.7224    0.4334   -0.0000 ]
[  -0.7224   -0.7224   -0.4337   -0.0003 ]
```

**Шаг 6: + A^12 / 12! =**
```
[   0.1401    0.1401    0.1051    0.0000 ]
[   0.0000    0.0000   -0.0000   -0.0000 ]
[  -0.1051   -0.1051   -0.0701    0.0000 ]
[   0.1051    0.1051    0.0701    0.0000 ]
```

**Шаг 7: - A^14 / 14! =**
```
[  -0.0139   -0.0139   -0.0108   -0.0000 ]
[  -0.0000   -0.0000    0.0000    0.0000 ]
[   0.0108    0.0108    0.0077   -0.0000 ]
[  -0.0108   -0.0108   -0.0077   -0.0000 ]
```

**Шаг 8: + A^16 / 16! =**
```
[   0.0010    0.0010    0.0008    0.0000 ]
[   0.0000    0.0000   -0.0000   -0.0000 ]
[  -0.0008   -0.0008   -0.0006    0.0000 ]
[   0.0008    0.0008    0.0006    0.0000 ]
```

**Шаг 9: - A^18 / 18! =**
```
[  -0.0001   -0.0001   -0.0000   -0.0000 ]
[  -0.0000   -0.0000    0.0000    0.0000 ]
[   0.0000    0.0000    0.0000   -0.0000 ]
[  -0.0000   -0.0000   -0.0000   -0.0000 ]
```

### Результат — приближённое значение cos(A):
```
[   0.1032   -0.4371   -0.1996   -0.9564 ]
[   0.0000    0.5403    0.9564    0.9564 ]
[  -0.7568   -0.7568   -1.4104    0.0000 ]
[   0.7568    0.7568    0.9943   -0.4161 ]
```