import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

class p_spline:
    def __init__(self, x=None, y=None, knots=None, degree=3, penalty_order=2, lambda_=1.0):
        """
        Инициализация объекта p_spline.

        Параметры:
        x (array-like): Данные независимой переменной.
        y (array-like): Данные зависимой переменной.
        knots (array-like): Последовательность узлов для B-сплайнов.
        degree (int): Степень базисных функций B-сплайна.
        penalty_order (int): Порядок разностного штрафа.
        lambda_ (float): Параметр сглаживания.
        """
        self.x = x
        self.y = y
        self.knots = knots
        self.degree = degree
        self.penalty_order = penalty_order
        self.lambda_ = lambda_
        self.coefficients = None
        self.spline = None

        if self.x is not None and self.y is not None:
            self.fit()

    def _difference_matrix(self, n_bases, d):
        """
        Создает разностную матрицу порядка d.

        Параметры:
        n_bases (int): Количество базисных функций B-сплайна.
        d (int): Порядок разности.

        Возвращает:
        ndarray: Разностная матрица.
        """
        D = np.eye(n_bases)
        for _ in range(d):
            D = np.diff(D, n=1, axis=0)
        return D

    def fit(self):
        """
        Аппроксимирует P-сплайн к предоставленным данным с использованием пенализованных наименьших квадратов.
        """
        if self.knots is None:
            num_internal_knots = max(int(len(self.x) / 4), 4)
            self.knots = np.linspace(min(self.x), max(self.x), num_internal_knots)
            self.knots = np.concatenate((
                [self.x[0]] * self.degree,
                self.knots,
                [self.x[-1]] * self.degree
            ))

        n = len(self.x)
        t = self.knots
        k = self.degree
        n_bases = len(t) - k - 1

        B = np.zeros((n, n_bases))
        for i in range(n_bases):
            c = np.zeros(n_bases)
            c[i] = 1
            spline = BSpline(t, c, k)
            B[:, i] = spline(self.x)

        D = self._difference_matrix(n_bases, self.penalty_order)
        P = self.lambda_ * D.T @ D
        BtB = B.T @ B
        Bty = B.T @ self.y

        A = BtB + P
        c = np.linalg.solve(A, Bty)

        self.coefficients = c
        self.spline = BSpline(t, c, k)

    def predict(self, x_new):
        """
        Предсказывает значения y для новых значений x с использованием аппроксимированного сплайна.

        Параметры:
        x_new (array-like): Новые значения x.

        Возвращает:
        ndarray: Предсказанные значения y.
        """
        if self.spline is None:
            raise ValueError("Модель еще не аппроксимирована.")
        return self.spline(x_new)

    def plot(self, x_new=None, num_points=None):
        """
        Строит график аппроксимированного P-сплайна вместе с исходными данными.

        Параметры:
        x_new (array-like): Значения X для построения сплайна.
        """
        if self.spline is None:
            raise ValueError("Модель еще не аппроксимирована.")
        if x_new is None:
            x_new = np.linspace(min(self.x), max(self.x), 100)
        y_new = self.predict(x_new)
        plt.figure()
        plt.plot(self.x, self.y, 'o', label='Данные')
        plt.plot(x_new, y_new, label='P-сплайн')
        if num_points:
            plt.title(f'Количество точек: {num_points}')
        plt.legend()
        plt.show()

    def method(self, *args, **kwargs):
        """
        Выполняет различные задачи в зависимости от количества аргументов.

        - Без аргументов: строит сплайн.
        - Один аргумент: предполагает, что это x_new, и возвращает предсказания.
        - Более одного аргумента: выводит аргументы.
        """
        if len(args) == 0:
            self.plot()
        elif len(args) == 1:
            x_new = args[0]
            return self.predict(x_new)
        else:
            print("Метод вызван с аргументами:", args)



def plot_spline(example=(0, 10, 100)):
    x = np.linspace(example[0], example[1], example[2])
    y = np.sin(x) + np.random.normal(0, 0.5, size=len(x))
    pspline = p_spline(x, y)

    # Передаем количество точек в метод plot
    pspline.plot(num_points=example[2])
    """
    pspline.method()

    # Предсказываем новые значения
    x_new = np.linspace(0, 10, 100)
    y_new = pspline.predict(x_new)

    # Используем method()
    pspline.method()  # Строит сплайн
    y_pred = pspline.method(x_new)  # Возвращает предсказания'
"""







