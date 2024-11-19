import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
# Базовый класс Spline
class spline:
    def __init__(self, knots, degree, coefficients=None, dimension=1):
        """
        Инициализация базового класса Spline.

        Параметры:
        - knots (array-like): Узлы сплайна.
        - degree (int): Степень сплайна.
        - coefficients (array-like, optional): Коэффициенты сплайна.
        - dimension (int): Размерность сплайна.
        """
        self.knots = np.array(knots)
        self.degree = degree
        self.dimension = dimension
        self.coefficients = np.array(coefficients) if coefficients is not None else None

    def evaluate(self, x):
        """
        Метод для вычисления значения сплайна в точке x.
        Должен быть переопределен в подклассе.
        """
        raise NotImplementedError("Этот метод должен быть переопределен в подклассе.")

    def plot_spline(self, x_range, num_points=100):
        """
        Построение графика сплайна в указанном диапазоне.

        Параметры:
        - x_range (tuple): Кортеж (min_x, max_x) для диапазона построения.
        - num_points (int): Количество точек для построения графика.
        """
        x_vals = np.linspace(x_range[0], x_range[1], num_points)
        y_vals = self.evaluate(x_vals)

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label=f"{self.__class__.__name__} сплайн")
        if self.coefficients is not None:
            # Отображаем узлы сплайна
            plt.scatter(self.knots, self.evaluate(self.knots), color='red', label="Узлы сплайна")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Построение {self.__class__.__name__} сплайна")
        plt.legend()
        plt.grid(True)
        plt.show()

    @classmethod
    def create_p_spline(cls, x, y, knots=None, degree=3, penalty_order=2, lambda_=1.0, dimension=1):
        """
        Фабричный метод для создания объекта p_spline.

        Параметры:
        - x (array-like): Данные независимой переменной.
        - y (array-like): Данные зависимой переменной.
        - knots (array-like, optional): Узлы сплайна. Если не заданы, будут рассчитаны автоматически.
        - degree (int): Степень сплайна.
        - penalty_order (int): Порядок разностного штрафа.
        - lambda_ (float): Параметр сглаживания.
        - dimension (int): Размерность сплайна.

        Возвращает:
        - p_spline: Экземпляр подкласса p_spline.
        """
        return p_spline(x, y, knots, degree, penalty_order, lambda_, dimension)

# Подкласс линейный сплайн
class linear_spline(spline):
    def __init__(self, knots, coefficients):
        """
        Инициализация класса LinearSpline.

        Параметры:
        - knots (array-like): Узлы линейного сплайна.
        - coefficients (array-like): Значения в узлах (они же коэффициенты для линейного сплайна).
        """
        super().__init__(knots, degree=1, coefficients=coefficients)

    def evaluate(self, x):
        """
        Вычисление значения линейного сплайна в точке x с помощью линейной интерполяции.
        """
        return np.interp(x, self.knots, self.coefficients)

    def plot_spline(self, **kwargs):
        """
        Построение линейного сплайна, соединяя каждый узел отрезками напрямую.

        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.knots, self.coefficients, 'b-', label='LinearSpline сплайн')  # Линия синего цвета
        plt.plot(self.knots, self.coefficients, 'ro', label="Узлы сплайна")  # Точки узлов красные
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Построение LinearSpline сплайна")
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_linear_spline(start=0, stop=10, num=5):
        """
        Статический метод для генерации случайных точек и построения линейного сплайна.

        Параметры:
        - start (float): Начальное значение диапазона x.
        - stop (float): Конечное значение диапазона x.
        - num (int): Количество случайных точек.
        """
        # Генерация случайных значений для x и y
        x_data = np.sort(np.random.uniform(start, stop, num))  # Случайные значения x
        y_data = np.random.uniform(0, 10, num)                 # Случайные значения y

        # Создание объекта линейного сплайна
        linearspline = linear_spline(x_data, y_data)

        # Построение графика линейного сплайна
        linearspline.plot_spline()

# Подкласс p_spline
class p_spline(spline):
    def __init__(self, x, y, knots=None, degree=3, penalty_order=2, lambda_=1.0, dimension=1):
        """
        Инициализация объекта p_spline.

        Параметры:
        - x (array-like): Данные независимой переменной.
        - y (array-like): Данные зависимой переменной.
        - knots (array-like, optional): Узлы сплайна.
        - degree (int): Степень сплайна.
        - penalty_order (int): Порядок разностного штрафа.
        - lambda_ (float): Параметр сглаживания.
        - dimension (int): Размерность сплайна.
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.penalty_order = penalty_order
        self.lambda_ = lambda_
        self.spline = None

        self.knots = knots
        self.degree = degree

        # Инициализируем граничные условия как None
        self.boundary_conditions = None

        # Выполняем подгонку сплайна к данным
        self.fit()

        # Инициализируем базовый класс с найденными параметрами
        super().__init__(knots=self.knots, degree=self.degree, coefficients=self.coefficients, dimension=dimension)

    def _difference_matrix(self, n_bases, d):
        """
        Создает разностную матрицу порядка d.

        Параметры:
        - n_bases (int): Количество базисных функций B-сплайна.
        - d (int): Порядок разности.

        Возвращает:
        - ndarray: Разностная матрица.
        """
        D = np.eye(n_bases)
        for _ in range(d):
            D = np.diff(D, n=1, axis=0)
        return D

    def set_boundary_conditions(self, bc_type, bc_values=None):
        """
        Задает граничные условия для сплайна.

        Параметры:
        - bc_type (str): Тип граничных условий ('natural', 'clamped').
        - bc_values (dict, optional): Значения производных для граничных условий.
            Для 'clamped' требуется {'left': value, 'right': value}.
            Для 'natural' не нужны дополнительные значения.
        """
        if bc_type not in ['natural', 'clamped']:
            raise ValueError("Поддерживаемые типы граничных условий: 'natural', 'clamped'.")

        if bc_type == 'clamped':
            if (bc_values is None or
                'left' not in bc_values or
                'right' not in bc_values):
                raise ValueError("Для 'clamped' граничных условий необходимо предоставить 'left' и 'right' значения производных.")

        self.boundary_conditions = {
            'type': bc_type,
            'values': bc_values
        }

        # После установки граничных условий необходимо повторно выполнить подгонку
        self.fit()

    def fit(self, penalty_fun=None):
        """
        Аппроксимирует P-сплайн к данным с учетом функции штрафа.

        Параметры:
        - penalty_fun (callable, optional): Функтор для модификации разностной матрицы (например, sin, cos).
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

        # Создаем базисную матрицу B
        B = np.zeros((n, n_bases))
        for i in range(n_bases):
            c = np.zeros(n_bases)
            c[i] = 1
            spline = BSpline(t, c, k)
            B[:, i] = spline(self.x)

        # Создаем разностную матрицу
        D = self._difference_matrix(n_bases, self.penalty_order)

        # Применяем функтор к разностной матрице, если он задан
        if penalty_fun is not None:
            D = penalty_fun(D)

        # Создаем штрафную матрицу P
        P = self.lambda_ * D.T @ D

        # Основная система уравнений: (B^T B + P) c = B^T y
        BtB = B.T @ B
        Bty = B.T @ self.y
        A = BtB + P
        rhs = Bty.copy()

        # Обработка граничных условий
        if self.boundary_conditions is not None:
            bc_type = self.boundary_conditions['type']
            if bc_type == 'natural':
                # Вторая производная на концах равна нулю
                # Вычисляем вторые производные базисных функций на концах
                # Для каждого базисного сплайна вычисляем его вторую производную на границе
                B_der2_left = np.array([
                    BSpline(t, np.eye(n_bases)[i], k).derivative(2)(self.x[0])
                    for i in range(n_bases)
                ])
                B_der2_right = np.array([
                    BSpline(t, np.eye(n_bases)[i], k).derivative(2)(self.x[-1])
                    for i in range(n_bases)
                ])

                # Добавляем эти условия в систему
                A = np.vstack([A, B_der2_left, B_der2_right])
                rhs = np.hstack([rhs, 0, 0])

            elif bc_type == 'clamped':
                # Первая производная на концах задана
                bc_values = self.boundary_conditions['values']
                # Вычисляем первые производные базисных функций на концах
                B_der1_left = np.array([
                    BSpline(t, np.eye(n_bases)[i], k).derivative(1)(self.x[0])
                    for i in range(n_bases)
                ])
                B_der1_right = np.array([
                    BSpline(t, np.eye(n_bases)[i], k).derivative(1)(self.x[-1])
                    for i in range(n_bases)
                ])

                # Добавляем эти условия в систему
                A = np.vstack([A, B_der1_left, B_der1_right])
                rhs = np.hstack([rhs, bc_values['left'], bc_values['right']])

        # Решаем систему уравнений с учетом граничных условий
        try:
            c = np.linalg.lstsq(A, rhs, rcond=None)[0]
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Ошибка при решении системы уравнений: {e}")

        self.coefficients = c
        self.spline = BSpline(t, c, k)

    def evaluate(self, x):
        """
        Метод для вычисления значения сплайна в точке x.
        """
        if self.spline is None:
            raise ValueError("Сплайн еще не аппроксимирован.")
        return self.spline(x)

    def predict(self, x_new):
        """
        Предсказывает значения y для новых значений x с использованием аппроксимированного сплайна.

        Параметры:
        - x_new (array-like): Новые значения x.

        Возвращает:
        - ndarray: Предсказанные значения y.
        """
        return self.evaluate(x_new)

    def plot_spline(self, x_range=None, num_points=100):
        """
        Переопределение метода для построения графика сплайна вместе с исходными данными.

        Параметры:
        - x_range (tuple, optional): Диапазон (min_x, max_x) для построения графика. Если не задан, используется диапазон данных.
        - num_points (int): Количество точек для построения графика.
        """
        if x_range is None:
            x_range = (min(self.x), max(self.x))
        x_vals = np.linspace(x_range[0], x_range[1], num_points)
        y_vals = self.evaluate(x_vals)

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label=f"{self.__class__.__name__} сплайн")
        plt.scatter(self.x, self.y, color='red', label="Данные")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Построение {self.__class__.__name__} сплайна")
        plt.legend()
        plt.grid(True)
        plt.show()

    def method_specific_to_p_spline(self):
        """
        Пример метода, специфичного для p_spline.
        """
        #print("Это метод, специфичный для p_spline.")

    @staticmethod
    def plot_p_spline(
            start=0, stop=10, num=100, boundary_conditions=None, clamped_values=None,
            penalty_fun=None, point_gen_func="random", power_exp=2, noise_variance=0.0
    ):
        """
        Построение P-сплайна с выбором метода генерации точек и добавлением шума.

        Параметры:
        - start, stop (float): Диапазон значений x.
        - num (int): Количество точек.
        - boundary_conditions (str): Тип граничных условий ('natural', 'clamped').
        - clamped_values (dict): Значения производных для 'clamped' условий.
        - penalty_fun (callable): Функтор для модификации разностной матрицы.
        - point_gen_func (str): Метод генерации точек ('random', 'sin', 'cos', 'exp', 'power').
        - power_exp (float): Экспонента для метода 'power'.
        - noise_variance (float): Дисперсия шума (0.0 = без шума).
        """
        np.random.seed(None)  # Для случайных точек

        # Генерация точек x и y в зависимости от выбранного метода
        if point_gen_func == "random":
            x_data = np.sort(np.random.uniform(low=start, high=stop, size=num))
            y_data = np.sin(x_data) + np.random.normal(0, 0.2, size=len(x_data))
        elif point_gen_func == "sin":
            x_data = np.sort(np.random.uniform(low=start, high=stop, size=num))  # Неравноудаленные точки
            y_data = np.sin(x_data)
        elif point_gen_func == "cos":
            x_data = np.sort(np.random.uniform(low=start, high=stop, size=num))  # Неравноудаленные точки
            y_data = np.cos(x_data)
        elif point_gen_func == "exp":
            x_data = np.sort(np.random.uniform(low=start, high=stop, size=num))  # Неравноудаленные точки
            y_data = np.exp(x_data / stop)
        elif point_gen_func == "power":
            x_data = np.sort(np.random.uniform(low=start, high=stop, size=num))  # Неравноудаленные точки
            y_data = x_data ** power_exp
            print(x_data)
        else:
            raise ValueError("Неподдерживаемый метод генерации точек: " + str(point_gen_func))

        # Добавляем шум к данным, если задана доля шума
        if noise_variance > 0.0:
            noise_variance = noise_variance / 100  # Преобразуем из процентов в долю
            # Вычисляем L2-норму функции y
            y_norm = np.sqrt(np.sum(y_data ** 2))  # ||y||_2
            # Генерируем шум
            noise_stddev = noise_variance * y_norm  # Масштабируем шум по L2-норме
            noise = np.random.normal(loc=0.0, scale=noise_stddev, size=num)
            # Добавляем шум к данным
            y_data += noise

        # Создание объекта p_spline
        spline_p = spline.create_p_spline(
            x=x_data,
            y=y_data,
            degree=3,
            penalty_order=2,
            lambda_=1.0
        )
        # Выполняем подгонку с функцией штрафа
        spline_p.fit(penalty_fun=penalty_fun)

        # Построение графика с учетом граничных условий
        if boundary_conditions == 1:
            spline_p.set_boundary_conditions(bc_type='natural')
            spline_p.plot_spline(x_range=(start, stop), num_points=200)
            print("Сплайн с граничными условиями 'natural':")
        elif boundary_conditions == 2:
            if clamped_values is None:
                clamped_values = {'left': 0.0, 'right': 0.0}  # Значения по умолчанию
            spline_p.set_boundary_conditions(bc_type='clamped', bc_values=clamped_values)
            spline_p.plot_spline(x_range=(start, stop), num_points=200)
            print(f"Сплайн с граничными условиями 'clamped': {clamped_values}")
        else:
            spline_p.plot_spline(x_range=(start, stop), num_points=200)
            print("Сплайн без граничных условий:")

        # Использование специфичного метода p_spline
        spline_p.method_specific_to_p_spline()


# Для отладки
if __name__ == "__main__":
    #p_spline.plot_p_spline()
    #linear_spline.plot_linear_spline(0, 30, 100)
    pass





