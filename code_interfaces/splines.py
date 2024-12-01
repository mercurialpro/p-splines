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
        if bc_type not in [None,'natural', 'clamped','cyclic']:
            raise ValueError("Поддерживаемые типы граничных условий: 'natural', 'clamped', 'cyclic'.")

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
            # Преобразуем элементы в положительные значения
            D = np.abs(D)

        # Создаем штрафную матрицу P
        P = self.lambda_ * D.T @ D

        # Основная система уравнений: (B^T B + P) c = B^T y
        BtB = B.T @ B
        Bty = B.T @ self.y
        A = BtB + P
        rhs = Bty.copy()

        # Сохраняем систему как атрибуты объекта
        self.A = A
        self.rhs = rhs

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
                self.A = np.vstack([A, B_der2_left, B_der2_right])
                self.rhs = np.hstack([rhs, 0, 0])

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
                self.A = np.vstack([A, B_der1_left, B_der1_right])
                self.rhs = np.hstack([rhs, bc_values['left'], bc_values['right']])

            elif bc_type == 'cyclic':
                self.set_cyclic_boundary_conditions()

        # Решаем систему уравнений с учетом граничных условий
        try:
            c = np.linalg.lstsq(self.A, self.rhs, rcond=None)[0]
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Ошибка при решении системы уравнений: {e}")

        self.coefficients = c
        self.spline = BSpline(self.knots, c, self.degree)

    def set_cyclic_boundary_conditions(self):
        """
		Задает циклические граничные условия для сплайна.
		"""
        if self.knots is None or self.coefficients is None:
            raise ValueError("Сплайн не был инициализирован корректно.")

        n_bases = len(self.coefficients)
        t = self.knots
        k = self.degree

        # Условия на совпадение значений сплайна на концах
        B_start = np.array([
            BSpline(t, np.eye(n_bases)[i], k)(self.x[0])
            for i in range(n_bases)
        ])
        B_end = np.array([
            BSpline(t, np.eye(n_bases)[i], k)(self.x[-1])
            for i in range(n_bases)
        ])
        continuity_row = B_start - B_end

        # Условия на совпадение первой производной
        B_der1_start = np.array([
            BSpline(t, np.eye(n_bases)[i], k).derivative(1)(self.x[0])
            for i in range(n_bases)
        ])
        B_der1_end = np.array([
            BSpline(t, np.eye(n_bases)[i], k).derivative(1)(self.x[-1])
            for i in range(n_bases)
        ])
        derivative1_row = B_der1_start - B_der1_end

        # Нормализация строковых условий
        continuity_row /= np.linalg.norm(continuity_row)
        derivative1_row /= np.linalg.norm(derivative1_row)

        # Увеличиваем вес условий цикличности
        weight = 1e3
        continuity_row *= weight
        derivative1_row *= weight

        # Логирование
        #print("Continuity row:", continuity_row)
        #print("Derivative1 row:", derivative1_row)

        # Обновляем матрицу A и правую часть rhs
        self.A = np.vstack([self.A, continuity_row, derivative1_row])
        self.rhs = np.hstack([self.rhs, 0, 0])

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


        if boundary_conditions == "cyclic":
            x_data = np.sort(np.concatenate([x_data, np.array([start, stop])]))
            y_data = np.concatenate([y_data, [y_data[0], y_data[-1]]])  # Добавляем значения на концах



        # Добавляем шум к данным, если задана доля шума
            if noise_variance > 0.0:
                noise_variance = noise_variance / 100  # Преобразуем из процентов в долю
                y_norm = np.sqrt(np.sum(y_data ** 2))  # ||y||_2
                noise_stddev = noise_variance * y_norm  # Масштабируем шум по L2-норме
                noise = np.random.normal(loc=0.0, scale=noise_stddev, size=len(y_data))  # Размер совпадает с y_data
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
        print(f"Сплайн с граничными условиями {boundary_conditions}:")
        spline_p.set_boundary_conditions(bc_type=boundary_conditions, bc_values=clamped_values)
        spline_p.plot_spline(x_range=(start, stop), num_points=200)

        if boundary_conditions == 'cyclic':
            # Вывод значений сплайна и его производной на концах
            S_start = spline_p.evaluate(x_data[0])
            S_end = spline_p.evaluate(x_data[-1])
            S_prime_start = spline_p.spline.derivative(1)(x_data[0])
            S_prime_end = spline_p.spline.derivative(1)(x_data[-1])

            print(f"S(x_start) = {S_start}, S(x_end) = {S_end}")
            print(f"S'(x_start) = {S_prime_start}, S'(x_end) = {S_prime_end}")

        # Использование специфичного метода p_spline
        spline_p.method_specific_to_p_spline()

# Подкласс BSpline для B-сплайнов
class b_spline(spline):
    def __init__(self, degree, control_points):
        self.control_points = control_points
        self.degree = degree
        super().__init__([], degree)
        self.knots = self.generate_knots()  # Генерация узлового вектора

    def generate_knots(self):
        """
        Автоматическая генерация узлового вектора.
        """
        n = len(self.control_points)  # Количество контрольных точек
        m = n + self.degree + 1  # Количество узлов
        knots = [0] * (self.degree + 1)  # Начальные узлы

        # Промежуточные узлы распределены
        interior_knots = np.linspace(1, n - self.degree - 3, m - 2 * (self.degree + 1))   # degree - 1
        # interior_knots = np.linspace(0, n - self.degree, m - 2 * (self.degree + 1))
        knots.extend(interior_knots)
        knots.extend([n - self.degree - 1] * (self.degree + 1))  # Конечные узлы
        return np.array(knots)

    def basis_function(self, i, k, t):
        if k == 0:
            return 1.0 if self.knots[i] <= t < self.knots[i + 1] else 0.0
        else:
            coeff1 = 0.0
            if self.knots[i + k] != self.knots[i]:
                coeff1 = (t - self.knots[i]) / (self.knots[i + k] - self.knots[i]) * self.basis_function(i, k - 1, t)
            coeff2 = 0.0
            if self.knots[i + k + 1] != self.knots[i + 1]:
                coeff2 = (self.knots[i + k + 1] - t) / (
                        self.knots[i + k + 1] - self.knots[i + 1]) * self.basis_function(i + 1, k - 1, t)
            return coeff1 + coeff2

    def evaluate(self, t):
        n = len(self.control_points) - 1
        result = np.zeros((len(self.control_points[0]),))

        for i in range(n + 1):
            b = self.basis_function(i, self.degree, t)
            result += b * np.array(self.control_points[i])

        return result

    def plot(self):
        t_values = np.linspace(self.knots[self.degree], self.knots[-self.degree - 1], 100)
        # t_values = np.linspace(self.knots[degree], self.knots[-degree - 1], 100)
        spline_points = np.array([self.evaluate(t) for t in t_values])
        spline_points[-1] = spline_points[-2]

        plt.figure(figsize=(8, 6))
        plt.plot(spline_points[:, 0], spline_points[:, 1], label='B-Сплайн', color='blue')

        control_points = np.array(self.control_points)
        plt.plot(control_points[:, 0], control_points[:, 1], 'ro--', label='Контрольные точки')

        plt.title("B-Сплайн")
        plt.xlabel("Ось X")
        plt.ylabel("Ось Y")
        plt.legend()
        plt.grid()
        plt.axis("equal")
        plt.show()


# Генерация случайных контрольных точек
    def generate_random_control_points(n, x_range=(0, 10), y_range=(0, 10)):
        """
        Генерирует n случайных контрольных точек.
        """
        x_coords = np.sort(np.random.uniform(x_range[0], x_range[1], n))
        y_coords = np.random.uniform(y_range[0], y_range[1], n)
        return list(zip(x_coords, y_coords))

    @staticmethod
    def plot_b_spline(degree=2, num=2):
        control_points = b_spline.generate_random_control_points(num)
        spline = b_spline(degree, control_points)
        spline.plot()


# Для отладки
if __name__ == "__main__":
    #p_spline.plot_p_spline()
    pass





