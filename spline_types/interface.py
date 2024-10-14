from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from spline_types.pspline import plot_spline

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Загружаем интерфейс из файла .ui
        uic.loadUi("spline_types/main.ui", self)
        # Привязываем кнопку "Выполнить" к методу
        self.buttonExecute.clicked.connect(self.on_execute)
        self.pushButton.clicked.connect(self.label_now)
        self.buttonExit.clicked.connect(QApplication.quit)

    # Метод, который проверяет, выбран ли p-сплайн, и выполняет действие
    def on_execute(self):
        # p-сплайн
        if self.buttonRSpline.isChecked():
            # действия пока не готовы
            #if self.buttonArcLength.isChecked():
            if self.buttonExample.isChecked():
                self.label_now()
            elif self.VariableButton.isChecked():
                self.run_slider_window()
            else:
                    self.label_output.setText("Выберите пример.")
            """
            elif self.buttonNearestPoint.isChecked():
                if self.buttonExample.isChecked():
                    self.label_now()
                elif self.VariableButton.isChecked():
                    self.run_slider_window()
                else:
                    self.label_output.setText("Выберите пример.")

            elif self.buttonDerivatives.isChecked():
                if self.buttonExample.isChecked():
                    self.label_now()
                elif self.VariableButton.isChecked():
                    self.run_slider_window()
                else:
                    self.label_output.setText("Выберите пример.")
            else:
                self.label_output.setText("Выберите действие.")
            """
        # остальные сплайны
        elif self.buttonZSpline.isChecked() or self.buttonBSpline.isChecked():
            self.label_output.setText("Z и B сплайны пока в разработке.")
        else:
            self.label_output.setText("Выберите сплайн.")

    # Метод, который вызывается при выборе p-сплайн и нажатии "Выполнить"
    def label_now(self):
        plot_spline()
        self.label_output.setText("График p-сплайн был выбран и нажата кнопка Выполнить.")

    # Метод для запуска окна с ползунками
    def run_slider_window(self):
        self.slider_window = SliderWindow()
        self.slider_window.show()

class SliderWindow(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi('spline_types/variable.ui', self)

        # Привязка сигналов к методам
        self.slider_start.valueChanged.connect(self.update_start)
        self.slider_stop.valueChanged.connect(self.update_stop)
        self.slider_num.valueChanged.connect(self.update_num)
        self.button_apply.clicked.connect(self.validate_values)

    def update_start(self, value):
        """Обновление начальной точки"""
        self.label_start.setText(f"Start: {value}")

    def update_stop(self, value):
        """Обновление конечной точки"""
        self.label_stop.setText(f"Stop: {value}")

    def update_num(self, value):
        """Обновление количества точек"""
        self.label_num.setText(f"Num: {value}")

    def validate_values(self):
        """Проверка условий и вывод значений"""
        start = self.slider_start.value()
        stop = self.slider_stop.value()
        num = self.slider_num.value()

        if start >= stop:
            print("Ошибка: Start должен быть меньше Stop!")
        elif num < 2:
            print("Ошибка: количество точек должен быть больше или равно 2!")
        else:
            print(f"Start: {start}, Stop: {stop}, Num: {num}")
            plot_spline((start, stop, num))
            self.close()

def start():
    # Создаём экземпляр QApplication один раз
    app = QApplication([])

    # Создаём главное окно
    window = MainWindow()
    window.show()

    # Запускаем цикл событий только один раз
    app.exec()