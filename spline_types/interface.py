from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QListWidgetItem
from spline_types.splines import plot_p_spline

class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()

		# Загружаем интерфейс из файла .ui
		uic.loadUi("spline_types/main.ui", self)

		# Подключение кнопок выполнить, тест и выход к соответствующим методам
		self.buttonExecute.clicked.connect(self.on_execute)
		self.pushButton.clicked.connect(self.handle_p_spline)
		self.buttonExit.clicked.connect(QApplication.quit)

		# Список радиокнопок для типов сплайнов
		self.spline_buttons = [
			self.p_radioButton,
			self.linear_radioButton,
			self.quadratic_radioButton,
			self.cubic_radioButton,
			self.z_radioButton,
			self.b_radioButton
		]

		# Подключаем каждую радиокнопку к общему обработчику связи вида сплайна и действия
		for button in self.spline_buttons:
			button.toggled.connect(self.update_listWidget)

		# Подключаем событие для отслеживания изменений в QListWidget с действиями
		self.listWidget.itemChanged.connect(self.on_item_checked)

		# Инициализация начальных элементов списка
		self.update_listWidget()

	def update_listWidget(self):
		"""Обновляет элементы QListWidget в зависимости от выбранного типа сплайна."""
		# Очищаем предыдущие элементы
		self.listWidget.clear()

		# Добавляем элементы в зависимости от выбранного сплайна
		items = []
		if self.p_radioButton.isChecked():
			items = ["Опции для p-сплайна"]#, "Опция 2 для p-сплайна", "Опция 3 для p-сплайна"]
		elif self.linear_radioButton.isChecked():
			items = ["Опции для линейного сплайна"]
		elif self.quadratic_radioButton.isChecked():
			items = ["Опции для квадратичного сплайна"]
		elif self.cubic_radioButton.isChecked():
			items = ["Опции для кубического сплайна"]
		elif self.z_radioButton.isChecked():
			items = ["Опции для z-сплайна"]#, "Опция 2 для z-сплайна"]
		elif self.b_radioButton.isChecked():
			items = ["Опции для b-сплайна"]#, "Опция 2 для b-сплайна", "Опция 3 для b-сплайна", "Опция 4 для b-сплайна"]

		# Добавляем элементы в QListWidget с флажками
		for item_text in items:
			item = QListWidgetItem(item_text)
			item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
			item.setCheckState(Qt.CheckState.Unchecked)
			self.listWidget.addItem(item)

	def on_item_checked(self, item):
		"""Обработчик изменения состояния флажка в QListWidget."""
		state = "Выбранный" if item.checkState() == Qt.CheckState.Checked else "Снята отметка"
		print(f"{state} элемент: {item.text()}")

	def on_execute(self):
		"""Проверяет, выбран ли p-сплайн, и выполняет соответствующее действие."""
		if self.p_radioButton.isChecked():
			if self.ExampleButton.isChecked():
				self.handle_p_spline()
			elif self.VariableButton.isChecked():
				self.run_slider_window()
			else:
				self.label_output.setText("Выберите пример.")

		elif self.linear_radioButton.isChecked():
			self.handle_linear_spline()
			self.development()
		elif self.quadratic_radioButton.isChecked():
			self.handle_quadratic_spline()
			self.development()
		elif self.cubic_radioButton.isChecked():
			self.handle_cubic_spline()
			self.development()

		elif self.z_radioButton.isChecked():
			self.handle_z_spline()
			self.development()

		elif self.b_radioButton.isChecked():
			self.handle_b_spline()
			self.development()

		else:
			self.label_output.setText("Выберите тип сплайна.")


	def development(self):
		self.label_output.setText("все, кроме p-сплайна, пока в разработке.")

	def handle_p_spline(self):
		"""Выводит сообщение и отображает график для p-сплайна."""
		plot_p_spline()
		self.label_output.setText("График p-сплайн был выбран и нажата кнопка Выполнить.")

	def handle_linear_spline(self):
		pass

	def handle_quadratic_spline(self):
		pass
	def handle_cubic_spline(self):
		pass

	def handle_z_spline(self):
		pass

	def handle_b_spline(self):
		pass

	#слайдер пока реализован только для p-сплайна
	def run_slider_window(self):
		"""Открывает окно с ползунками для выбора значений."""
		self.slider_window = SliderWindow()
		self.slider_window.show()


class SliderWindow(QWidget):
	def __init__(self):
		super().__init__()
		uic.loadUi('spline_types/p_variable.ui', self)
		# Привязка сигналов ползунков и кнопки к методам
		self.slider_start.valueChanged.connect(self.update_start)
		self.slider_stop.valueChanged.connect(self.update_stop)
		self.slider_num.valueChanged.connect(self.update_num)
		self.button_apply.clicked.connect(self.validate_values)



	def update_start(self, value):
		"""Обновление начальной точки."""
		self.label_start.setText(f"Start: {value}")

	def update_stop(self, value):
		"""Обновление конечной точки."""
		self.label_stop.setText(f"Stop: {value}")

	def update_num(self, value):
		"""Обновление количества точек."""
		self.label_num.setText(f"Num: {value}")
	def update_bc(self, value):
		"""Обновление количества точек."""
		self.label_num.setText(f"Num: {value}")

	def validate_values(self):
		"""Проверка и вывод значений ползунков."""
		start, stop, num = self.slider_start.value(), self.slider_stop.value(), self.slider_num.value()
		if self.radioButton_1.isChecked():
			bc = None
		elif self.radioButton_2.isChecked():
			bc = 1
		elif self.radioButton_3.isChecked():
			bc = 2
		if start >= stop:
			print("Ошибка: Start должен быть меньше Stop!")
		elif num < 2:
			print("Ошибка: количество точек должно быть больше или равно 2!")
		else:
			print(f"Start: {start}, Stop: {stop}, Num: {num}")
			plot_p_spline(start, stop, num, bc)
			self.close()


def start():
	"""Создает и запускает приложение."""
	app = QApplication([])

	window = MainWindow()
	window.show()

	app.exec()

start()