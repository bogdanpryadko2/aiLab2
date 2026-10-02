
import math
import random
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class FunctionSpec:
    key: str
    title: str
    bounds: tuple[float, float]
    global_minimum: tuple[float, float]
    global_value: float


@dataclass(frozen=True)
class GradientParams:
    function_key: str
    start_x: float
    start_y: float
    learning_rate: float
    tolerance: float
    max_iterations: int
    stop_by_gradient: bool
    stop_by_function_change: bool


@dataclass
class GradientResult:
    function_key: str
    points: np.ndarray
    values: np.ndarray
    gradient_norms: np.ndarray
    iterations: int
    converged: bool
    reason: str

    @property
    def final_point(self):
        return self.points[-1]

    @property
    def final_value(self):
        return float(self.values[-1])


@dataclass(frozen=True)
class AnnealingParams:
    function_key: str
    start_x: float
    start_y: float
    initial_temperature: float
    min_temperature: float
    cooling_rate: float
    step_scale: float
    steps_per_temperature: int
    max_iterations: int
    seed: int


@dataclass
class AnnealingResult:
    function_key: str
    points: np.ndarray
    accepted_points: np.ndarray
    best_points: np.ndarray
    best_values: np.ndarray
    temperatures: np.ndarray
    iterations: int

    @property
    def final_point(self):
        return self.best_points[-1]

    @property
    def final_value(self):
        return float(self.best_values[-1])


FUNCTIONS = {
    "rastrigin": FunctionSpec(
        "rastrigin",
        "Растригина",
        (-5.12, 5.12),
        (0.0, 0.0),
        0.0,
    ),
    "ackley": FunctionSpec(
        "ackley",
        "Экли",
        (-5.0, 5.0),
        (0.0, 0.0),
        0.0,
    ),
    "rosenbrock": FunctionSpec(
        "rosenbrock",
        "Розенброка",
        (-2.0, 2.0),
        (1.0, 1.0),
        0.0,
    ),
}


PRESETS = {
    "rastrigin": [
        ("Базовый", 3.5, 3.5, 0.01, 1e-6, 3000),
        ("Локальный минимум", 2.8, 2.8, 0.01, 1e-6, 3000),
        ("Малый шаг", -4.0, 4.0, 0.001, 1e-6, 7000),
        ("Большой шаг", 3.5, -3.5, 0.03, 1e-6, 1500),
    ],
    "ackley": [
        ("Базовый", 3.0, 3.0, 0.03, 1e-6, 3000),
        ("Далёкая точка", -4.0, 4.0, 0.02, 1e-6, 5000),
        ("Малый шаг", 2.5, -3.5, 0.005, 1e-6, 6000),
        ("Большой шаг", -3.5, -3.5, 0.10, 1e-6, 1500),
    ],
    "rosenbrock": [
        ("Классический старт", -1.2, 1.0, 0.001, 1e-6, 20000),
        ("Из оврага", -1.5, 1.5, 0.001, 1e-6, 20000),
        ("Малый шаг", -1.2, 1.0, 0.0002, 1e-6, 40000),
        ("Большой шаг", -1.2, 1.0, 0.003, 1e-6, 8000),
    ],
}


def rastrigin(x, y):
    return 20.0 + x * x + y * y - 10.0 * (
        math.cos(2.0 * math.pi * x) + math.cos(2.0 * math.pi * y)
    )


def rastrigin_gradient(x, y):
    dx = 2.0 * x + 20.0 * math.pi * math.sin(2.0 * math.pi * x)
    dy = 2.0 * y + 20.0 * math.pi * math.sin(2.0 * math.pi * y)
    return np.array([dx, dy], dtype=float)


def ackley(x, y):
    r = math.sqrt((x * x + y * y) / 2.0)
    c = (math.cos(2.0 * math.pi * x) + math.cos(2.0 * math.pi * y)) / 2.0
    return (
        -20.0 * math.exp(-0.2 * r)
        - math.exp(c)
        + math.e
        + 20.0
    )


def ackley_gradient(x, y):
    radius_sq = (x * x + y * y) / 2.0
    radius = math.sqrt(radius_sq)

    if radius < 1e-15:
        radial_x = 0.0
        radial_y = 0.0
    else:
        factor = 2.0 * math.exp(-0.2 * radius) / radius
        radial_x = factor * x
        radial_y = factor * y

    c = (math.cos(2.0 * math.pi * x) + math.cos(2.0 * math.pi * y)) / 2.0
    exp_c = math.exp(c)

    trig_x = math.pi * exp_c * math.sin(2.0 * math.pi * x)
    trig_y = math.pi * exp_c * math.sin(2.0 * math.pi * y)

    return np.array([radial_x + trig_x, radial_y + trig_y], dtype=float)


def rosenbrock(x, y):
    return 100.0 * (y - x * x) ** 2 + (1.0 - x) ** 2


def rosenbrock_gradient(x, y):
    dx = -400.0 * x * (y - x * x) + 2.0 * (x - 1.0)
    dy = 200.0 * (y - x * x)
    return np.array([dx, dy], dtype=float)


VALUE_FUNCTIONS = {
    "rastrigin": rastrigin,
    "ackley": ackley,
    "rosenbrock": rosenbrock,
}

GRADIENT_FUNCTIONS = {
    "rastrigin": rastrigin_gradient,
    "ackley": ackley_gradient,
    "rosenbrock": rosenbrock_gradient,
}


def evaluate_grid(function_key, x_grid, y_grid):
    if function_key == "rastrigin":
        return (
            20.0
            + x_grid**2
            + y_grid**2
            - 10.0
            * (
                np.cos(2.0 * np.pi * x_grid)
                + np.cos(2.0 * np.pi * y_grid)
            )
        )

    if function_key == "ackley":
        radius = np.sqrt((x_grid**2 + y_grid**2) / 2.0)
        cosine = (
            np.cos(2.0 * np.pi * x_grid)
            + np.cos(2.0 * np.pi * y_grid)
        ) / 2.0
        return (
            -20.0 * np.exp(-0.2 * radius)
            - np.exp(cosine)
            + np.e
            + 20.0
        )

    return 100.0 * (y_grid - x_grid**2) ** 2 + (1.0 - x_grid) ** 2


def run_gradient_descent(params):
    value_function = VALUE_FUNCTIONS[params.function_key]
    gradient_function = GRADIENT_FUNCTIONS[params.function_key]

    point = np.array([params.start_x, params.start_y], dtype=float)
    points = [point.copy()]
    values = [value_function(point[0], point[1])]
    gradient_norms = []

    converged = False
    reason = "Достигнуто максимальное число итераций"

    for _ in range(params.max_iterations):
        gradient = gradient_function(point[0], point[1])
        gradient_norm = float(np.linalg.norm(gradient))
        gradient_norms.append(gradient_norm)

        if not np.isfinite(gradient_norm):
            reason = "Расходимость: норма градиента стала нечисловой"
            break

        if params.stop_by_gradient and gradient_norm < params.tolerance:
            converged = True
            reason = "Критерий остановки: норма градиента < ε"
            break

        new_point = point - params.learning_rate * gradient

        if not np.all(np.isfinite(new_point)) or np.linalg.norm(new_point) > 1e8:
            reason = "Расходимость: координаты вышли за допустимый диапазон"
            break

        new_value = value_function(new_point[0], new_point[1])

        if not np.isfinite(new_value) or abs(new_value) > 1e16:
            reason = "Расходимость: значение функции стало слишком большим"
            break

        points.append(new_point.copy())
        values.append(new_value)

        if (
            params.stop_by_function_change
            and abs(values[-1] - values[-2]) < params.tolerance
        ):
            point = new_point
            converged = True
            reason = "Критерий остановки: |f(k+1) - f(k)| < ε"
            break

        point = new_point

    if len(gradient_norms) < len(points):
        last_gradient = gradient_function(points[-1][0], points[-1][1])
        gradient_norms.append(float(np.linalg.norm(last_gradient)))

    return GradientResult(
        params.function_key,
        np.asarray(points, dtype=float),
        np.asarray(values, dtype=float),
        np.asarray(gradient_norms, dtype=float),
        len(points) - 1,
        converged,
        reason,
    )


def run_simulated_annealing(params):
    rng = random.Random(params.seed)
    value_function = VALUE_FUNCTIONS[params.function_key]
    low, high = FUNCTIONS[params.function_key].bounds

    current = np.array([params.start_x, params.start_y], dtype=float)
    current_value = value_function(current[0], current[1])
    best = current.copy()
    best_value = current_value

    points = [current.copy()]
    accepted_points = [current.copy()]
    best_points = [best.copy()]
    best_values = [best_value]
    temperatures = [params.initial_temperature]

    temperature = params.initial_temperature
    iterations = 0

    while (
        temperature > params.min_temperature
        and iterations < params.max_iterations
    ):
        for _ in range(params.steps_per_temperature):
            if iterations >= params.max_iterations:
                break

            candidate = current + np.array(
                [
                    rng.gauss(0.0, params.step_scale),
                    rng.gauss(0.0, params.step_scale),
                ]
            )
            candidate = np.clip(candidate, low, high)
            candidate_value = value_function(candidate[0], candidate[1])
            delta = candidate_value - current_value

            accepted = (
                delta <= 0.0
                or rng.random() < math.exp(-delta / temperature)
            )

            if accepted:
                current = candidate
                current_value = candidate_value
                accepted_points.append(current.copy())

            if current_value < best_value:
                best = current.copy()
                best_value = current_value

            points.append(current.copy())
            best_points.append(best.copy())
            best_values.append(best_value)
            temperatures.append(temperature)
            iterations += 1

        temperature *= params.cooling_rate

    return AnnealingResult(
        params.function_key,
        np.asarray(points, dtype=float),
        np.asarray(accepted_points, dtype=float),
        np.asarray(best_points, dtype=float),
        np.asarray(best_values, dtype=float),
        np.asarray(temperatures, dtype=float),
        iterations,
    )


class SolverThread(QThread):
    gradient_completed = Signal(object)
    comparison_completed = Signal(object, object)
    failed = Signal(str)

    def __init__(self, gradient_params, annealing_params=None):
        super().__init__()
        self.gradient_params = gradient_params
        self.annealing_params = annealing_params

    def run(self):
        try:
            gradient_result = run_gradient_descent(self.gradient_params)

            if self.annealing_params is None:
                self.gradient_completed.emit(gradient_result)
                return

            annealing_result = run_simulated_annealing(self.annealing_params)
            self.comparison_completed.emit(
                gradient_result,
                annealing_result,
            )
        except Exception as error:
            self.failed.emit(str(error))


class ResultCharts(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(figsize=(9, 7))
        self.axes = self.figure.subplots(2, 2)
        super().__init__(self.figure)
        self.draw_empty()

    def draw_empty(self):
        for axis in self.axes.flat:
            axis.clear()
            axis.grid(alpha=0.25)
        self.axes[0, 0].set_title("Ландшафт функции и траектория")
        self.axes[0, 1].set_title("3D-поверхность")
        self.axes[1, 0].set_title("Значение целевой функции")
        self.axes[1, 1].set_title("Норма градиента")
        self.figure.tight_layout()
        self.draw_idle()

    def draw_gradient(self, result):
        self._draw(result, None)

    def draw_comparison(self, gradient_result, annealing_result):
        self._draw(gradient_result, annealing_result)

    def _draw(self, gradient_result, annealing_result):
        spec = FUNCTIONS[gradient_result.function_key]
        low, high = spec.bounds

        grid = np.linspace(low, high, 180)
        x_grid, y_grid = np.meshgrid(grid, grid)
        z_grid = evaluate_grid(spec.key, x_grid, y_grid)

        contour_axis = self.axes[0, 0]
        surface_axis = self.axes[0, 1]
        loss_axis = self.axes[1, 0]
        norm_axis = self.axes[1, 1]

        for axis in self.axes.flat:
            axis.clear()

        if spec.key == "rosenbrock":
            positive = np.maximum(z_grid, 1e-8)
            levels = np.geomspace(
                max(1e-4, float(np.nanmin(positive))),
                max(10.0, float(np.nanpercentile(positive, 96))),
                28,
            )
            contour_axis.contour(x_grid, y_grid, z_grid, levels=levels)
        else:
            contour_axis.contour(x_grid, y_grid, z_grid, levels=32)

        points = gradient_result.points
        contour_axis.plot(
            points[:, 0],
            points[:, 1],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label="Градиентный спуск",
        )
        contour_axis.scatter(
            [points[0, 0]],
            [points[0, 1]],
            marker="s",
            s=45,
            label="Старт",
        )

        if annealing_result is not None:
            accepted = annealing_result.accepted_points

            if len(accepted) > 600:
                indices = np.linspace(
                    0,
                    len(accepted) - 1,
                    600,
                    dtype=int,
                )
                accepted_to_draw = accepted[indices]
            else:
                accepted_to_draw = accepted

            contour_axis.plot(
                accepted_to_draw[:, 0],
                accepted_to_draw[:, 1],
                linewidth=0.9,
                alpha=0.70,
                label="Отжиг: принятые состояния",
            )

            best_x, best_y = annealing_result.final_point
            contour_axis.scatter(
                [best_x],
                [best_y],
                marker="D",
                s=70,
                label="Отжиг: лучшее решение",
            )

        contour_axis.scatter(
            [spec.global_minimum[0]],
            [spec.global_minimum[1]],
            marker="*",
            s=130,
            label="Глобальный минимум",
        )
        contour_axis.set_xlim(low, high)
        contour_axis.set_ylim(low, high)
        contour_axis.set_xlabel("X")
        contour_axis.set_ylabel("Y")
        contour_axis.set_title("Линии уровня и траектория")
        contour_axis.grid(alpha=0.2)
        contour_axis.legend(fontsize=8)

        surface_axis.remove()
        surface_axis = self.figure.add_subplot(2, 2, 2, projection="3d")
        self.axes[0, 1] = surface_axis

        stride = 3
        surface_axis.plot_surface(
            x_grid[::stride, ::stride],
            y_grid[::stride, ::stride],
            z_grid[::stride, ::stride],
            alpha=0.78,
            linewidth=0,
            antialiased=True,
        )
        z_path = np.array(
            [
                VALUE_FUNCTIONS[spec.key](x, y)
                for x, y in points
            ],
            dtype=float,
        )
        surface_axis.plot(
            points[:, 0],
            points[:, 1],
            z_path,
            linewidth=2,
            label="Градиентный спуск",
        )

        if annealing_result is not None:
            accepted = annealing_result.accepted_points

            if len(accepted) > 350:
                indices = np.linspace(
                    0,
                    len(accepted) - 1,
                    350,
                    dtype=int,
                )
                accepted_3d = accepted[indices]
            else:
                accepted_3d = accepted

            accepted_z = np.array(
                [
                    VALUE_FUNCTIONS[spec.key](x, y)
                    for x, y in accepted_3d
                ],
                dtype=float,
            )
            surface_axis.plot(
                accepted_3d[:, 0],
                accepted_3d[:, 1],
                accepted_z,
                linewidth=1.2,
                alpha=0.8,
                label="Алгоритм отжига",
            )

            best_x, best_y = annealing_result.final_point
            best_z = annealing_result.final_value
            surface_axis.scatter(
                [best_x],
                [best_y],
                [best_z],
                marker="D",
                s=45,
            )

        surface_axis.set_title("3D-поверхность и путь")
        surface_axis.legend(fontsize=8)
        surface_axis.set_xlabel("X")
        surface_axis.set_ylabel("Y")
        surface_axis.set_zlabel("f(X, Y)")

        loss_axis.plot(
            np.arange(len(gradient_result.values)),
            gradient_result.values,
            label="Градиентный спуск",
        )
        if annealing_result is not None:
            loss_axis.plot(
                np.arange(len(annealing_result.best_values)),
                annealing_result.best_values,
                label="Алгоритм отжига",
            )
            loss_axis.legend(fontsize=8)
        loss_axis.set_xlabel("Итерация")
        loss_axis.set_ylabel("f(X, Y)")
        loss_axis.set_title("Изменение целевой функции")
        loss_axis.grid(alpha=0.25)

        norm_axis.plot(
            np.arange(len(gradient_result.gradient_norms)),
            gradient_result.gradient_norms,
        )
        norm_axis.set_xlabel("Итерация")
        norm_axis.set_ylabel("||∇f||")
        norm_axis.set_title("Норма градиента")
        norm_axis.grid(alpha=0.25)

        self.figure.tight_layout()
        self.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None

        self.setWindowTitle(
            "Лабораторная работа — метод градиентного спуска"
        )
        self.resize(1450, 920)

        self._build_ui()
        self._function_changed()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QHBoxLayout(root)

        controls = QVBoxLayout()
        controls_container = QWidget()
        controls_container.setMaximumWidth(380)
        controls_container.setLayout(controls)

        params_group = QGroupBox("Параметры градиентного спуска")
        params_layout = QFormLayout(params_group)

        self.function_box = QComboBox()
        for key, spec in FUNCTIONS.items():
            self.function_box.addItem(spec.title, key)

        self.preset_box = QComboBox()

        self.start_x = self._double_spin(-1000.0, 1000.0, 4, 0.1)
        self.start_y = self._double_spin(-1000.0, 1000.0, 4, 0.1)
        self.learning_rate = self._double_spin(
            0.000001, 10.0, 6, 0.001
        )
        self.tolerance = self._double_spin(
            0.000000001, 1.0, 9, 0.000001
        )
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(1, 200000)

        self.stop_gradient = QCheckBox("По норме градиента")
        self.stop_gradient.setChecked(True)
        self.stop_function = QCheckBox("По изменению функции")
        self.stop_function.setChecked(True)

        params_layout.addRow("Тестовая функция", self.function_box)
        params_layout.addRow("Готовый тест", self.preset_box)
        params_layout.addRow("Начальная X", self.start_x)
        params_layout.addRow("Начальная Y", self.start_y)
        params_layout.addRow("Скорость обучения α", self.learning_rate)
        params_layout.addRow("Точность ε", self.tolerance)
        params_layout.addRow("Макс. итераций", self.max_iterations)
        params_layout.addRow(self.stop_gradient)
        params_layout.addRow(self.stop_function)

        compare_group = QGroupBox("Сравнение с алгоритмом отжига")
        compare_layout = QFormLayout(compare_group)

        self.compare_enabled = QCheckBox(
            "Запустить оба алгоритма на одной функции"
        )
        self.compare_enabled.setChecked(False)

        self.sa_temperature = self._double_spin(
            0.001, 10000.0, 3, 1.0
        )
        self.sa_min_temperature = self._double_spin(
            0.000001, 1000.0, 6, 0.001
        )
        self.sa_cooling = self._double_spin(
            0.001, 0.999999, 5, 0.001
        )
        self.sa_step = self._double_spin(
            0.0001, 20.0, 4, 0.05
        )
        self.sa_steps_per_temp = QSpinBox()
        self.sa_steps_per_temp.setRange(1, 10000)
        self.sa_seed = QSpinBox()
        self.sa_seed.setRange(0, 999999)

        self.sa_temperature.setValue(20.0)
        self.sa_min_temperature.setValue(0.01)
        self.sa_cooling.setValue(0.97)
        self.sa_step.setValue(0.25)
        self.sa_steps_per_temp.setValue(40)
        self.sa_seed.setValue(42)

        compare_layout.addRow(self.compare_enabled)
        compare_layout.addRow("Tmax", self.sa_temperature)
        compare_layout.addRow("Tmin", self.sa_min_temperature)
        compare_layout.addRow("Коэффициент охлаждения", self.sa_cooling)
        compare_layout.addRow("Масштаб случайного шага", self.sa_step)
        compare_layout.addRow(
            "Шагов при температуре",
            self.sa_steps_per_temp,
        )
        compare_layout.addRow("Seed", self.sa_seed)

        self.start_button = QPushButton("Запустить эксперимент")
        self.start_button.setMinimumHeight(42)

        self.status = QLabel("Готово")
        self.status.setWordWrap(True)

        self.formula = QLabel()
        self.formula.setWordWrap(True)

        controls.addWidget(params_group)
        controls.addWidget(compare_group)
        controls.addWidget(self.start_button)
        controls.addWidget(self.status)
        controls.addWidget(self.formula)
        controls.addStretch()

        self.charts = ResultCharts()

        main_layout.addWidget(controls_container)
        main_layout.addWidget(self.charts, 1)

        self.function_box.currentIndexChanged.connect(
            self._function_changed
        )
        self.preset_box.currentIndexChanged.connect(
            self._preset_changed
        )
        self.start_button.clicked.connect(self.start)

    @staticmethod
    def _double_spin(minimum, maximum, decimals, step):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setKeyboardTracking(False)
        return spin

    def _function_changed(self):
        key = self.function_box.currentData()

        self.preset_box.blockSignals(True)
        self.preset_box.clear()
        for name, *_ in PRESETS[key]:
            self.preset_box.addItem(name)
        self.preset_box.blockSignals(False)

        self.preset_box.setCurrentIndex(0)
        self._preset_changed()

        formulas = {
            "rastrigin": (
                "Растригина: f(x,y)=20+x²+y²−10(cos(2πx)+cos(2πy)); "
                "глобальный минимум: (0,0), f=0."
            ),
            "ackley": (
                "Экли: f(x,y)=−20·exp(−0.2·sqrt((x²+y²)/2)) "
                "−exp((cos(2πx)+cos(2πy))/2)+e+20; "
                "глобальный минимум: (0,0), f=0."
            ),
            "rosenbrock": (
                "Розенброка: f(x,y)=100(y−x²)²+(1−x)²; "
                "глобальный минимум: (1,1), f=0."
            ),
        }
        self.formula.setText(formulas[key])

    def _preset_changed(self):
        key = self.function_box.currentData()
        index = self.preset_box.currentIndex()
        if index < 0:
            return

        _, x, y, rate, tolerance, iterations = PRESETS[key][index]
        self.start_x.setValue(x)
        self.start_y.setValue(y)
        self.learning_rate.setValue(rate)
        self.tolerance.setValue(tolerance)
        self.max_iterations.setValue(iterations)

    def _gradient_params(self):
        if not self.stop_gradient.isChecked() and not self.stop_function.isChecked():
            raise ValueError(
                "Выберите хотя бы один критерий остановки."
            )

        return GradientParams(
            function_key=self.function_box.currentData(),
            start_x=self.start_x.value(),
            start_y=self.start_y.value(),
            learning_rate=self.learning_rate.value(),
            tolerance=self.tolerance.value(),
            max_iterations=self.max_iterations.value(),
            stop_by_gradient=self.stop_gradient.isChecked(),
            stop_by_function_change=self.stop_function.isChecked(),
        )

    def _annealing_params(self):
        if self.sa_temperature.value() <= self.sa_min_temperature.value():
            raise ValueError("Для отжига Tmax должна быть больше Tmin.")

        return AnnealingParams(
            function_key=self.function_box.currentData(),
            start_x=self.start_x.value(),
            start_y=self.start_y.value(),
            initial_temperature=self.sa_temperature.value(),
            min_temperature=self.sa_min_temperature.value(),
            cooling_rate=self.sa_cooling.value(),
            step_scale=self.sa_step.value(),
            steps_per_temperature=self.sa_steps_per_temp.value(),
            max_iterations=self.max_iterations.value(),
            seed=self.sa_seed.value(),
        )

    def start(self):
        try:
            gradient_params = self._gradient_params()
            annealing_params = (
                self._annealing_params()
                if self.compare_enabled.isChecked()
                else None
            )
        except ValueError as error:
            QMessageBox.warning(self, "Ошибка", str(error))
            return

        self.start_button.setEnabled(False)
        self.status.setText("Выполняется...")

        self.worker = SolverThread(
            gradient_params,
            annealing_params,
        )
        self.worker.gradient_completed.connect(
            self._gradient_completed
        )
        self.worker.comparison_completed.connect(
            self._comparison_completed
        )
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _gradient_completed(self, result):
        self.charts.draw_gradient(result)
        self.start_button.setEnabled(True)

        x, y = result.final_point
        spec = FUNCTIONS[result.function_key]
        distance = math.dist((x, y), spec.global_minimum)

        self.status.setText(
            f"{result.reason}\n"
            f"Итераций: {result.iterations}\n"
            f"Найденная точка: ({x:.8f}, {y:.8f})\n"
            f"f = {result.final_value:.10f}\n"
            f"Расстояние до глобального минимума: {distance:.8f}"
        )

    def _comparison_completed(self, gd, sa):
        self.charts.draw_comparison(gd, sa)
        self.start_button.setEnabled(True)

        spec = FUNCTIONS[gd.function_key]
        gx, gy = gd.final_point
        sx, sy = sa.final_point

        gd_distance = math.dist((gx, gy), spec.global_minimum)
        sa_distance = math.dist((sx, sy), spec.global_minimum)

        self.status.setText(
            "Сравнение на одной и той же тестовой функции\n"
            f"Градиентный спуск: f={gd.final_value:.10f}, "
            f"точка=({gx:.6f}, {gy:.6f}), "
            f"итераций={gd.iterations}, "
            f"расстояние до глобального минимума={gd_distance:.6f}\n"
            f"Алгоритм отжига: f={sa.final_value:.10f}, "
            f"точка=({sx:.6f}, {sy:.6f}), "
            f"итераций={sa.iterations}, "
            f"расстояние до глобального минимума={sa_distance:.6f}\n"
            f"Остановка градиентного спуска: {gd.reason}"
        )

    def _failed(self, message):
        self.start_button.setEnabled(True)
        QMessageBox.critical(self, "Ошибка выполнения", message)
        self.status.setText("Ошибка выполнения.")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
