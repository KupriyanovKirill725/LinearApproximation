import numpy as np
class LinearApproximation:
    
    def __init__(self):
        self.slope = None
        self.intercept = None
        self.mse = None
        self.rmse = None
        self.r_squared = None
        
    def fit(self, x, y):
        # Проверка входных данных
        if len(x) != len(y):
            raise ValueError("Длины массивов x и y должны совпадать!")
        
        if len(x) < 2:
            raise ValueError("Для аппроксимации нужно минимум 2 точки!")
        
        # Преобразование в numpy массивы
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        
        # Вычисление средних
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Вычисление коэффициентов
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if abs(denominator) < 1e-10:
            raise ValueError("Все значения x одинаковы, невозможно построить регрессию!")
        
        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean
        
        # Расчёт метрик качества
        self._calculate_metrics(x, y)
        
        return self.slope, self.intercept
    
    def predict(self, x):
        if self.slope is None or self.intercept is None:
            raise ValueError("Модель не обучена. Вызовите fit()")
        return self.slope * x + self.intercept
    
    def _calculate_metrics(self, x, y):
        y_pred = self.predict(x)
        n = len(y)
        
        # Среднеквадратичная ошибка (MSE)
        self.mse = np.sum((y - y_pred) ** 2) / n
        
        # Корень из среднеквадратичной ошибки (RMSE)
        self.rmse = np.sqrt(self.mse)
        
        # Коэффициент детерминации R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot < 1e-10:
            self.r_squared = 1.0
        else:
            self.r_squared = 1 - (ss_res / ss_tot)
    
    def get_equation(self):
        return f"y = {self.slope:.6f} * x + {self.intercept:.6f}"
    
    def get_metrics(self):
        return {
            'slope': self.slope,
            'intercept': self.intercept,
            'mse': self.mse,
            'rmse': self.rmse,
            'r_squared': self.r_squared
        }
