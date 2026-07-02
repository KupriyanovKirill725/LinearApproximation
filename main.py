import sys
import numpy as np

from approximation import LinearApproximation

def get_user_data():
    print("\n" + "-" * 70)
    print("РУЧНОЙ ВВОД ДАННЫХ ДЛЯ АППРОКСИМАЦИИ")
    print("-" * 70)
    print("\nИнструкция:")
    print("  • Вводите пары чисел 'x, y' через запятую")
    print("  • Используйте целые числа")
    print("  • Для завершения ввода введите пустую строку")
    print("  • Минимум 2 точки для построения регрессии")
    print("-" * 70)
    
    x = []
    y = []
    counter = 1
    
    while True:
        try:
            user_input = input(f"\nТочка {counter} (x, y): ").strip()
            
            # Проверка на завершение
            if user_input == "":
                if counter == 1:
                    print("\n Вы не ввели ни одной точки!")
                    return None, None
                break
            
            # Разделение ввода
            parts = user_input.split(',')
            if len(parts) != 2:
                print("Ошибка! Введите два числа через запятую! (пример: 5, 10)")
                continue
            
            # Парсинг чисел
            x_val_str = parts[0].strip()
            y_val_str = parts[1].strip()
            
            # Проверка, что это целые числа
            try:
                x_val = int(x_val_str)
                y_val = int(y_val_str)
            except ValueError:
                print(f"Ошибка! Введите целые числа! (получено: '{x_val_str}', '{y_val_str}')")
                continue
            
            # Добавление точек
            x.append(float(x_val))
            y.append(float(y_val))
            counter += 1
            
            # Вывод подтверждения
            print(f" Точка ({x_val}, {y_val}) добавлена")
            
        except KeyboardInterrupt:
            print("\n\n Ввод прерван пользователем")
            return None, None
        except Exception as e:
            print(f"Непредвиденная ошибка: {e}")
            continue
    
    print("\n" + "-" * 70)
    print(f"Ввод завершён! Всего точек: {len(x)}")
    print("-" * 70)
    
    # Вывод введённых данных
    print("\n Введённые данные:")
    print(f"{'№':<4} {'X':<10} {'Y':<10}")
    print("-" * 25)
    for i in range(len(x)):
        print(f"{i+1:<4} {x[i]:<10.0f} {y[i]:<10.0f}")
    print("-" * 25)
    
    return np.array(x), np.array(y)

def main():
    print("\n" + "-" * 70)
    print("ЛИНЕЙНАЯ АППРОКСИМАЦИЯ МЕТОДОМ НАИМЕНЬШИХ КВАДРАТОВ")
    print("-" * 70)
    print("Версия: 1.0 (с R² и RMSE)")
    print("Разработка с использованием системы контроля версий Git")
    print("=" * 70)
    
    # Ввод данных
    x, y = get_user_data()
    
    if x is None or y is None:
        print("\n Работа завершена из-за отсутствия данных.")
        sys.exit(1)
    
    if len(x) < 2:
        print(f"\n Недостаточно данных для аппроксимации (нужно минимум 2 точки, введено {len(x)})")
        sys.exit(1)
    
    # Создание и обучение модели
    model = LinearApproximation()
    
    try:
        print("\n Выполняется аппроксимация...")
        model.fit(x, y)
        print("Аппроксимация выполнена успешно!")
    except Exception as e:
        print(f"\n Ошибка при обучении модели: {e}")
        sys.exit(1)
    
    # Получение метрик
    metrics = model.get_metrics()
    
    # Вывод результатов
    print("\n" + "-" * 70)
    print("Результаты аппроксимации")
    print("-" * 70)
    
    # Уравнение регрессии
    print(f"\n УРАВНЕНИЕ РЕГРЕССИИ:")
    print(f"   {model.get_equation()}")
    
    # Метрики качества
    print(f"\n МЕТРИКИ КАЧЕСТВА:")
    print(f"   R²  (Коэффициент детерминации):  {metrics['r_squared']:.6f}")
    print(f"   RMSE (Корень из MSE):            {metrics['rmse']:.6f}")
    
    # Оценка качества
    print("\n ОЦЕНКА КАЧЕСТВА МОДЕЛИ:")
    r2 = metrics['r_squared']
    
    if r2 >= 0.95:
        quality = "ОТЛИЧНО"
        comment = "Модель очень хорошо описывает данные"
    elif r2 >= 0.85:
        quality = "ХОРОШО"
        comment = "Модель хорошо описывает данные"
    elif r2 >= 0.7:
        quality = "УДОВЛЕТВОРИТЕЛЬНО"
        comment = "Модель приемлемо описывает данные"
    else:
        quality = "ПЛОХО"
        comment = "Модель плохо описывает данные, возможна нелинейная зависимость"
    
    print(f"   Качество: {quality}")
    print(f"   Комментарий: {comment}")
    
    print("\n" + "=" * 70)
    
    # Завершение
    print("\n РАБОТА ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 70)

if __name__ == "__main__":
    main()
