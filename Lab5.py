import numpy as np
import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog
import matplotlib.pyplot as plt

# Функція активації (порогова)
def activation_function(value):
    return 1 if value >= 0 else 0

# Збереження останніх результатів
last_results = []
variant_count = 1 

def train_perceptron():
    global last_results, variant_count, weights_history, bias_history
    
    # Отримуємо значення кількості входів та навчальних наборів з текстових полів
    try:
        num_inputs = int(input_inputs.get()) if input_inputs.get() else 5
        num_samples = int(input_samples.get()) if input_samples.get() else 5
        
        # Перевірка, чи числа є додатними
        if num_inputs <= 0 or num_samples <= 0:
            raise ValueError("Введені значення повинні бути додатними.")
    except ValueError:
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, "Будь ласка, введіть правильні значення для кількості входів та наборів.\n")
        return
    
    # Генерація навчальних даних
    X = np.random.randint(0, 2, (num_samples, num_inputs))
    Y = np.random.randint(0, 2, num_samples)
    
    weights = np.random.uniform(-1, 1, num_inputs)
    bias = np.random.uniform(-1, 1)
    
    learning_rate = 0.1
    max_epochs = 100
    
    epoch = 0
    weights_history = [weights.copy()]
    bias_history = [bias] 
    
    for epoch in range(max_epochs):
        total_error = 0
        for i in range(num_samples):
            net_input = np.dot(X[i], weights) + bias
            prediction = activation_function(net_input)
            error = Y[i] - prediction
            total_error += abs(error)
            
            weights += learning_rate * error * X[i]
            bias += learning_rate * error
        
        weights_history.append(weights.copy())
        bias_history.append(bias)
        
        if total_error == 0:
            break
    
    result = f"Варіант {variant_count}:\n" \
             f"Параметри навчання:\n" \
             f"Кількість входів: {num_inputs}\n" \
             f"Кількість навчальних наборів: {num_samples}\n" \
             f"Навчальна матриця (X):\n{X}\n" \
             f"Очікувані результати (Y):\n{Y}\n" \
             f"Кількість епох: {epoch+1}\n" \
             f"Фінальні вагові коефіцієнти:\n{weights}\n" \
             f"Фінальне значення зміщення: {bias}\n"
    
    last_results.append((variant_count, epoch + 1, weights, bias, X, Y, num_inputs, num_samples, learning_rate, max_epochs))
    if len(last_results) > 4:
        last_results.pop(0)
    
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, result)
    
    variant_count += 1
    
# Функція для збереження результатів у файл
def save_results_to_file():
    if len(last_results) == 0:
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, "Немає результатів для збереження.\n")
        return
    
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'w') as file:
            # Запис основних результатів
            for r in last_results:
                file.write(f"Варіант {r[0]}:\n"
                           f"Параметри навчання:\n"
                           f"Кількість входів: {r[6]}\n"
                           f"Кількість навчальних наборів: {r[7]}\n"
                           f"Навчальна матриця (X):\n{r[4]}\n"
                           f"Очікувані результати (Y):\n{r[5]}\n"
                           f"Кількість епох: {r[1]}\n"
                           f"Фінальні вагові коефіцієнти:\n{r[2]}\n"
                           f"Фінальне значення зміщення: {r[3]}\n\n")
            
            # Додаємо аналіз останніх варіантів
            analysis_text = ""
            avg_epochs = np.mean([r[1] for r in last_results])
            min_epochs = min(last_results, key=lambda x: x[1])
            max_epochs = max(last_results, key=lambda x: x[1])
            
            for r in last_results:
                analysis_text += f"\nВаріант {r[0]}: {r[1]} епох,\n" \
                                 f"Зміщення: {r[3]:.4f}, \n" \
                                 f"Кількість входів: {len(r[4][0])}, \n" \
                                 f"Кількість навчальних наборів: {len(r[4])},\n"
            
            analysis_text += f"\nСередня кількість епох: {avg_epochs:.2f}\n"
            analysis_text += f"Найшвидше навчання: Варіант {min_epochs[0]} ({min_epochs[1]} епох)\n"
            analysis_text += f"Найтриваліше навчання: Варіант {max_epochs[0]} ({max_epochs[1]} епох)\n"
            
            file.write("\n\nАНАЛІЗ ОСТАННІХ ВАРІАНТІВ:\n")
            file.write(analysis_text)
        
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, f"Результати успішно збережені в {file_path}\n")


# Функція для перегляду останніх результатів
def show_last_results():
    results_window = tk.Toplevel(root)
    results_window.title("Останні результати")
    results_text = scrolledtext.ScrolledText(results_window, width=90, height=30)
    results_text.pack(pady=10)
    results_text.insert(tk.END, "\n\n".join([f"Варіант {r[0]}:\n"
                                            f"Параметри навчання:\n"
                                            f"Кількість входів: {r[6]}\n"
                                            f"Кількість навчальних наборів: {r[7]}\n"
                                            f"Навчальна матриця (X):\n{r[4]}\n"
                                            f"Очікувані результати (Y):\n{r[5]}\n"
                                            f"Кількість епох: {r[1]}\n"
                                            f"Фінальні вагові коефіцієнти:\n{r[2]}\n"
                                            f"Фінальне значення зміщення: {r[3]}\n" for r in last_results]))
    results_text.config(state=tk.DISABLED)

# Функція для аналізу
def analyze_last_results():
    if len(last_results) < 2:
        return
    
    analysis_window = tk.Toplevel(root)
    analysis_window.title("Аналіз останніх варіантів")
    analysis_text = scrolledtext.ScrolledText(analysis_window, width=90, height=30)
    analysis_text.pack(pady=10)
    
    avg_epochs = np.mean([r[1] for r in last_results])
    min_epochs = min(last_results, key=lambda x: x[1])
    max_epochs = max(last_results, key=lambda x: x[1])
    
    analysis_text.insert(tk.END, "Огляд останніх варіантів:\n")
    for r in last_results:
        analysis_text.insert(tk.END, f"\nВаріант {r[0]}: {r[1]} епох,\n"
                                    f"Зміщення: {r[3]:.4f}, \n"
                                    f"Кількість входів: {len(r[4][0])}, \n"
                                    f"Кількість навчальних наборів: {len(r[4])},\n")
    
    analysis_text.insert(tk.END, f"\nСередня кількість епох: {avg_epochs:.2f}\n")
    analysis_text.insert(tk.END, f"Найшвидше навчання: Варіант {min_epochs[0]} ({min_epochs[1]} епох)\n")
    analysis_text.insert(tk.END, f"Найтриваліше навчання: Варіант {max_epochs[0]} ({max_epochs[1]} епох)\n")
    analysis_text.config(state=tk.DISABLED)

# Створення графічного інтерфейсу
root = tk.Tk()
root.title("Перцептрон")
root.geometry("700x550")
root.configure(bg="#f0f0f0")

frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(pady=10)

# Поля для введення кількості входів та навчальних наборів
input_label1 = tk.Label(frame, text="Кількість входів (за замовчуванням 5):")
input_label1.grid(row=1, column=0, padx=5)

input_inputs = tk.Entry(frame)
input_inputs.grid(row=1, column=1, padx=5)

input_label2 = tk.Label(frame, text="Кількість навчальних наборів (за замовчуванням 5):")
input_label2.grid(row=2, column=0, padx=5)

input_samples = tk.Entry(frame)
input_samples.grid(row=2, column=1, padx=5)

train_button = ttk.Button(frame, text="Навчити", command=train_perceptron)
train_button.grid(row=0, column=0, padx=5)

show_results_button = ttk.Button(frame, text="Переглянути останні", command=show_last_results)
show_results_button.grid(row=0, column=1, padx=5)

analyze_button = ttk.Button(frame, text="Аналіз останніх", command=analyze_last_results)
analyze_button.grid(row=0, column=2, padx=5)

save_button = ttk.Button(frame, text="Зберегти результати", command=save_results_to_file)
save_button.grid(row=0, column=3, padx=5)

output_text = scrolledtext.ScrolledText(root, width=80, height=24, font=("Arial", 10))
output_text.pack(pady=10)

root.mainloop()
