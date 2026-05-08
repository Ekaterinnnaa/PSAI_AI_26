import numpy as np
import matplotlib.pyplot as plt
import itertools

N = 11
np.random.seed(42)

X_all = np.array(list(itertools.product([0, 1], repeat=N)), dtype=float)
y_all = np.any(X_all == 1, axis=1).astype(float)

print(f"Всего примеров в таблице истинности: {len(X_all)}")
print(f"Класс 1: {int(y_all.sum())} | Класс 0: {int((1 - y_all).sum())}\n")

indices = np.random.permutation(len(X_all))
train_size = int(0.8 * len(X_all))
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]

print(f"Обучающая выборка: {len(X_train)} примеров (класс 1: {int(y_train.sum())})")
print(f"Тестовая выборка: {len(X_test)} примеров (класс 1: {int(y_test.sum())})\n")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def total_error_mse(X, y, w, b):
    y_pred = sigmoid(X @ w + b)
    return 0.5 * np.sum((y - y_pred) ** 2)


def total_error_bce(X, y, w, b):
    y_pred = sigmoid(X @ w + b)
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def accuracy(X, y, w, b):
    y_pred = sigmoid(X @ w + b)
    return np.mean((y_pred >= 0.5).astype(float) == y)


def compute_adaptive_alpha_mse(x):
    return 1.0 / (1.0 + np.sum(x ** 2))


def train(X_tr, y_tr, X_te, y_te, loss='mse', adaptive=False, alpha_fixed=0.5,
          max_epochs=5000, tol_mse=0.01, tol_bce=0.05):
    tol = tol_mse if loss == 'mse' else tol_bce

    rng = np.random.default_rng(seed=42)
    w = rng.uniform(-0.5, 0.5, size=N)
    b = rng.uniform(-1.0, 0.0)

    train_errors = []
    test_errors = []
    epoch = 0

    for epoch in range(max_epochs):
        indices = rng.permutation(len(X_tr))

        for i in indices:
            xi, yi = X_tr[i], y_tr[i]
            z = np.dot(xi, w) + b
            out = sigmoid(z)

            if loss == 'mse':
                delta = (yi - out) * out * (1 - out)
                lr = alpha_fixed
                if adaptive:
                    lr = compute_adaptive_alpha_mse(xi)
            else:
                delta = (yi - out)
                if adaptive:
                    lr = compute_adaptive_alpha_mse(xi)
                else:
                    lr = alpha_fixed

            w += lr * delta * xi
            b += lr * delta

        if loss == 'mse':
            err_tr = total_error_mse(X_tr, y_tr, w, b)
            err_te = total_error_mse(X_te, y_te, w, b)
        else:
            err_tr = total_error_bce(X_tr, y_tr, w, b)
            err_te = total_error_bce(X_te, y_te, w, b)

        train_errors.append(err_tr)
        test_errors.append(err_te)

        if err_tr <= tol:
            print(f"  Остановка на эпохе {epoch + 1}. Ошибка: {err_tr:.6f}")
            break

    if epoch == max_epochs - 1 and err_tr > tol:
        print(f"  Достигнут лимит эпох ({max_epochs}). Ошибка: {err_tr:.6f}")

    return w, b, train_errors, test_errors, epoch + 1


print("=" * 70)
print("Обучение конфигураций (max_epochs=5000)")
print("Критерии: MSE tol=0.01, BCE tol=0.05")
print("=" * 70)

print("\nA. MSE + фиксированный шаг (α=0.5)")
w_mse_fixed, b_mse_fixed, err_mse_fixed, _, ep_mse_fixed = train(
    X_train, y_train, X_test, y_test, loss='mse', adaptive=False,
    alpha_fixed=0.5, max_epochs=5000
)

print("\nB. MSE + адаптивный шаг")
w_mse_adapt, b_mse_adapt, err_mse_adapt, _, ep_mse_adapt = train(
    X_train, y_train, X_test, y_test, loss='mse', adaptive=True,
    alpha_fixed=0.5, max_epochs=5000
)

print("\nC. BCE + фиксированный шаг (α=0.1)")
w_bce_fixed, b_bce_fixed, err_bce_fixed, _, ep_bce_fixed = train(
    X_train, y_train, X_test, y_test, loss='bce', adaptive=False,
    alpha_fixed=0.1, max_epochs=5000
)

print("\nD. BCE + адаптивный шаг (по формуле из ЛР№2)")
w_bce_adapt, b_bce_adapt, err_bce_adapt, _, ep_bce_adapt = train(
    X_train, y_train, X_test, y_test, loss='bce', adaptive=True,
    alpha_fixed=0.5, max_epochs=5000
)

print("\nE. BCE + фиксированный шаг (α=0.5) - для сравнения")
w_bce_fixed2, b_bce_fixed2, err_bce_fixed2, _, ep_bce_fixed2 = train(
    X_train, y_train, X_test, y_test, loss='bce', adaptive=False,
    alpha_fixed=0.5, max_epochs=5000
)


def evaluate_all(w, b):
    acc_train = accuracy(X_train, y_train, w, b)
    acc_test = accuracy(X_test, y_test, w, b)
    acc_full = accuracy(X_all, y_all, w, b)
    return acc_train, acc_test, acc_full


print("\n" + "=" * 90)
print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
print("=" * 90)
print(f"{'Конфигурация':<40} {'Эпохи':<10} {'Acc Train':<12} {'Acc Test':<12} {'Acc Full':<12}")
print("-" * 90)

for name, w, b, ep in [("MSE + фиксированный (α=0.5)", w_mse_fixed, b_mse_fixed, ep_mse_fixed),
                       ("MSE + адаптивный", w_mse_adapt, b_mse_adapt, ep_mse_adapt),
                       ("BCE + фиксированный (α=0.1)", w_bce_fixed, b_bce_fixed, ep_bce_fixed),
                       ("BCE + адаптивный (формула из ЛР№2)", w_bce_adapt, b_bce_adapt, ep_bce_adapt),
                       ("BCE + фиксированный (α=0.5)", w_bce_fixed2, b_bce_fixed2, ep_bce_fixed2)]:
    acc_tr, acc_te, acc_full = evaluate_all(w, b)
    print(f"{name:<40} {ep:<10} {acc_tr:<12.4f} {acc_te:<12.4f} {acc_full:<12.4f}")

plt.figure(figsize=(14, 8))

max_epochs_show = min(2500, len(err_mse_fixed))

plt.plot(range(1, len(err_mse_fixed[:max_epochs_show]) + 1),
         err_mse_fixed[:max_epochs_show],
         label='MSE + фиксированный (α=0.5)', color='blue', linestyle='-', linewidth=2)

plt.plot(range(1, len(err_mse_adapt[:max_epochs_show]) + 1),
         err_mse_adapt[:max_epochs_show],
         label='MSE + адаптивный', color='blue', linestyle='--', linewidth=2)

plt.plot(range(1, len(err_bce_fixed[:max_epochs_show]) + 1),
         err_bce_fixed[:max_epochs_show],
         label='BCE + фиксированный (α=0.1)', color='red', linestyle='-', linewidth=2)

plt.plot(range(1, len(err_bce_adapt[:max_epochs_show]) + 1),
         err_bce_adapt[:max_epochs_show],
         label='BCE + адаптивный', color='orange', linestyle='--', linewidth=2)

plt.plot(range(1, len(err_bce_fixed2[:max_epochs_show]) + 1),
         err_bce_fixed2[:max_epochs_show],
         label='BCE + фиксированный (α=0.5)', color='green', linestyle='-', linewidth=2)

plt.xlabel('Номер эпохи', fontsize=12)
plt.ylabel('Суммарная ошибка Es', fontsize=12)
plt.yscale('log')
plt.title('Сравнение сходимости MSE и BCE для логической функции OR (n=11)', fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.6)

plt.axhline(y=0.01, color='blue', linestyle=':', linewidth=1.5, label='Критерий MSE (0.01)')
plt.axhline(y=0.05, color='red', linestyle=':', linewidth=1.5, label='Критерий BCE (0.05)')

plt.legend(fontsize=9)
plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("ФИНАЛЬНЫЕ ЗНАЧЕНИЯ ОШИБОК:")
print("=" * 50)
print(f"MSE + фиксированный: {err_mse_fixed[-1]:.6f}")
print(f"MSE + адаптивный: {err_mse_adapt[-1]:.6f}")
print(f"BCE + фиксированный (α=0.1): {err_bce_fixed[-1]:.6f}")
print(f"BCE + адаптивный: {err_bce_adapt[-1]:.6f}")
print(f"BCE + фиксированный (α=0.5): {err_bce_fixed2[-1]:.6f}")


def interactive_check(w, b, name):
    print(f"\n--- Проверка сети: {name} ---")
    print(f"Введите {N} чисел (0 или 1) через пробел, либо 'q' для выхода.")
    while True:
        user_input = input("Вход: ")
        if user_input.lower() == 'q':
            break
        parts = user_input.split()
        if len(parts) != N:
            print(f"Ошибка: требуется {N} значений.")
            continue
        try:
            vec = np.array([int(p) for p in parts], dtype=float)
        except ValueError:
            print("Ошибка: вводите только 0 или 1.")
            continue

        prob = sigmoid(np.dot(vec, w_bce_fixed2) + b_bce_fixed2)
        pred_class = 1 if prob >= 0.5 else 0
        true_class = 1 if np.any(vec == 1) else 0

        print(f"  Вероятность ŷ: {prob:.6f}")
        print(f"  Предсказанный класс: {pred_class}")
        if pred_class == true_class:
            print("  Результат: Совпадает с таблицей истинности ✓")
        else:
            print("  Результат: Расхождение с таблицей истинности ✗")


interactive_check(w_bce_fixed2, b_bce_fixed2, "BCE + фиксированный шаг (α=0.5)")