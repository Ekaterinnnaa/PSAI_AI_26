defmodule LogisticRegressor do
  @moduledoc
  defstruct [:w1, :w2, :bias, :lr, :target_accuracy]

  def new(lr \\ 0.1, target_accuracy \\ 0.01) do
    %LogisticRegressor{
      w1: :rand.uniform() * 0.2 - 0.1,
      w2: :rand.uniform() * 0.2 - 0.1,
      bias: :rand.uniform() * 0.2 - 0.1,
      lr: lr,
      target_accuracy: target_accuracy
    }
  end

  # SIGMOID
  def sigmoid(z), do: 1.0 / (1.0 + :math.exp(-z))

  # z=w1*x1+w2*x2+bias*(-1.0)
  def predict_proba(model, x1, x2) do
    z = model.w1 * x1 + model.w2 * x2 + model.bias * -1.0
    sigmoid(z)
  end

  # Enum.reduce — это замена циклу for p in dataset sum +=
  def calculate_bce(model, dataset) do
    # защита от log0 чтобы программа не упала
    eps = 1.0e-15
    n = length(dataset)

    sum =
      Enum.reduce(dataset, 0.0, fn p, acc ->
        y_hat = predict_proba(model, p.x1, p.x2)
        y_hat_safe = max(eps, min(1.0 - eps, y_hat))
        acc + (p.label * :math.log(y_hat_safe) + (1.0 - p.label) * :math.log(1.0 - y_hat_safe))
      end)

    # возвращаем среднее значение
    -(sum / n)
  end

  # Одна эпоха: полный проход по данным с корректировкой весов.
  def train_epoch(model, dataset, use_adaptive \\ false) do
    # Расчет шага обучения (Learning Rate).
    # Если включен Adaptive Step, считаем 1 / среднюю энергию сигналов.
    current_lr =
      if use_adaptive do
        avg_energy =
          Enum.reduce(dataset, 0.0, fn p, acc ->
            acc + (:math.pow(p.x1, 2) + :math.pow(p.x2, 2) + :math.pow(-1.0, 2))
          end) / length(dataset)

        1.0 / avg_energy
      else
        model.lr
      end

    # ГРАДИЕНТНЫЙ СПУСК:
    # В Elixir переменные неизменяемы, поэтому Enum.reduce здесь работает так:
    # Берем модель -> применяем одну точку данных -> получаем НОВУЮ модель ->
    # -> передаем её на следующую точку. В конце получаем финальную версию за эпоху.
    new_model =
      Enum.reduce(dataset, model, fn p, %LogisticRegressor{} = m ->
        y_hat = predict_proba(m, p.x1, p.x2)
        # Направление и величина ошибки
        error = y_hat - p.label

        # Синтаксис %{ m | ... } создает копию структуры 'm' с новыми значениями.
        %{
          m
          | w1: m.w1 - current_lr * error * p.x1,
            w2: m.w2 - current_lr * error * p.x2,
            bias: m.bias - current_lr * error * -1.0
        }
      end)

    # Возвращаем кортеж обновленная_модель ошибка_на_этой_эпохе
    {new_model, calculate_bce(new_model, dataset)}
  end

  # В Elixir НЕТ WHILE ТАК ЧТО МЫ ИСПОЛЬЗУЕМ РЕКУРСИЮ ДЛЯ ОБУЧЕНИЯ
  def train(model, dataset, use_adaptive, epoch, history) do
    # Выполняем шаг обучения
    {updated_model, bce} = train_epoch(model, dataset, use_adaptive)

    # Вывод в консоль каждые 100 итераций
    if rem(epoch, 100) == 0 do
      IO.puts("Epoch #{epoch}: BCE = #{Float.round(bce, 6)}")
    end

    if bce <= model.target_accuracy or epoch >= 5000 do
      {updated_model, Enum.reverse([bce | history])}
    else
      train(updated_model, dataset, use_adaptive, epoch + 1, [bce | history])
    end
  end
end

dataset = [
  %{x1: 6, x2: 2, label: 0},
  %{x1: -6, x2: 2, label: 0},
  %{x1: 6, x2: -2, label: 1},
  %{x1: -6, x2: -2, label: 0}
]

initial_model = LogisticRegressor.new(0.1, 0.01)

IO.puts("Starting training (Adaptive Step)...")
{final_model, _history} = LogisticRegressor.train(initial_model, dataset, true, 1, [])

IO.puts("\nFinal Weights:")
IO.inspect(final_model)

test_point = {6.0, -2.0}
prob = LogisticRegressor.predict_proba(final_model, elem(test_point, 0), elem(test_point, 1))
class = if prob >= 0.5, do: 1, else: 0

IO.puts("\nTest Point #{inspect(test_point)}:")
IO.puts("Probability: #{Float.round(prob, 4)}")
IO.puts("Predicted Class: #{class}")
