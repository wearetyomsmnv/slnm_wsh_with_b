  ```
@wearetyomsmnv && @ivolake
```

# Инструмент для распознавания речи и adversarial атак

## Описание

Этот инструмент предназначен для проведения adversarial атак на системы распознавания речи, использующие модель Vosk. Он позволяет создавать искаженные аудиофайлы, которые могут привести к ошибкам в распознавании речи, а также обнаруживать такие атаки.

## Требования

- Python 3.7+
- Библиотеки: numpy, torch, librosa, soundfile, vosk, scipy, matplotlib
- Модель Vosk для русского языка

## Установка

1. Установите необходимые библиотеки:

```
pip install -r requirements.txt
```

2. Скачайте модель Vosk для русского языка с сайта https://alphacephei.com/vosk/models и распакуйте ее в папку `vosk-model-ru-0.22` (или укажите путь к модели при запуске скрипта).




# PGD атаки

# 1. Базовая PGD атака
python script.py --input_file voice.wav \
                 --model_path vosk-model-ru \
                 --output_file pgd_basic.wav \
                 --attack pgd \
                 --epsilon 0.1 \
                 --alpha 0.01 \
                 --num_iter 40

# 2. PGD атака с увеличенной силой (больше epsilon)
python script.py --input_file voice.wav \
                 --model_path vosk-model-ru \
                 --output_file pgd_strong.wav \
                 --attack pgd \
                 --epsilon 0.3 \
                 --alpha 0.02 \
                 --num_iter 40

# 3. PGD атака с большим количеством итераций
python script.py --input_file voice.wav \
                 --model_path vosk-model-ru \
                 --output_file pgd_many_iter.wav \
                 --attack pgd \
                 --epsilon 0.1 \
                 --alpha 0.005 \
                 --num_iter 100

# 4. PGD атака с автоматической настройкой
python script.py --input_file voice.wav \
                 --model_path vosk-model-ru \
                 --output_file pgd_auto.wav \
                 --attack pgd \
                 --epsilon 0.1 \
                 --alpha 0.01 \
                 --num_iter 40 \
                 --auto_tune

# Carlini-Wagner атаки

# 1. Базовая Carlini-Wagner атака
python script.py --input_file voice.wav \
                 --model_path vosk-model-ru \
                 --output_file cw_basic.wav \
                 --attack carlini \
                 --num_iter 100 \
                 --learning_rate 0.01

# 2. Carlini-Wagner атака с большим количеством итераций
python script.py --input_file voice.wav \
                 --model_path vosk-model-ru \
                 --output_file cw_long.wav \
                 --attack carlini \
                 --num_iter 500 \
                 --learning_rate 0.005

# 3. Carlini-Wagner атака с автоматической настройкой
python script.py --input_file voice.wav \
                 --model_path vosk-model-ru \
                 --output_file cw_auto.wav \
                 --attack carlini \
                 --num_iter 100 \
                 --learning_rate 0.01 \
                 --auto_tune

# 4. Carlini-Wagner атака с другой частотой дискретизации
python script.py --input_file voice_44100.wav \
                 --model_path vosk-model-ru \
                 --output_file cw_44100.wav \
                 --attack carlini \
                 --num_iter 100 \
                 --learning_rate 0.01 \
                 --sample_rate 44100

# Сравнение атак (запуск нескольких атак последовательно)
for attack in pgd carlini; do
  for epsilon in 0.1 0.2 0.3; do
    python script.py --input_file voice.wav \
                     --model_path vosk-model-ru \
                     --output_file ${attack}_eps${epsilon}.wav \
                     --attack $attack \
                     --epsilon $epsilon \
                     --num_iter 100
  done
done


## Примеры использования

Ниже приведены различные примеры использования инструмента с разными параметрами и сценариями.

### 1. Базовое использование с FGSM атакой

```bash
python script_name.py --input_file speech.wav --output_file adversarial_speech.wav --attack fgsm
```

Этот пример применяет FGSM атаку к файлу speech.wav и сохраняет результат в adversarial_speech.wav.

### 2. Использование PGD атаки с настройкой параметров

```bash
python script.py --input_file speech.wav --output_file pgd_adversarial.wav --attack pgd --epsilon 0.05 --alpha 0.005 --num_iter 20
```

Здесь мы используем PGD атаку с меньшим значением epsilon, меньшим шагом alpha и большим количеством итераций.

### 3. Применение атаки Карлини-Вагнера

```bash
python script.py --input_file speech.wav --output_file cw_adversarial.wav --attack cw --num_iter 50 --learning_rate 0.005
```

Этот пример демонстрирует использование атаки Карлини-Вагнера с увеличенным количеством итераций и уменьшенной скоростью обучения.

### 4. Изменение порога обнаружения adversarial примеров

```bash
python script.py --input_file speech.wav --output_file adversarial.wav --attack fgsm --threshold 0.05
```

Здесь мы уменьшаем порог обнаружения adversarial примеров, что может привести к более чувствительному обнаружению.

### 5. Использование другой модели Vosk

```bash
python script.py --input_file speech.wav --output_file adversarial.wav --model_path path/to/custom/vosk/model
```

Этот пример показывает, как использовать другую модель Vosk, расположенную в нестандартном месте.

### 6. Обработка аудио с другой частотой дискретизации

```bash
python script.py --input_file high_quality_speech.wav --output_file adversarial.wav --attack fgsm --sample_rate 44100
```

Здесь мы обрабатываем аудиофайл с более высокой частотой дискретизации (44.1 кГц вместо стандартных 16 кГц).

### 7. Сравнение разных методов атак

Чтобы сравнить эффективность разных методов атак, вы можете выполнить следующие команды и сравнить результаты:

```bash
python script.py --input_file speech.wav --output_file fgsm_adversarial.wav --attack fgsm
python script.py --input_file speech.wav --output_file pgd_adversarial.wav --attack pgd
python script.py --input_file speech.wav --output_file cw_adversarial.wav --attack cw
```

### 8. Анализ влияния epsilon на FGSM атаку

Чтобы понять, как параметр epsilon влияет на результаты FGSM атаки, попробуйте разные значения:

```bash
python script.py --input_file speech.wav --output_file fgsm_low.wav --attack fgsm --epsilon 0.01
python script.py --input_file speech.wav --output_file fgsm_medium.wav --attack fgsm --epsilon 0.1
python script.py --input_file speech.wav --output_file fgsm_high.wav --attack fgsm --epsilon 0.5
```

Сравните получившиеся аудиофайлы и транскрипции, чтобы увидеть, как изменение epsilon влияет на качество звука и эффективность атаки.

После выполнения этих примеров, вы можете:
1. Сравнить исходные и искаженные транскрипции
2. Оценить качество звука искаженных аудиофайлов
3. Проанализировать графики различий, созданные для каждого примера
4. Сравнить значения KL-дивергенции и другие численные показатели



### Параметры командной строки

- `--input_file`: Путь к входному аудиофайлу (по умолчанию: "1.wav")
- `--output_file`: Путь для сохранения искаженного аудиофайла (по умолчанию: "adversarial_audio.wav")
- `--model_path`: Путь к модели Vosk (по умолчанию: "vosk-model-ru-0.22")
- `--attack`: Метод атаки (выбор из 'fgsm', 'pgd', 'cw', по умолчанию: 'fgsm')
- `--epsilon`: Эпсилон для атак FGSM и PGD (по умолчанию: 0.1)
- `--alpha`: Альфа для атаки PGD (по умолчанию: 0.01)
- `--num_iter`: Количество итераций для атак PGD и CW (по умолчанию: 10)
- `--learning_rate`: Скорость обучения для атаки CW (по умолчанию: 0.01)
- `--threshold`: Порог для обнаружения adversarial примеров (по умолчанию: 0.1)
- `--sample_rate`: Частота дискретизации аудио (по умолчанию: 16000)

## Функциональность

1. Загрузка и транскрибирование исходного аудиофайла
2. Применение выбранного метода adversarial атаки (FGSM, PGD или Carlini-Wagner)
3. Транскрибирование искаженного аудио
4. Сравнение исходной и искаженной транскрипций
5. Обнаружение adversarial примеров
6. Визуализация различий между исходным и искаженным аудио
7. Вычисление численных различий между исходным и искаженным аудио

## Вывод

Скрипт выводит следующую информацию:

- Исходная транскрипция
- Искаженная транскрипция
- Сходство между транскрипциями
- Результат обнаружения adversarial примера
- Значение KL-дивергенции
- Средние абсолютные различия в форме волны, спектрограмме и MFCC

Также создается график различий между исходным и искаженным аудио, который сохраняется в файл 'audio_differences.png'.



Что касается модели Vosk, вам необходимо скачать модель для русского языка с сайта https://alphacephei.com/vosk/models. Выберите модель "vosk-model-ru-0.22" или более новую версию, если она доступна. После скачивания распакуйте архив в папку "vosk-model-ru-0.22" в том же каталоге, где находится ваш скрипт, или укажите путь к модели при запуске скрипта с помощью параметра `--model_path`.
