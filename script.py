import numpy as np
import torch
import librosa
import soundfile as sf
from vosk import Model, KaldiRecognizer
from scipy.stats import entropy
import json
import os
import argparse
import matplotlib.pyplot as plt


def load_audio(file_path, sample_rate=16000):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    print(f"Загружено аудио из {file_path} с частотой {sr} Гц.")
    print(f"Форма аудио: {audio.shape}, Тип данных: {audio.dtype}, Мин/Макс значения: {audio.min()}/{audio.max()}")
    return audio, sr


def transcribe_audio(audio, recognizer):
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # Преобразование в моно, если стерео
    audio = (audio * 32767).astype(np.int16)  # Преобразование в 16-битный формат
    print(f"Подготовленное аудио - Форма: {audio.shape}, Тип: {audio.dtype}, Мин/Макс: {audio.min()}/{audio.max()}")

    recognizer.AcceptWaveform(audio.tobytes())
    result = json.loads(recognizer.FinalResult())
    print(f"Результат распознавания: {result}")

    if 'text' not in result or not result['text']:
        print("Ошибка: Текст не распознан.")
        return None, 0.0

    confidence = result.get('confidence', 0.0)
    print(f"Распознанный текст: {result['text']}, Уверенность: {confidence}")
    return result['text'], confidence


def fgsm_attack(data, epsilon):
    perturbed_data = data + epsilon * np.sign(get_gradient(data))
    return np.clip(perturbed_data, -1, 1)


def pgd_attack(data, epsilon, alpha, num_iter):
    perturbed_data = data.copy()
    for _ in range(num_iter):
        grad = get_gradient(perturbed_data)
        perturbed_data += alpha * np.sign(grad)
        perturbation = np.clip(perturbed_data - data, -epsilon, epsilon)
        perturbed_data = np.clip(data + perturbation, -1, 1)
    return perturbed_data


def carlini_wagner_attack(data, num_iter=100, learning_rate=0.01):
    perturbed_data = data.copy()
    for _ in range(num_iter):
        perturbed_data += np.random.normal(0, 0.01, data.shape)
        perturbed_data = np.clip(perturbed_data, -1, 1)
    return perturbed_data


def get_gradient(audio):
    # Здесь должен быть ваш код для вычисления градиента
    return np.random.randn(*audio.shape)


def detect_adversarial(original_audio, perturbed_audio, threshold=0.1):
    original_features = librosa.feature.mfcc(y=original_audio)
    perturbed_features = librosa.feature.mfcc(y=perturbed_audio)
    kl_div = entropy(original_features.flatten(), perturbed_features.flatten())
    return kl_div > threshold, kl_div


def plot_difference(original_audio, perturbed_audio, sr, output_file):
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(original_audio, label='Original')
    plt.plot(perturbed_audio, label='Perturbed')
    plt.plot(perturbed_audio - original_audio, label='Difference')
    plt.title('Waveform Difference')
    plt.legend()

    plt.subplot(3, 1, 2)
    D_original = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
    D_perturbed = librosa.amplitude_to_db(np.abs(librosa.stft(perturbed_audio)), ref=np.max)
    plt.imshow(D_perturbed - D_original, aspect='auto', origin='lower', cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram Difference')

    plt.subplot(3, 1, 3)
    mfcc_original = librosa.feature.mfcc(y=original_audio)
    mfcc_perturbed = librosa.feature.mfcc(y=perturbed_audio)
    plt.imshow(mfcc_perturbed - mfcc_original, aspect='auto', origin='lower', cmap='coolwarm')
    plt.colorbar()
    plt.title('MFCC Difference')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def calculate_attack_success_probability(original_confidence, perturbed_confidence):
    if original_confidence == 0:
        return 1.0 if perturbed_confidence == 0 else 0.0
    return max(0, (original_confidence - perturbed_confidence) / original_confidence)


def auto_tune_attack(audio, recognizer, attack_type, initial_params, target_confidence_drop=0.2, max_iterations=10):
    original_transcription, original_confidence = transcribe_audio(audio, recognizer)

    params = initial_params.copy()
    best_params = None
    best_confidence_drop = 0

    for _ in range(max_iterations):
        if attack_type == "fgsm":
            perturbed_audio = fgsm_attack(audio, params['epsilon'])
        elif attack_type == "pgd":
            perturbed_audio = pgd_attack(audio, params['epsilon'], params['alpha'], params['num_iter'])
        elif attack_type == "carlini":
            perturbed_audio = carlini_wagner_attack(audio, params['num_iter'], params['learning_rate'])

        _, perturbed_confidence = transcribe_audio(perturbed_audio, recognizer)
        confidence_drop = original_confidence - perturbed_confidence

        if confidence_drop > best_confidence_drop and confidence_drop <= target_confidence_drop:
            best_params = params.copy()
            best_confidence_drop = confidence_drop

        if confidence_drop >= target_confidence_drop:
            break

        # Увеличиваем параметры, если искажение недостаточно сильное
        for key in params:
            params[key] *= 1.5

    return best_params if best_params else params


def main(args):
    if not os.path.exists(args.model_path):
        print(f"Пожалуйста, скачайте модель Vosk для русского языка и распакуйте в папку {args.model_path}")
        return

    model = Model(args.model_path)
    recognizer = KaldiRecognizer(model, args.sample_rate)

    audio, sr = load_audio(args.input_file)

    # Распознавание текста из оригинального аудио
    print("Начинаем распознавание оригинального аудио...")
    original_transcription, original_confidence = transcribe_audio(audio, recognizer)

    if original_transcription is None:
        print("Не удалось распознать оригинальное аудио. Программа завершается.")
        return

    print(f"Оригинальная транскрипция: {original_transcription}")
    print(f"Уверенность в оригинальном распознавании: {original_confidence}")

    # Автоматический подбор параметров атаки
    if args.auto_tune:
        print("Автоматический подбор параметров атаки...")
        initial_params = {
            'epsilon': args.epsilon,
            'alpha': args.alpha,
            'num_iter': args.num_iter,
            'learning_rate': args.learning_rate
        }
        best_params = auto_tune_attack(audio, recognizer, args.attack, initial_params)
        print(f"Подобранные параметры: {best_params}")

        # Обновляем аргументы наилучшими параметрами
        for key, value in best_params.items():
            setattr(args, key, value)

    # Применение атаки
    print(f"Применение атаки {args.attack}...")
    if args.attack == "fgsm":
        perturbed_audio = fgsm_attack(audio, args.epsilon)
    elif args.attack == "pgd":
        perturbed_audio = pgd_attack(audio, args.epsilon, args.alpha, args.num_iter)
    elif args.attack == "carlini":
        perturbed_audio = carlini_wagner_attack(audio, args.num_iter, args.learning_rate)

    # Сохранение искаженного аудио
    sf.write(args.output_file, perturbed_audio, sr)
    print(f"Искаженное аудио сохранено в {args.output_file}")

    # Попытка распознать искаженное аудио
    print("Попытка распознать искаженное аудио...")
    perturbed_transcription, perturbed_confidence = transcribe_audio(perturbed_audio, recognizer)

    if perturbed_transcription is None:
        print("Не удалось распознать искаженное аудио.")
        perturbed_confidence = 0.0
    else:
        print(f"Транскрипция искаженного аудио: {perturbed_transcription}")
        print(f"Уверенность в распознавании искаженного аудио: {perturbed_confidence}")

    # Расчет вероятности успеха атаки
    attack_success_probability = calculate_attack_success_probability(original_confidence, perturbed_confidence)
    print(f"Вероятность успеха атаки: {attack_success_probability:.2%}")

    # Визуализация результатов
    plot_difference(audio, perturbed_audio, sr, 'difference_plot.png')
    print("График различий сохранен в difference_plot.png")

    # Обнаружение состязательного примера
    is_adversarial, kl_div = detect_adversarial(audio, perturbed_audio)
    print(f"Обнаружен состязательный пример: {is_adversarial}, KL-расхождение: {kl_div}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Путь к входному аудиофайлу')
    parser.add_argument('--model_path', type=str, required=True, help='Путь к модели Vosk')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Частота дискретизации')
    parser.add_argument('--output_file', type=str, required=True, help='Путь к выходному аудиофайлу с искажением')
    parser.add_argument('--attack', type=str, choices=['fgsm', 'pgd', 'carlini'], required=True,
                        help='Тип атаки (fgsm, pgd или carlini)')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Параметр для FGSM и PGD атаки')
    parser.add_argument('--alpha', type=float, default=0.01, help='Шаг для PGD атаки')
    parser.add_argument('--num_iter', type=int, default=100, help='Количество итераций для PGD и Carlini атаки')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Скорость обучения для Carlini-Wagner атаки')
    parser.add_argument('--auto_tune', action='store_true', help='Автоматический подбор параметров атаки')

    args = parser.parse_args()

    main(args)