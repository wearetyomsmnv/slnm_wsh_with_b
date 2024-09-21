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
from difflib import SequenceMatcher

def load_audio(file_path, sample_rate=16000):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr

def transcribe_audio(audio, recognizer):
    recognizer.AcceptWaveform(audio.astype(np.int16).tobytes())
    result = json.loads(recognizer.Result())
    return result['text']

def fgsm_attack(data, epsilon, data_grad):
    perturbed_data = data + epsilon * np.sign(data_grad)
    return np.clip(perturbed_data, -1, 1)

def pgd_attack(data, epsilon, alpha, num_iter, data_grad):
    perturbed_data = data.copy()
    for _ in range(num_iter):
        grad = get_gradient(perturbed_data)
        perturbed_data = perturbed_data + alpha * np.sign(grad)
        perturbation = np.clip(perturbed_data - data, -epsilon, epsilon)
        perturbed_data = np.clip(data + perturbation, -1, 1)
    return perturbed_data

def carlini_wagner_attack(data, target, num_iter=100, learning_rate=0.01):
    perturbed_data = data.copy()
    for _ in range(num_iter):
        perturbed_data += np.random.normal(0, 0.01, data.shape)
        perturbed_data = np.clip(perturbed_data, -1, 1)
    return perturbed_data

def get_gradient(audio):
    return np.random.randn(*audio.shape)

def detect_adversarial(original_audio, perturbed_audio, threshold=0.1):
    original_features = librosa.feature.mfcc(y=original_audio)
    perturbed_features = librosa.feature.mfcc(y=perturbed_audio)
    kl_div = entropy(original_features.flatten(), perturbed_features.flatten())
    return kl_div > threshold, kl_div

def plot_difference(original_audio, perturbed_audio, sr, output_file):
    plt.figure(figsize=(15, 10))

    # Plot waveform difference
    plt.subplot(3, 1, 1)
    plt.plot(original_audio, label='Original')
    plt.plot(perturbed_audio, label='Perturbed')
    plt.plot(perturbed_audio - original_audio, label='Difference')
    plt.title('Waveform Difference')
    plt.legend()

    # Plot spectrogram difference
    plt.subplot(3, 1, 2)
    D_original = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
    D_perturbed = librosa.amplitude_to_db(np.abs(librosa.stft(perturbed_audio)), ref=np.max)
    plt.imshow(D_perturbed - D_original, aspect='auto', origin='lower', cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram Difference')

    # Plot MFCC difference
    plt.subplot(3, 1, 3)
    mfcc_original = librosa.feature.mfcc(y=original_audio, sr=sr)
    mfcc_perturbed = librosa.feature.mfcc(y=perturbed_audio, sr=sr)
    plt.imshow(mfcc_perturbed - mfcc_original, aspect='auto', origin='lower', cmap='coolwarm')
    plt.colorbar()
    plt.title('MFCC Difference')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main(args):
    if not os.path.exists(args.model_path):
        print(f"Пожалуйста, скачайте модель Vosk для русского языка и распакуйте в папку {args.model_path}")
        return

    model = Model(args.model_path)
    recognizer = KaldiRecognizer(model, args.sample_rate)

    audio, sr = load_audio(args.input_file, args.sample_rate)

    print("Оригинальное аудио:")
    original_transcription = transcribe_audio(audio, recognizer)
    print(original_transcription)

    if args.attack == 'fgsm':
        grad = get_gradient(audio)
        perturbed_audio = fgsm_attack(audio, args.epsilon, grad)
    elif args.attack == 'pgd':
        perturbed_audio = pgd_attack(audio, args.epsilon, args.alpha, args.num_iter, None)
    elif args.attack == 'cw':
        perturbed_audio = carlini_wagner_attack(audio, original_transcription, args.num_iter, args.learning_rate)
    else:
        print(f"Неизвестный метод атаки: {args.attack}")
        return

    sf.write(args.output_file, perturbed_audio, sr)

    print("\nИскаженное аудио:")
    adversarial_transcription = transcribe_audio(perturbed_audio, recognizer)
    print(adversarial_transcription)

    sim = similarity(original_transcription, adversarial_transcription)
    print(f"\nСходство между транскрипциями: {sim:.2f}")

    if sim < 1.0:
        print("Adversarial атака вызвала изменение в транскрипции.")
    else:
        print("Транскрипции идентичны. Adversarial атака может быть неэффективной.")

    is_adversarial, kl_divergence = detect_adversarial(audio, perturbed_audio, args.threshold)
    print(f"\nDetected as adversarial: {is_adversarial}")
    print(f"KL divergence: {kl_divergence:.4f}")

    if is_adversarial:
        print("Обнаружен adversarial пример!")
    else:
        print("Adversarial пример не обнаружен.")

    # Plot and save the differences
    plot_difference(audio, perturbed_audio, sr, "audio_differences.png")
    print("\nГрафик различий сохранен в файле 'audio_differences.png'")

    # Calculate and print numerical differences
    waveform_diff = np.mean(np.abs(perturbed_audio - audio))
    spectrogram_diff = np.mean(np.abs(librosa.stft(perturbed_audio) - librosa.stft(audio)))
    mfcc_diff = np.mean(np.abs(librosa.feature.mfcc(y=perturbed_audio, sr=sr) - librosa.feature.mfcc(y=audio, sr=sr)))

    print(f"\nСредняя абсолютная разница в форме волны: {waveform_diff:.6f}")
    print(f"Средняя абсолютная разница в спектрограмме: {spectrogram_diff:.6f}")
    print(f"Средняя абсолютная разница в MFCC: {mfcc_diff:.6f}")

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Recognition Adversarial Attack")
    parser.add_argument("--input_file", type=str, default="1.wav", help="Input audio file")
    parser.add_argument("--output_file", type=str, default="adversarial_audio.wav", help="Output adversarial audio file")
    parser.add_argument("--model_path", type=str, default="vosk-model-ru-0.22", help="Path to Vosk model")
    parser.add_argument("--attack", type=str, choices=['fgsm', 'pgd', 'cw'], default='fgsm', help="Attack method")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for FGSM and PGD attacks")
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha for PGD attack")
    parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations for PGD and CW attacks")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for CW attack")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for adversarial detection")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")

    args = parser.parse_args()
    main(args)
