import wave
import json
from vosk import Model, KaldiRecognizer

# Загрузка модели
model = Model("vosk-model-ru-0.22")
rec = KaldiRecognizer(model, 16000)

# Открытие аудиофайла
wf = wave.open("1.wav", "rb")
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
    print("Аудиофайл должен быть моно, 16 бит и 16000 Гц.")
    exit(1)

# Распознавание речи
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        result = rec.Result()
        print(result)
    else:
        print(rec.PartialResult())

# Получение окончательного результата
print(rec.FinalResult())
