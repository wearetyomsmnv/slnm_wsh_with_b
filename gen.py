from gtts import gTTS
from pydub import AudioSegment
import os

# Задайте текст, который вы хотите преобразовать в речь
text = "Привет! Это пример генерации аудиосемпла на русском языке."

# Укажите язык
language = 'ru'

# Генерация аудиофайла в формате MP3
tts = gTTS(text=text, lang=language, slow=False)
mp3_file = "sample_audio.mp3"
tts.save(mp3_file)

# Конвертация MP3 в WAV
wav_file = "sample_audio.wav"
audio = AudioSegment.from_mp3(mp3_file)
audio.export(wav_file, format="wav")