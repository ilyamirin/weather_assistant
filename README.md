# 🌤️ Локальный ассистент погоды (Transformers)

Небольшое приложение на Streamlit, которое использует локальную модель Hugging Face Transformers для диалогового общения и вызова инструмента получения погоды (tool calling) через OpenWeather API.

Основные файлы:
- `main.py` — интерфейс Streamlit и логика диалога с моделью/инструментами.
- `config.py` — конфигурация: API-ключ погоды, имя модели и флаг поддержки tools.

## Возможности
- Локальный запуск без облака, с использованием `transformers` и `torch`.
- Поддержка "инструментов" (tool calling) через подсказку (prompt): модель может запрашивать вызов функции `get_weather`.
- Кеширование модели и токенизатора, использование chat template при наличии у модели.

## Требования
См. `requirements.txt`. Вкратце:
- Python 3.9+ рекомендуется.
- Библиотеки: `streamlit`, `requests`, `transformers`, `accelerate`, `safetensors`, `sentencepiece`, `torch`.
- Для ускорения на GPU (CUDA) установите подходящую сборку PyTorch (см. ниже).

## Установка
1) Клонируйте или скопируйте проект.

2) Создайте и активируйте виртуальное окружение (рекомендуется):
- Windows (PowerShell):
  - `python -m venv .venv`
  - `.venv\Scripts\Activate.ps1`

3) Установите зависимости:
```
pip install -r requirements.txt
```

4) (Опционально) Установка PyTorch с поддержкой CUDA:
- На Windows/ПК с NVIDIA GPU лучше установить колесо с сайта PyTorch, чтобы получить правильный билд CUDA.
- Инструкции: https://pytorch.org/get-started/locally/
- Пример (замените версии под вашу среду):
```
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

## Конфигурация
Откройте `config.py` и настройте:
- `OPENWEATHER_API_KEY` — ваш ключ OpenWeather. Получить можно бесплатно на https://openweathermap.org/.
- `TRANSFORMERS_MODEL_NAME` — имя модели на Hugging Face, например:
  - `Qwen/Qwen3-4B-Instruct-2507` (по умолчанию в текущей версии кода)
  - `Qwen/Qwen2.5-7B-Instruct`
  - `meta-llama/Llama-3.1-8B-Instruct`
  - `mistralai/Mistral-7B-Instruct-v0.3`
- `MODEL_SUPPORTS_TOOLS` — если `True`, модель будет пытаться вызывать инструмент `get_weather` (через протокол JSON в ответе). Если `False` — обычные текстовые ответы без инструментов.

Примечания:
- При первом запуске модель и токенизатор будут скачаны и закешированы в локальный кэш HF (обычно `~/.cache/huggingface`).
- Некоторые модели требуют дополнительных файлов токенизации (поэтому в зависимостях есть `sentencepiece`).

## Запуск
В корне проекта выполните:
```
streamlit run main.py
```
Приложение откроется в браузере. Задавайте вопросы о погоде, например: «Какая погода в Париже?»

## Как работает tool calling
- В `main.py` определён инструмент `get_weather(city: str)` и его спецификация в массиве `tools`.
- При включенном `MODEL_SUPPORTS_TOOLS` в system prompt добавляется инструкция возвращать JSON вида:
```
{"tool_calls":[{"id":"toolcall_1","function":{"name":"get_weather","arguments":"{\"city\":\"Москва\"}"}}]}
```
- Если модель вернула `tool_calls`, приложение вызывает соответствующую функцию, добавляет ответ инструмента в историю и затем запрашивает у модели финальный текстовый ответ (уже без tools).

## Частые проблемы и решения
- Долгая загрузка/ошибки памяти: выберите более компактную модель (например, 4B/7B) или запустите на CPU.
- CUDA не используется: убедитесь, что установлена подходящая сборка PyTorch с CUDA и драйверы GPU, и что `torch.cuda.is_available()` возвращает `True`.
- Ошибка при скачивании модели: проверьте соединение или выполните вход в HF, если модель требует аутентификации: `huggingface-cli login`.
- OpenWeather возвращает ошибку: проверьте `OPENWEATHER_API_KEY` и корректность названия города.

## Структура проекта
```
weather_assistant/
├─ main.py
├─ config.py
├─ requirements.txt
└─ README.md
```

## Лицензия
Этот проект предоставляется «как есть». Проверьте лицензионные условия используемых моделей и библиотек.
