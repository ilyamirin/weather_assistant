import streamlit as st
import requests
import json

# --- Настройки ---
OLLAMA_API_BASE = "http://localhost:11434"  # Локальный Ollama сервер
MODEL_NAME = "llama3:instruct"

# --- Функция для получения погоды ---
def get_weather(city: str) -> str:
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric&lang=ru"
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            description = data["weather"][0]["description"]
            humidity = data["main"]["humidity"]
            return f"В городе {city} сейчас {temp:.1f}°C, ощущается как {feels_like:.1f}°C. {description.capitalize()}. Влажность: {humidity}%."
        else:
            return f"Не удалось получить погоду для {city}. Ошибка: {data.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Ошибка при запросе погоды: {str(e)}"

# --- Инструменты (в формате, похожем на OpenAI) ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Получить текущую погоду в указанном городе",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Название города, например, Москва, Berlin, Tokyo"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# --- Поддержка tool calling через Ollama ---
def format_messages_with_tools(messages, tools):
    """
    Форматирует сообщения для Ollama с поддержкой tools.
    Ollama понимает поле 'tools' и может вызывать функции.
    """
    return {
        "model": MODEL_NAME,
        "messages": messages,
        "tools": tools,
        "stream": False
    }

def call_ollama(messages, tools=None):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "format": "json" if tools else None,  # Не обязательно, но помогает
    }
    if tools:
        payload["tools"] = tools

    try:
        response = requests.post(f"{OLLAMA_API_BASE}/api/chat", json=payload)
        response.raise_for_status()
        result = response.json()
        return result["message"]
    except Exception as e:
        return {"role": "assistant", "content": f"Ошибка связи с Ollama: {str(e)}"}

# --- Streamlit UI ---
st.set_page_config(page_title="🌤️ Локальный ассистент погоды (Ollama)", page_icon="🌤️")
st.title("🌤️ Локальный ассистент погоды")
st.markdown("Работает на **llama3:instruct** через Ollama. Никакого облака!")

# --- Проверка доступности Ollama ---
try:
    requests.get(OLLAMA_API_BASE)
    st.success("✅ Ollama доступна")
except:
    st.error("❌ Не удалось подключиться к Ollama. Убедитесь, что Ollama запущена: `ollama serve`")
    st.stop()

# --- Ключ от OpenWeather ---
OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]  # Или вставьте напрямую (не рекомендуется)

# --- История сообщений ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "Ты — умный ассистент. Ты можешь вызывать функции. "
                "Если пользователь спрашивает о погоде, используй инструмент get_weather. "
                "Отвечай на русском языке."
            )
        }
    ]

# --- Отображение истории ---
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# --- Ввод пользователя ---
if prompt := st.chat_input("Спросите о погоде, например: 'Какая погода в Париже?'"):
    # Добавляем сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Отправляем запрос в Ollama
    with st.spinner("Ассистент думает..."):
        response_msg = call_ollama(st.session_state.messages, tools)

        # Проверяем, содержит ли ответ вызов функции
        if hasattr(response_msg, "tool_calls") or ("tool_calls" in response_msg and response_msg["tool_calls"]):
            # В текущей версии Ollama tool_calls приходят в виде dict
            tool_calls = response_msg.get("tool_calls", [])
            st.session_state.messages.append(response_msg)

            available_functions = {"get_weather": get_weather}

            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                if function_name == "get_weather":
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                        city = args["city"]
                        result = get_weather(city)

                        # Добавляем результат вызова инструмента
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": function_name,
                            "content": result,
                        })
                    except Exception as e:
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": function_name,
                            "content": f"Ошибка обработки аргументов: {e}",
                        })

            # Запрашиваем финальный ответ после инструмента
            final_response = call_ollama(st.session_state.messages)
            st.session_state.messages.append(final_response)
            with st.chat_message("assistant"):
                st.write(final_response["content"])
        else:
            # Простой ответ без инструмента
            st.session_state.messages.append(response_msg)
            with st.chat_message("assistant"):
                st.write(response_msg["content"])
