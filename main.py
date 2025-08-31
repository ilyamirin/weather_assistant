import streamlit as st
import requests
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import OPENWEATHER_API_KEY, TRANSFORMERS_MODEL_NAME, MODEL_SUPPORTS_TOOLS

# --- Настройки для Transformers ---
DEFAULT_TEMPERATURE = 0.2
MAX_NEW_TOKENS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

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

# --- Инициализация модели Transformers ---
@st.cache_resource(show_spinner=True)
def _load_model_and_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE != "cuda":
        model = model.to(DEVICE)
    return tok, model


def _build_system_prompt(tools_spec, supports_tools: bool) -> str:
    base = (
        "Ты — умный ассистент. Отвечай на русском языке."
    )
    if supports_tools and tools_spec:
        tool_desc = (
            "Ты умеешь вызывать функции (tools). Если вопрос связан с погодой, вызови функцию get_weather.\n"
            "Схема инструмента: " + json.dumps(tools_spec, ensure_ascii=False) + "\n\n"
            "Если необходимо вызвать инструмент, верни строго JSON вида:\n"
            "{\"tool_calls\":[{\"id\":\"toolcall_1\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Москва\\\"}\"}}]}\n"
            "Без каких-либо пояснений вокруг. Если инструмент не нужен, ответь обычным текстом."
        )
        return base + "\n\n" + tool_desc
    return base


def _apply_chat_template_or_concat(tok: AutoTokenizer, messages: list, system_prompt: str) -> str:
    # Try to use chat template if available
    msgs = []
    inserted_system = False
    for m in messages:
        if m["role"] == "system":
            inserted_system = True
    if not inserted_system and system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    for m in messages:
        msgs.append(m)
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback: simple concatenation
        parts = []
        if system_prompt:
            parts.append(f"[СИСТЕМА]\n{system_prompt}\n")
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role in ("user", "assistant"):
                parts.append(f"[{role.upper()}]\n{content}\n")
        parts.append("[ASSISTANT]\n")
        return "\n".join(parts)


def _maybe_extract_tool_calls(text: str):
    # Try to find a JSON block containing "tool_calls"
    try:
        # Direct JSON
        data = json.loads(text)
        if isinstance(data, dict) and "tool_calls" in data:
            return data.get("tool_calls")
    except Exception:
        pass
    # Regex search for {..."tool_calls": ...}
    match = re.search(r"\{[^\{\}]*\"tool_calls\"\s*:\s*\[[\s\S]*?\]\s*\}", text)
    if match:
        try:
            data = json.loads(match.group(0))
            return data.get("tool_calls")
        except Exception:
            return None
    return None

def _generate_assistant_message(messages, tools_spec=None, supports_tools: bool = True):
    tok, model = _load_model_and_tokenizer(TRANSFORMERS_MODEL_NAME)
    system_prompt = _build_system_prompt(tools_spec, supports_tools)
    prompt_text = _apply_chat_template_or_concat(tok, messages, system_prompt)

    inputs = tok(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=DEFAULT_TEMPERATURE,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    full_text = tok.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant continuation if chat template didn't do it
    # Heuristic: take the tail after the prompt_text
    if full_text.startswith(prompt_text):
        assistant_text = full_text[len(prompt_text):].strip()
    else:
        assistant_text = full_text.strip()

    # Try detect tool calls
    tool_calls = _maybe_extract_tool_calls(assistant_text) if supports_tools else None

    if tool_calls:
        return {"role": "assistant", "content": "", "tool_calls": tool_calls}
    else:
        return {"role": "assistant", "content": assistant_text}

# --- Streamlit UI ---
st.set_page_config(page_title="🌤️ Локальный ассистент погоды (Transformers)", page_icon="🌤️")
st.title("🌤️ Локальный ассистент погоды")
st.markdown(f"Работает на модели Transformers: **{TRANSFORMERS_MODEL_NAME}**. Локально, без облака!")

# --- Проверка поддержки tools в модели ---
if MODEL_SUPPORTS_TOOLS:
    st.info("🧰 Режим инструментов включён: модель будет пытаться вызывать функции.")
else:
    st.warning("ℹ️ Модель не поддерживает вызов инструментов: ответы будут без tool-calling.")

# --- Ключ от OpenWeather ---
# Ранее: OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]

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

    # Генерация ответа через Transformers
    with st.spinner("Ассистент думает..."):
        response_msg = _generate_assistant_message(
            st.session_state.messages,
            tools_spec=tools,
            supports_tools=MODEL_SUPPORTS_TOOLS,
        )

        # Проверяем, содержит ли ответ вызов функции
        tool_calls = response_msg.get("tool_calls") if isinstance(response_msg, dict) else None
        if MODEL_SUPPORTS_TOOLS and tool_calls:
            st.session_state.messages.append(response_msg)

            for tool_call in tool_calls:
                function_name = tool_call.get("function", {}).get("name")
                if function_name == "get_weather":
                    try:
                        args_raw = tool_call.get("function", {}).get("arguments", "{}")
                        # arguments может быть JSON-строкой или уже dict
                        if isinstance(args_raw, str):
                            args = json.loads(args_raw)
                        else:
                            args = args_raw or {}
                        city = args.get("city", "")
                        result = get_weather(city)

                        # Добавляем результат вызова инструмента
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", "toolcall_1"),
                            "name": function_name,
                            "content": result,
                        })
                    except Exception as e:
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", "toolcall_1"),
                            "name": function_name,
                            "content": f"Ошибка обработки аргументов: {e}",
                        })

            # Запрашиваем финальный ответ после инструмента
            final_response = _generate_assistant_message(
                st.session_state.messages,
                tools_spec=None,
                supports_tools=False,
            )
            st.session_state.messages.append(final_response)
            with st.chat_message("assistant"):
                st.write(final_response.get("content", ""))
        else:
            # Простой ответ без инструмента
            st.session_state.messages.append(response_msg)
            with st.chat_message("assistant"):
                st.write(response_msg.get("content", ""))
