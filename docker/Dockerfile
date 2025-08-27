# Базовый образ с Python 3.11
FROM python:3.11-slim

# # Устанавливаем системные зависимости для Pillow и OpenCV     
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        fonts-dejavu-core \
        git \
        wget \
        curl \
        unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Установка pip и обновление
RUN python -m pip install --upgrade pip setuptools wheel

# Создаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Экспортируем переменные окружения
ENV BOT_TOKEN=

# Команда запуска
CMD ["python", "bot.py"]
