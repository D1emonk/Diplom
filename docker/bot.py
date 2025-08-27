import os
import cv2
import asyncio
import numpy as np
from ultralytics import YOLO
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from PIL import Image, ImageDraw, ImageFont

# ==== Настройки ====
BOT_TOKEN = os.environ.get("BOT_TOKEN")
if BOT_TOKEN is None:
    raise ValueError("⚠️ Не задан токен бота! Установи переменную окружения BOT_TOKEN")

SAVE_DIR = "./telegram_predictions"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== Словарь классов GTSRB ====
CLASS_NAMES = {
    0: "Ограничение скорости (20 км/ч)",
    1: "Ограничение скорости (30 км/ч)",
    2: "Ограничение скорости (50 км/ч)",
    3: "Ограничение скорости (60 км/ч)",
    4: "Ограничение скорости (70 км/ч)",
    5: "Ограничение скорости (80 км/ч)",
    6: "Отмена ограничения (80 км/ч)",
    7: "Ограничение скорости (100 км/ч)",
    8: "Ограничение скорости (120 км/ч)",
    9: "Обгон запрещен",
    10: "Обгон грузовикам запрещен",
    11: "Перекресток со знаком приоритета",
    12: "Главная дорога",
    13: "Уступи дорогу",
    14: "СТОП",
    15: "Движение запрещено",
    16: "Движение грузовиков запрещено",
    17: "Въезд запрещен",
    18: "Опасность",
    19: "Опасный поворот налево",
    20: "Опасный поворот направо",
    21: "Двойной поворот",
    22: "Неровная дорога",
    23: "Скользкая дорога",
    24: "Сужение дороги справа",
    25: "Дорожные работы",
    26: "Светофорное регулирование",
    27: "Пешеходный переход",
    28: "Дети",
    29: "Велосипедисты",
    30: "Осторожно: лёд/снег",
    31: "Дикие животные",
    32: "Окончание всех ограничений",
    33: "Поворот направо",
    34: "Поворот налево",
    35: "Движение прямо",
    36: "Движение прямо или направо",
    37: "Движение прямо или налево",
    38: "Объезд препятствия справа",
    39: "Объезд препятствия слева",
    40: "Круговое движение",
    41: "Конец зоны обгона",
    42: "Конец зоны запрещения обгона грузовым автомобилям",
}

# ==== Загрузка моделей ====
sign_classifier = YOLO("classific.pt")
sign_detector = YOLO("detector.pt")

# ==== Хэндлеры ====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет 👋 Отправь мне фото или видео дорожного знака, и я попробую его распознать!"
    )

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        file = await update.message.photo[-1].get_file()
        image_path = "input.jpg"
        await file.download_to_drive(image_path)
        image = cv2.imread(image_path)

        det_res = sign_detector.predict(image, verbose=False)
        detected_signs_list = []

        if det_res and len(det_res) > 0 and det_res[0].boxes is not None:
            boxes = det_res[0].boxes
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image_pil)

            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 24)
            except:
                font = ImageFont.load_default()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                cls_res = sign_classifier.predict(crop, task="classify", verbose=False)
                if cls_res and hasattr(cls_res[0], "probs") and cls_res[0].probs is not None:
                    label_idx = int(cls_res[0].probs.top1)
                    model_label_name = int(cls_res[0].names[label_idx])
                    label_name = CLASS_NAMES.get(model_label_name, f"Класс {model_label_name}")
                else:
                    label_name = "Не распознано"

                detected_signs_list.append(label_name)
                draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)
                draw.text((x1, y1 - 25), label_name, font=font, fill=(0, 255, 0))

            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        output_path = "output.jpg"
        cv2.imwrite(output_path, image)

        if detected_signs_list:
            text = "Найденные знаки: " + ", ".join(detected_signs_list)
        else:
            text = "Знаки не распознаны"

        await update.message.reply_text(text)
        await update.message.reply_photo(photo=open(output_path, "rb"))

    except Exception as e:
        await update.message.reply_text(f"Ошибка обработки: {e}")

async def video_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        file = await update.message.video.get_file()
        input_path = "input.mp4"
        await file.download_to_drive(input_path)

        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = "output.mp4"
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            det_res = sign_detector.predict(frame, verbose=False)
            if det_res and len(det_res) > 0 and det_res[0].boxes is not None:
                boxes = det_res[0].boxes
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                font = ImageFont.truetype("arial.ttf", 20)

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    cls_res = sign_classifier.predict(crop, task="classify", verbose=False)
                    if cls_res and hasattr(cls_res[0], "probs") and cls_res[0].probs is not None:
                        label_idx = int(cls_res[0].probs.top1)
                        model_label_name = int(cls_res[0].names[label_idx])
                        label_name = CLASS_NAMES.get(model_label_name, f"Класс {model_label_name}")
                    else:
                        label_name = "Не распознано"

                    draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)
                    draw.text((x1, y1 - 25), label_name, font=font, fill=(255, 0, 0))

                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            out.write(frame)

        cap.release()
        out.release()
        await update.message.reply_video(video=open(output_path, "rb"))

    except Exception as e:
        await update.message.reply_text(f"Ошибка обработки видео: {e}")

# ==== Запуск бота ====
if __name__ == "__main__":
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.VIDEO, video_handler))

    # Асинхронный запуск
    asyncio.run(app.run_polling())
