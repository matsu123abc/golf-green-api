from fastapi import FastAPI
import cv2
import numpy as np
import json
import os
import requests
from io import BytesIO
from azure.storage.blob import BlobServiceClient

app = FastAPI()

# Blob Storage
connection_string = os.getenv("BLOB_CONNECTION_STRING")
container_name = os.getenv("GREEN_CONTAINER_NAME")

# Blob 上の画像 URL（例）
# https://{storage}.blob.core.windows.net/{container}/green_1.png
BLOB_IMAGE_URL = os.getenv("GREEN_IMAGE_URL_1")  # ← 環境変数で設定推奨


def load_image_from_blob(url: str):
    """Blob Storage の PNG を読み込んで OpenCV 画像に変換"""
    response = requests.get(url)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def color_to_height(pixel):
    b, g, r = pixel

    if b > g and b > r:
        return 0  # 青

    if g > r and g > b:
        return 2  # 緑

    if r > b and g > b:
        return 4  # 黄

    if r > g and g > b:
        return 6  # オレンジ

    if r > 150 and g < 80:
        return 7  # 赤

    return 0


@app.get("/generate/green/1")
def generate_green_1():
    # === 1. Blob から画像読み込み ===
    img = load_image_from_blob(BLOB_IMAGE_URL)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # === 2. 色域マスク（青〜赤） ===
    lower = np.array([10, 30, 30])
    upper = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # === 3. 最大領域（グリーン）抽出 ===
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # === 4. 切り抜き ===
    crop = img[y:y+h, x:x+w]

    # === 5. 36×36 にリサイズ ===
    resized = cv2.resize(crop, (36, 36), interpolation=cv2.INTER_AREA)

    # === 6. 色 → 高さ値 ===
    height_map = []
    for row in resized:
        height_row = []
        for pixel in row:
            height_row.append(color_to_height(pixel))
        height_map.append(height_row)

    # === 7. JSON 生成 ===
    json_data = {
        "green_id": 1,
        "grid_width": 36,
        "grid_height": 36,
        "cell_size_yards": 1.0,
        "heights": height_map,
        "pin_positions": {}
    }

    # ローカル保存
    with open("green_1.json", "w") as f:
        json.dump(json_data, f, indent=2)

    # === 8. Blob Storage にアップロード ===
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container = blob_service.get_container_client(container_name)
    blob = container.get_blob_client("green_1.json")

    with open("green_1.json", "rb") as f:
        blob.upload_blob(f, overwrite=True)

    return {"status": "green_1.json generated & uploaded"}


@app.get("/")
def root():
    return {"message": "USA FastAPI is running"}
