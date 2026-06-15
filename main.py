from fastapi import FastAPI
import cv2
import numpy as np
import json
import os
import requests
from io import BytesIO
from azure.storage.blob import BlobServiceClient
from fastapi.responses import HTMLResponse

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


@app.get("/pin", response_class=HTMLResponse)
def pin():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Pin Position Setter</title>
<style>
  body { margin: 0; background: #222; color: white; text-align: center; }
  #canvas { touch-action: manipulation; }
</style>
</head>
<body>

<h2>グリーンのピン位置をタップして登録</h2>

<canvas id="canvas" width="360" height="360"></canvas>

<p id="info"></p>

<script>
// 2D グリーン画像（36×36 の高さマップを色変換した PNG）
const greenImageUrl = "https://pcbdiagnosisrga8a5.blob.core.windows.net/course-maps/green_1.png";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// 画像読み込み
const img = new Image();
img.src = greenImageUrl;
img.onload = () => {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
};

// タップ位置を取得
canvas.addEventListener("click", function(e) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / 10);  // 360px → 36 グリッド
    const y = Math.floor((e.clientY - rect.top) / 10);

    // ピン位置を赤丸で描画
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.arc(x * 10, y * 10, 6, 0, Math.PI * 2);
    ctx.fillStyle = "red";
    ctx.fill();

    document.getElementById("info").innerText =
        `ピン位置: (${x}, ${y}) を登録しました`;

    // サーバーに送信
    fetch("/set_pin/1", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x: x, y: y })
    });
});
</script>

</body>
</html>
"""

@app.post("/set_pin/{green_id}")
def set_pin(green_id: int, pos: dict):
    x = pos["x"]
    y = pos["y"]

    # green_1.json を読み込み
    blob_client = container_client.get_blob_client(f"green_{green_id}.json")
    data = json.loads(blob_client.download_blob().readall())

    # ピン位置を更新
    data["pin_positions"]["today"] = [x, y]

    # 上書き保存
    blob_client.upload_blob(json.dumps(data), overwrite=True)

    return {"status": "pin updated", "pin": [x, y]}


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Green 1 - 3D View</title>
<style>
  body { margin: 0; overflow: hidden; background: #222; }
  canvas { display: block; }
</style>
</head>
<body>

<script src="https://cdn.jsdelivr.net/npm/three@0.152.2/build/three.min.js"></script>

<script>
async function loadGreenData() {
  const url = "https://pcbdiagnosisrga8a5.blob.core.windows.net/course-maps/green_1.json";
  const res = await fetch(url);
  return await res.json();
}

async function main() {
  const data = await loadGreenData();
  const heights = data.heights;
  const W = data.grid_width;
  const H = data.grid_height;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, -60, 40);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  const light = new THREE.DirectionalLight(0xffffff, 1);
  light.position.set(30, -30, 50);
  scene.add(light);

  const ambient = new THREE.AmbientLight(0x888888);
  scene.add(ambient);

  const geometry = new THREE.PlaneGeometry(36, 36, W - 1, H - 1);

  const verts = geometry.attributes.position;
  for (let i = 0; i < verts.count; i++) {
    const x = i % W;
    const y = Math.floor(i / W);
    const h = heights[y][x] * 0.3;
    verts.setZ(i, h);
  }
  verts.needsUpdate = true;
  geometry.computeVertexNormals();

  const material = new THREE.MeshLambertMaterial({
    color: 0x55aa55,
    side: THREE.DoubleSide
  });

  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);
 
  function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
  }
  animate();
}

main();
</script>

</body>
</html>
"""
