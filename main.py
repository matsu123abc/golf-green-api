from fastapi import FastAPI
import cv2
import numpy as np
import json
import os
import requests
from io import BytesIO
from azure.storage.blob import BlobServiceClient
from fastapi.responses import HTMLResponse
from openai import AzureOpenAI

app = FastAPI()

# ===== Blob Storage =====
connection_string = os.getenv("BLOB_CONNECTION_STRING")
container_name = os.getenv("GREEN_CONTAINER_NAME")
blob_service = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service.get_container_client(container_name)

# 画像 URL（36×36 の元画像）
BLOB_IMAGE_URL = os.getenv("GREEN_IMAGE_URL_1")


def load_image_from_blob(url: str):
    response = requests.get(url)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def color_to_height(pixel):
    b, g, r = pixel
    hsv = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    # 青 (H ≈ 100〜130)
    if 90 <= h <= 130:
        return 0

    # 緑 (H ≈ 40〜85)
    if 40 <= h <= 85:
        return 2

    # 黄 (H ≈ 20〜35)
    if 20 <= h <= 35:
        return 4

    # オレンジ (H ≈ 10〜20)
    if 10 <= h <= 20:
        return 6

    # 赤 (H ≈ 0〜10 or 160〜180)
    if h < 10 or h > 160:
        return 7

    return 0


# ============================================================
# GPT 戦略ロジック
# ============================================================
def gpt_strategy(heights, pin):

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    prompt = f"""
あなたはプロキャディです。
以下は 36×36 のグリーン高低差データです。

heights = {heights}
pin = {pin}

あなたの役割は「ピン位置の周囲の傾斜を読み取り、地形を詳しく解説し、その地形を踏まえて戦略を立案すること」です。

必ず以下の順番で返答してください：

1. ピン周囲の傾斜の詳細な解説  
   - ピンの左右の高さ差  
   - ピンの手前・奥の高さ差  
   - どちらが受けているか  
   - どちらが下っているか  
   - 傾斜ベクトルの向き（例：右下方向に下り）  
   - 危険方向（球が止まりにくい方向）  
   - 安全方向（球が止まりやすい方向）

2. その地形を踏まえた戦略  
   - どの方向から攻めるべきか（手前/奥/左右）  
   - その理由（傾斜・高さ差を根拠に）  
   - 絶対に避けるべき方向  
   - 球が止まりやすいエリアの特徴  

返答は必ず次の JSON 形式のみで返してください：

{
  "slope_analysis": "ここに傾斜の詳細解説",
  "strategy": "ここに戦略"
}
"""

    try:
        res = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        raw = res.choices[0].message.content.strip()

        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        json_text = raw[json_start:json_end]
        json_text = json_text.replace("```json", "").replace("```", "").strip()

        return json.loads(json_text)

    except Exception as e:
        return {
            "best_landing_spot": None,
            "line": "",
            "danger_zone": "",
            "strategy": f"GPTエラー: {str(e)}"
        }


# ============================================================
# AI 戦略 API
# ============================================================
@app.post("/ai_strategy/1")
def ai_strategy_green1():

    blob_client = container_client.get_blob_client("green_1.json")
    data = json.loads(blob_client.download_blob().readall())

    heights = data["heights"]
    pin = data["pin_positions"].get("today", None)

    if pin is None:
        return {"error": "ピン位置が登録されていません"}

    return gpt_strategy(heights, pin)


# ============================================================
# Green1 JSON 生成
# ============================================================
@app.get("/generate/green/1")
def generate_green_1():
    img = load_image_from_blob(BLOB_IMAGE_URL)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([10, 30, 30])
    upper = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    crop = img[y:y+h, x:x+w]
    resized = cv2.resize(crop, (36, 36), interpolation=cv2.INTER_AREA)

    height_map = []
    for row in resized:
        height_row = []
        for pixel in row:
            height_row.append(color_to_height(pixel))
        height_map.append(height_row)

    json_data = {
        "green_id": 1,
        "grid_width": 36,
        "grid_height": 36,
        "cell_size_yards": 1.0,
        "heights": height_map,
        "pin_positions": {}
    }

    blob = container_client.get_blob_client("green_1.json")
    blob.upload_blob(json.dumps(json_data), overwrite=True)

    return {"status": "green_1.json generated & uploaded"}


# ============================================================
# ピン位置保存 API
# ============================================================
@app.post("/set_pin/{green_id}")
def set_pin(green_id: int, pos: dict):
    x = pos["x"]
    y = pos["y"]

    blob_client = container_client.get_blob_client(f"green_{green_id}.json")
    data = json.loads(blob_client.download_blob().readall())

    data["pin_positions"]["today"] = [x, y]

    blob_client.upload_blob(json.dumps(data), overwrite=True)

    return {"status": "pin updated", "pin": [x, y]}


# ============================================================
# 起動画面：統合 UI（ピン登録＋AI戦略）
# ============================================================
@app.get("/", response_class=HTMLResponse)
def green1():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Green 1 - ピン登録 & AI戦略</title>
<style>
  body { background:#222; color:white; font-size:20px; text-align:center; margin:0; padding:10px; }
  #canvas { touch-action: manipulation; border:1px solid #555; }
  button {
    font-size:22px; padding:12px 24px; margin-top:10px;
    background:#4CAF50; border:none; color:white; border-radius:6px;
    width:90%;
  }
  #result {
    margin-top:20px; padding:15px; background:#333; border-radius:8px;
    white-space:pre-wrap; text-align:left;
  }
  iframe {
    width:100%;
    height:400px;
    border:1px solid #555;
    border-radius:8px;
    margin-top:20px;
  }
</style>
</head>
<body>

<h2>Green 1 - ピン登録 & AI戦略</h2>

<canvas id="canvas" width="360" height="360"></canvas>

<p id="info">ピン位置をタップしてください</p>

<button id="saveBtn" style="display:none;">この位置を登録する</button>
<button id="aiBtn" style="display:none; background:#2196F3;">AI に戦略を聞く</button>

<div id="result"></div>

<h3 style="margin-top:30px;">3D グリーン（参考表示）</h3>
<iframe src="/green1/3d"></iframe>

<script>
let selectedX = null;
let selectedY = null;

const greenImageUrl = "https://pcbdiagnosisrga8a5.blob.core.windows.net/course-maps/green_1.png";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const saveBtn = document.getElementById("saveBtn");
const aiBtn = document.getElementById("aiBtn");

const img = new Image();
img.src = greenImageUrl;
img.onload = () => {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
};

canvas.addEventListener("click", function(e) {
    const rect = canvas.getBoundingClientRect();
    selectedX = Math.floor((e.clientX - rect.left) / 10);
    selectedY = Math.floor((e.clientY - rect.top) / 10);

    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.arc(selectedX * 10, selectedY * 10, 6, 0, Math.PI * 2);
    ctx.fillStyle = "red";
    ctx.fill();

    document.getElementById("info").innerText =
        `選択中のピン位置: (${selectedX}, ${selectedY})`;

    saveBtn.style.display = "block";
});

saveBtn.addEventListener("click", async function() {
    await fetch("/set_pin/1", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ x: selectedX, y: selectedY })
    });

    document.getElementById("info").innerText =
        "ピン位置 (" + selectedX + ", " + selectedY + ") を登録しました！";
    
    aiBtn.style.display = "block";
});

aiBtn.addEventListener("click", async function() {
    document.getElementById("result").innerText = "AI が戦略を計算中です…";

    const res = await fetch("/ai_strategy/1", { method: "POST" });
    const data = await res.json();

    // --- 新仕様：傾斜解説＋戦略のみ表示 ---
    let text = "";
    text += "⛰️ 傾斜の解説:\\n" + data.slope_analysis + "\\n\\n";
    text += "🧠 戦略:\\n" + data.strategy;

    document.getElementById("result").innerText = text;

    // ピン位置だけ再描画
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.arc(selectedX * 10, selectedY * 10, 6, 0, Math.PI * 2);
    ctx.fillStyle = "red";
    ctx.fill();
});
</script>

</body>
</html>
"""

# ============================================================
# 3D 表示（確認用）
# ============================================================
@app.get("/green1/3d", response_class=HTMLResponse)
def green1_3d():
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
