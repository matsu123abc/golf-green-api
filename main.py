import os
import json
import time
import logging
from io import BytesIO

import cv2
import numpy as np
import requests
from json import JSONDecodeError

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI

# ---------------------------
# Logging
# ---------------------------
logger = logging.getLogger("green_app")
logging.basicConfig(level=logging.INFO)

# ---------------------------
# App & Blob setup
# ---------------------------
app = FastAPI()

connection_string = os.getenv("BLOB_CONNECTION_STRING")
container_name = os.getenv("GREEN_CONTAINER_NAME")
if not connection_string or not container_name:
    logger.warning("BLOB_CONNECTION_STRING or GREEN_CONTAINER_NAME not set in environment")

blob_service = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service.get_container_client(container_name)

BLOB_IMAGE_URL = os.getenv("GREEN_IMAGE_URL_1", "")

# ---------------------------
# Utility: load image from URL
# ---------------------------
def load_image_from_blob(url: str):
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from blob URL")
    return img

# ---------------------------
# Utility: safe blob JSON load
# ---------------------------
def safe_load_json_from_blob(blob_name: str):
    try:
        blob_client = container_client.get_blob_client(blob_name)
        raw = blob_client.download_blob().readall()
        return json.loads(raw)
    except Exception as e:
        logger.exception("Failed to load JSON from blob: %s", blob_name)
        raise

# ---------------------------
# Color -> height mapping
# ---------------------------
def color_to_height(pixel):
    # pixel is BGR
    b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Blue (approx)
    if 90 <= h <= 130:
        return 0
    # Green
    if 40 <= h <= 85:
        return 2
    # Yellow
    if 20 <= h <= 35:
        return 4
    # Orange
    if 10 <= h <= 20:
        return 6
    # Red
    if h < 10 or h > 160:
        return 7
    return 0

# ---------------------------
# GPT 戦略ロジック（堅牢版）
# ---------------------------
def gpt_strategy(heights, pin, max_retries=2, retry_delay=1.0, timeout_seconds=30):
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
   - 傾斜ベクトルの向き  
   - 危険方向  
   - 安全方向

2. その地形を踏まえた戦略  
   - どの方向から攻めるべきか  
   - その理由  
   - 避けるべき方向  
   - 球が止まりやすいエリア

返答は必ず次の JSON 形式のみで返してください：

{{
  "slope_analysis": "ここに傾斜の詳細解説",
  "strategy": "ここに戦略"
}}
"""

    attempt = 0
    last_exception = None

    while attempt <= max_retries:
        try:
            attempt += 1
            logger.info("Calling AzureOpenAI (attempt %d)", attempt)
            res = client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            raw = res.choices[0].message.content.strip()
            logger.info("GPT raw response (truncated): %s", raw[:1000])

            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start == -1 or json_end == 0 or json_end <= json_start:
                raise ValueError("GPTレスポンスに JSON 部分が見つかりませんでした。")

            json_text = raw[json_start:json_end]
            json_text = json_text.replace("```json", "").replace("```", "").strip()

            parsed = json.loads(json_text)

            if not isinstance(parsed, dict):
                raise ValueError("パース結果がオブジェクトではありません。")
            if "slope_analysis" not in parsed or "strategy" not in parsed:
                raise ValueError("必要なキー(slope_analysis/strategy)が含まれていません。")

            return parsed

        except (JSONDecodeError, ValueError) as e:
            logger.error("JSON parse/validation error: %s", str(e))
            last_exception = e
            break
        except Exception as e:
            logger.exception("GPT call failed on attempt %d: %s", attempt, str(e))
            last_exception = e
            if attempt <= max_retries:
                time.sleep(retry_delay)
                continue
            break

    logger.error("gpt_strategy failed after %d attempts", attempt)
    return {
        "slope_analysis": f"GPTエラーが発生しました: {str(last_exception)}",
        "strategy": "戦略を生成できませんでした。"
    }

# ============================================================
# AI 戦略 API（汎用化：1〜18）
# ============================================================
@app.post("/ai_strategy/{green_id}")
def ai_strategy_green(green_id: int):
    blob_name = f"green_{green_id}.json"
    try:
        data = safe_load_json_from_blob(blob_name)
    except Exception as e:
        logger.exception("Failed to read %s", blob_name)
        return JSONResponse(status_code=500, content={"error": f"{blob_name} を読み込めませんでした"})

    heights = data.get("heights")
    pin = data.get("pin_positions", {}).get("today", None)

    if pin is None:
        return JSONResponse(status_code=400, content={"error": "ピン位置が登録されていません"})

    try:
        result = gpt_strategy(heights, pin)
    except Exception as e:
        logger.exception("gpt_strategy raised unexpected exception")
        return JSONResponse(status_code=500, content={
            "slope_analysis": f"GPT内部エラー: {str(e)}",
            "strategy": "戦略を生成できませんでした。"
        })

    return {
        "slope_analysis": result.get("slope_analysis", "解析エラー"),
        "strategy": result.get("strategy", "戦略エラー")
    }

# ============================================================
# Green1 JSON 生成（そのまま）
# ============================================================
@app.get("/generate/green/1")
def generate_green_1():
    if not BLOB_IMAGE_URL:
        logger.error("GREEN_IMAGE_URL_1 not set")
        raise HTTPException(status_code=500, detail="GREEN_IMAGE_URL_1 が設定されていません")

    try:
        img = load_image_from_blob(BLOB_IMAGE_URL)
    except Exception as e:
        logger.exception("Failed to load image from blob URL")
        raise HTTPException(status_code=500, detail="画像の読み込みに失敗しました")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([10, 30, 30])
    upper = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.error("No contours found in image")
        raise HTTPException(status_code=500, detail="グリーン領域が検出できませんでした")

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    crop = img[y:y+h, x:x+w]
    resized = cv2.resize(crop, (36, 36), interpolation=cv2.INTER_AREA)

    height_map = []
    for row in resized:
        height_row = []
        for pixel in row:
            height_row.append(color_to_height(pixel))
        height_row.append
        height_map.append(height_row)

    json_data = {
        "green_id": 1,
        "grid_width": 36,
        "grid_height": 36,
        "cell_size_yards": 1.0,
        "heights": height_map,
        "pin_positions": {}
    }

    try:
        blob = container_client.get_blob_client("green_1.json")
        blob.upload_blob(json.dumps(json_data), overwrite=True)
    except Exception as e:
        logger.exception("Failed to upload green_1.json")
        raise HTTPException(status_code=500, detail="green_1.json のアップロードに失敗しました")

    return {"status": "green_1.json generated & uploaded"}

# ============================================================
# ピン位置保存 API（既に汎用）
# ============================================================
@app.post("/set_pin/{green_id}")
def set_pin(green_id: int, pos: dict):
    try:
        x = int(pos.get("x"))
        y = int(pos.get("y"))
    except Exception:
        raise HTTPException(status_code=400, detail="x,y must be integers")

    if not (0 <= x < 36 and 0 <= y < 36):
        raise HTTPException(status_code=400, detail="x,y out of range (0-35)")

    blob_name = f"green_{green_id}.json"
    try:
        data = safe_load_json_from_blob(blob_name)
    except Exception:
        raise HTTPException(status_code=500, detail="green JSON を読み込めませんでした")

    if "pin_positions" not in data:
        data["pin_positions"] = {}

    data["pin_positions"]["today"] = [x, y]

    try:
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(json.dumps(data), overwrite=True)
    except Exception:
        logger.exception("Failed to upload updated pin to blob")
        raise HTTPException(status_code=500, detail="ピン位置の保存に失敗しました")

    return {"status": "pin updated", "pin": [x, y]}

# ============================================================
# 起動画面：統合 UI（18ホール対応）
# ============================================================
@app.get("/", response_class=HTMLResponse)
def green_ui():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Green - ピン登録 & AI戦略</title>

<style>
  :root{
    --bg:#111;
    --panel:#1b1b1b;
    --accent:#2196F3;
    --ok:#4CAF50;
    --text:#fff;
    --touch:56px;
  }

  body{
    margin:0;
    background:var(--bg);
    color:var(--text);
    font-family:system-ui, -apple-system, "Hiragino Kaku Gothic ProN", sans-serif;
    padding:10px;
  }

  .header{
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:10px;
  }

  .title{
    font-size:18px;
    margin:0;
  }

  .map-wrap{
    width:100%;
    display:flex;
    justify-content:center;
  }

  canvas#canvas{
    width:100%;
    max-width:420px;
    height:auto;
    border:1px solid #555;
    border-radius:8px;
    background:#000;
    touch-action:manipulation;
  }

  #result{
    margin-top:10px;
    padding:12px;
    background:var(--panel);
    border-radius:8px;
    font-size:14px;
    white-space:pre-wrap;
    text-align:left;
    max-height:28vh;
    overflow:auto;
  }

  .action-bar{
    position:fixed;
    left:0;
    right:0;
    bottom:env(safe-area-inset-bottom, 0);
    display:flex;
    gap:8px;
    padding:8px;
    background:rgba(0,0,0,0.6);
    justify-content:center;
    z-index:9999;
  }

  .btn{
    height:var(--touch);
    min-width:44%;
    border-radius:10px;
    font-size:18px;
    border:none;
    color:var(--text);
  }

  .btn-save{ background:var(--ok); }
  .btn-ai{ background:var(--accent); }

  #toggle3d{
    margin-top:10px;
    width:100%;
    max-width:420px;
    background:#333;
    color:#fff;
    border-radius:8px;
    padding:10px;
    border:none;
  }

  /* 親ページ側：iframe を画面比で大きくする */
  iframe {
  width: 100%;
  /* max-width を削除または十分大きくする */
  max-width: none;
  /* 高さは画面の70%をデフォルトに */
  height: 70vh;
  border-radius: 8px;
  border: 1px solid #555;
  display: none;
  margin-top: 8px;
  box-sizing: border-box;
  }
  
</style>
</head>

<body>

<div class="header">
  <h2 class="title">Green - ピン登録 & AI戦略</h2>
  <select id="holeSelect" style="font-size:16px; padding:6px;">
    <option value="1">Hole 1</option>
    <option value="2">Hole 2</option>
    <option value="3">Hole 3</option>
    <option value="4">Hole 4</option>
    <option value="5">Hole 5</option>
    <option value="6">Hole 6</option>
    <option value="7">Hole 7</option>
    <option value="8">Hole 8</option>
    <option value="9">Hole 9</option>
    <option value="10">Hole 10</option>
    <option value="11">Hole 11</option>
    <option value="12">Hole 12</option>
    <option value="13">Hole 13</option>
    <option value="14">Hole 14</option>
    <option value="15">Hole 15</option>
    <option value="16">Hole 16</option>
    <option value="17">Hole 17</option>
    <option value="18">Hole 18</option>
  </select>
</div>

<div class="map-wrap">
  <canvas id="canvas" width="360" height="360"></canvas>
</div>

<p id="info" style="font-size:14px;">ピン位置をタップしてください</p>

<div id="result"></div>

<button id="toggle3d">3D 表示を開く</button>
<iframe id="view3d" src="/green/1/3d"></iframe>

<div class="action-bar">
  <button id="saveBtn" class="btn btn-save" style="display:none;">登録</button>
  <button id="aiBtn" class="btn btn-ai" style="display:none;">AI戦略</button>
</div>

<script>
let selectedX = null;
let selectedY = null;
let currentHole = 1;

let greenImageUrl = "https://pcbdiagnosisrga8a5.blob.core.windows.net/course-maps/green_1.png";

const holeSelect = document.getElementById("holeSelect");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const saveBtn = document.getElementById("saveBtn");
const aiBtn = document.getElementById("aiBtn");
const iframe = document.getElementById("view3d");
const toggle3d = document.getElementById("toggle3d");

const img = new Image();
img.src = greenImageUrl;
img.onload = () => redraw();

function redraw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  if(selectedX !== null){
    ctx.beginPath();
    ctx.arc(selectedX * 10, selectedY * 10, 6, 0, Math.PI * 2);
    ctx.fillStyle = "red";
    ctx.fill();
  }
}

holeSelect.addEventListener("change", function() {
  currentHole = parseInt(this.value);
  greenImageUrl = "https://pcbdiagnosisrga8a5.blob.core.windows.net/course-maps/green_" + currentHole + ".png";
  img.src = greenImageUrl;

  iframe.src = "/green/" + currentHole + "/3d";

  selectedX = null;
  selectedY = null;
  saveBtn.style.display = "none";
  aiBtn.style.display = "none";
  document.getElementById("info").innerText = "ピン位置をタップしてください";
  document.getElementById("result").innerText = "";
});

canvas.addEventListener("click", function(e) {
  const rect = canvas.getBoundingClientRect();
  selectedX = Math.floor((e.clientX - rect.left) / 10);
  selectedY = Math.floor((e.clientY - rect.top) / 10);

  redraw();

  document.getElementById("info").innerText =
      `選択中のピン位置: (${selectedX}, ${selectedY})`;

  saveBtn.style.display = "block";
});

saveBtn.addEventListener("click", async function() {
  await fetch("/set_pin/" + currentHole, {
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

  const res = await fetch("/ai_strategy/" + currentHole, { method: "POST" });
  const data = await res.json();

  let text = "";
  text += "⛰️ 傾斜の解説:\\n" + data.slope_analysis + "\\n\\n";
  text += "🧠 戦略:\\n" + data.strategy;

  document.getElementById("result").innerText = text;

  redraw();
});

toggle3d.addEventListener("click", function(){
  const currentDisplay = window.getComputedStyle(iframe).display;

  if (currentDisplay === "none") {
    // 表示する
    iframe.style.display = "block";

    // 親ページ側で高さを確実に確保（スマホ縦画面向け）
    iframe.style.height = Math.max(window.innerHeight * 0.75, 320) + "px";

    // 幅も念のため確保（親の制約がある場合に上書き）
    iframe.style.width = "100%";

    toggle3d.innerText = "3D 表示を閉じる";
  } else {
    iframe.style.display = "none";
    toggle3d.innerText = "3D 表示を開く";
  }
});

</script>

</body>
</html>
"""

# ============================================================
# 3D 表示（汎用：1〜18）
# ============================================================
@app.get("/green/{green_id}/3d", response_class=HTMLResponse)
def green_3d(green_id: int):
    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Green {green_id} - 3D View</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ margin: 0; overflow: hidden; background: #222; color: #fff; }}
  canvas {{ display: block; }}
  .error {{ padding:20px; color:#fff; background:#600; font-family:system-ui; }}
</style>
</head>
<body>

<script src="https://unpkg.com/three@0.152.2/build/three.min.js"></script>

<script>
async function loadGreenData() {{
  const url = "https://pcbdiagnosisrga8a5.blob.core.windows.net/course-maps/green_{green_id}.json";
  const res = await fetch(url);
  if (!res.ok) throw new Error("HTTP " + res.status);
  return await res.json();
}}

async function main() {{
  let data;
  try {{
    data = await loadGreenData();
  }} catch (e) {{
    document.body.innerHTML = "<div class='error'>JSON 読み込み失敗: " + e + "</div>";
    return;
  }}

  const heights = data.heights;
  const W = data.grid_width || 36;
  const H = data.grid_height || 36;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
      45,
      window.innerWidth / window.innerHeight,
      0.1,
      2000
  );
  camera.position.set(0, -120, 200);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // ★★★ ここが欠けていたため動いていなかった ★★★
  const geometry = new THREE.PlaneGeometry(36, 36, W - 1, H - 1);
  const verts = geometry.attributes.position;
  const HEIGHT_SCALE = 0.08;

  for (let i = 0; i < verts.count; i++) {{
    const x = i % W;
    const y = Math.floor(i / W);
    const h = heights[y][x] * HEIGHT_SCALE;
    verts.setZ(i, h);
  }}
  verts.needsUpdate = true;
  geometry.computeVertexNormals();
  // ★★★★★★★★★★★★★★★★★★★★★★★★★★★

  const material = new THREE.MeshLambertMaterial({ color: 0x55aa55 });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.rotation.x = -Math.PI / 2;
  scene.add(mesh);

  const wire = new THREE.Mesh(
    geometry.clone(),
    new THREE.MeshBasicMaterial({ color:0x003300, wireframe:true, opacity:0.25, transparent:true })
  );
  wire.rotation.x = -Math.PI / 2;
  scene.add(wire);

  const light = new THREE.DirectionalLight(0xffffff, 1);
  light.position.set(30, -30, 50);
  scene.add(light);
  scene.add(new THREE.AmbientLight(0x888888));

  function animate() {{
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
  }}
  animate();

  window.addEventListener('resize', () => {{
    const w = window.innerWidth;
    const h = window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }});
}}

main();
</script>

</body>
</html>
"""

