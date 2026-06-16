import os
import json
import time
import logging
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests
from json import JSONDecodeError

from fastapi import FastAPI, HTTPException, Query
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

# 単体用の環境変数（従来互換）
BLOB_IMAGE_URL = os.getenv("GREEN_IMAGE_URL_1", "")

# 一括生成用パターン（例: https://.../green_{id}.png）
GREEN_IMAGE_URL_PATTERN = os.getenv("GREEN_IMAGE_URL_PATTERN", "")

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
# AI 戦略 API（汎用）
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
# 汎用: 画像から JSON を生成して Blob にアップロードする関数
# ============================================================
def generate_green_from_url(green_id: int, url: str):
    """
    指定 URL の画像を読み込み、36x36 にリサイズして color_to_height で JSON を作成し
    Blob に green_{id}.json としてアップロードする。例外は呼び出し元で処理する。
    """
    logger.info("Generating green %d from %s", green_id, url)
    img = load_image_from_blob(url)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([10, 30, 30])
    upper = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"green_{green_id}.png: グリーン領域が検出できませんでした")

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
        "green_id": green_id,
        "grid_width": 36,
        "grid_height": 36,
        "cell_size_yards": 1.0,
        "heights": height_map,
        "pin_positions": {}
    }

    blob_name = f"green_{green_id}.json"
    blob = container_client.get_blob_client(blob_name)
    blob.upload_blob(json.dumps(json_data), overwrite=True)
    logger.info("Uploaded %s", blob_name)
    return {"green_id": green_id, "status": "ok", "blob": blob_name}

# ============================================================
# 一括生成エンドポイント
# ============================================================
@app.get("/generate/greens")
def generate_greens(start: int = 1, end: int = 18, concurrency: int = Query(None, description="同時処理数")):
    """
    一括生成エンドポイント。
    使用例: /generate/greens?start=1&end=18
    concurrency: 同時処理数（未指定なら環境変数 GENERATE_CONCURRENCY、無ければ直列）
    """
    pattern = GREEN_IMAGE_URL_PATTERN
    if not pattern:
        raise HTTPException(status_code=500, detail="GREEN_IMAGE_URL_PATTERN が設定されていません")

    if concurrency is None:
        try:
            concurrency = int(os.getenv("GENERATE_CONCURRENCY", "0"))
        except Exception:
            concurrency = 0

    ids = list(range(start, end + 1))
    results = []
    errors = []

    if concurrency and concurrency > 1:
        logger.info("Generating greens concurrently: %d workers", concurrency)
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            future_to_id = {}
            for gid in ids:
                # pattern に合わせて id を埋める（必要なら zfill を使う）
                url = pattern.format(id=gid)
                future = ex.submit(generate_green_from_url, gid, url)
                future_to_id[future] = gid

            for fut in as_completed(future_to_id):
                gid = future_to_id[fut]
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as e:
                    logger.exception("Failed to generate green %d", gid)
                    errors.append({"green_id": gid, "error": str(e)})
    else:
        # 直列処理
        for gid in ids:
            url = pattern.format(id=gid)
            try:
                res = generate_green_from_url(gid, url)
                results.append(res)
            except Exception as e:
                logger.exception("Failed to generate green %d", gid)
                errors.append({"green_id": gid, "error": str(e)})

    return {"generated": results, "errors": errors}

# ============================================================
# Green1 JSON 生成（従来互換エンドポイント）
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
# ピン位置保存 API
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
# 起動画面：統合 UI（ホール選択対応）
# ============================================================
@app.get("/", response_class=HTMLResponse)
def green1():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Green - ピン登録 & AI戦略</title>
<style>
  body { background:#222; color:white; font-size:20px; text-align:center; margin:0; padding:10px; }
  #canvas { touch-action: manipulation; border:1px solid #555; background:#111; }
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
  label { font-size:18px; display:block; margin:8px 0; }
  select { font-size:18px; padding:6px; }
</style>
</head>
<body>

<h2>Green - ピン登録 & AI戦略</h2>

<label for="holeSelect">ホール選択:
  <select id="holeSelect"></select>
</label>

<canvas id="canvas" width="360" height="360"></canvas>

<p id="info">ピン位置をタップしてください</p>

<button id="saveBtn" style="display:none;">この位置を登録する</button>
<button id="aiBtn" style="display:none; background:#2196F3;">AI に戦略を聞く</button>

<div id="result"></div>

<h3 style="margin-top:30px;">3D グリーン（参考表示）</h3>
<iframe id="view3d" src="/green/3d?hole=1"></iframe>

<script>
(function(){
  // ホール選択を生成
  const holeSelect = document.getElementById("holeSelect");
  for (let i = 1; i <= 18; i++) {
    const opt = document.createElement("option");
    opt.value = i;
    opt.text = "Hole " + i;
    holeSelect.appendChild(opt);
  }

  let selectedX = null;
  let selectedY = null;
  let currentHole = 1;

  const IMAGE_URL_PATTERN = "https://pcbdiagnosisrga8a5.blob.core.windows.net/course-maps/green_{id}.png";
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const saveBtn = document.getElementById("saveBtn");
  const aiBtn = document.getElementById("aiBtn");
  const info = document.getElementById("info");
  const resultDiv = document.getElementById("result");
  const iframe = document.getElementById("view3d");

  let img = new Image();

  function loadHole(holeId) {
    currentHole = holeId;
    selectedX = null;
    selectedY = null;
    saveBtn.style.display = "none";
    aiBtn.style.display = "none";
    info.innerText = "ピン位置をタップしてください";
    resultDiv.innerText = "";

    const url = IMAGE_URL_PATTERN.replace("{id}", holeId);
    img = new Image();
    img.crossOrigin = "anonymous";
    img.src = url;
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      // 既存のピンがあれば表示
      fetch(`/green_${holeId}.json`).then(()=>{}).catch(()=>{});
      // iframe 更新（汎用 3D エンドポイント）
      iframe.src = `/green/3d?hole=${holeId}`;
    };
    img.onerror = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#444";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "white";
      ctx.fillText("画像を読み込めませんでした", 10, 20);
    };
  }

  // 初期ロード
  loadHole(1);
  holeSelect.value = "1";

  holeSelect.addEventListener("change", function() {
    const holeId = parseInt(this.value);
    loadHole(holeId);
  });

  canvas.addEventListener("click", function(e) {
    const rect = canvas.getBoundingClientRect();
    selectedX = Math.floor((e.clientX - rect.left) / 10);
    selectedY = Math.floor((e.clientY - rect.top) / 10);

    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.arc(selectedX * 10, selectedY * 10, 6, 0, Math.PI * 2);
    ctx.fillStyle = "red";
    ctx.fill();

    info.innerText = `選択中のピン位置: (${selectedX}, ${selectedY})`;

    saveBtn.style.display = "block";
  });

  saveBtn.addEventListener("click", async function() {
    if (selectedX === null || selectedY === null) return;
    const res = await fetch(`/set_pin/${currentHole}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x: selectedX, y: selectedY })
    });
    if (!res.ok) {
      const txt = await res.text();
      info.innerText = "ピン保存エラー: " + txt;
      return;
    }
    info.innerText = "ピン位置 (" + selectedX + ", " + selectedY + ") を登録しました！";
    aiBtn.style.display = "block";
  });

  aiBtn.addEventListener("click", async function() {
    resultDiv.innerText = "AI が戦略を計算中です…";

    const res = await fetch(`/ai_strategy/${currentHole}`, { method: "POST" });
    if (!res.ok) {
      const text = await res.text();
      resultDiv.innerText = "サーバーエラー: " + text;
      return;
    }

    const data = await res.json();
    if (!data.slope_analysis || !data.strategy) {
      resultDiv.innerText = "レスポンス形式が不正です";
      return;
    }

    let text = "";
    text += "⛰️ 傾斜の解説:\n" + data.slope_analysis + "\n\n";
    text += "🧠 戦略:\n" + data.strategy;
    resultDiv.innerText = text;

    // ピン再描画
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    if (selectedX !== null && selectedY !== null) {
      ctx.beginPath();
      ctx.arc(selectedX * 10, selectedY * 10, 6, 0, Math.PI * 2);
      ctx.fillStyle = "red";
      ctx.fill();
    }
  });

})();
</script>

</body>
</html>
"""

# ============================================================
# 汎用 3D 表示（クエリでホール指定）
# ============================================================
@app.get("/green/3d", response_class=HTMLResponse)
def green_3d(hole: int = Query(1, ge=1, le=99)):
    # この HTML はクライアント側で green_{hole}.json を fetch する
    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Green {hole} - 3D View</title>
<style>
  body {{ margin: 0; overflow: hidden; background: #222; }}
  canvas {{ display: block; }}
</style>
</head>
<body>

<script src="https://cdn.jsdelivr.net/npm/three@0.152.2/build/three.min.js"></script>

<script>
(async function() {{
  const hole = {hole};
  const url = `https://pcbdiagnosisrga8a5.blob.core.windows.net/course-maps/green_${{hole}}.json`;

  async function loadGreenData() {{
    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to load JSON");
    return await res.json();
  }}

  try {{
    const data = await loadGreenData();
    const heights = data.heights;
    const W = data.grid_width;
    const H = data.grid_height;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, -60, 40);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(30, -30, 50);
    scene.add(light);

    const ambient = new THREE.AmbientLight(0x888888);
    scene.add(ambient);

    const geometry = new THREE.PlaneGeometry(36, 36, W - 1, H - 1);

    const verts = geometry.attributes.position;
    for (let i = 0; i < verts.count; i++) {{
      const x = i % W;
      const y = Math.floor(i / W);
      const h = heights[y][x] * 0.3;
      verts.setZ(i, h);
    }}
    verts.needsUpdate = true;
    geometry.computeVertexNormals();

    const material = new THREE.MeshLambertMaterial({{
      color: 0x55aa55,
      side: THREE.DoubleSide
    }});

    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    function animate() {{
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    }}
    animate();
  }} catch (err) {{
    document.body.style.color = "white";
    document.body.style.padding = "20px";
    document.body.innerText = "3D データの読み込みに失敗しました: " + err.message;
  }}
}})();
</script>

</body>
</html>
"""
