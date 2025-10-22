# app/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import cv2, os, tempfile, logging, sys
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
import mediapipe as mp  # core (for Image + Hands)

# Alias for the core mediapipe Image (NOT vision.Image)
MPImage = mp.Image

app = FastAPI()

# --- logging ---
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# --- baked-in model path (copied during image build) ---
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/app/models/face_landmarker_v2_with_blendshapes.task"
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/process-motion")
async def process_motion(
    phrase: str = Form(...),
    detectionArea: str = Form(...),   # "face"/"eyes" or "hand"/"hands"
    videoFile: UploadFile = File(...),
):
    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
        tf.write(await videoFile.read())
        temp_video_path = tf.name

    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Guard bogus or extreme FPS (e.g., 0 or 1000 from some encoders)
    if fps <= 0 or fps > 240:
        fps = 30.0

    logger.info("Video loaded: %s | fps=%.2f | frames=%d", temp_video_path, fps, total)

    # ~25ms step; ensure interval >= 1
    frame_interval = max(1, int(round(fps * 0.025)))
    frame_idx = 0

    motion_data = {}
    face_blendshapes = []
    hand_series = []

    face_landmarker = None
    hand_model = None

    try:
        if detectionArea in {"face", "eyes"}:
            # MediaPipe Tasks FaceLandmarker in VIDEO mode
            base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                num_faces=1,
                running_mode=vision.RunningMode.VIDEO,
            )
            face_landmarker = vision.FaceLandmarker.create_from_options(options)

        elif detectionArea in {"hand", "hands"}:
            # Classic Solutions Hands (works well in a frame loop)
            mp_hands = mp.solutions.hands
            hand_model = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
            )
        else:
            return JSONResponse(status_code=400, content={"error": "Invalid detectionArea"})

        while cap.isOpened():
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % frame_interval == 0:
                t_ms = int((frame_idx / fps) * 1000)  # VIDEO mode timestamp (monotonic)
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                if face_landmarker:
                    mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result: FaceLandmarkerResult = face_landmarker.detect_for_video(mp_image, t_ms)

                    values = {}
                    if result.face_blendshapes and len(result.face_blendshapes) > 0:
                        for bs in result.face_blendshapes[0]:
                            values[bs.category_name] = float(bs.score)
                    face_blendshapes.append({"timestamp_ms": t_ms, "values": values})

                else:
                    # Hands path
                    results = hand_model.process(rgb) if hand_model else None
                    frame_obj = {"timestamp_ms": t_ms, "right_hand": None, "left_hand": None}
                    if results and results.multi_hand_landmarks:
                        for i, hlm in enumerate(results.multi_hand_landmarks):
                            coords = [[float(l.x), float(l.y), float(l.z)] for l in hlm.landmark]
                            if i == 0:
                                frame_obj["right_hand"] = coords
                            elif i == 1:
                                frame_obj["left_hand"] = coords
                    hand_series.append(frame_obj)

            frame_idx += 1

    finally:
        cap.release()
        try:
            os.unlink(temp_video_path)
        except Exception:
            pass
        if hand_model:
            hand_model.close()
        if face_landmarker:
            face_landmarker.close()

    if face_landmarker:
        motion_data["face_blendshapes"] = face_blendshapes
    else:
        motion_data["hand_landmarks"] = hand_series

    return {
        "phrase": phrase,
        "detectionArea": detectionArea,
        "motion_data": motion_data,
    }
