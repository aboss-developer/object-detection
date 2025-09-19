import cv2
import json
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11m.pt")

# Stream video
video_path = "https://ITSStreamingBR2.dotd.la.gov/public/shr-cam-030.streams/playlist.m3u8"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video stream.")
    exit()

# Get FPS (fallback to 30 if not available)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 30.0

results_list = []
frame_num = 0
frames_to_save_json = 100  # Save every 100 frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Error: Could not read frame from stream. Exiting...")
        break

    frame_num += 1
    timestamp = frame_num / fps  # Convert frame number to seconds

    # Run YOLO detection
    results = model(frame, verbose=False)[0]  # Take first result

    for box in results.boxes:
        cls_id = int(box.cls[0])        # Class ID
        label = model.names[cls_id]     # Class name
        conf = float(box.conf[0])       # Confidence

        # Bounding box (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Save results to JSON
        results_list.append({
            "frame": frame_num,
            "timestamp_sec": round(timestamp, 2),
            "class_id": cls_id,
            "class_name": label,
            "confidence": round(conf, 2),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
        })

        # Draw bounding box on frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Put label + confidence text
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save detections every N frames
    if frame_num % frames_to_save_json == 0:
        with open("live_detections.json", "w") as f:
            json.dump(results_list, f, indent=4)
        print(f"✅ JSON log updated at frame {frame_num}.")

    # Show live detections with boxes
    cv2.imshow("YOLO Live Detection", frame)

    # Quit with 'q'
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Streaming stopped.")
