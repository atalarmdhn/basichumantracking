import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# debug
print("Program Human Tracking dengan DeepSORT dimulai...")
print("Tekan tombol 'q' untuk keluar.")

# Model yolov8 akan diunduh otomatis saat pertama kali dijalankan
try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Error saat memuat model YOLO: {e}")
    print("Pastikan Anda terhubung ke internet untuk mengunduh model saat pertama kali.")
    exit()

# Inisialisasi tracker DeepSORT
# max_age: berapa frame sebuah track dipertahankan tanpa ada deteksi baru
tracker = DeepSort(max_age=30)

# Mengakses webcam
cap = cv2.VideoCapture(0)

# Looping utama untuk memproses video
while cap.isOpened():
    # Membaca satu frame dari video
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame. Program berhenti.")
        break

    # 1. DETEKSI oleh YOLOv8
    # Lakukan deteksi objek pada frame
    results = model(frame, stream=True)

    # List untuk menyimpan hasil deteksi yang akan dikirim ke DeepSORT
    detections = []

    # Loop melalui hasil deteksi dari YOLO
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Dapatkan koordinat bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Dapatkan kelas objek dan confidence score
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Filter hanya untuk kelas 'person' (kelas 0 di dataset COCO)
            # dan confidence di atas 0.5
            if cls == 0 and conf > 0.5:
                # Format deteksi untuk DeepSORT: [left, top, width, height], confidence, class
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, cls))

    # 2. TRACKING oleh DeepSORT
    # Update tracker dengan deteksi yang sudah difilter
    tracks = tracker.update_tracks(detections, frame=frame)

    # Loop melalui hasil tracking
    for track in tracks:
        # Lewati track yang belum terkonfirmasi
        if not track.is_confirmed():
            continue

        # Dapatkan ID unik dan bounding box dari track
        track_id = track.track_id
        ltrb = track.to_ltrb() # Format: [left, top, right, bottom]
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # Gambar bounding box dan ID pada frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Tampilkan hasil frame
    cv2.imshow("Human Tracking dengan DeepSORT | Tekan 'q' untuk keluar", frame)

    # Hentikan program jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan sumber video dan tutup semua jendela
print("Program dihentikan.")
cap.release()
cv2.destroyAllWindows()