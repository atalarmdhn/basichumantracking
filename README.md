# Pelacak Manusia Real-time dengan YOLOv8 & DeepSORT

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Proyek ini adalah implementasi sistem **deteksi dan pelacakan manusia (human tracking) secara real-time** menggunakan kekuatan **YOLOv8** untuk deteksi objek presisi dan **DeepSORT** untuk pelacakan yang tangguh. Input video diproses langsung dari **webcam**, dan hasilnya ditampilkan di layar lengkap dengan **kotak pembatas (bounding box)** serta **ID unik** untuk setiap individu yang terlacak.

<img width="682" height="549" alt="Screenshot 2025-08-13 155559" src="https://github.com/user-attachments/assets/4d2710f6-e0f7-48f4-a357-b71a61e25a12" />

---

## 📌 Fitur Utama

-   **Deteksi Manusia Akurat**: Menggunakan model YOLOv8 yang sudah terlatih pada dataset COCO untuk mengenali manusia dengan presisi tinggi.
-   **Pelacakan Multi-Objek**: Dengan DeepSORT, sistem mampu mempertahankan ID unik untuk setiap individu, bahkan saat mereka bergerak, tumpang tindih, atau sempat keluar dari frame.
-   **Proses Real-Time**: Didesain untuk bekerja secara efisien menggunakan input langsung dari webcam.
-   **ID Pelacakan Unik**: Memberikan label ID numerik yang persisten untuk setiap orang yang terdeteksi.
-   **Filter Deteksi**: Untuk meningkatkan efisiensi dan akurasi, deteksi difilter berdasarkan:
    -   Kelas objek: Hanya `person` (COCO class ID = 0).
    -   Tingkat kepercayaan (confidence score): Hanya deteksi dengan skor > 0.5 yang diproses.

---

## ⚙️ Alur Kerja Program

Sistem bekerja dalam tiga tahap utama pada setiap frame video:

1.  **Deteksi (YOLOv8)**: Model YOLOv8 menganalisis frame untuk menemukan lokasi semua manusia dan menghasilkan koordinat kotak pembatas (*bounding box*) untuk masing-masing.
2.  **Pelacakan (DeepSORT)**: Data kotak pembatas dari YOLOv8 diteruskan ke DeepSORT. Algoritma ini kemudian menggunakan *Kalman Filter* dan model *Re-Identification* untuk menetapkan ID unik ke setiap deteksi dan melacak pergerakannya antar-frame.
3.  **Visualisasi (OpenCV)**: Frame video asli digabungkan dengan hasil dari DeepSORT (kotak pembatas dan ID) lalu ditampilkan ke layar pengguna secara real-time.

---

## 🚀 Instalasi & Pengaturan

Pastikan Anda memiliki **Python 3.8 atau yang lebih baru**.

1.  **Clone repository ini:**
    ```bash
    git clone https://github.com/atalarmdhn/basichumantracking.git
    ```

2.  **Instal semua dependensi yang dibutuhkan:**
    Anda bisa menginstalnya satu per satu atau menggunakan `requirements.txt` jika tersedia.
    ```bash
    pip install ultralytics deep-sort-realtime opencv-python
    ```
    *Catatan: Saat program dijalankan pertama kali, `ultralytics` akan mengunduh bobot model YOLOv8 secara otomatis. Pastikan ada koneksi internet.*

---

## ▶️ Cara Menjalankan

Setelah instalasi selesai, jalankan skrip utama dari terminal:

```bash
python human_track.py
