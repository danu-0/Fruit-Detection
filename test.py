from ultralytics import YOLO
import cv2
from flask import Flask, render_template, request, jsonify, send_file
import os
import io

# Inisialisasi aplikasi Flask
app = Flask(__name__)

model = YOLO('./runs/detect/train3/weights/best.pt')

# Fungsi untuk memproses gambar
def process_image(image_path, process_type):
    # Baca gambar menggunakan OpenCV
    img = cv2.imread(image_path)

    if process_type == "grayscale":
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif process_type == "binary":
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, processed_img = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY)
    elif process_type == "edges":
        processed_img = cv2.Canny(img, 100, 200)
    else:
        return None

    # Konversi gambar hasil proses ke format biner
    _, buffer = cv2.imencode('.jpg', processed_img)
    return io.BytesIO(buffer)

# Fungsi untuk melakukan deteksi objek
def detect_image(image_path):
    # Baca gambar menggunakan OpenCV
    img = cv2.imread(image_path)

    # Deteksi objek menggunakan model YOLO
    results = model(image_path)

    # Ambil hasil deteksi (bounding boxes, confidence, labels)
    boxes = results[0].boxes
    labels = results[0].names
    detected_objects = []

    for box in boxes:
        conf = box.conf[0]  # Confidence score
        cls = int(box.cls[0])  # ID kelas (integer)

        if conf > 0.3:  # Threshold untuk confidence
            label = labels[cls]
            detected_objects.append((label, conf))

    # Jika tidak ada objek terdeteksi
    if not detected_objects:
        return "Data tidak ada", None

    # Tambahkan bounding box pada gambar
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat bounding box
        conf = box.conf[0]
        cls = int(box.cls[0])
        label = labels[cls]
        color = (0, 255, 0)  # Warna hijau untuk bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Konversi gambar hasil prediksi ke format biner
    _, buffer = cv2.imencode('.jpg', img)
    return None, io.BytesIO(buffer)

# Route utama untuk menampilkan form upload gambar
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk upload dan memproses gambar
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    # Simpan file gambar sementara di memori
    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)

    # Jenis proses yang diminta
    process_type = request.form.get('process_type', 'detect')

    # Lakukan proses sesuai jenis
    try:
        if process_type == "detect":
            message, result_image = detect_image(image_path)
            if message == "Data tidak ada":
                return jsonify({"message": "Data tidak ada"})
        else:
            result_image = process_image(image_path, process_type)
    except Exception as e:
        return jsonify({"error": f"Error during processing: {str(e)}"})

    # Kirim hasil gambar ke pengguna
    return send_file(result_image, mimetype='image/jpeg')

# Menjalankan aplikasi Flask
if __name__ == "__main__":
    # Pastikan folder uploads ada
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
