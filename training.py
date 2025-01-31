from ultralytics import YOLO
import torch

# Konfigurasi pelatihan
data_path = 'D:/Camp/PYTHON/citra_dataset/dataset/data.yaml'
batch_size = 8  # Sesuaikan dengan kapasitas GPU atau RAM laptop Anda
epochs = 50  # Jumlah epoch, bisa disesuaikan sesuai kecepatan pelatihan
imgsz = 640  # Ukuran gambar untuk input model (bisa sesuaikan dengan kapasitas GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Memilih perangkat (GPU jika tersedia)

# Membuat model YOLOv8
model = YOLO('yolov8s.pt')  # Menggunakan model yang lebih besar

# Memulai pelatihan
model.train(data=data_path,
            batch=batch_size,
            epochs=epochs,
            imgsz=imgsz,  # Gunakan 'imgsz' bukannya 'img_size'
            device=device,
            augment=True)  # Aktifkan augmentasi data

# Simpan model yang telah dilatih
model.save('trained_model.pt')

print(f"Training selesai! Model disimpan di: trained_model.pt")
