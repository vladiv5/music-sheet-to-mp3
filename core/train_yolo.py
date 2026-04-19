from ultralytics import YOLO

# --- MODEL SELECTION ---
# RTX 4060 (8GB VRAM): YOLOv8s is the sweet spot.
# - nano (n) is too weak for 15 classes and dense small objects (noteheads).
# - medium (m) risks OOM at imgsz=1280 with batch=8 on 8GB VRAM.
# - small (s) gives a strong accuracy boost over nano while staying safe on memory.
model = YOLO('yolov8s.pt')

print("Starting YOLO training on big_dataset (2200 images, 15 classes)...")
print("Model  : YOLOv8s")
print("imgsz  : 1280  — critical for small objects like noteheads on music sheets")
print("batch  : 8     — safe ceiling for RTX 4060 8GB at this resolution")

results = model.train(
    data='/app/dataset/big_dataset/data.yaml',

    # --- RESOLUTION ---
    # Music sheets have tiny, densely packed symbols (noteheads ≈ 20-30px at 640).
    # Doubling to 1280 gives 4x more pixels → dramatically improves small-object recall.
    imgsz=1280,

    # --- TRAINING SCHEDULE ---
    epochs=100,           # enough headroom for 2200 images; early stopping will trim it
    patience=20,          # stop if val loss doesn't improve for 20 consecutive epochs

    # --- BATCH & MEMORY ---
    batch=4,              # reduced from 8: extra safety margin for system RAM at imgsz=1280
    cache=False,          # do NOT cache images in RAM — this was causing OOM

    # --- OPTIMIZER ---
    optimizer='AdamW',    # more stable than default SGD on medium-sized datasets
    lr0=0.001,            # standard AdamW starting LR
    weight_decay=0.0005,

    # --- AUGMENTATION ---
    # Ultralytics applies mosaic, flips, HSV jitter etc. by default — keep them on.
    augment=True,

    # --- OUTPUT ---
    project='/app/tests/current_test/runs',
    name='omr_nivel1_v8s_1280',   # name encodes model + resolution for easy comparison
    
    # --- HARDWARE ---
    device=0,             # GPU 0 (the RTX 4060)
    workers=2,            # CRITICAL: was 8 → OOM killer was killing DataLoader workers
                          # 8 workers × 1280px images = too much system RAM inside Docker
)

print("Training complete!")
print(f"Best weights saved to: {results.save_dir}/weights/best.pt")