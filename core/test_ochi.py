from ultralytics import YOLO

# I load my newly trained model from the 6th run
model = YOLO('/app/tests/current_test/runs/omr_nivel06/weights/best.pt')

# I run inference but I drastically lower the confidence threshold to 5% (0.05)
# This forces the AI to show me even its wildest guesses!
print("I am analyzing the music sheets with lowered confidence...")
results = model.predict(
    source='/app/dataset/minor_dataset/dataset_1/train/images', 
    save=True,
    conf=0.05, # <-- MAGIA ESTE AICI
    project='/app/tests/inference_results',
    name='second_test'
)

print("Done! I have saved the new results.")