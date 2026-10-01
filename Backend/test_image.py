from PIL import Image

try:
    from Backend.image_predict import predict_histopathology
except ModuleNotFoundError:
    from image_predict import predict_histopathology


image_path = r"Data\histopathology\lung_n\lungn1.jpeg"

image = Image.open(image_path)

result = predict_histopathology(image)

print("\n" + "=" * 60)
print("HISTOPATHOLOGY PREDICTION")
print("=" * 60)

for key, value in result.items():
    print(f"{key}: {value}")