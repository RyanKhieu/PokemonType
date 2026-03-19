import os

import gradio as gr
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps


np.set_printoptions(suppress=True)


def get_model_path():
	candidates = ["keras_model.h5", "keras_Model.h5"]
	for candidate in candidates:
		if os.path.exists(candidate):
			return candidate
	raise FileNotFoundError("Could not find model file. Expected keras_model.h5 or keras_Model.h5")


def load_labels(path="labels.txt"):
	with open(path, "r", encoding="utf-8") as labels_file:
		labels = []
		for line in labels_file:
			line = line.strip()
			if not line:
				continue
			parts = line.split(maxsplit=1)
			labels.append(parts[1] if len(parts) > 1 else parts[0])
	return labels


model = load_model(get_model_path(), compile=False)
class_names = load_labels("labels.txt")


def predict_pokemon_type(image: Image.Image):
	if image is None:
		return "Please upload an image.", {}

	data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
	image = image.convert("RGB")
	image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

	image_array = np.asarray(image)
	normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
	data[0] = normalized_image_array

	prediction = model.predict(data, verbose=0)[0]
	index = int(np.argmax(prediction))
	confidence = float(prediction[index])
	predicted_class = class_names[index] if index < len(class_names) else str(index)

	confidence_table = {name: float(score) for name, score in zip(class_names, prediction)}
	result_text = f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}"
	return result_text, confidence_table


app = gr.Interface(
	fn=predict_pokemon_type,
	inputs=gr.Image(type="pil", label="Upload Pokemon Image"),
	outputs=[
		gr.Textbox(label="Result"),
		gr.Label(label="All Class Scores", num_top_classes=2),
	],
	title="Pokemon Type Classifier",
	description="Upload an image to classify it as Fire or Water.",
)


if __name__ == "__main__":
	app.launch()
