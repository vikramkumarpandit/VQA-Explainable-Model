# ===========================================================
# gradcam_vqa.py — Grad-CAM Visualization for Explainable VQA
# ===========================================================
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt

import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import json, os
from prepare_data import setup_v2
from model_v2 import build_vqa_model   # ✅ import your architecture builder


# -----------------------------------------------------------
# Custom NotEqual layer (for TF >= 2.16 compatibility)
# -----------------------------------------------------------
class NotEqual(Layer):
    def __init__(self, **kwargs):
        super(NotEqual, self).__init__(**kwargs)
    def call(self, inputs):
        x, y = inputs
        return K.not_equal(x, y)


# -----------------------------------------------------------
# Grad-CAM generator
# -----------------------------------------------------------
def make_gradcam_heatmap(img_array, question_seq, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array, question_seq])
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # This line converts heatmap to a numpy array:
    heatmap = np.maximum(heatmap, 0)
    
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
        
    return heatmap


# -----------------------------------------------------------
# 1️⃣ Load metadata and rebuild model architecture
# -----------------------------------------------------------
print("\n--- Rebuilding architecture and loading weights ---")

# minimal dataset load to get parameters
_, _, _, _, _, _, _, vocab_size, num_answers, _, _, _, max_seq_len, _, _, _, _, _ = setup_v2()
im_shape = (128, 128, 3)

# rebuild architecture
model = build_vqa_model(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    num_answers=num_answers,
    im_shape=im_shape,
    trainable_resnet=False
)

# load weights only
with custom_object_scope({'NotEqual': NotEqual}):
    model.load_weights("vqa_model_final.h5")

print("✅ Model architecture rebuilt and weights loaded successfully!")


# -----------------------------------------------------------
# 2️⃣ Load tokenizer and answers
# -----------------------------------------------------------
with open("artifacts/tokenizer.json") as f:
    tokenizer_json = json.load(f)
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

with open("artifacts/answers.json") as f:
    ans_data = json.load(f)
all_answers = ans_data["answers"]
idx_to_answer = {int(k): v for k, v in ans_data["idx_to_answer"].items()}


# -----------------------------------------------------------
# 3️⃣ Prepare one test sample
# -----------------------------------------------------------
print("\n--- Preparing test image and question ---")

(
    train_X_ims, train_X_seqs, train_Y,
    test_X_ims, test_X_seqs, test_Y,
    im_shape, vocab_size, num_answers,
    all_answers, answer_to_idx, idx_to_answer,
    max_seq_len, tokenizer,
    train_image_paths, test_image_paths,
    train_image_ids, test_image_ids
) = setup_v2()


# --- Determine correct image path ---
print("🔍 Getting first image ID from 'train_image_ids'...")
first_image_id = train_image_ids[0]
print(f"✅ First image ID: {first_image_id}")

print("🔍 Looking up path in 'train_image_paths' map...")
image_path = train_image_paths[first_image_id]
print("✅ Selected image path:", image_path)


image = load_img(image_path, target_size=(128, 128))
img_array = img_to_array(image)
img_array = np.expand_dims(resnet_preprocess(img_array), axis=0)

question = "What color is the object?"
seq = tokenizer.texts_to_sequences([question])
padded_seq = pad_sequences(seq, maxlen=20, padding='post')



# -----------------------------------------------------------
# 4️⃣ Predict
# -----------------------------------------------------------
preds = model.predict([img_array, padded_seq])
pred_class = np.argmax(preds[0])
pred_answer = all_answers[pred_class]
print(f"🧠 Predicted Answer: {pred_answer}")


# -----------------------------------------------------------
# 5️⃣ Generate Grad-CAM
# -----------------------------------------------------------
print("\n--- Generating Grad-CAM heatmap ---")
# ✅ ✅ ✅ THE FIX: Corrected the variable name ✅ ✅ ✅
last_conv_layer_name = "conv5_block3_out"  # last conv layer of ResNet50
heatmap = make_gradcam_heatmap(img_array, padded_seq, model, last_conv_layer_name)


# -----------------------------------------------------------
# 6️⃣ Overlay and save
# -----------------------------------------------------------
def overlay_heatmap(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

overlay = overlay_heatmap(image_path, heatmap)
cv2.imwrite("gradcam_output.png", overlay)
print("✅ Grad-CAM saved as 'gradcam_output.png'")


# -----------------------------------------------------------
# 7️⃣ Display results
# -----------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(load_img(image_path, target_size=(128, 128)))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title(f"Grad-CAM — Focus on: {pred_answer}")
plt.axis("off")

plt.tight_layout()
plt.show()