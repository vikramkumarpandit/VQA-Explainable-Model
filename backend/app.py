from flask import Flask, jsonify, request
from flask_cors import CORS
from prepare_data import setup_v2
from model_v2 import build_vqa_model
from gradcam_vqa import make_gradcam_heatmap
from easy_vqa import get_train_questions, get_test_questions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import numpy as np
import os, random, cv2, shutil

# ------------------------------------------------------
# Flask setup
# ------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ------------------------------------------------------
# Ensure dataset images are available in /static
# ------------------------------------------------------
def copy_easy_vqa_images():
    """Copy Easy-VQA dataset images into static/easy_vqa_images if missing."""
    from easy_vqa import get_train_image_paths, get_test_image_paths
    train_paths = get_train_image_paths()
    test_paths = get_test_image_paths()

    base_dir = "static/easy_vqa_images"
    os.makedirs(f"{base_dir}/train", exist_ok=True)
    os.makedirs(f"{base_dir}/test", exist_ok=True)

    if len(os.listdir(f"{base_dir}/train")) == 0:
        print("📂 Copying Easy-VQA train images...")
        for src in train_paths.values():
            shutil.copy(src, f"{base_dir}/train/")
    if len(os.listdir(f"{base_dir}/test")) == 0:
        print("📂 Copying Easy-VQA test images...")
        for src in test_paths.values():
            shutil.copy(src, f"{base_dir}/test/")
    print("✅ Easy-VQA images ready in /static folder")

copy_easy_vqa_images()

# ------------------------------------------------------
# Load model + metadata once
# ------------------------------------------------------
print("🔹 Loading Easy-VQA dataset and model...")

(
    _train_X_ims, _train_X_seqs, _train_Y,
    _test_X_ims, _test_X_seqs, _test_Y,
    im_shape, vocab_size, num_answers,
    all_answers, answer_to_idx, idx_to_answer,
    max_seq_len, tokenizer,
    train_image_paths, test_image_paths,
    train_image_ids, test_image_ids
) = setup_v2()

train_qs, train_answers, train_ids = get_train_questions()
test_qs, test_answers, test_ids = get_test_questions()

model = build_vqa_model(vocab_size, max_seq_len, num_answers, im_shape, trainable_resnet=False)
model.load_weights("vqa_model_final.h5")

print("✅ Model and dataset loaded successfully!")

# ------------------------------------------------------
# API: Random Image
# ------------------------------------------------------
@app.route("/api/random_image", methods=["GET"])
def random_image():
    """Return random image URL from static folder."""
    folder = random.choice(["static/easy_vqa_images/train", "static/easy_vqa_images/test"])
    image_file = random.choice(os.listdir(folder))
    image_url = f"/{folder}/{image_file}"
    return jsonify({"image_path": image_url})

# ------------------------------------------------------
# API: Random Question
# ------------------------------------------------------
@app.route("/api/random_question", methods=["GET"])
def random_question():
    """Return random question from Easy-VQA dataset."""
    all_questions = train_qs + test_qs
    question = random.choice(all_questions)
    return jsonify({"question": question})

# ------------------------------------------------------
# API: Predict + GradCAM
# ------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict_vqa():
    data = request.json
    image_path = data.get("image_path")
    question = data.get("question")

    if not image_path or not question:
        return jsonify({"error": "Missing image_path or question"}), 400

    # Convert /static/... path → actual local file path
    if image_path.startswith("/"):
        image_path = image_path.lstrip("/")
    image_path = os.path.join(os.getcwd(), image_path)

    # Preprocess image
    image = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(image)
    img_array = np.expand_dims(resnet_preprocess(img_array), axis=0)

    # Preprocess question
    seq = tokenizer.texts_to_sequences([question])
    padded_seq = pad_sequences(seq, maxlen=max_seq_len, padding='post')

    # Predict
    preds = model.predict([img_array, padded_seq])
    pred_idx = np.argmax(preds[0])
    pred_answer = all_answers[pred_idx]

    # Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, padded_seq, model, "conv5_block3_out")
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    heatmap = cv2.resize(heatmap, (128, 128))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    os.makedirs("static", exist_ok=True)

    # Cleanup old Grad-CAMs
    for f in os.listdir("static"):
        if f.startswith("gradcam_") and f.endswith(".png"):
            try:
                os.remove(os.path.join("static", f))
            except:
                pass

    # Save new Grad-CAM
    unique_id = str(random.randint(10000, 99999))
    out_filename = f"gradcam_{unique_id}.png"
    out_path = os.path.join("static", out_filename)
    cv2.imwrite(out_path, overlay)

    return jsonify({
        "answer": pred_answer,
        "heatmap_url": f"/static/{out_filename}"
    })

# ------------------------------------------------------
# Run Flask (for Render)
# ------------------------------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    print(f"✅ Starting server on port {port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
