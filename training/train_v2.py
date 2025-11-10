# ===========================================================
# train_v2.py — Train the ResNet + LSTM Explainable VQA Model
# ===========================================================

from prepare_data import setup_v2
from model_v2 import build_vqa_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# -----------------------------------------------------------
# 1️⃣ Load and prepare data
# -----------------------------------------------------------
print("\n--- Loading and preparing data ---")
# ✅ ✅ ✅ THIS IS THE CORRECTION: Added 4 underscores at the end ✅ ✅ ✅
train_X_ims, train_X_seqs, train_Y, \
test_X_ims, test_X_seqs, test_Y, \
im_shape, vocab_size, num_answers, \
all_answers, answer_to_idx, idx_to_answer, \
max_seq_len, tokenizer, _, _, _, _ = setup_v2()

# -----------------------------------------------------------
# 2️⃣ Build the model
# -----------------------------------------------------------
print("\n--- Building the ResNet + LSTM model ---")
model = build_vqa_model(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    num_answers=num_answers,
    im_shape=im_shape,
    trainable_resnet=False  # keep frozen for faster demo training
)

# -----------------------------------------------------------
# 3️⃣ Setup model checkpoints and early stopping
# -----------------------------------------------------------
os.makedirs("checkpoints", exist_ok=True)
checkpoint_path = os.path.join("checkpoints", "best_model.h5")

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,      # stop if no improvement for 3 epochs
    restore_best_weights=True
)

# -----------------------------------------------------------
# 4️⃣ Train the model
# -----------------------------------------------------------
print("\n--- Training started ---")
history = model.fit(
    [train_X_ims, train_X_seqs],
    train_Y,
    validation_data=([test_X_ims, test_X_seqs], test_Y),
    epochs=3,          # small number for testing
    batch_size=8,      # lower batch size for less memory usage
    shuffle=True,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# -----------------------------------------------------------
# 5️⃣ Save the final model
# -----------------------------------------------------------
model.save("vqa_model_final.h5")
print("\n✅ Training complete! Model saved as 'vqa_model_final.h5'")

# -----------------------------------------------------------
# 6️⃣ Display summary metrics
# -----------------------------------------------------------
final_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"🧠 Final Training Accuracy: {final_acc:.3f}")
print(f"🧩 Final Validation Accuracy: {val_acc:.3f}")