# ===========================================================
# prepare_data.py — Easy-VQA data preparation (optimized version)
# Works with: model_v2.py (ResNet50 + LSTM)
# ===========================================================

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import numpy as np
import json
import os
from easy_vqa import get_train_questions, get_test_questions, get_train_image_paths, get_test_image_paths, get_answers


# -----------------------------------------------------------
# Helper: Read Easy-VQA data (subset + path maps)
# -----------------------------------------------------------
def _read_easy_vqa():
    # ✅ Load only a small subset for low-memory testing
    train_qs, train_answers, train_image_ids = get_train_questions()[:500]
    test_qs, test_answers, test_image_ids = get_test_questions()[:100]

    # ✅ Get image paths
    train_image_paths = get_train_image_paths()
    test_image_paths = get_test_image_paths()
    all_answers = get_answers()

    return (train_qs, train_answers, train_image_ids,
            test_qs, test_answers, test_image_ids,
            train_image_paths, test_image_paths, all_answers)


# -----------------------------------------------------------
# Main function: setup_v2
# -----------------------------------------------------------
def setup_v2(image_size=128, max_seq_len=20, save_artifacts=True, artifacts_dir='artifacts'):
    """
    Prepares Easy-VQA data for the ResNet + LSTM model.
    Returns preprocessed train/test data arrays and metadata.
    """

    print("\n--- Loading Easy-VQA data ---")
    train_qs, train_answers, train_image_ids, \
    test_qs, test_answers, test_image_ids, \
    train_image_paths, test_image_paths, all_answers = _read_easy_vqa()

    print(f"Train Qs: {len(train_qs)}, Test Qs: {len(test_qs)}")

    # -------------------------------------------------------
    # Build answer mappings
    # -------------------------------------------------------
    answer_to_idx = {a: i for i, a in enumerate(all_answers)}
    idx_to_answer = {i: a for a, i in answer_to_idx.items()}
    num_answers = len(all_answers)

    # -------------------------------------------------------
    # Image preprocessing (resize + normalize for ResNet)
    # -------------------------------------------------------
    print(f"\n--- Processing images ({image_size}x{image_size}, ResNet preprocessing) ---")

    def load_and_process_image(image_path):
        im = load_img(image_path, target_size=(image_size, image_size))
        im = img_to_array(im)
        im = resnet_preprocess(im)
        return im

    def read_images(image_ids, image_paths):
        ims = {}
        for img_id in image_ids:
            ims[img_id] = load_and_process_image(image_paths[img_id])
        return ims

    train_imgs_dict = read_images(train_image_ids, train_image_paths)
    test_imgs_dict = read_images(test_image_ids, test_image_paths)

    train_X_ims = np.array([train_imgs_dict[id] for id in train_image_ids])
    test_X_ims = np.array([test_imgs_dict[id] for id in test_image_ids])

    im_shape = train_X_ims[0].shape
    print(f"Read {len(train_X_ims)} train + {len(test_X_ims)} test images, shape={im_shape}")

    # -------------------------------------------------------
    # Question preprocessing (tokenization + padding)
    # -------------------------------------------------------
    print("\n--- Tokenizing questions (Embedding + LSTM input) ---")
    tokenizer = Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts(train_qs)
    vocab_size = len(tokenizer.word_index) + 1

    train_seq = tokenizer.texts_to_sequences(train_qs)
    test_seq = tokenizer.texts_to_sequences(test_qs)
    train_X_seqs = pad_sequences(train_seq, maxlen=max_seq_len, padding='post', truncating='post')
    test_X_seqs = pad_sequences(test_seq, maxlen=max_seq_len, padding='post', truncating='post')

    print(f"Vocab Size: {vocab_size}, Example Q sequence: {train_X_seqs[0]}")

    # -------------------------------------------------------
    # Answer encoding (one-hot vectors)
    # -------------------------------------------------------
    train_ans_idx = [answer_to_idx[a] for a in train_answers]
    test_ans_idx = [answer_to_idx[a] for a in test_answers]

    train_Y = to_categorical(train_ans_idx, num_classes=num_answers)
    test_Y = to_categorical(test_ans_idx, num_classes=num_answers)

    # -------------------------------------------------------
    # Save tokenizer + answer mappings
    # -------------------------------------------------------
    if save_artifacts:
        os.makedirs(artifacts_dir, exist_ok=True)
        with open(os.path.join(artifacts_dir, 'tokenizer.json'), 'w') as f:
            json.dump(tokenizer.to_json(), f)
        with open(os.path.join(artifacts_dir, 'answers.json'), 'w') as f:
            json.dump({'answers': all_answers,
                       'answer_to_idx': answer_to_idx,
                       'idx_to_answer': idx_to_answer}, f)

    print("\n✅ Data preparation complete!")

    # ✅ ✅ ✅ THIS IS THE CORRECTED RETURN STATEMENT ✅ ✅ ✅
    return (train_X_ims, train_X_seqs, train_Y,
            test_X_ims, test_X_seqs, test_Y,
            im_shape, vocab_size, num_answers,
            all_answers, answer_to_idx, idx_to_answer,
            max_seq_len, tokenizer,
            # Added these four values:
            train_image_paths, test_image_paths,
            train_image_ids, test_image_ids
           )


# -----------------------------------------------------------
# Run test when file executed directly
# -----------------------------------------------------------
if __name__ == "__main__":
    print("🔹 Testing setup_v2() ...")
    data = setup_v2()
    print("✅ Data preparation complete!")