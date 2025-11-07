# ===========================================================
# model_v2.py — ResNet50 + LSTM Model for Explainable VQA
# ===========================================================
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding, LSTM, concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# -----------------------------------------------------------
# Build the ResNet + LSTM VQA model
# -----------------------------------------------------------
def build_vqa_model(vocab_size, max_seq_len, num_answers, im_shape=(224, 224, 3), trainable_resnet=False):
    """
    Creates a dual-branch model:
    1️⃣ Image branch → ResNet50 backbone
    2️⃣ Question branch → Embedding + LSTM
    Then merges both and outputs softmax probabilities over answers.
    """

    print("\n--- Building ResNet50 + LSTM VQA model ---")

    # -------------------------------------------------------
    # 1️⃣ Image branch — Feature extraction with ResNet50
    # -------------------------------------------------------
    image_input = Input(shape=im_shape, name="image_input")

    # Pretrained ResNet50 (ImageNet weights)
    resnet_base = ResNet50(include_top=False, weights='imagenet', input_tensor=image_input)
    resnet_base.trainable = trainable_resnet  # freeze weights to speed up training

    # Global pooling instead of Flatten (reduces params)
    image_features = GlobalAveragePooling2D()(resnet_base.output)
    image_features = Dense(512, activation='relu')(image_features)
    image_features = Dropout(0.3)(image_features)

    # -------------------------------------------------------
    # 2️⃣ Question branch — Embedding + LSTM
    # -------------------------------------------------------
    question_input = Input(shape=(max_seq_len,), name="question_input")

    # Convert word indices to embeddings
    x = Embedding(input_dim=vocab_size, output_dim=256, input_length=max_seq_len, mask_zero=True)(question_input)
    x = LSTM(256, return_sequences=False)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)

    # -------------------------------------------------------
    # 3️⃣ Combine both branches
    # -------------------------------------------------------
    combined = concatenate([image_features, x])
    combined = Dense(512, activation='relu')(combined)
    combined = Dropout(0.4)(combined)
    output = Dense(num_answers, activation='softmax', name="output")(combined)

    # -------------------------------------------------------
    # 4️⃣ Build and compile model
    # -------------------------------------------------------
    model = Model(inputs=[image_input, question_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    return model
