const randomImageBtn = document.getElementById("random-image-btn");
const randomQuestionBtn = document.getElementById("random-question-btn");
const predictBtn = document.getElementById("predict-btn");

const imageEl = document.getElementById("vqa-image");
const questionInput = document.getElementById("question-input");
const answerBox = document.getElementById("answer-box");
const heatmapEl = document.getElementById("heatmap-image");

const BACKEND_URL = "http://127.0.0.1:5000";

// --- Load random image ---
async function loadRandomImage() {
  answerBox.textContent = "";
  heatmapEl.src = "";
  try {
    const res = await fetch(`${BACKEND_URL}/api/random_image`);
    const data = await res.json();
    imageEl.src = `${BACKEND_URL}${data.image_path}`;
  } catch (err) {
    console.error("Error fetching random image:", err);
  }
}

// --- Load random question ---
async function loadRandomQuestion() {
  answerBox.textContent = "";
  try {
    const res = await fetch(`${BACKEND_URL}/api/random_question`);
    const data = await res.json();
    questionInput.value = data.question;
  } catch (err) {
    console.error("Error fetching random question:", err);
  }
}

// --- Predict answer + GradCAM ---
async function predictAnswer(e) {
  e.preventDefault();
  e.stopPropagation();

  const imagePath = imageEl.src.replace(BACKEND_URL, "");
  const question = questionInput.value.trim();
  if (!imagePath || !question) {
    alert("Please load both an image and a question first!");
    return;
  }

  answerBox.textContent = "Predicting...";
  heatmapEl.src = "";

  try {
    const res = await fetch(`${BACKEND_URL}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_path: imagePath, question }),
    });

    const data = await res.json();
    if (data.answer) {
      answerBox.textContent = `Answer: ${data.answer}`;
      heatmapEl.src = `${BACKEND_URL}${data.heatmap_url}`;
    } else {
      answerBox.textContent = "Prediction failed.";
    }
  } catch (err) {
    console.error("Prediction error:", err);
  }
}

// --- Add event listeners ---
randomImageBtn.addEventListener("click", loadRandomImage);
randomQuestionBtn.addEventListener("click", loadRandomQuestion);
predictBtn.addEventListener("click", predictAnswer);
