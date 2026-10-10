const escapeText = (value) => String(value ?? "").replace(/[&<>\"]/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;"
}[char]));

const COMPLETED_KEY = "genai-intelligence-completed-articles";

function completedArticles() {
  try { return new Set(JSON.parse(localStorage.getItem(COMPLETED_KEY) || "[]")); }
  catch { return new Set(); }
}

function markArticleCompleted(articleId) {
  const completed = completedArticles();
  completed.add(articleId);
  localStorage.setItem(COMPLETED_KEY, JSON.stringify([...completed]));
}

function renderStudy(container, data) {
  container.replaceChildren();
  const heading = document.createElement("h3");
  heading.textContent = "Your saved study note";
  container.append(heading);

  for (const [label, value] of [
    ["What happened", data.study.briefing],
    ["Why it matters", data.study.why_it_matters],
    ["Caveat", data.study.caveat]
  ]) {
    const paragraph = document.createElement("p");
    paragraph.innerHTML = `<strong>${escapeText(label)}:</strong> ${escapeText(value)}`;
    container.append(paragraph);
  }

  if (data.study.key_points?.length) {
    const subhead = document.createElement("h4");
    subhead.textContent = "Key points";
    container.append(subhead);
    const list = document.createElement("ul");
    for (const point of data.study.key_points) {
      const item = document.createElement("li");
      item.textContent = point;
      list.append(item);
    }
    container.append(list);
  }

  const quizHeading = document.createElement("h4");
  quizHeading.textContent = "Quick quiz";
  container.append(quizHeading);
  for (const [index, question] of data.study.quiz.entries()) {
    const group = document.createElement("fieldset");
    const legend = document.createElement("legend");
    legend.textContent = `${index + 1}. ${question.question}`;
    group.append(legend);
    const name = `quiz-${crypto.randomUUID()}`;
    question.choices.forEach((choice, choiceIndex) => {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = name;
      input.value = String(choiceIndex);
      label.append(input, document.createTextNode(` ${choice}`));
      group.append(label, document.createElement("br"));
    });
    const check = document.createElement("button");
    check.type = "button";
    check.textContent = "Check answer";
    const feedback = document.createElement("p");
    check.addEventListener("click", () => {
      const selected = group.querySelector(`input[name="${name}"]:checked`);
      if (!selected) {
        feedback.textContent = "Choose an answer first.";
        return;
      }
      const correct = Number(selected.value) === question.answer_index;
      feedback.textContent = `${correct ? "Correct. " : "Not quite. "}${question.explanation || ""}`;
      feedback.className = correct ? "answer" : "error";
    });
    group.append(check, feedback);
    container.append(group);
  }

  const flashcardHeading = document.createElement("h4");
  flashcardHeading.textContent = "Flashcards";
  container.append(flashcardHeading);
  for (const card of data.study.flashcards) {
    const details = document.createElement("details");
    const front = document.createElement("summary");
    front.textContent = card.front;
    const back = document.createElement("p");
    back.textContent = card.back;
    details.append(front, back);
    container.append(details);
  }
}

async function loadProgress(button) {
  if (completedArticles().has(button.dataset.id)) button.textContent = "Completed · review study kit";
  try {
    const response = await fetch(`/api/study/${encodeURIComponent(button.dataset.id)}`);
    if (!response.ok) return;
    const data = await response.json();
    if (data.study) renderStudy(document.getElementById(`study-${button.dataset.id}`), data);
  } catch { /* A temporarily unavailable database should not break reading. */ }
}

document.querySelectorAll("button.complete").forEach((button) => {
  loadProgress(button);
  button.addEventListener("click", async () => {
    button.disabled = true;
    button.textContent = "Creating your saved study kit…";
    const target = document.getElementById(`study-${button.dataset.id}`);
    try {
      const response = await fetch(`/api/study/${encodeURIComponent(button.dataset.id)}`, { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not create study kit");
      markArticleCompleted(button.dataset.id);
      button.textContent = "Completed · review study kit";
      renderStudy(target, data);
    } catch (error) {
      button.textContent = "Mark read & create study kit";
      target.innerHTML = `<p class="error">${escapeText(error.message)}</p>`;
    } finally {
      button.disabled = false;
    }
  });
});
