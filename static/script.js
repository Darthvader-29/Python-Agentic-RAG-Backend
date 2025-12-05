const chatWindow = document.getElementById("chat-window");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const uploadBtn = document.getElementById("upload-btn");
const fileInput = document.getElementById("file-input");
const webToggle = document.getElementById("web-toggle");

let SESSION_ID = crypto.randomUUID();
let uploadedFileKeys = [];

// --- Helpers ---
function addMessage(text, role = "bot") {
  const div = document.createElement("div");
  div.className = `message ${role === "user" ? "user" : "bot"}`;
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }
  return res.json();
}

// --- Chat Logic ---
async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  addMessage(text, "user");
  userInput.value = "";

  try {
    const data = await postJSON("/api/chat", {
      message: text,
      session_id: SESSION_ID,
      web_search_allowed: webToggle.checked,
    });

    if (data.session_id) {
      SESSION_ID = data.session_id;
    }

    addMessage(data.answer, "bot");
  } catch (err) {
    console.error(err);
    addMessage("Error while contacting backend.", "bot");
  }
}

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});

// --- Upload Flow (S3 via backend) ---
uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  addMessage(`Uploading ${file.name}...`, "bot");

  try {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      throw new Error(await res.text());
    }

    const data = await res.json();

    if (data.session_id) {
      SESSION_ID = data.session_id;
    }
    if (data.s3_key) {
      uploadedFileKeys.push(data.s3_key);
    }

    addMessage(`File ${file.name} uploaded and ingestion started.`, "bot");
  } catch (err) {
    console.error(err);
    addMessage("Upload or ingestion failed.", "bot");
  } finally {
    fileInput.value = "";
  }
});

// --- Cleanup on tab close ---
window.addEventListener("beforeunload", () => {
  if (!SESSION_ID) return;

  const payload = JSON.stringify({
    session_id: SESSION_ID,
    file_keys: uploadedFileKeys,
  });

  const blob = new Blob([payload], { type: "application/json" });
  navigator.sendBeacon("/api/cleanup", blob);
});
