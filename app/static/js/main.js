/**
 * Landing page interactions + upload + multi-document chat logic.
 */
(function () {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const uploadBtn = document.getElementById("upload-btn");
  const uploadStatus = document.getElementById("upload-status");
  const uploadSpinner = document.getElementById("upload-spinner");

  const landingView = document.getElementById("landing-view");
  const chatView = document.getElementById("chat-view");
  const documentSelect = document.getElementById("document-select");
  const chatMessages = document.getElementById("chat-messages");
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const chatSendBtn = document.getElementById("chat-send-btn");
  const newUploadBtn = document.getElementById("new-upload-btn");

  if (!dropzone || !fileInput || !uploadBtn) return;

  let conversationHistory = [];
  let selectedDocument = null;
  const SOURCE_DELIMITER = "\n@@SOURCES@@\n";

  function openFilePicker() { fileInput.click(); }
  function setActive(active) { dropzone.classList.toggle("dropzone--active", active); }

  uploadBtn.addEventListener("click", openFilePicker);
  dropzone.addEventListener("click", (e) => { if (e.target !== fileInput) openFilePicker(); });
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openFilePicker(); }
  });
  ["dragenter", "dragover"].forEach((t) => dropzone.addEventListener(t, (e) => { e.preventDefault(); setActive(true); }));
  ["dragleave", "drop"].forEach((t) => dropzone.addEventListener(t, (e) => { e.preventDefault(); setActive(false); }));
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) { fileInput.files = e.dataTransfer.files; handleUpload(file); }
  });
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) handleUpload(file);
  });

  function setUploading(isUploading, message) {
    uploadSpinner.hidden = !isUploading;
    uploadStatus.textContent = message || "";
  }

  async function handleUpload(file) {
    setUploading(true, "Uploading document...");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/upload", { method: "POST", body: formData });
      let data = {};
      try { data = await response.json(); } catch (e) {}

      if (response.ok) {
        setUploading(false, "");
        await refreshDocumentList();
        selectDocument(data.source);
        enterChatView();
      } else {
        setUploading(false, "Error: " + (data.detail || "Upload failed. Please try a different file."));
      }
    } catch (err) {
      setUploading(false, "Could not reach the server. Check your connection and try again.");
    }
    fileInput.value = "";
  }

  async function refreshDocumentList() {
    try {
      const response = await fetch("/documents");
      const data = await response.json();
      const documents = data.documents || [];
      documentSelect.innerHTML = "";
      documents.forEach((doc) => {
        const option = document.createElement("option");
        option.value = doc.source;
        option.textContent = doc.filename;
        documentSelect.appendChild(option);
      });
      return documents;
    } catch (err) { return []; }
  }

  function selectDocument(source) {
    selectedDocument = source;
    if (source) documentSelect.value = source;
  }

  function enterChatView() {
    landingView.hidden = true;
    chatView.hidden = false;
    window.scrollTo({ top: 0, behavior: "smooth" });
    chatInput.disabled = false;
    chatSendBtn.disabled = false;
    chatMessages.innerHTML = "";
    conversationHistory = [];

    const label = documentSelect.selectedOptions[0] ? documentSelect.selectedOptions[0].textContent : "this document";
    addMessage("bot", "I've read through \"" + label + "\". Ask me anything about it.");
    chatInput.focus();
  }

  documentSelect.addEventListener("change", () => {
    selectedDocument = documentSelect.value;
    chatMessages.innerHTML = "";
    conversationHistory = [];
    const label = documentSelect.selectedOptions[0] ? documentSelect.selectedOptions[0].textContent : "this document";
    addMessage("bot", "Switched to \"" + label + "\". Ask me anything about it.");
    chatInput.focus();
  });

  function getUniquePages(sources) {
    if (!sources || !sources.length) return [];
    const seen = new Set();
    const pages = [];
    sources.forEach((s) => {
      if (!seen.has(s.page)) { seen.add(s.page); pages.push(s.page); }
    });
    return pages;
  }

  function appendSourceInfo(bubble, sourceType, sources) {
    if (!sources || !sources.length) return;

    if (sourceType === "web") {
      const wrap = document.createElement("div");
      wrap.className = "chat-bubble__sources chat-bubble__sources--web";
      const label = document.createElement("div");
      label.textContent = "Web results:";
      wrap.appendChild(label);
      sources.forEach((s) => {
        const link = document.createElement("a");
        link.href = s.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = s.title || s.url;
        wrap.appendChild(link);
      });
      bubble.appendChild(wrap);
    } else if (sourceType === "document") {
      const uniquePages = getUniquePages(sources);
      if (!uniquePages.length) return;
      const sourceEl = document.createElement("div");
      sourceEl.className = "chat-bubble__sources";
      sourceEl.textContent = "Sources: page " + uniquePages.join(", ");
      bubble.appendChild(sourceEl);
    }
  }

  function addMessage(sender, text, sourceType, sources) {
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble chat-bubble--" + sender;

    const textEl = document.createElement("div");
    textEl.textContent = text;
    bubble.appendChild(textEl);

    if (sender === "bot") appendSourceInfo(bubble, sourceType, sources);

    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return bubble;
  }

  function addTypingIndicator() {
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble chat-bubble--bot chat-bubble--typing";
    bubble.innerHTML = "<span></span><span></span><span></span>";
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return bubble;
  }

  async function streamAnswer(question) {
    const typingBubble = addTypingIndicator();
    let botBubble = null;
    let botTextEl = null;
    let buffer = "";

    try {
      const response = await fetch("/ask/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history: conversationHistory, document: selectedDocument }),
      });

      if (!response.ok || !response.body) {
        typingBubble.remove();
        addMessage("bot", "Something went wrong. Please try again.");
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        if (!botBubble) {
          typingBubble.remove();
          botBubble = document.createElement("div");
          botBubble.className = "chat-bubble chat-bubble--bot";
          botTextEl = document.createElement("div");
          botBubble.appendChild(botTextEl);
          chatMessages.appendChild(botBubble);
        }

        const visibleText = buffer.includes(SOURCE_DELIMITER) ? buffer.split(SOURCE_DELIMITER)[0] : buffer;
        botTextEl.textContent = visibleText;
        chatMessages.scrollTop = chatMessages.scrollHeight;
      }

      let finalAnswer = buffer;
      let sourceType = "none";
      let sources = [];

      if (buffer.includes(SOURCE_DELIMITER)) {
        const parts = buffer.split(SOURCE_DELIMITER);
        finalAnswer = parts[0];
        try {
          const parsed = JSON.parse(parts[1]);
          sourceType = parsed.type || "none";
          sources = parsed.sources || [];
        } catch (e) {}
      }

      if (botBubble) appendSourceInfo(botBubble, sourceType, sources);

      conversationHistory.push({ question, answer: finalAnswer });
      if (conversationHistory.length > 6) conversationHistory = conversationHistory.slice(-6);
    } catch (err) {
      typingBubble.remove();
      addMessage("bot", "Could not reach the server. Check your connection and try again.");
    }
  }

  if (chatForm) {
    chatForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const question = chatInput.value.trim();
      if (!question) return;

      addMessage("user", question);
      chatInput.value = "";
      chatInput.disabled = true;
      chatSendBtn.disabled = true;

      await streamAnswer(question);

      chatInput.disabled = false;
      chatSendBtn.disabled = false;
      chatInput.focus();
    });
  }

  if (newUploadBtn) {
    newUploadBtn.addEventListener("click", () => {
      chatView.hidden = true;
      landingView.hidden = false;
      conversationHistory = [];
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }
})();
