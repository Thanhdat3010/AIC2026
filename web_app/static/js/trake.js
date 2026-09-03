/**
 * TRAKE 4-Track Studio & Monotonic Alignment Validator
 * Quản lý bóc tách sự kiện con E1..E4 và gán khung hình trực quan từ Video Player.
 */

class TrakeStudio {
  constructor() {
    this.container = document.getElementById("trake-studio-container");
    this.slotsContainer = document.getElementById("trake-slots");
    this.eventList = [];
    this.assignedFrames = {}; // event_index -> frame_idx

    this.initEvents();
  }

  initEvents() {
    document.getElementById("btn-save-trake")?.addEventListener("click", () => this.saveTrakeToSubmission());
    document.getElementById("btn-sort-trake")?.addEventListener("click", () => this.autoSortMonotonic());

    // Bắt phím tắt 1, 2, 3, 4 để snap nhanh
    window.addEventListener("keydown", (e) => {
      // Chỉ bắt khi không focus vào input text
      if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
      if (['1', '2', '3', '4'].includes(e.key)) {
        const idx = parseInt(e.key) - 1;
        if (idx < this.eventList.length && window.videoInspector) {
          this.assignFrame(idx, window.videoInspector.currentFrameIdx);
        }
      }
    });
  }

  parseSubevents(queryText) {
    if (!queryText) return [];
    const lines = queryText.split('\n').map(l => l.trim()).filter(Boolean);
    let events = [];

    // 1. Tìm theo tiền tố E1, E2, Sự kiện 1, Cảnh 1
    const prefixRegex = /^(?:sự kiện|event|bước|cảnh|scene|giai đoạn|e)\s*\d+[\s:.-]*/i;
    for (const l of lines) {
      if (prefixRegex.test(l)) {
        const cleaned = l.replace(prefixRegex, '').trim();
        if (cleaned) events.push(cleaned);
      }
    }

    // 2. Tìm inline regex
    if (events.length === 0) {
      const inlineRegex = /(?:[eE]\d+|sự kiện\s*\d+|cảnh\s*\d+|event\s*\d+)[:\s.-]+([^;\n\.]+(?:[\.\?!](?![eE]\d+|sự kiện|cảnh|event))*)/gi;
      let match;
      while ((match = inlineRegex.exec(queryText)) !== null) {
        if (match[1] && match[1].trim().length > 5) {
          events.push(match[1].trim());
        }
      }
    }

    // 3. Fallback phân tách bằng dấu chấm phẩy hoặc dấu phẩy
    if (events.length === 0) {
      const parts = queryText.split(/;\s*|\n|(?<=[\.\?!])\s+/).map(p => p.trim()).filter(p => p.length > 8);
      if (parts.length >= 2) {
        events = parts.slice(0, 4);
      } else {
        events = ["Sự kiện 1 (Bắt đầu)", "Sự kiện 2 (Diễn biến)", "Sự kiện 3 (Kết thúc)"];
      }
    }

    return events.slice(0, 5); // Tối đa 4-5 sự kiện
  }

  setupQuery(queryText, taskType) {
    if (taskType !== "TRAKE") {
      if (this.container) this.container.style.display = "none";
      return;
    }

    if (this.container) this.container.style.display = "flex";
    this.eventList = this.parseSubevents(queryText);
    this.assignedFrames = {};
    this.renderSlots();
  }

  renderSlots() {
    if (!this.slotsContainer) return;
    this.slotsContainer.innerHTML = "";

    this.eventList.forEach((evText, idx) => {
      const currentVal = this.assignedFrames[idx] !== undefined ? this.assignedFrames[idx] : "---";
      const slotEl = document.createElement("div");
      slotEl.className = "trake-slot";
      slotEl.id = `trake-slot-${idx}`;
      slotEl.innerHTML = `
        <span class="slot-label">E${idx + 1} (${idx + 1}):</span>
        <span class="slot-desc" title="${evText}">${evText}</span>
        <span class="badge-time" id="slot-val-${idx}">${currentVal}</span>
        <button class="btn-slot-snap" onclick="window.trakeStudio.assignCurrentFromPlayer(${idx})">Gán [Phím ${idx + 1}]</button>
      `;
      this.slotsContainer.appendChild(slotEl);
    });

    this.validateMonotonic();
  }

  assignCurrentFromPlayer(idx) {
    if (window.videoInspector) {
      this.assignFrame(idx, window.videoInspector.currentFrameIdx);
    }
  }

  assignFrame(idx, frameIdx) {
    this.assignedFrames[idx] = frameIdx;
    const valEl = document.getElementById(`slot-val-${idx}`);
    if (valEl) valEl.textContent = `#${frameIdx}`;
    this.validateMonotonic();
  }

  validateMonotonic() {
    let isValid = true;
    const frameValues = [];

    for (let i = 0; i < this.eventList.length; i++) {
      const f = this.assignedFrames[i];
      if (f !== undefined) {
        frameValues.push({ idx: i, frame: f });
      }
    }

    // Kiểm tra thứ tự tăng dần
    for (let j = 0; j < frameValues.length - 1; j++) {
      if (frameValues[j].frame >= frameValues[j + 1].frame) {
        isValid = false;
        break;
      }
    }

    const statusEl = document.getElementById("trake-validation-status");
    if (statusEl) {
      if (frameValues.length < 2) {
        statusEl.textContent = "Chờ gán frames...";
        statusEl.style.color = "#94a3b8";
      } else if (isValid) {
        statusEl.textContent = "✅ Chuỗi hợp lệ (Tăng dần theo thời gian)";
        statusEl.style.color = "#34d399";
      } else {
        statusEl.textContent = "⚠️ Cảnh báo: Lệch thứ tự thời gian!";
        statusEl.style.color = "#ef4444";
      }
    }
  }

  autoSortMonotonic() {
    const vals = Object.values(this.assignedFrames).sort((a, b) => a - b);
    vals.forEach((f, idx) => {
      if (idx < this.eventList.length) {
        this.assignedFrames[idx] = f;
      }
    });
    this.renderSlots();
  }

  saveTrakeToSubmission() {
    const frames = [];
    for (let i = 0; i < this.eventList.length; i++) {
      if (this.assignedFrames[i] !== undefined) {
        frames.push(this.assignedFrames[i]);
      } else {
        alert(`Bạn chưa gán khung hình cho Sự kiện E${i + 1}!`);
        return;
      }
    }

    const videoId = window.videoInspector?.currentVideoId || (window.app?.currentQueryData?.video_id || "");
    if (!videoId) {
      alert("Vui lòng chọn Video trước khi lưu chuỗi TRAKE!");
      return;
    }

    if (window.app) {
      window.app.saveManualSubmission(videoId, frames[0], {
        trake_frames: frames.join(",")
      });
    }
  }
}

window.trakeStudio = new TrakeStudio();
