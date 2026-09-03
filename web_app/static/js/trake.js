/**
 * TRAKE 4-Track Quick Studio Engine (Pre-populated & Interactive)
 * Tự động nạp chuỗi frames từ kết quả trước/ứng viên và cho phép hiệu chỉnh từng sự kiện con.
 */

class TrakeStudio {
  constructor() {
    this.assignedFrames = [null, null, null, null];
    this.subeventTitles = ["Sự kiện 1", "Sự kiện 2", "Sự kiện 3", "Sự kiện 4"];
    this.currentVideoId = null;

    this.initEvents();
  }

  initEvents() {
    document.getElementById("btn-save-trake-now")?.addEventListener("click", () => this.saveTrakeToSubmission());

    // Phím tắt 1, 2, 3, 4 để gán nhanh frame từ player
    window.addEventListener("keydown", (e) => {
      if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
      if (['1', '2', '3', '4'].includes(e.key)) {
        const idx = parseInt(e.key) - 1;
        this.assignCurrentFromPlayer(idx);
      }
    });
  }

  parseSubeventTitles(queryText) {
    if (!queryText) return ["Sự kiện 1", "Sự kiện 2", "Sự kiện 3", "Sự kiện 4"];
    const lines = queryText.split('\n').map(l => l.trim()).filter(Boolean);
    let titles = [];

    const prefixRegex = /^(?:sự kiện|event|bước|cảnh|scene|giai đoạn|e)\s*\d+[\s:.-]*/i;
    for (const l of lines) {
      if (prefixRegex.test(l)) {
        const cleaned = l.replace(prefixRegex, '').trim();
        if (cleaned) titles.push(cleaned.slice(0, 35));
      }
    }

    if (titles.length === 0) {
      const inlineRegex = /(?:[eE]\d+|sự kiện\s*\d+|cảnh\s*\d+|event\s*\d+)[:\s.-]+([^;\n\.]+(?:[\.\?!](?![eE]\d+|sự kiện|cảnh|event))*)/gi;
      let match;
      while ((match = inlineRegex.exec(queryText)) !== null) {
        if (match[1] && match[1].trim().length > 3) {
          titles.push(match[1].trim().slice(0, 35));
        }
      }
    }

    while (titles.length < 4) {
      titles.push(`Sự kiện ${titles.length + 1}`);
    }
    return titles.slice(0, 4);
  }

  setupQuery(queryText) {
    this.subeventTitles = this.parseSubeventTitles(queryText);
    this.assignedFrames = [null, null, null, null];

    // Cập nhật nhãn nút bấm
    for (let i = 0; i < 4; i++) {
      const btn = document.querySelector(`button[onclick="window.trakeStudio.assignCurrentFromPlayer(${i})"] strong`);
      if (btn) {
        btn.textContent = `📌 E${i + 1}: ${this.subeventTitles[i]}`;
      }
      const valEl = document.getElementById(`trake-val-${i}`);
      if (valEl) valEl.textContent = "---";
    }

    this.validateMonotonic();
  }

  setInitialFrames(framesList, videoId = null) {
    if (!framesList || framesList.length === 0) return;
    if (videoId) this.currentVideoId = videoId;

    this.assignedFrames = [null, null, null, null];
    framesList.forEach((f, idx) => {
      if (idx < 4 && f !== null && !isNaN(f)) {
        this.assignedFrames[idx] = parseInt(f);
        const valEl = document.getElementById(`trake-val-${idx}`);
        if (valEl) {
          valEl.textContent = `#${f} (Click để xem)`;
          valEl.style.cursor = "pointer";
          valEl.title = `Click để tua video đến Frame #${f}`;
          valEl.onclick = (e) => {
            e.stopPropagation();
            this.seekToSlot(idx);
          };
        }
      }
    });

    this.validateMonotonic();
  }

  seekToSlot(slotIdx) {
    const f = this.assignedFrames[slotIdx];
    if (f !== null && window.videoInspector) {
      const vid = this.currentVideoId || window.videoInspector.currentVideoId;
      if (vid) {
        window.videoInspector.loadVideo(vid, f);
        window.app?.showToast(`Đang soi Sự kiện E${slotIdx + 1} tại Frame #${f}`);
      }
    }
  }

  assignCurrentFromPlayer(slotIdx) {
    if (!window.videoInspector || window.videoInspector.currentFrameIdx === undefined) return;
    const f = window.videoInspector.currentFrameIdx;
    this.assignedFrames[slotIdx] = f;
    this.currentVideoId = window.videoInspector.currentVideoId;

    const valEl = document.getElementById(`trake-val-${slotIdx}`);
    if (valEl) {
      valEl.textContent = `#${f} (Click để xem)`;
      valEl.style.cursor = "pointer";
      valEl.onclick = (e) => {
        e.stopPropagation();
        this.seekToSlot(slotIdx);
      };
    }

    this.validateMonotonic();
    window.app?.showToast(`Đã gán Frame #${f} vào E${slotIdx + 1} (${this.subeventTitles[slotIdx] || ''})`);
  }

  validateMonotonic() {
    const statusEl = document.getElementById("trake-val-status");
    if (!statusEl) return;

    const validFrames = this.assignedFrames.filter(f => f !== null);
    if (validFrames.length < 2) {
      statusEl.textContent = "Chờ gán...";
      statusEl.style.color = "#94a3b8";
      return;
    }

    let isStrictlyIncreasing = true;
    for (let i = 0; i < validFrames.length - 1; i++) {
      if (validFrames[i] >= validFrames[i + 1]) {
        isStrictlyIncreasing = false;
        break;
      }
    }

    if (isStrictlyIncreasing) {
      statusEl.textContent = `✅ Hợp lệ (${validFrames.length} events)`;
      statusEl.style.color = "#34d399";
    } else {
      statusEl.textContent = "⚠️ Cảnh báo: Lệch thứ tự thời gian!";
      statusEl.style.color = "#ef4444";
    }
  }

  async saveTrakeToSubmission() {
    const valid = this.assignedFrames.filter(f => f !== null);
    if (valid.length < 2) {
      alert("Vui lòng gán ít nhất 2 sự kiện trước khi lưu!");
      return;
    }

    const videoId = this.currentVideoId || window.videoInspector?.currentVideoId;
    if (!videoId) {
      alert("Chưa chọn video!");
      return;
    }

    const currentQ = window.app?.currentQueryData;
    if (!currentQ) return;

    try {
      const res = await fetch("/api/contest/override_rank1", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          output_package: window.app.selectedOutputPkg,
          query_id: currentQ.id,
          task_type: "TRAKE",
          video_id: videoId,
          frame_idx: valid[0],
          trake_frames: valid.join(",")
        })
      });

      const data = await res.json();
      if (data.status === "success") {
        window.app?.showToast(`👑 Đã lưu chuỗi TRAKE [${valid.join(", ")}] vào câu ${currentQ.id}!`);
        window.app?.loadCurrentSubmissionData(currentQ.id);
      }
    } catch (e) {
      alert("Lỗi khi lưu TRAKE: " + e);
    }
  }
}

window.trakeStudio = new TrakeStudio();
