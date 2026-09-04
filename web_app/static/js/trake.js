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
    if (!queryText) return ["Sự kiện 1", "Sự kiện 2", "Sự kiện 3"];
    const lines = queryText.split('\n').map(l => l.trim()).filter(Boolean);
    let titles = [];

    const prefixRegex = /^(?:sự kiện|event|bước|cảnh|scene|giai đoạn|e)\s*\d+[\s:.-]*/i;
    for (const l of lines) {
      if (prefixRegex.test(l)) {
        const cleaned = l.replace(prefixRegex, '').trim();
        if (cleaned) titles.push(cleaned.slice(0, 45));
      }
    }

    if (titles.length === 0) {
      const inlineRegex = /(?:[eE]\d+|sự kiện\s*\d+|cảnh\s*\d+|event\s*\d+)[:\s.-]+([^;\n\.]+(?:[\.\?!](?![eE]\d+|sự kiện|cảnh|event))*)/gi;
      let match;
      while ((match = inlineRegex.exec(queryText)) !== null) {
        if (match[1] && match[1].trim().length > 3) {
          titles.push(match[1].trim().slice(0, 45));
        }
      }
    }

    // Không ép lên 4 nếu đề chỉ có 2 hoặc 3 events!
    if (titles.length === 0) {
      titles = ["Sự kiện 1", "Sự kiện 2", "Sự kiện 3"];
    }
    return titles;
  }

  setupQuery(queryText) {
    this.subeventTitles = this.parseSubeventTitles(queryText);
    const n = this.subeventTitles.length;
    this.assignedFrames = new Array(n).fill(null);

    for (let i = 0; i < 4; i++) {
      const itemEl = document.getElementById(`trake-item-${i}`);
      if (itemEl) {
        if (i < n) {
          itemEl.style.display = "flex";
          const descEl = document.getElementById(`trake-desc-${i}`);
          if (descEl) descEl.textContent = this.subeventTitles[i];
          const valEl = document.getElementById(`trake-val-${i}`);
          if (valEl) valEl.textContent = "#---";
        } else {
          itemEl.style.display = "none";
        }
      }
    }

    this.validateMonotonic();
  }

  setInitialFrames(framesList, videoId = null) {
    if (!framesList || framesList.length === 0) return;
    if (videoId) this.currentVideoId = videoId;

    const n = this.subeventTitles.length || 3;
    this.assignedFrames = new Array(n).fill(null);
    const effectiveFrames = framesList.slice(0, n);

    effectiveFrames.forEach((f, idx) => {
      if (f !== null && !isNaN(f)) {
        this.assignedFrames[idx] = parseInt(f);
        const valEl = document.getElementById(`trake-val-${idx}`);
        if (valEl) {
          valEl.textContent = `#${f}`;
        }
      }
    });

    for (let i = 0; i < 4; i++) {
      const itemEl = document.getElementById(`trake-item-${i}`);
      if (itemEl) {
        itemEl.style.display = (i < n) ? "flex" : "none";
      }
    }

    this.validateMonotonic();
  }

  seekToSlot(slotIdx) {
    const f = this.assignedFrames[slotIdx];
    if (f !== null && window.videoInspector) {
      const vid = this.currentVideoId || window.videoInspector.currentVideoId;
      if (vid) {
        window.videoInspector.loadVideo(vid, f);
        window.app?.showToast(`👁️ Đang xem video E${slotIdx + 1} tại Frame #${f}`);
      }
    } else {
      window.app?.showToast(`Sự kiện E${slotIdx + 1} chưa có frame để xem!`);
    }
  }

  assignCurrentFromPlayer(slotIdx) {
    if (!window.videoInspector || window.videoInspector.currentFrameIdx === undefined) return;
    const f = window.videoInspector.currentFrameIdx;
    this.assignedFrames[slotIdx] = f;
    this.currentVideoId = window.videoInspector.currentVideoId;

    const valEl = document.getElementById(`trake-val-${slotIdx}`);
    if (valEl) {
      valEl.textContent = `#${f}`;
    }

    this.validateMonotonic();
    window.app?.showToast(`📌 Đã gán Frame #${f} vào E${slotIdx + 1}!`);
  }

  validateMonotonic() {
    const statusEl = document.getElementById("trake-val-status");
    if (!statusEl) return;

    const n = this.subeventTitles.length || 3;
    const validFrames = this.assignedFrames.filter(f => f !== null);
    if (validFrames.length < 2) {
      statusEl.textContent = `Chờ gán (0/${n})...`;
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
      statusEl.textContent = `✅ Hợp lệ (${validFrames.length}/${n} events)`;
      statusEl.style.color = "#34d399";
    } else {
      statusEl.textContent = "⚠️ Lệch thứ tự thời gian!";
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
