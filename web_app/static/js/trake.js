/**
 * TRAKE 4-Track Quick Studio Engine
 */

class TrakeStudio {
  constructor() {
    this.assignedFrames = [null, null, null, null];
    this.initEvents();
  }

  initEvents() {
    document.getElementById("btn-save-trake-now")?.addEventListener("click", () => this.saveTrakeToSubmission());

    // Phím tắt 1, 2, 3, 4
    window.addEventListener("keydown", (e) => {
      if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
      if (['1', '2', '3', '4'].includes(e.key)) {
        const idx = parseInt(e.key) - 1;
        this.assignCurrentFromPlayer(idx);
      }
    });
  }

  setupQuery(queryText) {
    this.assignedFrames = [null, null, null, null];
    for (let i = 0; i < 4; i++) {
      const el = document.getElementById(`trake-val-${i}`);
      if (el) el.textContent = "---";
    }
    this.validateMonotonic();
  }

  assignCurrentFromPlayer(slotIdx) {
    if (!window.videoInspector || window.videoInspector.currentFrameIdx === undefined) return;
    const f = window.videoInspector.currentFrameIdx;
    this.assignedFrames[slotIdx] = f;

    const valEl = document.getElementById(`trake-val-${slotIdx}`);
    if (valEl) valEl.textContent = `#${f}`;

    this.validateMonotonic();
    window.app?.showToast(`Đã gán Frame #${f} vào Sự kiện E${slotIdx + 1}`);
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
      statusEl.textContent = "✅ Tăng dần";
      statusEl.style.color = "#34d399";
    } else {
      statusEl.textContent = "⚠️ Lệch thứ tự!";
      statusEl.style.color = "#ef4444";
    }
  }

  async saveTrakeToSubmission() {
    const valid = this.assignedFrames.filter(f => f !== null);
    if (valid.length < 2) {
      alert("Vui lòng gán ít nhất 2 sự kiện trước khi lưu!");
      return;
    }

    const videoId = window.videoInspector?.currentVideoId;
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
        window.app?.showToast(`Đã lưu chuỗi TRAKE [${valid.join(", ")}] vào câu ${currentQ.id}!`);
        window.app?.loadCurrentSubmissionData(currentQ.id);
      }
    } catch (e) {
      alert("Lỗi khi lưu TRAKE: " + e);
    }
  }
}

window.trakeStudio = new TrakeStudio();
