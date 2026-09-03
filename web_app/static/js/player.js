/**
 * Frame-Accurate Video Player & Keyframe Scrubber Engine
 * Tích hợp tính năng 1-Click Chốt Frame làm Rank 1 ngay khi đang xem video.
 */

class VideoInspector {
  constructor() {
    this.videoEl = document.getElementById("video-element");
    this.frameDisplay = document.getElementById("current-frame-val");
    this.timeDisplay = document.getElementById("current-time-val");
    this.filmstripScroll = document.getElementById("filmstrip-scroll");
    this.lockBtn = document.getElementById("btn-lock-current-frame");
    
    this.currentVideoId = null;
    this.currentFrameIdx = 0;
    this.fps = 25.0;

    this.initEvents();
  }

  initEvents() {
    if (!this.videoEl) return;

    this.videoEl.addEventListener("timeupdate", () => {
      const time = this.videoEl.currentTime;
      this.currentFrameIdx = Math.round(time * this.fps);
      this.updateDisplays();
    });

    // Các nút tua thời gian & frame
    document.getElementById("btn-step-back-5s")?.addEventListener("click", () => this.stepSeconds(-5));
    document.getElementById("btn-step-fwd-5s")?.addEventListener("click", () => this.stepSeconds(5));
    document.getElementById("btn-step-back-1f")?.addEventListener("click", () => this.stepFrame(-1));
    document.getElementById("btn-step-fwd-1f")?.addEventListener("click", () => this.stepFrame(1));
    document.getElementById("btn-toggle-play")?.addEventListener("click", () => this.togglePlay());

    // NÚT CHỐT FRAME HIỆN TẠI LÀM RANK #1
    this.lockBtn?.addEventListener("click", () => this.lockCurrentFrameAsRank1());

    // Nút chốt cho QA
    document.getElementById("btn-lock-qa")?.addEventListener("click", () => {
      const qaAns = document.getElementById("qa-input-field")?.value.trim() || "";
      this.lockCurrentFrameAsRank1(qaAns);
    });

    // Bắt phím tắt Enter để chốt nhanh
    window.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
        e.preventDefault();
        this.lockCurrentFrameAsRank1();
      }
    });
  }

  loadVideo(videoId, targetFrameIdx = 0) {
    if (!videoId) return;
    this.currentVideoId = videoId;
    this.currentFrameIdx = targetFrameIdx;
    
    const streamUrl = `/api/media/video_stream/${videoId}`;
    this.videoEl.src = streamUrl;
    
    const targetTime = Math.max(0, targetFrameIdx / this.fps);
    this.videoEl.currentTime = targetTime;
    this.updateDisplays();

    // Nạp filmstrip
    this.loadSurroundingFilmstrip(videoId, targetFrameIdx);
  }

  stepSeconds(sec) {
    if (!this.videoEl) return;
    this.videoEl.currentTime = Math.max(0, this.videoEl.currentTime + sec);
  }

  stepFrame(delta) {
    if (!this.videoEl) return;
    this.videoEl.pause();
    const newTime = Math.max(0, this.videoEl.currentTime + (delta / this.fps));
    this.videoEl.currentTime = newTime;
    this.currentFrameIdx = Math.round(newTime * this.fps);
    this.updateDisplays();
  }

  togglePlay() {
    if (!this.videoEl) return;
    if (this.videoEl.paused) {
      this.videoEl.play();
    } else {
      this.videoEl.pause();
    }
  }

  updateDisplays() {
    if (this.frameDisplay) {
      this.frameDisplay.textContent = `#${this.currentFrameIdx}`;
    }
    if (this.timeDisplay && this.videoEl) {
      const t = this.videoEl.currentTime;
      const m = Math.floor(t / 60);
      const s = Math.floor(t % 60);
      const ms = Math.floor((t % 1) * 1000);
      this.timeDisplay.textContent = `[${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}]`;
    }
    if (this.lockBtn) {
      this.lockBtn.innerHTML = `<span>👑 CHỐT FRAME #${this.currentFrameIdx} NÀY LÀM RANK #1</span>`;
    }
  }

  async lockCurrentFrameAsRank1(qaAnswer = "") {
    if (!this.currentVideoId) {
      alert("Chưa có video nào được chọn để chốt frame!");
      return;
    }

    const currentQ = window.app?.currentQueryData;
    if (!currentQ) {
      alert("Vui lòng chọn câu hỏi trong danh sách trước khi chốt!");
      return;
    }

    try {
      const res = await fetch("/api/contest/override_rank1", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          output_package: window.app.selectedOutputPkg,
          query_id: currentQ.id,
          task_type: currentQ.task_type,
          video_id: this.currentVideoId,
          frame_idx: this.currentFrameIdx,
          qa_answer: qaAnswer
        })
      });

      const data = await res.json();
      if (data.status === "success") {
        // Phản hồi trực quan trên nút
        if (this.lockBtn) {
          this.lockBtn.classList.add("success");
          this.lockBtn.innerHTML = `<span>✅ ĐÃ CHỐT FRAME #${this.currentFrameIdx} THÀNH CÔNG!</span>`;
          setTimeout(() => {
            this.lockBtn.classList.remove("success");
            this.updateDisplays();
          }, 1500);
        }

        // Hiện toast
        window.app?.showToast(`Đã chốt Video ${this.currentVideoId} (#${this.currentFrameIdx}) làm Rank #1!`);

        // Tải lại bảng kết quả ngay lập tức
        window.app?.loadCurrentSubmissionData(currentQ.id);
      }
    } catch (e) {
      alert("Lỗi khi chốt frame: " + e);
    }
  }

  async loadSurroundingFilmstrip(videoId, frameIdx) {
    if (!this.filmstripScroll) return;
    try {
      const res = await fetch(`/api/media/surrounding/${videoId}/${frameIdx}?count=10`);
      const data = await res.json();
      const frames = data.surrounding_frames || [];

      this.filmstripScroll.innerHTML = "";
      frames.forEach(f => {
        const thumb = document.createElement("div");
        thumb.className = `strip-item ${f === frameIdx ? 'active' : ''}`;
        thumb.innerHTML = `
          <img src="/api/media/keyframe/${videoId}/${f}" loading="lazy" alt="frame ${f}" />
          <span>#${f}</span>
        `;
        thumb.addEventListener("click", () => {
          this.currentFrameIdx = f;
          this.videoEl.currentTime = f / this.fps;
          this.updateDisplays();
          this.filmstripScroll.querySelectorAll('.strip-item').forEach(el => el.classList.remove('active'));
          thumb.classList.add('active');
        });
        this.filmstripScroll.appendChild(thumb);
      });
    } catch (e) {
      console.warn("Lỗi load filmstrip:", e);
    }
  }
}

window.videoInspector = new VideoInspector();
