/**
 * Frame-Accurate Video Player & Keyframe Scrubber Engine
 * Hỗ trợ tua chuẩn từng Frame Index và hiển thị dải phim ngữ cảnh (Filmstrip).
 */

class VideoInspector {
  constructor() {
    this.videoEl = document.getElementById("video-element");
    this.frameDisplay = document.getElementById("current-frame-val");
    this.timeDisplay = document.getElementById("current-time-val");
    this.filmstripScroll = document.getElementById("filmstrip-scroll");
    this.currentVideoId = null;
    this.currentFrameIdx = 0;
    this.fps = 25.0; // Chuẩn FPS mặc định của video AIC (25 fps)

    this.initEvents();
  }

  initEvents() {
    if (!this.videoEl) return;

    this.videoEl.addEventListener("timeupdate", () => {
      const time = this.videoEl.currentTime;
      this.currentFrameIdx = Math.round(time * this.fps);
      this.updateDisplays();
    });

    // Các nút tua frame
    document.getElementById("btn-prev-frame")?.addEventListener("click", () => this.stepFrame(-1));
    document.getElementById("btn-next-frame")?.addEventListener("click", () => this.stepFrame(1));
    document.getElementById("btn-prev-5f")?.addEventListener("click", () => this.stepFrame(-5));
    document.getElementById("btn-next-5f")?.addEventListener("click", () => this.stepFrame(5));
    document.getElementById("btn-toggle-play")?.addEventListener("click", () => this.togglePlay());
  }

  loadVideo(videoId, targetFrameIdx = 0) {
    this.currentVideoId = videoId;
    this.currentFrameIdx = targetFrameIdx;
    
    // Cập nhật nguồn video stream từ backend
    const streamUrl = `/api/media/video_stream/${videoId}`;
    this.videoEl.src = streamUrl;
    
    const targetTime = targetFrameIdx / this.fps;
    this.videoEl.currentTime = targetTime;
    this.updateDisplays();

    // Tải dải phim ngữ cảnh xung quanh frame mục tiêu
    this.loadSurroundingFilmstrip(videoId, targetFrameIdx);
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
  }

  async loadSurroundingFilmstrip(videoId, frameIdx) {
    if (!this.filmstripScroll) return;
    try {
      const res = await fetch(`/api/media/surrounding/${videoId}/${frameIdx}?count=8`);
      const data = await res.json();
      const frames = data.surrounding_frames || [];

      this.filmstripScroll.innerHTML = "";
      frames.forEach(f => {
        const thumb = document.createElement("div");
        thumb.className = `filmstrip-thumb ${f === frameIdx ? 'active' : ''}`;
        thumb.innerHTML = `
          <img src="/api/media/keyframe/${videoId}/${f}" loading="lazy" alt="frame ${f}" />
          <span>#${f}</span>
        `;
        thumb.addEventListener("click", () => {
          this.currentFrameIdx = f;
          this.videoEl.currentTime = f / this.fps;
          this.updateDisplays();
          // Cập nhật active class
          this.filmstripScroll.querySelectorAll('.filmstrip-thumb').forEach(el => el.classList.remove('active'));
          thumb.classList.add('active');
        });
        this.filmstripScroll.appendChild(thumb);
      });
    } catch (e) {
      console.warn("Không thể nạp filmstrip:", e);
    }
  }
}

window.videoInspector = new VideoInspector();
