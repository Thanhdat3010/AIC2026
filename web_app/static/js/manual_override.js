/**
 * Manual Override Modal Manager
 * Quản lý hộp thoại ghim thủ công Video ID + Frame Index lên Rank 1 hoặc thêm vào cuối danh sách.
 */

class ManualOverrideModal {
  constructor() {
    this.modal = document.getElementById("modal-override");
    this.inputVid = document.getElementById("override-vid");
    this.inputFrame = document.getElementById("override-frame");
    this.inputQa = document.getElementById("override-qa");
    this.inputTrake = document.getElementById("override-trake");
    this.qaGroup = document.getElementById("override-qa-group");
    this.trakeGroup = document.getElementById("override-trake-group");

    this.initEvents();
  }

  initEvents() {
    document.getElementById("btn-open-override")?.addEventListener("click", () => this.open());
    document.getElementById("btn-close-override")?.addEventListener("click", () => this.close());
    document.getElementById("btn-confirm-override")?.addEventListener("click", () => this.submitOverride());
  }

  open(prefillVid = "", prefillFrame = 0) {
    if (!this.modal) return;
    const taskType = window.app?.currentQueryData?.task_type || "KIS";

    if (this.inputVid) this.inputVid.value = prefillVid || (window.videoInspector?.currentVideoId || "");
    if (this.inputFrame) this.inputFrame.value = prefillFrame || (window.videoInspector?.currentFrameIdx || 0);

    // Ẩn hiện các trường theo task
    if (this.qaGroup) this.qaGroup.style.display = (taskType === "QA") ? "flex" : "none";
    if (this.trakeGroup) this.trakeGroup.style.display = (taskType === "TRAKE") ? "flex" : "none";

    this.modal.classList.add("open");
  }

  close() {
    if (this.modal) this.modal.classList.remove("open");
  }

  async submitOverride() {
    const vid = this.inputVid?.value.trim();
    const frame = parseInt(this.inputFrame?.value) || 0;
    const qaAns = this.inputQa?.value.trim() || "";
    const trakeFrames = this.inputTrake?.value.trim() || "";

    if (!vid) {
      alert("Vui lòng nhập Video ID!");
      return;
    }

    const currentQuery = window.app?.currentQueryData;
    if (!currentQuery) {
      alert("Vui lòng chọn câu hỏi trong danh sách trước khi ghim!");
      return;
    }

    try {
      const res = await fetch("/api/contest/override_rank1", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          output_package: window.app.selectedOutputPkg,
          query_id: currentQuery.id,
          task_type: currentQuery.task_type,
          video_id: vid,
          frame_idx: frame,
          qa_answer: qaAns,
          trake_frames: trakeFrames
        })
      });
      const data = await res.json();
      if (data.status === "success") {
        this.close();
        alert(`👑 Đã ghim thành công ${vid} (frame ${frame}) lên Rank #1!`);
        // Nạp lại danh sách nộp bài
        window.app.loadCurrentSubmissionData(currentQuery.id);
      } else {
        alert("Lỗi khi ghim: " + JSON.stringify(data));
      }
    } catch (e) {
      alert("Lỗi kết nối máy chủ: " + e);
    }
  }
}

window.manualOverrideModal = new ManualOverrideModal();
