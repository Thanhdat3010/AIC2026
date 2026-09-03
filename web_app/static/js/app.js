/**
 * AIC 2026 Championship Console - Core Application Coordinator
 * Quản lý vòng đời tìm kiếm, hiển thị danh sách 100+ card ứng viên và tương tác thi đấu.
 */

class AppController {
  constructor() {
    this.selectedOutputPkg = "sotuyen1";
    this.selectedQueryPkg = "SOTUYEN1-bo-de-thi";
    this.currentQueryData = null;
    this.currentResults = [];

    this.omnibarInput = document.getElementById("omnibar-input");
    this.queryListEl = document.getElementById("query-list");
    this.cardsGridEl = document.getElementById("cards-grid");
    this.activeQueryTextEl = document.getElementById("active-query-text");
    this.activeQueryTagEl = document.getElementById("active-query-tag");
    this.statsLatencyEl = document.getElementById("stats-latency");
    this.statsCountEl = document.getElementById("stats-count");
    this.outputSelectEl = document.getElementById("select-output-pkg");
    this.querySelectEl = document.getElementById("select-query-pkg");

    this.initEvents();
    this.loadPackages();
  }

  initEvents() {
    // Nút tìm kiếm
    document.getElementById("btn-search")?.addEventListener("click", () => this.executeSearch());

    // Nút tải zip
    document.getElementById("btn-download-zip")?.addEventListener("click", () => {
      window.location.href = `/api/contest/download_zip?output_package=${this.selectedOutputPkg}`;
    });

    // Nút Hoàn Tác (Undo)
    document.getElementById("btn-undo")?.addEventListener("click", () => this.executeUndo());

    // Thay đổi package
    this.outputSelectEl?.addEventListener("change", (e) => {
      this.selectedOutputPkg = e.target.value;
      this.loadQueries();
    });
    this.querySelectEl?.addEventListener("change", (e) => {
      this.selectedQueryPkg = e.target.value;
      this.loadQueries();
    });

    // Hệ thống phím tắt toàn cục (Competitive Hotkeys)
    window.addEventListener("keydown", (e) => {
      // Ctrl + K hoặc /: Nhảy nhanh vào ô tìm kiếm
      if ((e.ctrlKey && e.key.toLowerCase() === 'k') || (e.key === '/' && document.activeElement.tagName !== 'INPUT')) {
        e.preventDefault();
        this.omnibarInput?.focus();
        this.omnibarInput?.select();
      }

      // Ctrl + Enter: Thực thi tìm kiếm
      if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        this.executeSearch();
      }

      // Ctrl + Z: Hoàn tác
      if (e.ctrlKey && e.key.toLowerCase() === 'z' && document.activeElement.tagName !== 'INPUT') {
        e.preventDefault();
        this.executeUndo();
      }

      // Space: Play/Pause Video (khi không gõ chữ)
      if (e.code === 'Space' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
        e.preventDefault();
        window.videoInspector?.togglePlay();
      }

      // [ và ]: Lùi/Tiến 1 frame
      if (e.key === '[' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
        window.videoInspector?.stepFrame(-1);
      }
      if (e.key === ']' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
        window.videoInspector?.stepFrame(1);
      }

      // Tab: Mở modal ghim nhanh hoặc snap
      if (e.key === 'Tab' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
        e.preventDefault();
        if (window.manualOverrideModal && window.videoInspector) {
          window.manualOverrideModal.open(window.videoInspector.currentVideoId, window.videoInspector.currentFrameIdx);
        }
      }
    });
  }

  async loadPackages() {
    try {
      const res = await fetch("/api/contest/packages");
      const data = await res.json();

      if (this.outputSelectEl) {
        this.outputSelectEl.innerHTML = data.output_packages.map(p => `<option value="${p}">${p}</option>`).join("");
        this.outputSelectEl.value = data.default_output;
        this.selectedOutputPkg = data.default_output;
      }

      if (this.querySelectEl) {
        this.querySelectEl.innerHTML = data.query_packages.map(p => `<option value="${p}">${p}</option>`).join("");
        this.querySelectEl.value = data.default_query;
        this.selectedQueryPkg = data.default_query;
      }

      this.loadQueries();
    } catch (e) {
      console.error("Lỗi nạp danh sách packages:", e);
    }
  }

  async loadQueries() {
    if (!this.selectedQueryPkg || !this.selectedOutputPkg) return;
    try {
      const res = await fetch(`/api/contest/queries?query_package=${this.selectedQueryPkg}&output_package=${this.selectedOutputPkg}`);
      const data = await res.json();
      const queries = data.queries || [];

      // Cập nhật tiến độ
      const progressEl = document.getElementById("contest-progress");
      if (progressEl) progressEl.textContent = `${data.completed} / ${data.total} câu`;

      if (!this.queryListEl) return;
      this.queryListEl.innerHTML = "";

      queries.forEach((q, idx) => {
        const item = document.createElement("div");
        item.className = "query-item";
        item.id = `q-item-${q.id}`;
        
        const tagClass = q.task_type.toLowerCase() === 'qa' ? 'tag-qa' : (q.task_type.toLowerCase() === 'trake' ? 'tag-trake' : 'tag-kis');
        const statusClass = q.is_completed ? 'status-done' : 'status-pending';
        const statusText = q.is_completed ? '✅ Đã lưu' : '⏳ Chưa nộp';

        item.innerHTML = `
          <div class="query-header">
            <span class="query-tag ${tagClass}">${q.task_type}</span>
            <span class="query-status ${statusClass}">${statusText}</span>
          </div>
          <div style="font-weight: 700; font-size: 0.88rem; color: #f8fafc;">${q.id}</div>
          <div class="query-snippet">${q.content}</div>
        `;

        item.addEventListener("click", () => this.selectQuery(q));
        this.queryListEl.appendChild(item);
      });

      // Tự động chọn câu đầu tiên
      if (queries.length > 0 && !this.currentQueryData) {
        this.selectQuery(queries[0]);
      }
    } catch (e) {
      console.error("Lỗi nạp queries:", e);
    }
  }

  selectQuery(q) {
    this.currentQueryData = q;
    
    // Highlight item
    this.queryListEl?.querySelectorAll('.query-item').forEach(el => el.classList.remove('active'));
    document.getElementById(`q-item-${q.id}`)?.classList.add('active');

    // Cập nhật Omnibar
    if (this.omnibarInput) this.omnibarInput.value = q.content;
    if (this.activeQueryTextEl) this.activeQueryTextEl.textContent = `[${q.id}]: "${q.content}"`;
    if (this.activeQueryTagEl) {
      this.activeQueryTagEl.textContent = `${q.task_type} TASK`;
      this.activeQueryTagEl.className = `query-tag tag-${q.task_type.toLowerCase()}`;
    }

    // Thiết lập TRAKE Studio nếu là TRAKE
    if (window.trakeStudio) {
      window.trakeStudio.setupQuery(q.content, q.task_type);
    }

    // Thông báo cho đồng đội qua WebSocket
    if (window.collabClient) {
      window.collabClient.notifyQuerySelect({ id: q.id, task_type: q.task_type, content: q.content });
    }

    // Nạp kết quả đã lưu trước đó nếu có
    this.loadCurrentSubmissionData(q.id);
  }

  async loadCurrentSubmissionData(queryId) {
    try {
      const res = await fetch(`/api/contest/submission_data?output_package=${this.selectedOutputPkg}&query_id=${queryId}`);
      const data = await res.json();
      if (data.exists && data.rows && data.rows.length > 0) {
        // Render các dòng đã nộp
        const formattedResults = data.rows.map((parts, idx) => {
          return {
            rank: idx + 1,
            video_id: parts[0],
            frame_idx: parseInt(parts[1]) || 0,
            answer: parts[2] ? parts[2].replace(/"/g, '') : '',
            score: 1.0 - (idx * 0.005)
          };
        });
        this.renderCards(formattedResults);
      } else {
        // Chưa có kết quả, tự động chạy tìm kiếm A8_SOTA
        this.executeSearch();
      }
    } catch (e) {
      this.executeSearch();
    }
  }

  async executeSearch() {
    const query = this.omnibarInput?.value.trim();
    if (!query) return;

    const taskType = this.currentQueryData ? this.currentQueryData.task_type.toLowerCase() : "auto";
    if (this.statsLatencyEl) this.statsLatencyEl.textContent = "Đang tìm kiếm...";

    try {
      const res = await fetch("/api/search/auto", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          task_type: taskType,
          top_k: 100,
          config_name: "A8_SOTA"
        })
      });

      const data = await res.json();
      if (data.status === "success") {
        this.currentResults = data.results || [];
        if (this.statsLatencyEl) this.statsLatencyEl.textContent = `${data.latency_ms} ms`;
        if (this.statsCountEl) this.statsCountEl.textContent = `${this.currentResults.length} ứng viên`;

        this.renderCards(this.currentResults);

        // Tự động lưu kết quả vào gói bài nộp
        if (this.currentQueryData) {
          this.saveCurrentResults(this.currentResults);
        }
      }
    } catch (e) {
      if (this.statsLatencyEl) this.statsLatencyEl.textContent = "Lỗi kết nối!";
      console.error(e);
    }
  }

  renderCards(results) {
    if (!this.cardsGridEl) return;
    this.cardsGridEl.innerHTML = "";

    results.forEach((c, idx) => {
      const rank = c.rank || (idx + 1);
      const isR1 = (rank === 1);
      const isTop5 = (rank <= 5 && !isR1);

      const cardEl = document.createElement("div");
      cardEl.className = `card-item ${isR1 ? 'rank-1' : (isTop5 ? 'rank-top5' : 'rank-normal')}`;
      
      const badgeClass = isR1 ? 'badge-rank-1' : (isTop5 ? 'badge-rank-top5' : 'badge-rank-normal');
      const badgeText = isR1 ? '👑 Rank 1' : (isTop5 ? `🥈 #${rank}` : `#${rank}`);

      const timeSec = (c.frame_idx / 25).toFixed(1);
      const m = Math.floor(timeSec / 60);
      const s = Math.floor(timeSec % 60);
      const timeStr = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;

      let qaHtml = "";
      if (c.answer) {
        qaHtml = `<div class="qa-answer-box">💬 Đáp án: "${c.answer}"</div>`;
      }

      cardEl.innerHTML = `
        <div class="card-thumb-wrapper" onclick="window.app.previewCandidate('${c.video_id}', ${c.frame_idx})">
          <img class="card-thumb-img" src="/api/media/keyframe/${c.video_id}/${c.frame_idx}" loading="lazy" alt="${c.video_id}" />
          <span class="badge-rank ${badgeClass}">${badgeText}</span>
          <span class="badge-time">${timeStr} (#${c.frame_idx})</span>
        </div>
        <div class="card-details">
          <div class="card-video-info">
            <span class="card-vid-name">${c.video_id}</span>
            <span class="card-frame-idx">Frame: ${c.frame_idx}</span>
          </div>
          ${qaHtml}
          <div class="card-actions">
            <button class="btn-card-action" onclick="window.app.previewCandidate('${c.video_id}', ${c.frame_idx})">👁️ Soi Video</button>
            <button class="btn-card-action btn-set-rank1" onclick="window.app.quickSetRank1('${c.video_id}', ${c.frame_idx}, '${c.answer || ''}')">👑 Ghim R1</button>
          </div>
        </div>
      `;

      this.cardsGridEl.appendChild(cardEl);
    });

    // Mở video đầu tiên vào player
    if (results.length > 0 && window.videoInspector) {
      window.videoInspector.loadVideo(results[0].video_id, results[0].frame_idx);
    }
  }

  previewCandidate(videoId, frameIdx) {
    if (window.videoInspector) {
      window.videoInspector.loadVideo(videoId, frameIdx);
    }
  }

  async quickSetRank1(videoId, frameIdx, answer = "") {
    if (!this.currentQueryData) return;
    try {
      const res = await fetch("/api/contest/override_rank1", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          output_package: this.selectedOutputPkg,
          query_id: this.currentQueryData.id,
          task_type: this.currentQueryData.task_type,
          video_id: videoId,
          frame_idx: frameIdx,
          qa_answer: answer
        })
      });
      const data = await res.json();
      if (data.status === "success") {
        this.loadCurrentSubmissionData(this.currentQueryData.id);
        alert(`👑 Đã ghim ${videoId} (#${frameIdx}) lên Rank 1!`);
      }
    } catch (e) {
      console.error("Lỗi ghim rank 1:", e);
    }
  }

  async saveCurrentResults(results) {
    if (!this.currentQueryData) return;
    try {
      await fetch("/api/contest/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          output_package: this.selectedOutputPkg,
          query_id: this.currentQueryData.id,
          task_type: this.currentQueryData.task_type,
          rows: results
        })
      });
      // Cập nhật trạng thái checkmark
      this.loadQueries();
    } catch (e) {
      console.warn("Lỗi auto-save:", e);
    }
  }

  async executeUndo() {
    if (!this.currentQueryData) return;
    try {
      const res = await fetch(`/api/contest/undo?output_package=${this.selectedOutputPkg}&query_id=${this.currentQueryData.id}`, { method: "POST" });
      const data = await res.json();
      if (data.status === "success") {
        alert("⏪ " + data.message);
        this.loadCurrentSubmissionData(this.currentQueryData.id);
      } else {
        alert(data.detail || "Không có thao tác để hoàn tác.");
      }
    } catch (e) {
      alert("Lỗi hoàn tác: " + e);
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  window.app = new AppController();
});
