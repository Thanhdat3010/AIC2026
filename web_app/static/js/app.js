/**
 * AIC 2026 Championship Console - Core Application Coordinator (Streamlined Edition)
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
    this.toastEl = document.getElementById("toast-notice");
    this.toastMsgEl = document.getElementById("toast-msg");

    this.initEvents();
    this.loadPackages();
  }

  initEvents() {
    document.getElementById("btn-search")?.addEventListener("click", () => this.executeSearch());

    document.getElementById("btn-download-zip")?.addEventListener("click", () => {
      window.location.href = `/api/contest/download_zip?output_package=${this.selectedOutputPkg}`;
    });

    document.getElementById("btn-undo")?.addEventListener("click", () => this.executeUndo());

    this.outputSelectEl?.addEventListener("change", (e) => {
      this.selectedOutputPkg = e.target.value;
      this.loadQueries();
    });

    this.querySelectEl?.addEventListener("change", (e) => {
      this.selectedQueryPkg = e.target.value;
      this.loadQueries();
    });

    // Hotkeys
    window.addEventListener("keydown", (e) => {
      if ((e.ctrlKey && e.key.toLowerCase() === 'k') || (e.key === '/' && document.activeElement.tagName !== 'INPUT')) {
        e.preventDefault();
        this.omnibarInput?.focus();
        this.omnibarInput?.select();
      }

      if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        this.executeSearch();
      }

      if (e.ctrlKey && e.key.toLowerCase() === 'z' && document.activeElement.tagName !== 'INPUT') {
        e.preventDefault();
        this.executeUndo();
      }

      if (e.code === 'Space' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
        e.preventDefault();
        window.videoInspector?.togglePlay();
      }

      if (e.key === '[' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
        window.videoInspector?.stepFrame(-1);
      }
      if (e.key === ']' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
        window.videoInspector?.stepFrame(1);
      }
    });
  }

  showToast(msg) {
    if (!this.toastEl || !this.toastMsgEl) return;
    this.toastMsgEl.textContent = msg;
    this.toastEl.style.display = "flex";
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => {
      this.toastEl.style.display = "none";
    }, 2500);
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
      console.error("Lỗi nạp packages:", e);
    }
  }

  async loadQueries() {
    if (!this.selectedQueryPkg || !this.selectedOutputPkg) return;
    try {
      const res = await fetch(`/api/contest/queries?query_package=${this.selectedQueryPkg}&output_package=${this.selectedOutputPkg}`);
      const data = await res.json();
      const queries = data.queries || [];

      const progressEl = document.getElementById("contest-progress");
      if (progressEl) progressEl.textContent = `${data.completed}/${data.total}`;

      const summaryEl = document.getElementById("questions-summary");
      if (summaryEl) summaryEl.textContent = `${data.completed}/${data.total} câu`;

      if (!this.queryListEl) return;
      this.queryListEl.innerHTML = "";

      queries.forEach((q) => {
        const item = document.createElement("div");
        item.className = "question-card";
        item.id = `q-card-${q.id}`;
        
        const tagType = q.task_type.toLowerCase();
        const statusClass = q.is_completed ? 'done' : 'pending';
        const statusText = q.is_completed ? '✅ Đã nộp' : '⏳ Chưa nộp';

        item.innerHTML = `
          <div class="q-card-top">
            <span class="q-tag ${tagType}">${q.task_type}</span>
            <span class="q-status ${statusClass}">${statusText}</span>
          </div>
          <div class="q-id">${q.id}</div>
          <div class="q-content">${q.content}</div>
        `;

        item.addEventListener("click", () => this.selectQuery(q));
        this.queryListEl.appendChild(item);
      });

      if (queries.length > 0 && !this.currentQueryData) {
        this.selectQuery(queries[0]);
      }
    } catch (e) {
      console.error("Lỗi nạp queries:", e);
    }
  }

  selectQuery(q) {
    this.currentQueryData = q;

    // Highlight card
    this.queryListEl?.querySelectorAll('.question-card').forEach(el => el.classList.remove('active'));
    document.getElementById(`q-card-${q.id}`)?.classList.add('active');

    // Omnibar & Banner Full Text
    if (this.omnibarInput) this.omnibarInput.value = q.content;
    const queryIdEl = document.getElementById("active-query-id");
    if (queryIdEl) queryIdEl.textContent = `📋 ĐỀ BÀI: ${q.id}`;
    if (this.activeQueryTextEl) this.activeQueryTextEl.textContent = q.content;
    if (this.activeQueryTagEl) {
      this.activeQueryTagEl.textContent = `${q.task_type} TASK`;
      this.activeQueryTagEl.className = `q-tag ${q.task_type.toLowerCase()}`;
    }

    const bannerBox = document.getElementById("query-banner-box");
    if (bannerBox) {
      const borderColors = { 'qa': '#10b981', 'trake': '#f59e0b', 'kis': '#38bdf8' };
      bannerBox.style.borderLeftColor = borderColors[q.task_type.toLowerCase()] || '#38bdf8';
    }

    // Ẩn hiện các trạm đặc thù (QA vs TRAKE)
    const qaBox = document.getElementById("qa-answer-station");
    const trakeBox = document.getElementById("trake-station");
    if (qaBox) qaBox.style.display = (q.task_type === "QA") ? "flex" : "none";
    if (trakeBox) trakeBox.style.display = (q.task_type === "TRAKE") ? "flex" : "none";

    if (window.trakeStudio && q.task_type === "TRAKE") {
      window.trakeStudio.setupQuery(q.content);
    }

    // Nạp kết quả
    this.loadCurrentSubmissionData(q.id);
  }

  async loadCurrentSubmissionData(queryId) {
    try {
      const res = await fetch(`/api/contest/submission_data?output_package=${this.selectedOutputPkg}&query_id=${queryId}`);
      const data = await res.json();
      if (data.exists && data.rows && data.rows.length > 0) {
        const formatted = data.rows.map((parts, idx) => {
          return {
            rank: idx + 1,
            video_id: parts[0],
            frame_idx: parseInt(parts[1]) || 0,
            answer: parts[2] ? parts[2].replace(/"/g, '') : '',
            score: 1.0 - (idx * 0.005)
          };
        });

        // Điền trước đáp án QA nếu có
        if (this.currentQueryData?.task_type === "QA" && formatted[0]?.answer) {
          const qaField = document.getElementById("qa-input-field");
          if (qaField) qaField.value = formatted[0].answer;
        }

        this.renderCards(formatted);
      } else {
        // Tự động tìm kiếm nếu chưa có kết quả
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
    if (this.statsLatencyEl) this.statsLatencyEl.textContent = "Đang tìm...";

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
        if (this.statsCountEl) this.statsCountEl.textContent = `${this.currentResults.length}`;

        this.renderCards(this.currentResults);

        if (this.currentQueryData) {
          this.saveCurrentResults(this.currentResults);
        }
      }
    } catch (e) {
      if (this.statsLatencyEl) this.statsLatencyEl.textContent = "Lỗi kết nối!";
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
      cardEl.className = `candidate-card ${isR1 ? 'rank-1' : (isTop5 ? 'rank-top5' : '')}`;
      
      const badgeClass = isR1 ? 'rank-pill-1' : (isTop5 ? 'rank-pill-top5' : 'rank-pill-normal');
      const badgeText = isR1 ? '👑 RANK 1' : `#${rank}`;

      const timeSec = (c.frame_idx / 25).toFixed(1);
      const m = Math.floor(timeSec / 60);
      const s = Math.floor(timeSec % 60);
      const timeStr = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;

      let qaHtml = "";
      if (c.answer) {
        qaHtml = `<div style="font-size: 0.76rem; font-weight: 700; color: #a7f3d0; background: rgba(16,185,129,0.15); padding: 3px 6px; border-radius: 4px;">💬 "${c.answer}"</div>`;
      }

      cardEl.innerHTML = `
        <div class="card-img-box" onclick="window.app.previewVideo('${c.video_id}', ${c.frame_idx})">
          <img class="card-img" src="/api/media/keyframe/${c.video_id}/${c.frame_idx}" loading="lazy" alt="${c.video_id}" />
          <span class="card-badge-rank ${badgeClass}">${badgeText}</span>
          <span class="card-badge-time">${timeStr} (#${c.frame_idx})</span>
        </div>
        <div class="card-body">
          <div class="card-meta">
            <span style="color: #fff;">${c.video_id}</span>
            <span style="color: #94a3b8; font-family: monospace; font-size: 0.8rem;">#${c.frame_idx}</span>
          </div>
          ${qaHtml}
          <div class="card-actions-row">
            <button class="btn-card" onclick="window.app.previewVideo('${c.video_id}', ${c.frame_idx})">👁️ Xem Video</button>
            <button class="btn-card pin-r1" onclick="window.app.quickPinRank1('${c.video_id}', ${c.frame_idx}, '${c.answer || ''}')">👑 Ghim R1</button>
          </div>
        </div>
      `;

      this.cardsGridEl.appendChild(cardEl);
    });

    if (results.length > 0 && window.videoInspector) {
      window.videoInspector.loadVideo(results[0].video_id, results[0].frame_idx);
    }
  }

  previewVideo(videoId, frameIdx) {
    if (window.videoInspector) {
      window.videoInspector.loadVideo(videoId, frameIdx);
    }
  }

  async quickPinRank1(videoId, frameIdx, answer = "") {
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
        this.showToast(`Đã ghim ${videoId} (#${frameIdx}) làm Rank #1!`);
        this.loadCurrentSubmissionData(this.currentQueryData.id);
      }
    } catch (e) {
      console.error(e);
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
      this.loadQueries();
    } catch (e) {
      console.warn("Lỗi lưu:", e);
    }
  }

  async executeUndo() {
    if (!this.currentQueryData) return;
    try {
      const res = await fetch(`/api/contest/undo?output_package=${this.selectedOutputPkg}&query_id=${this.currentQueryData.id}`, { method: "POST" });
      const data = await res.json();
      if (data.status === "success") {
        this.showToast("Đã hoàn tác thao tác gần nhất!");
        this.loadCurrentSubmissionData(this.currentQueryData.id);
      } else {
        alert(data.detail || "Không có dữ liệu hoàn tác.");
      }
    } catch (e) {
      alert("Lỗi hoàn tác: " + e);
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  window.app = new AppController();
});
