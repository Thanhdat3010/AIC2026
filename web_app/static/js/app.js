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
    this.sourceBadgeEl = document.getElementById("source-badge");
    this.btnForceResearch = document.getElementById("btn-force-research");

    // Khung Thêm / Chèn Thủ Công Clip & Frame Từ Bên Ngoài (Manual Override)
    this.btnToggleManual = document.getElementById("btn-toggle-manual");
    this.manualPanel = document.getElementById("manual-override-panel");
    this.manualInputVideo = document.getElementById("manual-input-video");
    this.manualInputFrame = document.getElementById("manual-input-frame");
    this.manualInputQA = document.getElementById("manual-input-qa");
    this.manualInputTRAKE = document.getElementById("manual-input-trake");
    this.manualGroupQA = document.getElementById("manual-group-qa");
    this.manualGroupTRAKE = document.getElementById("manual-group-trake");
    this.btnManualPinR1 = document.getElementById("btn-manual-pin-r1");
    this.btnManualAppend = document.getElementById("btn-manual-append");
    this.btnManualPreview = document.getElementById("btn-manual-preview");

    this.queryRequestCounter = 0;
    this.activeQueryToken = 0;

    this.initEvents();
    this.loadPackages();
  }

  initEvents() {
    document.getElementById("btn-search")?.addEventListener("click", () => this.executeSearch());
    this.btnForceResearch?.addEventListener("click", () => this.executeSearch());

    this.omnibarInput?.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        this.executeSearch();
      }
    });

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

    // Sự kiện Thêm / Chèn Thủ Công (Manual Override)
    this.btnToggleManual?.addEventListener("click", () => {
      if (!this.manualPanel) return;
      const isClosed = (this.manualPanel.style.display === "none");
      this.manualPanel.style.display = isClosed ? "flex" : "none";
      if (isClosed) {
        if (window.videoInspector?.currentVideoId && !this.manualInputVideo?.value) {
          this.manualInputVideo.value = window.videoInspector.currentVideoId;
          this.manualInputFrame.value = window.videoInspector.currentFrameIdx || 0;
        }
        this.manualInputVideo?.focus();
        this.manualInputVideo?.select();
      }
    });

    this.btnManualPreview?.addEventListener("click", () => {
      const vid = this.manualInputVideo?.value.trim().replace(/\.mp4$/i, "");
      const fidx = parseInt(this.manualInputFrame?.value.trim()) || 0;
      if (!vid) {
        alert("⚠️ Vui lòng nhập Mã Video (Video ID) để xem trước!");
        this.manualInputVideo?.focus();
        return;
      }
      if (window.videoInspector) {
        window.videoInspector.loadVideo(vid, fidx);
      }
    });

    this.btnManualPinR1?.addEventListener("click", () => this.submitManualCandidate("rank1"));
    this.btnManualAppend?.addEventListener("click", () => this.submitManualCandidate("append"));

    // Enter key trong các ô input thủ công tự động kích hoạt Ghim Rank #1
    [this.manualInputVideo, this.manualInputFrame, this.manualInputQA, this.manualInputTRAKE].forEach(inp => {
      inp?.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          this.submitManualCandidate("rank1");
        }
      });
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

      // SẮP XẾP CHUẨN SỐ TỰ NHIÊN: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11... 30
      queries.sort((a, b) => {
        return a.id.localeCompare(b.id, undefined, { numeric: true, sensitivity: 'base' });
      });

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

  showLoadingGrid(q, message) {
    if (!this.cardsGridEl) return;
    if (this.statsCountEl) this.statsCountEl.innerHTML = `<span class="loading-pulse">Đang nạp...</span>`;
    if (this.statsLatencyEl) this.statsLatencyEl.innerHTML = `<span class="loading-pulse">...</span>`;
    if (this.sourceBadgeEl) this.sourceBadgeEl.style.display = "none";
    if (this.btnForceResearch) this.btnForceResearch.style.display = "none";

    const taskType = q?.task_type || "SEARCH";
    const qid = q?.id || "Đang xử lý";
    const tagClass = taskType.toLowerCase();

    // Sinh 10 khung thẻ Skeleton với hiệu ứng shimmer chuyển động
    const skeletons = Array.from({ length: 10 }, (_, i) => `
      <div class="skeleton-card">
        <div class="skeleton-img">
          <div class="skeleton-shimmer"></div>
          <span class="skeleton-pill">#${i + 1}</span>
        </div>
        <div class="skeleton-body">
          <div class="skeleton-line full"></div>
          <div class="skeleton-line half"></div>
          <div class="skeleton-btn-row">
            <div class="skeleton-btn"></div>
            <div class="skeleton-btn"></div>
          </div>
        </div>
      </div>
    `).join("");

    this.cardsGridEl.innerHTML = `
      <div class="query-transition-notice">
        <div class="query-spinner"></div>
        <div class="query-notice-body">
          <div class="query-notice-header">
            <span>Đang chuyển sang câu hỏi:</span>
            <strong class="query-badge-qid">${qid}</strong>
            <span class="q-tag ${tagClass}">${taskType}</span>
          </div>
          <div class="query-notice-status" id="loading-status-text">${message || "Đang tải dữ liệu..."}</div>
        </div>
      </div>
      ${skeletons}
    `;
  }

  updateLoadingStatus(msg) {
    const el = document.getElementById("loading-status-text");
    if (el) el.textContent = msg;
  }

  clearSidebarLoading() {
    this.queryListEl?.querySelectorAll('.question-card.loading').forEach(el => {
      el.classList.remove('loading');
    });
  }

  selectQuery(q) {
    if (!q) return;
    this.currentQueryData = q;
    const requestToken = ++this.queryRequestCounter;
    this.activeQueryToken = requestToken;

    // Highlight card & hiển thị trạng thái loading ở sidebar
    this.queryListEl?.querySelectorAll('.question-card').forEach(el => {
      el.classList.remove('active', 'loading');
    });
    const activeCard = document.getElementById(`q-card-${q.id}`);
    if (activeCard) {
      activeCard.classList.add('active', 'loading');
    }

    // Tạm dừng video đang phát nếu có
    if (window.videoInspector && window.videoInspector.videoEl && !window.videoInspector.videoEl.paused) {
      window.videoInspector.videoEl.pause();
    }

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
    const lockR1Btn = document.getElementById("btn-lock-current-frame");

    if (qaBox) qaBox.style.display = (q.task_type === "QA") ? "flex" : "none";
    if (trakeBox) trakeBox.style.display = (q.task_type === "TRAKE") ? "flex" : "none";
    if (lockR1Btn) lockR1Btn.style.display = (q.task_type === "TRAKE") ? "none" : "flex";

    if (this.manualGroupQA) this.manualGroupQA.style.display = (q.task_type === "QA") ? "flex" : "none";
    if (this.manualGroupTRAKE) this.manualGroupTRAKE.style.display = (q.task_type === "TRAKE") ? "flex" : "none";

    if (window.trakeStudio && q.task_type === "TRAKE") {
      window.trakeStudio.setupQuery(q.content);
    }
    if (q.task_type === "QA") {
      const qaField = document.getElementById("qa-input-field");
      if (qaField) qaField.value = "";
    }

    // HIỂN THỊ NGAY GIAO DIỆN SKELETON ĐỂ TRÁNH LẪN LỘN KẾT QUẢ CŨ
    this.showLoadingGrid(q, "🔄 Đang truy xuất kết quả đã lưu & cấu hình ứng viên...");

    // Nạp kết quả
    this.loadCurrentSubmissionData(q.id, requestToken);
  }

  async loadCurrentSubmissionData(queryId, token = null) {
    try {
      const res = await fetch(`/api/contest/submission_data?output_package=${this.selectedOutputPkg}&query_id=${queryId}`);
      if (token && token !== this.activeQueryToken) return;

      const data = await res.json();
      if (token && token !== this.activeQueryToken) return;

      if (data.exists && data.rows && data.rows.length > 0) {
        const isTrake = (this.currentQueryData?.task_type === "TRAKE");
        const formatted = data.rows.map((parts, idx) => {
          const cleanParts = parts.map(x => (typeof x === 'string') ? x.replace(/^["']|["']$/g, '').trim() : x);
          const rawFrames = cleanParts.slice(1).map(x => parseInt(String(x).trim())).filter(x => !isNaN(x));
          return {
            rank: idx + 1,
            video_id: cleanParts[0],
            frame_idx: rawFrames.length > 0 ? rawFrames[0] : 0,
            event_frames: isTrake ? rawFrames : [],
            answer: (!isTrake && cleanParts[2]) ? cleanParts[2] : '',
            score: 1.0 - (idx * 0.005)
          };
        });

        // Điền trước đáp án QA nếu có
        if (this.currentQueryData?.task_type === "QA" && formatted[0]?.answer) {
          const qaField = document.getElementById("qa-input-field");
          if (qaField) qaField.value = formatted[0].answer;
        }

        // TỰ ĐỘNG GÁN SẴN CHUỖI TRAKE TỪ KẾT QUẢ ĐÃ LƯU TRƯỚC ĐÓ
        if (isTrake && formatted.length > 0 && formatted[0].event_frames.length > 0) {
          window.trakeStudio?.setInitialFrames(formatted[0].event_frames, formatted[0].video_id);
        }

        if (this.statsLatencyEl) this.statsLatencyEl.textContent = "0 ms (cache)";
        if (this.statsCountEl) this.statsCountEl.textContent = `${formatted.length}`;
        if (this.sourceBadgeEl) {
          this.sourceBadgeEl.textContent = "📦 Bài nộp có sẵn";
          this.sourceBadgeEl.className = "source-pill cache";
          this.sourceBadgeEl.style.display = "inline-flex";
        }
        if (this.btnForceResearch) this.btnForceResearch.style.display = "inline-flex";

        this.clearSidebarLoading();
        this.renderCards(formatted);
      } else {
        // Tự động tìm kiếm nếu chưa có kết quả
        this.updateLoadingStatus("🚀 Chưa có bài nộp lưu sẵn. Đang chạy AI A8_SOTA (SigLIP-2 + Gemini)...");
        this.executeSearch(token);
      }
    } catch (e) {
      if (token && token !== this.activeQueryToken) return;
      this.updateLoadingStatus("🚀 Chưa có bài nộp lưu sẵn. Đang chạy AI A8_SOTA (SigLIP-2 + Gemini)...");
      this.executeSearch(token);
    }
  }

  async executeSearch(token = null) {
    const query = this.omnibarInput?.value.trim();
    if (!query) return;

    if (!token) {
      token = ++this.queryRequestCounter;
      this.activeQueryToken = token;
      this.showLoadingGrid(this.currentQueryData, "🚀 Đang thực hiện tìm kiếm đa phương thức A8_SOTA...");
    }

    const taskType = this.currentQueryData ? this.currentQueryData.task_type.toLowerCase() : "auto";
    if (this.statsLatencyEl) this.statsLatencyEl.innerHTML = `<span class="loading-pulse">Đang tìm...</span>`;

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

      if (token && token !== this.activeQueryToken) return;

      const data = await res.json();
      if (token && token !== this.activeQueryToken) return;

      if (data.status === "success") {
        this.currentResults = data.results || [];
        if (this.statsLatencyEl) this.statsLatencyEl.textContent = `${data.latency_ms} ms`;
        if (this.statsCountEl) this.statsCountEl.textContent = `${this.currentResults.length}`;
        if (this.sourceBadgeEl) {
          this.sourceBadgeEl.textContent = "⚡ AI A8_SOTA";
          this.sourceBadgeEl.className = "source-pill live";
          this.sourceBadgeEl.style.display = "inline-flex";
        }
        if (this.btnForceResearch) this.btnForceResearch.style.display = "none";

        // Nhận diện bài toán KIS / QA / TRAKE
        const isTrake = (this.currentQueryData?.task_type === "TRAKE") || (data.task_type === "trake");
        const qaBox = document.getElementById("qa-answer-station");
        const trakeBox = document.getElementById("trake-station");
        const lockR1Btn = document.getElementById("btn-lock-current-frame");

        if (isTrake) {
          if (trakeBox) trakeBox.style.display = "flex";
          if (qaBox) qaBox.style.display = "none";
          if (lockR1Btn) lockR1Btn.style.display = "none";

          // Khởi tạo trạm TRAKE theo câu truy vấn thực tế
          window.trakeStudio?.setupQuery(query);

          // TỰ ĐỘNG GÁN SẴN CHUỖI TRAKE TỪ RANK 1 VỪA TÌM ĐƯỢC
          if (this.currentResults.length > 0 && this.currentResults[0].event_frames && this.currentResults[0].event_frames.length > 0) {
            window.trakeStudio?.setInitialFrames(this.currentResults[0].event_frames, this.currentResults[0].video_id);
          }
        } else if (data.task_type === "qa") {
          if (qaBox) qaBox.style.display = "flex";
          if (trakeBox) trakeBox.style.display = "none";
          if (lockR1Btn) lockR1Btn.style.display = "flex";
        }

        this.clearSidebarLoading();
        this.renderCards(this.currentResults);

        if (this.currentQueryData) {
          this.saveCurrentResults(this.currentResults);
        }
      }
    } catch (e) {
      if (token && token !== this.activeQueryToken) return;
      this.clearSidebarLoading();
      if (this.statsLatencyEl) this.statsLatencyEl.textContent = "Lỗi kết nối!";
    }
  }

  renderCards(results) {
    if (!this.cardsGridEl) return;
    this.cardsGridEl.innerHTML = "";

    const isTrake = (this.currentQueryData?.task_type === "TRAKE");

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

      let subHtml = "";
      let validFrames = [];
      if (isTrake && c.event_frames && c.event_frames.length > 0) {
        const expectedCount = window.trakeStudio?.subeventTitles?.length || c.event_frames.length;
        validFrames = c.event_frames.slice(0, expectedCount);
        subHtml = `<div style="font-size: 0.74rem; font-weight: 800; color: #fbbf24; background: rgba(245,158,11,0.15); padding: 3px 6px; border-radius: 4px; border: 1px solid rgba(245,158,11,0.3); font-family: monospace;">⏱️ Events: ${validFrames.map(f => '#' + f).join(' ➔ ')}</div>`;
      } else if (!isTrake && c.answer) {
        subHtml = `<div style="font-size: 0.76rem; font-weight: 700; color: #a7f3d0; background: rgba(16,185,129,0.15); padding: 3px 6px; border-radius: 4px;">💬 "${c.answer}"</div>`;
      }

      const evParam = (validFrames.length > 0) ? JSON.stringify(validFrames) : ((c.event_frames && c.event_frames.length > 0) ? JSON.stringify(c.event_frames) : 'null');

      cardEl.innerHTML = `
        <div class="card-img-box" onclick='window.app.previewVideo("${c.video_id}", ${c.frame_idx}, ${evParam})'>
          <img class="card-img" src="/api/media/keyframe/${c.video_id}/${c.frame_idx}" loading="lazy" alt="${c.video_id}" />
          <span class="card-badge-rank ${badgeClass}">${badgeText}</span>
          <span class="card-badge-time">${timeStr} (#${c.frame_idx})</span>
        </div>
        <div class="card-body">
          <div class="card-meta">
            <span style="color: #fff;">${c.video_id}</span>
            <span style="color: #94a3b8; font-family: monospace; font-size: 0.8rem;">#${c.frame_idx}</span>
          </div>
          ${subHtml}
          <div class="card-actions-row">
            <button class="btn-card" onclick='window.app.previewVideo("${c.video_id}", ${c.frame_idx}, ${evParam})'>👁️ Xem Video</button>
            <button class="btn-card pin-r1" onclick="window.app.quickPinRank1('${c.video_id}', ${c.frame_idx}, '${c.answer || ''}')">👑 Ghim R1</button>
          </div>
        </div>
      `;

      this.cardsGridEl.appendChild(cardEl);
    });

    if (results.length > 0 && window.videoInspector) {
      const topCand = results[0];
      this.previewVideo(topCand.video_id, topCand.frame_idx, topCand.event_frames);
    }
  }

  previewVideo(videoId, frameIdx, eventFrames = null) {
    if (window.videoInspector) {
      window.videoInspector.loadVideo(videoId, frameIdx);
    }
    const trakeStation = document.getElementById("trake-station");
    const isTrake = (this.currentQueryData?.task_type === "TRAKE") || (trakeStation && trakeStation.style.display !== "none");
    if (isTrake && eventFrames && eventFrames.length > 0) {
      window.trakeStudio?.setInitialFrames(eventFrames, videoId);
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

  async submitManualCandidate(position = "rank1") {
    if (!this.currentQueryData) {
      alert("⚠️ Vui lòng chọn câu hỏi trong danh sách trước khi thêm thủ công!");
      return;
    }

    const videoId = this.manualInputVideo?.value.trim().replace(/\.mp4$/i, "");
    if (!videoId) {
      alert("⚠️ Vui lòng nhập Mã Video (Video ID), ví dụ: L22_V022, L30_V047!");
      this.manualInputVideo?.focus();
      return;
    }

    const frameIdx = parseInt(this.manualInputFrame?.value.trim()) || 0;
    const taskType = this.currentQueryData.task_type;
    let qaAnswer = "";
    let trakeFrames = "";

    if (taskType === "QA") {
      qaAnswer = this.manualInputQA?.value.trim() || "";
      if (!qaAnswer) {
        alert("⚠️ Vui lòng nhập Đáp án QA cho câu hỏi này!");
        this.manualInputQA?.focus();
        return;
      }
    } else if (taskType === "TRAKE") {
      trakeFrames = this.manualInputTRAKE?.value.trim() || "";
      if (!trakeFrames) {
        trakeFrames = String(frameIdx);
      }
    }

    try {
      const res = await fetch("/api/contest/override_rank1", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          output_package: this.selectedOutputPkg,
          query_id: this.currentQueryData.id,
          task_type: taskType,
          video_id: videoId,
          frame_idx: frameIdx,
          qa_answer: qaAnswer,
          trake_frames: trakeFrames,
          position: position
        })
      });

      const data = await res.json();
      if (data.status === "success") {
        const posText = (position === "rank1") ? "Rank #1 (đẩy cũ xuống)" : "cuối danh sách";
        this.showToast(`🎉 Đã thêm ${videoId} (#${frameIdx}) vào ${posText}!`);

        // Tự động load và tua video sang video vừa chèn để xem ngay
        if (window.videoInspector) {
          window.videoInspector.loadVideo(videoId, frameIdx);
        }

        // Tự động cập nhật chuỗi TRAKE Studio nếu là câu TRAKE
        if (taskType === "TRAKE" && window.trakeStudio && trakeFrames) {
          const evList = trakeFrames.split(",").map(x => parseInt(x.trim())).filter(x => !isNaN(x));
          if (evList.length > 0) {
            window.trakeStudio.setInitialFrames(evList, videoId);
          }
        }

        // Load lại danh sách card từ file vừa lưu
        await this.loadCurrentSubmissionData(this.currentQueryData.id);
        this.loadQueries();
      } else {
        alert(data.detail || "Có lỗi khi lưu kết quả!");
      }
    } catch (e) {
      alert("Lỗi kết nối khi ghim video: " + e);
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
