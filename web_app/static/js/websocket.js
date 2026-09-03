/**
 * WebSocket Collaboration Client
 * Đồng bộ trạng thái phòng thi đấu thời gian thực giữa các máy tính trong đội.
 */

class CollaborationClient {
  constructor() {
    this.ws = null;
    this.statusEl = document.getElementById("team-status-text");
    this.room = "aic2026_room";
    this.username = "Máy " + Math.floor(Math.random() * 100 + 1);

    this.connect();
  }

  connect() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/collaborate?room=${this.room}`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      if (this.statusEl) this.statusEl.textContent = "Đã kết nối";
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      } catch (e) {
        console.warn("Lỗi phân tích WebSocket message:", e);
      }
    };

    this.ws.onclose = () => {
      if (this.statusEl) this.statusEl.textContent = "Mất kết nối (Đang thử lại...)";
      setTimeout(() => this.connect(), 3000);
    };
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      data.sender = this.username;
      this.ws.send(JSON.stringify(data));
    }
  }

  handleMessage(data) {
    const type = data.type;

    if (type === "member_joined" || type === "member_left" || type === "room_state_init") {
      const count = data.active_members_count || 1;
      if (this.statusEl) this.statusEl.textContent = `${count} thành viên online`;
    } else if (type === "query_sync") {
      console.log(`[TEAM SYNC] ${data.sender} đang xem câu:`, data.query_info);
    } else if (type === "candidate_shared") {
      const cand = data.candidate;
      if (cand) {
        alert(`🔔 [ĐỒNG ĐỘI ${data.sender} CHIA SẺ]:\nTìm thấy Video ${cand.video_id} (Frame ${cand.frame_idx})!`);
        if (window.videoInspector) {
          window.videoInspector.loadVideo(cand.video_id, cand.frame_idx);
        }
      }
    } else if (type === "submission_alert") {
      console.log(`[TEAM ALERT] ${data.sender} vừa lưu kết quả cho ${data.query_name}`);
    }
  }

  shareCandidate(cand) {
    this.send({
      type: "share_candidate",
      candidate: cand
    });
    alert(`Đã chia sẻ Video ${cand.video_id} (Frame ${cand.frame_idx}) đến toàn đội!`);
  }

  notifyQuerySelect(queryInfo) {
    this.send({
      type: "set_active_query",
      query_info: queryInfo
    });
  }
}

window.collabClient = new CollaborationClient();
