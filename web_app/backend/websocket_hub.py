import json
from typing import Dict, Set
from fastapi import WebSocket

class CollaborationHub:
    def __init__(self):
        # Room ID -> Set of active WebSocket connections
        self.active_rooms: Dict[str, Set[WebSocket]] = {}
        # Room ID -> Room state (current_query, shared_candidates, etc.)
        self.room_states: Dict[str, dict] = {}

    async def connect(self, websocket: WebSocket, room_id: str = "default"):
        await websocket.accept()
        if room_id not in self.active_rooms:
            self.active_rooms[room_id] = set()
            self.room_states[room_id] = {
                "active_query": None,
                "pinned_candidates": [],
                "recent_actions": []
            }
        self.active_rooms[room_id].add(websocket)
        # Gửi trạng thái phòng hiện tại cho client vừa kết nối
        await websocket.send_json({
            "type": "room_state_init",
            "room_id": room_id,
            "state": self.room_states[room_id],
            "active_members_count": len(self.active_rooms[room_id])
        })
        # Thông báo cho các thành viên khác
        await self.broadcast(room_id, {
            "type": "member_joined",
            "active_members_count": len(self.active_rooms[room_id])
        }, exclude=websocket)

    def disconnect(self, websocket: WebSocket, room_id: str = "default"):
        if room_id in self.active_rooms:
            self.active_rooms[room_id].discard(websocket)
            if not self.active_rooms[room_id]:
                del self.active_rooms[room_id]

    async def broadcast(self, room_id: str, message: dict, exclude: WebSocket = None):
        if room_id not in self.active_rooms:
            return
        dead_connections = set()
        for connection in self.active_rooms[room_id]:
            if connection == exclude:
                continue
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.add(connection)
        for dead in dead_connections:
            self.active_rooms[room_id].discard(dead)

    async def handle_message(self, websocket: WebSocket, room_id: str, data: dict):
        msg_type = data.get("type")
        state = self.room_states.setdefault(room_id, {})
        
        if msg_type == "set_active_query":
            state["active_query"] = data.get("query_info")
            await self.broadcast(room_id, {
                "type": "query_sync",
                "query_info": state["active_query"],
                "sender": data.get("sender", "Teammate")
            })
            
        elif msg_type == "share_candidate":
            cand = data.get("candidate")
            if cand:
                state.setdefault("pinned_candidates", []).append(cand)
                state["pinned_candidates"] = state["pinned_candidates"][-10:] # Keep last 10
            await self.broadcast(room_id, {
                "type": "candidate_shared",
                "candidate": cand,
                "sender": data.get("sender", "Teammate")
            })
            
        elif msg_type == "trake_sync":
            await self.broadcast(room_id, {
                "type": "trake_update",
                "event_data": data.get("event_data"),
                "sender": data.get("sender", "Teammate")
            })

        elif msg_type == "submission_notice":
            await self.broadcast(room_id, {
                "type": "submission_alert",
                "query_name": data.get("query_name"),
                "sender": data.get("sender", "Teammate")
            })

hub = CollaborationHub()
