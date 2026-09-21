# -*- coding: utf-8 -*-
"""
Phone Simulator Client (Edge Worker Emulator).
Simulates an Android Phone running an SLM worker:
- Connects to Host Laptop via WebSocket (/ws/npc/{agent_id})
- Handles PING / PONG keepalive
- Listens for GENERATE_REQUEST from Host Laptop
- Generates in-character dialogue conditioned on current Fast/Medium state
- Sends responses back over WebSocket
"""

from __future__ import annotations
import asyncio
import json
import time
from typing import Any
import websockets

from sandbox.persona import (
    Persona,
    get_default_personas,
    build_grounded_dialogue_prompt
)


class PhoneSimulator:
    def __init__(
        self,
        agent_id: str,
        host_url: str = "ws://127.0.0.1:8000",
        use_slm: bool = False,
        model = None,
        tokenizer = None,
        persona: Persona | None = None
    ):
        self.agent_id = agent_id.lower()
        self.ws_url = f"{host_url}/ws/npc/{self.agent_id}"
        self.running = False
        self.websocket = None
        self.use_slm = use_slm
        self.model = model
        self.tokenizer = tokenizer
        
        if persona is not None:
            self.persona = persona
        else:
            self.persona = get_default_personas().get(self.agent_id)

    async def stop(self):
        self.running = False
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass

    async def connect_and_run(self):
        self.running = True
        print(f"[{self.agent_id.upper()} Phone Simulator] Connecting to {self.ws_url}...")
        
        async with websockets.connect(self.ws_url) as ws:
            self.websocket = ws
            print(f"[{self.agent_id.upper()} Phone Simulator] Connected! Listening for state & dialogue requests...")

            while self.running:
                try:
                    message_raw = await ws.recv()
                    data = json.loads(message_raw)
                    msg_type = data.get("type")

                    if msg_type == "GENERATE_REQUEST":
                        turn_id = data.get("turn_id")
                        speaker = data.get("speaker", "Người lạ")
                        user_msg = data.get("message", "")
                        emotion = data.get("emotion", {})
                        trust = data.get("relationship_trust", 0.5)
                        conflict_mode = data.get("conflict_mode", "COLLABORATIVE")
                        action_intent = data.get("action_intent", "")
                        core_belief = data.get("core_belief", "")
                        retrieved_memories_text = data.get("retrieved_memories_text", "")
                        reflection_event = data.get("reflection_event")

                        print(f"[{self.agent_id.upper()} Phone Simulator] Received Turn {turn_id} from {speaker}: \"{user_msg}\"")
                        print(f"   -> Conditioned State: Anger={emotion.get('anger', 0):.2f}, Trust={trust:.2f}, Mode={conflict_mode}")
                        if reflection_event:
                            print(f"   -> [REFLECTION TRIGGERED] New Belief: \"{core_belief}\"")

                        # Generate response conditioned on Persona & State
                        t0 = time.time()
                        response_text = self._generate_dialogue(
                            user_msg=user_msg,
                            speaker=speaker,
                            emotion=emotion,
                            trust=trust,
                            conflict_mode=conflict_mode,
                            action_intent=action_intent,
                            core_belief=core_belief,
                            retrieved_memories_text=retrieved_memories_text
                        )
                        gen_time = (time.time() - t0) * 1000

                        reply_payload = {
                            "type": "GENERATE_RESPONSE",
                            "turn_id": turn_id,
                            "response": response_text,
                            "generation_time_ms": round(gen_time, 2)
                        }
                        await ws.send(json.dumps(reply_payload))
                        print(f"[{self.agent_id.upper()} Phone Simulator] Sent Response in {gen_time:.1f}ms: \"{response_text}\"")

                except websockets.ConnectionClosed:
                    print(f"[{self.agent_id.upper()} Phone Simulator] WebSocket connection closed.")
                    break

    def _generate_dialogue(
        self,
        user_msg: str,
        speaker: str = "Người lạ",
        emotion: dict | None = None,
        trust: float = 0.5,
        conflict_mode: str = "COLLABORATIVE",
        action_intent: str = "",
        core_belief: str = "",
        retrieved_memories_text: str = ""
    ) -> str:
        """Generates grounded in-character dialogue based on Persona, dynamic State & RAG memory."""
        emotion = emotion or {}
        anger = emotion.get("anger", 0.0)

        # Mode A: Real SLM execution on GPU/Edge device
        if self.use_slm and self.model is not None and self.tokenizer is not None and self.persona is not None:
            import torch
            prompt = build_grounded_dialogue_prompt(
                persona=self.persona,
                user_utterance=user_msg,
                speaker_name=speaker,
                emotion=emotion,
                trust=trust,
                conflict_mode=conflict_mode,
                action_intent=action_intent,
                core_belief=core_belief,
                retrieved_memories=retrieved_memories_text
            )
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            import re
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=60,
                    temperature=0.4,
                    top_p=0.85,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            # Clean up dialogue: take line up to closing quote or newline and remove any CJK characters
            clean_text = raw_text.split('"')[0].split('\n')[0].strip()
            clean_text = re.sub(r'[\u4e00-\u9fff]', '', clean_text).strip()
            if clean_text:
                return clean_text
            return raw_text.strip()

        # Mode B: Fast rule-based heuristic generation for quick test suites
        if self.agent_id == "alice":
            if "tự vệ" in core_belief or trust < 0.3 or anger >= 0.35:
                return f"Anh vừa nói gì cơ?! Tôi đã có lòng tốt giúp đỡ, vậy mà anh lại đối xử như thế. Mời anh rời khỏi trạm y tế ngay!"
            elif trust >= 0.5:
                return f"Tôi rất sẵn lòng giúp bạn. Đừng lo lắng, hãy ngồi xuống đây để tôi xem xét tình hình giúp bạn nhé."
            else:
                return f"Tôi có thể hỗ trợ một phần, nhưng đề nghị anh giữ thái độ lịch sự và tuân thủ nội quy của trạm y tế."

        elif self.agent_id == "bob":
            if anger >= 0.35 or trust < 0.3:
                return f"Biến đi chỗ khác! Tao không rảnh nghe mày lải nhải, bước thêm bước nữa là đừng trách tao độc ác!"
            elif trust >= 0.5:
                return f"Được rồi, nể tình lần này tao chia cho một ít, nhưng đừng có mà được đằng chân lân đằng đầu."
            else:
                return f"Không có gì miễn phí ở đây đâu! Muốn có thuốc hay đồ ăn thì phải mang thứ gì có giá trị ra mà đổi!"

        return "..."


async def main():
    import sys
    agent = sys.argv[1] if len(sys.argv) > 1 else "alice"
    sim = PhoneSimulator(agent_id=agent)
    await sim.connect_and_run()


if __name__ == "__main__":
    asyncio.run(main())
