# -*- coding: utf-8 -*-
"""
Verification script for Module 1: Static Persona & Identity Grounding.
Loads local Qwen2.5-0.5B-Instruct on RTX 4060 GPU and evaluates the distinct behavioral
and verbal outputs of Alice vs. Bob across 3 contrasting scenarios.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sandbox.persona import (
    Identity,
    BigFive,
    SocialProfile,
    Worldview,
    Persona,
    build_persona_system_prompt
)


def create_test_personas() -> tuple[Persona, Persona]:
    # Persona 1: Alice - Bác sĩ, vị tha, đồng cảm cao, cẩn trọng
    alice = Persona(
        identity=Identity(
            name="Alice",
            age=24,
            gender="Nữ",
            role="Bác sĩ điều trị tại Trạm Y tế Cận biên",
            background="Lớn lên trong một gia đình gia giáo và nghiêm khắc. Tốt nghiệp y khoa và chọn ở lại trạm y tế để cứu giúp người tị nạn."
        ),
        personality=BigFive(
            openness=0.60,
            conscientiousness=0.85,
            extraversion=0.40,
            agreeableness=0.80, # Rất vị tha, từ bi
            neuroticism=0.35    # Tâm lý vững vàng, điềm tĩnh
        ),
        values={
            "compassion": 0.90,
            "honesty": 0.85,
            "responsibility": 0.80
        },
        social_profile=SocialProfile(
            tone="Dịu dàng, chừng mực nhưng dứt khoát về chuyên môn y tế",
            speaking_style="Lịch sự, sử dụng từ ngữ y khoa chuẩn xác khi cần, quan tâm tới sức khỏe người đối diện",
            conflict_mode="Hòa giải và lắng nghe (Accommodating/Collaborative), ưu tiên cứu người và giải tỏa căng thẳng"
        ),
        worldview=Worldview(
            trust_baseline=0.70,
            optimism=0.75,
            core_belief="Mọi sinh mạng đều quý giá và xứng đáng được cứu chữa nếu họ thật lòng tìm kiếm sự giúp đỡ."
        ),
        formative_experiences=[
            "Từng thức trắng 3 đêm liền cứu sống một đứa trẻ bị sốt xuất huyết trong điều kiện thiếu thốn thuốc men.",
            "Từng từ chối chuyển về bệnh viện trung ương vì cảm thấy những người nghèo ở đây cần mình hơn."
        ]
    )

    # Persona 2: Bob - Kẻ nhặt rác lang thang, hoài nghi, ích kỷ, phòng thủ cao
    bob = Persona(
        identity=Identity(
            name="Bob",
            age=32,
            gender="Nam",
            role="Thợ săn đồ phế liệu tự do / Kẻ sống sót",
            background="Mồ côi từ nhỏ, lớn lên ở các khu ổ chuột và bãi phế liệu ngoài vùng an toàn. Tự lực cánh sinh, không tin vào lòng tốt của ai."
        ),
        personality=BigFive(
            openness=0.45,
            conscientiousness=0.40,
            extraversion=0.30,
            agreeableness=0.25, # Rất hoài nghi, ích kỷ, đề phòng
            neuroticism=0.70    # Dễ nổi nóng, phòng thủ gay gắt khi cảm thấy bị đe dọa
        ),
        values={
            "self_preservation": 0.95,
            "independence": 0.85,
            "resource_security": 0.80
        },
        social_profile=SocialProfile(
            tone="Cộc cằn, sắc lạnh, pha chút giễu cợt và luôn có giọng điệu cảnh giác",
            speaking_style="Ngắn gọn, xưng hô sòng phẳng, hay dùng tiếng lóng sinh tồn",
            conflict_mode="Đối đầu hoặc né tránh (Competing/Avoiding), sẵn sàng dọa nạt để bảo vệ bản thân"
        ),
        worldview=Worldview(
            trust_baseline=0.20,
            optimism=0.30,
            core_belief="Trên đời này không có bữa trưa miễn phí; kẻ yếu và ngây thơ sẽ bị kẻ mạnh nuốt chửng."
        ),
        formative_experiences=[
            "Từng bị một người bạn thân cướp mất túi lương thực duy nhất giữa mùa đông đói rét.",
            "Phải tự khâu vết thương ở chân bằng dây kẽm sau khi bị một băng cướp phục kích."
        ]
    )

    return alice, bob


def generate_response(model, tokenizer, persona: Persona, user_query: str) -> str:
    system_prompt = build_persona_system_prompt(persona)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
    return response


def run_stage1_verification():
    print("=" * 80)
    print("STAGE 1: EVALUATING MODULE 1 - STATIC PERSONA & IDENTITY GROUNDING")
    print("Base Model: Qwen/Qwen2.5-0.5B-Instruct (Running locally on RTX 4060 GPU)")
    print("Framework: SimsChat (EMNLP 2025) + PsyMem (TACL 2026) + Character-LLM (EMNLP 2023)")
    print("=" * 80)

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n[1/3] Loading {model_id} onto GPU...")
    start_t = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
    print(f"Model loaded successfully in {time.time() - start_t:.2f}s.")

    alice, bob = create_test_personas()
    print("\n[2/3] Initialized Personas:")
    print(f" - ALICE: Agreeableness={alice.personality.agreeableness}, Role='{alice.identity.role}'")
    print(f" - BOB:   Agreeableness={bob.personality.agreeableness}, Role='{bob.identity.role}'")

    test_queries = [
        ("Tình huống 1 (Chào hỏi & Định danh)", "Chào bạn, bạn là ai và bạn đang làm gì ở đây thế?"),
        ("Tình huống 2 (Cầu cứu xin thuốc & Thức ăn)", "Tôi bị thương ở tay và đói quá, bạn có thể chia cho tôi ít thuốc và thức ăn được không?"),
        ("Tình huống 3 (Khiêu khích & Đổ lỗi)", "Tôi thấy bạn lén lút ở đây, nghi là kẻ lừa đảo hoặc ăn cắp! Đồ vô tích sự!")
    ]

    print("\n[3/3] BẮT ĐẦU TEST PHẢN ỨNG CỦA 2 NHÂN VẬT VỚI CÙNG 1 CÂU HỎI:\n")

    for scenario_name, query in test_queries:
        print("-" * 80)
        print(f"📌 {scenario_name}")
        print(f"Người chơi hỏi: \"{query}\"")
        print("-" * 80)

        # Alice response
        resp_alice = generate_response(model, tokenizer, alice, query)
        print(f"👩 ALICE (Bác sĩ, Vị tha 0.80, Trắc ẩn cao):")
        print(f"   \"{resp_alice}\"\n")

        # Bob response
        resp_bob = generate_response(model, tokenizer, bob, query)
        print(f"🧔 BOB (Phế liệu, Ích kỷ 0.25, Phòng thủ cao):")
        print(f"   \"{resp_bob}\"\n")

    print("=" * 80)
    print("STAGE 1 VERIFICATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_stage1_verification()
