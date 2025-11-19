# llm_module.py
# 본 모듈은 Gemma 3 12B 언어모델을 활용하여 텍스트 생성 기능을 수행함
# 생성 프롬프트 및 시스템 지시문은 예선 단계에서 단순화하여 고정된 문자열을 사용함
# RAG 통합 및 동적 지식 기반 반영 기능은 후속 연구 단계로 이월함

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLMProcessor:
    # Gemma 3 12B 모델과 토크나이저를 초기화함
    def __init__(self, device="cuda"):
        model_name = "google/gemma-3-12b"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device
        )

    # 텍스트 생성 기능을 수행함
    def generate(self, user_input: str) -> str:
        # 아래 프롬프트 구성 요소는 예선 단계에서 단순화함
        system_prompt = "사용자 질문에 대해 간결하고 정확하게 답변할 것"
        input_text = f"<system>\n{system_prompt}\n</system>\n<user>\n{user_input}\n</user>"

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

        # 아래 코드는 후속 단계에서 RAG 기반 힌트를 포함하도록 확장 예정임
        # rag_context = retrieve_documents(user_input)  # 연구 단계에서 구현 예정
        # revised_prompt = integrate_context(input_text, rag_context)  # 연구 단계에서 구현 예정

if __name__ == "__main__":
    llm = LLMProcessor()
    print(llm.generate("청년 상담 연계 서비스의 장점을 설명하라"))
