⸻

## AI Processing Pipeline (LLM · RAG · STT)

본 프로젝트는 Gemma 3 12B 기반 LLM 생성 모듈, LangChain 기반 RAG 파이프라인, 그리고 Whisper Large v3 음성인식(STT) 모듈을 포함하는 통합 구조를 제공합니다.
예선 단계에서는 모듈 간 전체 파이프라인을 최소한의 형태로 구성하고, 후속 연구 단계에서 벡터스토어·지식 기반 통합·추론 최적화 등을 확장하도록 설계되었습니다.

⸻

### 구성 요소

본 리포지토리는 다음 세 가지 코어 모듈로 구성됩니다.

1) llm_module.py
-	사용 모델: Gemma 3 12B (google/gemma-3-12b)
-	역할: 사용자 입력에 대해 텍스트 생성 수행
-	토크나이저 및 모델을 로드하며 간단한 프롬프트 템플릿(<system>, <user>) 적용
-	예선 단계에서는 고정 시스템 프롬프트 사용
-	추후 RAG 문맥 삽입 기능 확장 예정

### 주요 기능
llm = LLMProcessor()
print(llm.generate("청년 상담 연계 서비스의 장점을 설명하라"))

⸻

2) rag_pipeline.py
- LangChain 기반 RAG 구조 정의
-	현 단계에서는 벡터스토어를 초기화하지 않고 인터페이스만 제공
-	후속 단계에서 FAISS, sentence transformers 기반 Embedding, RetrievalQA로 확장 예정

### 주요 기능

rag = RAGPipeline()
print(rag.query("청년 상담 데이터를 기반으로 주요 문제 유형을 설명하라"))

반환: “RAG 검색 기능은 예선 단계에서 비활성 상태로 설정함”

⸻

3) stt_module.py
-	사용 모델: Whisper Large v3
-	역할: 음성 파일 → 텍스트 변환(STT)
-	실험 재현성을 위해 옵션 고정
-	CUDA 사용 가능 시 fp16 인코딩 자동 활성화

주요 기능

stt = STTProcessor()
text = stt.transcribe("input_audio.wav")
print(text)
⸻

🧩 전체 구조 요약

├── llm_module.py        # Gemma 3 12B 기반 텍스트 생성
├── rag_pipeline.py      # LangChain RAG 파이프라인 (예선: 비활성)
└── stt_module.py        # Whisper Large v3 음성 인식
