# stt_module.py
# 본 모듈은 Whisper Large v3 모델을 활용하여 음성 입력을 텍스트로 변환하도록 구성함
# 실험 과정에서의 재현성을 확보하기 위해 음성 파일 경로 및 디코딩 옵션을 고정 설정함
# 모델 로드는 사전 학습 가중치를 활용하며, 로컬 캐시 사용으로 초기화 비용을 최소화함

import torch
import whisper

class STTProcessor:
    # Whisper Large v3 모델을 로드하는 초기화 절차를 수행함
    def __init__(self):
        self.model = whisper.load_model("large-v3")

    # 입력 음성 파일을 인식하여 텍스트로 변환함
    def transcribe(self, audio_path: str) -> str:
        result = self.model.transcribe(audio_path, fp16=torch.cuda.is_available())
        return result["text"]

# 실행 예시
if __name__ == "__main__":
    stt = STTProcessor()
    text = stt.transcribe("input_audio.wav")
    print(text)
