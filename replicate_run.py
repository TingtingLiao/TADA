# replicate_run.py
import json
import sys
import subprocess
import replicate
import os

def main():
    # 1) Replicate는 JSON 형태로 stdin에서 input을 전달
    raw_input = sys.stdin.read()
    inputs = json.loads(raw_input)

    # 여기서 필요한 값 추출
    # 예) config, text, negative
    config_path = inputs.get("config", "configs/default.yaml")
    text_prompt = inputs.get("prompt", "No prompt given")
    negative_prompt = inputs.get("negative", "")

    # 2) TADA run.py 스크립트를 subprocess로 호출
    #    --config, --text, --negative 등 인자 전달
    #    학습/추론이 끝나면 .obj가 저장될 것
    obj_filename = "tada_result.obj"  # TADA 내부에서 이 이름으로 저장한다고 가정
    cmd = [
        "python", "run.py",
        "--config", config_path,
        "--text", text_prompt,
        "--negative", negative_prompt,
        # 추가로 TADA에 필요한 인자를 계속 넣어줄 수 있음
    ]
    # run.py가 결과 .obj를 /app/tada_result.obj로 저장한다고 가정
    subprocess.run(cmd, check=True)

    # 3) .obj 파일 업로드 -> replicate.upload()
    #    replicate-python SDK가 설치되어 있어야 함 (pip install replicate)
    url = replicate.upload(obj_filename)

    # 4) 결과를 JSON으로 stdout에 출력
    print(json.dumps({
        "output": url
    }))

if __name__ == "__main__":
    main()
