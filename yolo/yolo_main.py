import os
import uvicorn
import subprocess
from glob import glob
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

class PathModel(BaseModel):
    path: str

@app.post("/yolov5_detect")
async def yolov5_detect(image_path: PathModel):
    path = image_path.path

    if not os.path.isfile(path):
        return {"error": "알맞는 경로가 아닙니다."}

    detect_script = "detect.py"
    label_yaml = "./data/koreafood.yaml"
    img_arg = "640"
    stride_arg = "30"
    conf_arg = "0.25"
    
    command = [
        "python", detect_script, 
        "--data", label_yaml,
        "--source", path,
        "--weights", "./yolo_best.onnx", 
        "--img", img_arg,
        "--save-txt",
        "--save-conf",
        "--hide-conf",
        "--vid-stride", stride_arg,
        "--save-crop",
        "--conf", conf_arg,              # confidence
    ]

    subprocess.run(command)
    
    max_conf = 0
    top_picture_path = ""
    txt_folder_path = "/home/yolov5/runs/detect/exp/labels/*.txt"
    txt_files = glob(txt_folder_path)
    
    # labels 내 conf 중 최대값 추출
    for txt in txt_files:
        with open(txt, 'r', encoding='utf-8') as file:  # 10 0.473177 0.386111 0.0869792 0.203704 0.365057\n
            first_line = float(file.readline().split(" ")[-1].replace("\n", ""))*100
            if max_conf < first_line:
                max_conf = first_line
                top_picture_path = "_".join(txt.split("/")[5].split("_")[0:2])
                # print(f"top_picture_path : {top_picture_path}") # ./runs/detect/exp/crops/ KETI_MULTIMODAL_0000001123 _7.jpg
    
    # crops 내 폴더 추출        
    folder_path_1 = "/home/yolov5/runs/detect/exp/crops/*"
    in_crops = glob(folder_path_1) # 깍두기, 기타...
    
    # 객체탐지된 폴더 내 사진 파일 검색
    for x in in_crops:
        temp = glob(x+"/*.jpg") 
        for y in temp:
            if top_picture_path in y: # y에 top_picture가 포함되면 절대경로 반환
                print(f"---------------- Y : {y} -----------------")
                return {"path" : y}
    return {"path" : ""}

if __name__ == "__main__":
    uvicorn.run("yolo_main:app", host="0.0.0.0", port=8000, log_level="info")

    
