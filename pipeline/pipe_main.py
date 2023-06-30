import io
import os
import cv2
import shutil
import random
import requests
import numpy as np
import pandas as pd
import streamlit as st
from pipeline import pipeline
from pydub import AudioSegment
from audio_recorder_streamlit import audio_recorder

def main():
    # 랜덤 숫자를 세션 상태 변수로 저장합니다.
    if 'random_number' not in st.session_state:
        result2 = pd.read_csv("./study/study_result2.csv")
        # DataFrame2 기준 영상 fix를 위해 index 랜덤 추출
        st.session_state["random_number"] = random.choice(result2.index) 

    if 'audio_bytes' not in st.session_state:
        st.session_state.audio_bytes = None
        
    # 전체 비디오(Dataframe1) 중 랜덤 비디오 하나 선택
    result = pd.read_csv("./study/study_result.csv")
    result2 = pd.read_csv("./study/study_result2.csv")

    # DataFrame2에서 샘플 영상 추출
    random_number_sep = st.session_state.random_number 
    sep_video = result2['video_path'][random_number_sep]
    video_num = sep_video.split("_")[2] 
    full_video_path = str(result[result['video_path'].str.contains(video_num)]['video_path'].iloc[0])
    
    # 선정된 Script 추출
    full_script_temp = eval(result[result['video_path']==full_video_path]['utterance_dict'].iloc[0]) # json
    sample_full_script = [v for _, v in full_script_temp.items()]
    sample_full_sentiment = str(result[result['video_path']==full_video_path]['sentiment'].iloc[0])

    # 비디오 출력   
    st.markdown("## 전체 영상")

    st.markdown(f"### < 이 영상의 분위기는🤔 : {sample_full_sentiment} >")
    st.video(full_video_path)

    # 대화내용 출력
    st.write(sample_full_script)

    # random_samples session_state 저장
    if "random_samples" not in st.session_state:
        sep_filter = sep_video.split(".")[0][:-1]
        sep_filter_2 = result2[result2['video_path'].str.contains(sep_filter)]['video_path'].tolist()
        st.session_state.random_samples = random.sample(sep_filter_2, 3)

    # 샘플 영상 1
    st.markdown("## 샘플 영상 # 1")
    st.video(st.session_state.random_samples[0])
    sample_sep_script = eval(result2[result2['video_path']==st.session_state.random_samples[0]]['utterance_dict'].iloc[0])['utterance0']
    st.write(sample_sep_script)


    img_file_buffer = st.camera_input("")

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Check the type of cv2_img:
        # Should output: <class 'numpy.ndarray'>
        st.write(type(cv2_img))

        # Check the shape of cv2_img:
        # Should output shape: (height, width, channels)
        st.write(cv2_img.shape)
    
    audio_bytes = audio_recorder(pause_threshold=20.0, sample_rate=41_000)

    if audio_bytes:
        st.session_state.audio_bytes = audio_bytes

        # 오디오를 실행하고 저장합니다.
        st.audio(st.session_state.audio_bytes, format="audio/wav")
        audio_data = AudioSegment.from_wav(io.BytesIO(st.session_state.audio_bytes))
        audio_save_path = "recorded_audio.mp3"
        audio_data.export(audio_save_path, format="mp3")
    
        pipe = pipeline()
        with st.spinner('Wait for it...'):
            score = pipe.text_recognition(sample_sep_script, audio_save_path)
            if score[0] > 100:
                score_temp = 100
            else:
                score_temp = score[0]
            st.write(f"당신의 말하기 점수는 : {score_temp:.1f}점 입니다.")
        with st.spinner('Wait for it...'):
            is_path = str(result[result['video_path']==full_video_path]['detect_path'].iloc[0])
            if is_path == "":
                st.write("객체가 검출되지 않았습니다.")
            else:
                st.image(is_path)
        # KETIMULTIMODAL0000001135 추천
        st.markdown("### 👇추천 영상👇")
        st.video("/home/pipeline/output_video/KETI_MULTIMODAL_0000001050_combined_video_1.mp4")
if __name__ == "__main__":
    main()

    
