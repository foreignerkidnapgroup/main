import os
import json
import shutil
import requests
import pandas as pd
from pathlib import Path
from config import get_secret
from pydub import AudioSegment
from konlpy.tag import Okt
from moviepy.editor import VideoFileClip

class pipeline() :
  def __init__(self) :
    # Clova Speech invoke URL
    self.clova_url = get_secret("CLOVA_SPEECH_URL")
    # Clova Speech secret key
    self.clova_key = get_secret("CLOVA_SPEECH_KEY")
    # 기본 디렉토리(현재 위치)
    BASE_DIR = Path(__file__).resolve().parent
    self.COMMON_URL = "172.23.144.1"
    self.NLP_URL = f"http://{self.COMMON_URL}:7000/sentiment_analysis"
    self.YOLO_URL = f"http://{self.COMMON_URL}:8000/yolov5_detect"
    # input 세팅
    self.input_video_path = BASE_DIR/"input_video"
    self.input_json_path = BASE_DIR/"input_script"
    # output 세팅
    self.output_video_path = BASE_DIR / "output_video"
    self.output_json_path = BASE_DIR / "output_script"
    self.output_audio_path= BASE_DIR / "output_audio"
    self.result_video_path = BASE_DIR / "study" / "video"
    self.result_sep_video_path = BASE_DIR / "study" / "video_sep"
    self.result_script_path = BASE_DIR / "study" / "script"
    self.result_script_sep_path = BASE_DIR / "study" / "script_sep"
    self.result_csv_path = BASE_DIR / "study" / "study_result.csv"
    self.result_csv_path_2 = BASE_DIR / "study" / "study_result2.csv"

    directory_check = [
      self.input_video_path, self.input_json_path, self.output_video_path, self.output_json_path, self.output_audio_path, \
      self.result_video_path, self.result_sep_video_path, self.result_script_path, self.result_script_sep_path ]
    
    # 디렉토리 유무 체크 후 없으면 생성
    for x in directory_check:
      if not os.path.exists(x):
          os.makedirs(x)
          
  # yolo 객체 검출(음식)
  def detect_food(self, input_video_path): # /home/pipeline/output_video/KETI_MULTIMODAL_0000000309_combined_video_1.mp4
    payload = {"path": input_video_path}
    response = requests.post(url=self.YOLO_URL, json=payload)
    return response  
  
  # 감정 분석 API
  def sentiment_inference(self, script):
    payload = {"text": script[:128]}
    response = requests.post(url=self.NLP_URL, json=payload)
    return response.json()['result']
  
  def object_detection_inference():
    return 0
  # 무비파이 추가 부분
  def get_mp4_files(self, directory):
      return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp4')]

  def get_json_files(self, directory):
      return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]

  def find_matching_mp4_file(self, json_file, mp4_files):
      json_file_name = os.path.splitext(os.path.basename(json_file))[0]
      for mp4_file in mp4_files:
          mp4_file_name = os.path.splitext(os.path.basename(mp4_file))[0]
          if mp4_file_name in json_file_name or json_file_name in mp4_file_name:
              return mp4_file
      return None

  def time_to_seconds(self, time_str):
      h, m, s = map(float, time_str.split(':'))
      return h * 3600 + m * 60 + s

  def movie_py(self, input_video_path, output_video_path, input_json_path, output_json_path, movies_cnt = 4, split=False):
    json_files = self.get_json_files(input_json_path)
    mp4_files = self.get_mp4_files(input_video_path)

    # 빈 데이터 프레임 생성 #이름이 df1 #구df_utterance
    # 여기 video_path가 구(file_name)임. 나중에 변수명 바꾸기 >>>>>>>>>>>>>>>>> 빈 데이터프레임 생성하는 위치 확인(더 먼저 나와도 되는지) ##오도은수정
    df1 = pd.DataFrame(columns = ['video_path', 'start_time', 'end_time', 'utterance_dict', 'audio_path', 'score_STT', 'sentiment', 'detect_path'])

    for json_file in json_files:
      print(f"Processing JSON file: {json_file}")

      matched_mp4_file = self.find_matching_mp4_file(json_file, mp4_files)
      if not matched_mp4_file:
          print("No matched MP4 file found.")
          continue

      print("Matched MP4 file:", matched_mp4_file)

      with open(json_file) as file:
          data = json.load(file)

      dialogue_infos = data['dialogue_infos']
      dialogue_infos.sort(key = lambda x : x['start_time'])
      input_video = VideoFileClip(matched_mp4_file)
      video_duration = input_video.duration
      start_time = 0
      end_time = 0
      cnt = 0
      u_dict = {}
      
      print('전체 대화 길이 :', len(dialogue_infos))
      for idx, dialogue_info in enumerate(dialogue_infos):
        print(idx, dialogue_info)
        if idx % movies_cnt == 0 :
          start_idx = idx
          u_dict = {}
          start_time = max(min(self.time_to_seconds(dialogue_info['start_time']), video_duration) - 0.3 , 0)
          if split :
            start_time = max(self.time_to_seconds(dialogue_info['start_time']) - 0.3 , 0)
            if idx == 0:
              global first_time
              first_time = start_time
            start_time -= first_time
            print('split | start: True', first_time, start_time)
            
        u_dict['utterance'+str(idx % movies_cnt)] = dialogue_info['utterance']
        
        if idx % movies_cnt == (movies_cnt-1) :
          cnt += 1
          end_idx = idx
          end_time = max(min(self.time_to_seconds(dialogue_info['end_time']), video_duration) + 0.3, 0)

          if split :
            end_time = max(self.time_to_seconds(dialogue_info['end_time']) + 0.3 , 0)
            end_time -= first_time
            print('split : True | END ', end_time, first_time)
          
          print('시간 : ', start_time, end_time)
          
          try:
            if start_time < end_time:
                final_clip = input_video.subclip(start_time, end_time)
                output_video_file = os.path.join(str(output_video_path), f"{os.path.splitext(os.path.basename(matched_mp4_file))[0]}_combined_video_{cnt}.mp4")

                final_clip.write_videofile(output_video_file)
                print(f"Saved video file: {output_video_file}")

                new_data = data.copy()
                new_data['dialogue_infos'] = dialogue_infos[start_idx:end_idx+1]
                
                new_json_file_path = os.path.join(str(output_json_path), f"{os.path.splitext(os.path.basename(matched_mp4_file))[0]}_combined_data_{cnt}.json")
                with open(new_json_file_path, "w") as new_file:
                    json.dump(new_data, new_file, ensure_ascii=False, indent=4)
                    
                temp_dict = {'video_path' : [output_video_file] , 'start_time' : [start_time], 'end_time' : [end_time], 'utterance_dict': [str(u_dict)], \
                              'audio_path' : [''], 'score_STT' : [0], 'sentiment' : ['']}
                df_temp = pd.DataFrame(temp_dict)
                df1 = pd.concat([df1, df_temp], axis=0, ignore_index=True)
                print(f"Saved JSON file: {new_json_file_path}")
          except:
            print(f"*************************{idx}번째 인덱스에서 문제가 생겼습니다.***************************")
            continue
    return df1

  # 비디오에서 음원 추출 함수
  def mp4towav(self, input_video_path, output_audio_path):
    # Load the mp4 file
    input_video_path = str(input_video_path)
    output_audio_path = str(output_audio_path)
    video = VideoFileClip(input_video_path)
    mp3 = input_video_path.replace('mp4','mp3')

    # Extract audio from video
    video.audio.write_audiofile(mp3)
    wav = mp3.replace('mp3','wav')
    save_path = output_audio_path + '/' + wav.split('/')[-1]
    AudioSegment.from_mp3(mp3).export(wav, format="wav", bitrate="16k")
    shutil.move(wav, save_path)
    return save_path


  # STT
  def clovaSTT(self, input_audio_path, completion='sync', callback=None, userdata=None, forbiddens=None, boostings=None,
                  wordAlignment=True, fullText=True, diarization=None):
      request_body = {
          'language': 'ko-KR',
          'completion': completion,
          'callback': callback,
          'userdata': userdata,
          'wordAlignment': wordAlignment,
          'fullText': fullText,
          'forbiddens': forbiddens,
          'boostings': boostings,
          'diarization': diarization,
      }
      headers = {
          'Accept': 'application/json;UTF-8',
          'X-CLOVASPEECH-API-KEY': self.clova_key
      }
      files = {
          'media': open(input_audio_path, 'rb'),
          'params': (None, json.dumps(request_body, ensure_ascii=False).encode('UTF-8'), 'application/json')
      }
      response = requests.post(headers=headers, url=self.clova_url + '/recognizer/upload', files=files)
      return response


  #STT score 
  def text_recognition(self, source_script, target):
      # User Video to STT
      res = self.clovaSTT(target)
      json_object = res.json()
      user_text = ''
      for seg in json_object['segments']:
          user_text += seg['text']

      # Comparison with System and User
      tagger = Okt()

      tag_user = tagger.pos(user_text)
      tag_system = tagger.morphs(source_script)

      total = max(len(tag_user), len(tag_system))

      penalty = ['Josa', 'Eomi', 'Punctuation']
      advance = ['Noun', 'Verb']

      cnt = 0
      for morph, pos in tag_user:
          if (morph in tag_system) and (pos in advance):
              tag_system.remove(morph)
              cnt += 1.25
          elif (morph in tag_system) and (pos in penalty):
              tag_system.remove(morph)
              cnt += 0.75
          elif morph in tag_system:
              tag_system.remove(morph)
              cnt += 1
      score = cnt/total
      score = score * 120
      if (score) > 100:
          return score, source_script, user_text
      else:
          return score, source_script, user_text

  def forward_pipeline(self) :
    # 무비파이로 영상 잘라는 대화셋 따로 저장 + 데이터프레임 형태로 정보 저장
    # 각 video별 폴더 만들어서 저장
    df1 = self.movie_py(self.input_video_path , self.output_video_path, self.input_json_path, self.output_json_path )
    # 각 비디오별 영상에서 음성파일 추출 및 형식 변환
    for idx in df1.index :
      input_video_path = df1.loc[idx, 'video_path']
      wav_path = self.mp4towav(input_video_path, self.output_audio_path)
      df1.loc[idx, 'audio_path'] = str(wav_path)
      # STT 함수에 넣어서 결과값 확인
      script_dict = eval(str(df1.loc[idx, 'utterance_dict']))
      script = script_dict['utterance0'] + ' ' + script_dict['utterance1'] + ' ' + script_dict['utterance2'] + ' ' + script_dict['utterance3']
      print(f"{idx+1}번째 STT 시작")
      stt_score = self.text_recognition(script, wav_path)[0]
      df1.loc[idx, 'score_STT'] = stt_score
      # KoElectra 추론
      # print(f"Sentiment 안에 들어가는 Script : {script}")
      print(f"{idx+1}번째 감성 분석 시작")
      sentiment = self.sentiment_inference(script)
      df1.loc[idx, 'sentiment'] = str(sentiment)      
      # Yolo TOP confidence(신뢰도) 추출
      print(f"{idx+1}번째 객체 탐지 시작 VIDEO_PATH : {input_video_path} ")
      detect_path = self.detect_food(input_video_path) # /home/pipeline/output_video/KETI_MULTIMODAL_0000001211_combined_video_3.mp4
      # print(f"DETECT_PATH : {detect_path}")
      # print(f"DETECT_PATH_TYPE : {type(detect_path)}")
      df1.loc[idx, 'detect_path'] = str(detect_path.json()['path'])

  # 각 대화셋별 점수 최소치 / TOP Rank 확인 후 학습영상 대화셋 선별 //score_total 기준 선정 top 2개 sort values
    df1 = df1.sort_values('score_STT',ascending=False)
    df1.to_csv(self.result_csv_path, index=False)
    df_top2 = df1.head(2)
    for idx in df_top2.index :
      video_path = str(df_top2.loc[idx, 'video_path'])
      shutil.copy(video_path, video_path.replace('output_video', 'study/video'))
      json_path = video_path.replace('output_video', 'output_script').replace('_video_', '_data_').replace('mp4', 'json')
      shutil.copy(json_path, json_path.replace('output_script', 'study/script').replace('_data_', '_video_'))

    df2 = self.movie_py(self.result_video_path , self.result_sep_video_path, self.result_script_path, self.result_script_sep_path, movies_cnt=1, split=True)
    df2 = df2.iloc[:, :4]
    df2.to_csv(self.result_csv_path_2, index=False)

if __name__ == "__main__":
  pipe = pipeline()
  pipe.forward_pipeline()
