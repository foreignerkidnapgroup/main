import pandas as pd
import numpy as np
import onnxruntime as ort
from time import time
from transformers import ElectraTokenizer

tokenizer = ElectraTokenizer.from_pretrained("skplanet/dialog-koelectra-small-discriminator")

input_text = "나는 돈이 많지만 어제도 병원에 갔다 왔어. 내 삶은 행복하지 않아."

start_inference = time()
inputs = tokenizer.encode(input_text, return_tensors='np')
ort_sess = ort.InferenceSession("skplanet-dialog-koelectra-small-discriminator.onnx")
outputs = ort_sess.run(None, {'input':inputs})
end_inference = time()

sub_mapping = ['기쁨', '고립된', '스트레스 받는', '당황', '배신당한', '환멸을 느끼는', '슬픔', '희생된',
       '열등감', '초조한', '분노', '조심스러운', '죄책감의', '억울한', '부끄러운', '회의적인', '한심한',
       '괴로워하는', '성가신', '짜증내는', '상처', '충격 받은']

print(outputs)
max_index = np.argmax(outputs[0])
print(sub_mapping[max_index])
# 0.366487979888916
print(f"Total inference time : {end_inference-start_inference}")