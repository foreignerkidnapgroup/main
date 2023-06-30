import uvicorn
import numpy as np
import onnxruntime as ort
from transformers import ElectraTokenizer
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

sub_mapping = ['기쁨', '고립된', '스트레스 받는', '당황', '배신당한', '환멸을 느끼는', '슬픔', '희생된',
    '열등감', '초조한', '분노', '조심스러운', '죄책감의', '억울한', '부끄러운', '회의적인', '한심한',
    '괴로워하는', '성가신', '짜증내는', '상처', '충격 받은']

tokenizer = ElectraTokenizer.from_pretrained("skplanet/dialog-koelectra-small-discriminator")
ort_sess = ort.InferenceSession("skplanet-dialog-koelectra-small-discriminator.onnx")

class TestString(BaseModel):
    text: str

def sentiment_analysis(test_string, tokenizer, ort_sess):
    input_ids = tokenizer.encode(test_string, return_tensors='np')
    outputs = ort_sess.run(None, {'input': input_ids})
    max_index = np.argmax(outputs[0])
    result = sub_mapping[max_index]
    return result

@app.post("/sentiment_analysis")
async def sentiment_analysis_api(test_string: TestString):
    result = sentiment_analysis(test_string.text, tokenizer, ort_sess)
    return {"result": result}

if __name__=="__main__":
        uvicorn.run("nlp_main:app", host="0.0.0.0", port=7000, log_level="info")