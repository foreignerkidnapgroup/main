import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification

model = ElectraForSequenceClassification.from_pretrained("skplanet/dialog-koelectra-small-discriminator", num_labels=22)
tokenizer = ElectraTokenizer.from_pretrained("skplanet/dialog-koelectra-small-discriminator")
model.load_state_dict(torch.load('pytorch_model.bin', map_location = 'cpu'))

# for param in model.parameters():
#     print(param)

model.eval()


test_string = "일은 왜 해도 해도 끝이 없을까? 화가 난다. 많이 힘드시겠어요. 주위에 의논할 상대가 있나요? 그냥 내가 해결하는 게 나아. 남들한테 부담 주고 싶지도 않고. 혼자 해결하기로 했군요. 혼자서 해결하기 힘들면 주위에 의논할 사람을 찾아보세요."
inputs = tokenizer.encode(test_string, return_tensors='pt')

# inputs = (batch_size, max_length) type=tensor
torch.onnx.export(
    model,
    inputs,
    "skplanet-dialog-koelectra-small-discriminator.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names = ['input'],
    output_names = ['output'],
    dynamic_axes = {
        'input' : {0 : 'batch_size', 1:"max_length"},
        'output' : { 0 : 'batch_size', 1:"max_length"}
    }
)
print()
print("Model has been converted to ONNX")