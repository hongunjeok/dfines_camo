import torch

# 🔹 checkpoint 경로 지정
ckpt_path = '/workspace/DEIM/bestest.pth'  # ← 여기에 실제 경로 입력

# 🔹 checkpoint 로드
checkpoint = torch.load(ckpt_path, map_location='cpu')

# 🔹 model state_dict 확인
state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

print("==> All parameter shapes in checkpoint:\n")
for name, param in state_dict.items():
    print(f"{name:50s} | shape: {tuple(param.shape)}")

# 🔹 decoder.cls_score 파라미터 shape만 따로 확인
print("\n==> decoder.cls_score.weight shape:")
print(state_dict.get('decoder.cls_score.weight', '⚠️ Not found'))

print("\n==> decoder.cls_score.bias shape:")
print(state_dict.get('decoder.cls_score.bias', '⚠️ Not found'))
