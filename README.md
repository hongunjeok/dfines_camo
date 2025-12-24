1. 학습 방법 : CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=7777 --nproc_per_node=2 train.py -c configs/deim_dfine/dfine_hgnetv2_s_camo.yml --use-amp --seed=0
 가. /workspace/DEIM/configs/deim_dfine 위치에 데이터셋 및 모델 사이즈별 yml 파일이 들어있음
 나. /workspace/DEIM/configs/dataset 위치에 데이터셋별 파일 경로 지정 필요
 다. /workspace/DEIM/configs/base 위치에 dataloader는 배치 사이즈, 이미지 사이즈 조절 가능
 라. /workspace/DEIM/configs/base 위치에 deim은 모델의 증강기법이나 큰 틀에서의 모델 조정(resize 말고는 별도로 건드리지 않음)
 마. /workspace/DEIM/configs/base 위치에 dfine_hgnetv2파일은 제일 하단의 DEIMCriterion을 조정하여 모델의 loss를 조정(loss_sem_proto가 제안한 loss)
2. 엔진
 가. /workspace/DEIM/engine/deim 위치의 deim_criterion에 loss 계산에 관한 것들이 들어있음
 나. /workspace/DEIM/engine/deim 위치에 deim은 모델의 전체 흐름을 통제(조금 수정함)
 다. /workspace/DEIM/engine/deim 위치에 hybridencoder는 인코더단의 모듈을 통제(역전파 과정에서 loss 사용을 위해 조금 수정)

* 졸업 후 ablation study : 논문 상에는 global에 모든 feature 벡터를 맞추는 것으로 실험을 하였지만, global<-medium, medium<-local이 훨씬 성능이 좋았음. 자세한 성능은 wandb 사진으로 대체함. 0.782가 현재 가장 좋은 성능임.
* <img width="1210" height="938" alt="image" src="https://github.com/user-attachments/assets/48221579-16dc-4dad-be37-e29735405813" />
