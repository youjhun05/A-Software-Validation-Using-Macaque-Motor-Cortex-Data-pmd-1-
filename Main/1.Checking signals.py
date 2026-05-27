# 필요 라이브러리 확인
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

print("모든 라이브러리 로딩 완료")

import os

data_path = 'Data'  # 실제 경로로 수정

for root, dirs, files in os.walk(data_path):
    for file in files:
        filepath = os.path.join(root, file)
        size_mb = os.path.getsize(filepath) / (1024*1024)
        print(f"{filepath}  ({size_mb:.1f} MB)")


# MM_S1_processed.mat Load
mat = sio.loadmat(
    'Data/data_and_scripts/source_data/processed/MM_S1_processed.mat',
    struct_as_record=False,
    squeeze_me=True
)

Data = mat['Data']

# Checking Structures
fields = [f for f in dir(Data) if not f.startswith('_')]
print("Data 필드:")
for f in fields:
    print(f"  {f}")


print("\n=== 첫 번째 구조 ===")
print(f"kinematics shape:      {Data.kinematics[0].shape}")
print(f"neural_data_M1 shape:  {Data.neural_data_M1[0].shape}")
print(f"neural_data_PMd shape: {Data.neural_data_PMd[0].shape}")