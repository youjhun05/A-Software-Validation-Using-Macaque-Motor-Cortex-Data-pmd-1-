import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d

# =========================================================
# 1. 데이터 로딩
# =========================================================
mat = sio.loadmat(
    'Data/data_and_scripts/source_data/processed/MM_S1_processed.mat',
    struct_as_record=False,
    squeeze_me=True
)
Data = mat['Data']
n_reaches = len(Data.kinematics)
print(f"총 reaches 수: {n_reaches}")

# =========================================================
# 2. 전체 데이터 행렬 구성
# =========================================================
all_spikes = []
all_vx     = []
all_vy     = []

for i in range(n_reaches):
    kin = Data.kinematics[i]          # (97 × 7)
    m1  = Data.neural_data_M1[i]      # (67 × 97)
    pmd = Data.neural_data_PMd[i]     # (94 × 97)

    # M1 + PMd 합치기
    neural = np.vstack([m1, pmd])     # (161 × 97)

    all_spikes.append(neural.T)       # (97 × 161)
    all_vx.append(kin[:, 2])         # x velocity
    all_vy.append(kin[:, 3])         # y velocity

X  = np.vstack(all_spikes)           # (전체 bins × 161)
Vx = np.concatenate(all_vx)          # (전체 bins,)
Vy = np.concatenate(all_vy)

print(f"\n=== 데이터 행렬 ===")
print(f"X shape:  {X.shape}")
print(f"Vx shape: {Vx.shape}")
print(f"Vy shape: {Vy.shape}")

# =========================================================
# 3. 희소성 분석 (논문 sparsity 주장 검증)
# =========================================================
sparsity   = np.mean(X == 0)
mean_fr    = X.mean() / 0.01         # 10ms bin → Hz
max_fr     = X.max()  / 0.01
total_spikes = np.sum(X > 0)

print(f"\n=== 희소성 분석 ===")
print(f"희소율:       {sparsity*100:.2f}%")
print(f"평균 발화율:  {mean_fr:.2f} Hz")
print(f"최대 발화율:  {max_fr:.0f} Hz")
print(f"총 스파이크:  {total_spikes:,}개")
print(f"논문 기준:    평균 9.2Hz, 희소율 >95%")

# # =========================================================
# # 4. 기저 디코딩 성능 측정 (r_neural)
# # =========================================================

# # 셔플을 하니까 시간 연속성 패턴을 인지 시키지 못함. 그래서 정확도가 수직 하락함.

# =========================================================
# [보완 1] 타임 랙(Time Lag) 피처 생성 함수
# =========================================================
def create_lagged_features(X, Y_vx, Y_vy, num_lags=10):
    X_lagged, Y_vx_lagged, Y_vy_lagged = [], [], []
    for t in range(num_lags, len(X)):
        # 과거 num_lags 만큼의 bin을 평탄화(Flatten)하여 1차원 피처로 만듦
        lagged_window = X[t-num_lags : t, :].flatten()
        X_lagged.append(lagged_window)
        Y_vx_lagged.append(Y_vx[t])
        Y_vy_lagged.append(Y_vy[t])
    return np.array(X_lagged), np.array(Y_vx_lagged), np.array(Y_vy_lagged)

# 피처 확장 적용 (20 bins = 200ms history)
X_lagged, Vx_lagged, Vy_lagged = create_lagged_features(X, Vx, Vy, num_lags=20)
print(f"확장된 피처 차원: {X_lagged.shape}")

# =========================================================
# 4. 기저 디코딩 성능 측정 (r_neural) - 개선판
# =========================================================
# [보완 2] shuffle=False 로 변경하여 시계열 누수(Leakage) 방지
kf = KFold(n_splits=5, shuffle=False)

r_neural_x = []
r_neural_y = []

print(f"\n=== 5-fold Cross Validation ===")

for fold, (train_idx, test_idx) in enumerate(kf.split(X_lagged)):
    
    X_train, X_test = X_lagged[train_idx], X_lagged[test_idx]
    Vx_train, Vx_test = Vx_lagged[train_idx], Vx_lagged[test_idx]
    Vy_train, Vy_test = Vy_lagged[train_idx], Vy_lagged[test_idx]

    # Ridge Regression 학습
    model_x = Ridge(alpha=1.0)
    model_y = Ridge(alpha=1.0)
    model_x.fit(X_train, Vx_train)
    model_y.fit(X_train, Vy_train)

    # 예측
    Vx_pred_raw = model_x.predict(X_test)
    Vy_pred_raw = model_y.predict(X_test)

    # [보완 3] 가우시안 스무딩 후처리 (sigma 값은 궤적 부드러움에 따라 3~5 내외 조정)
    Vx_pred_smooth = gaussian_filter1d(Vx_pred_raw, sigma=3)
    Vy_pred_smooth = gaussian_filter1d(Vy_pred_raw, sigma=3)

    # Correlation 계산 (스무딩된 예측값과 실제 정답 비교)
    rx, _ = pearsonr(Vx_test, Vx_pred_smooth)
    ry, _ = pearsonr(Vy_test, Vy_pred_smooth)

    r_neural_x.append(rx)
    r_neural_y.append(ry)

    print(f"  Fold {fold+1}: r_x={rx:.4f}  r_y={ry:.4f}")

r_neural_x_mean = np.mean(r_neural_x)
r_neural_y_mean = np.mean(r_neural_y)
r_neural   = (r_neural_x_mean + r_neural_y_mean) / 2

print(f"\n=== 개선된 r_neural 결과 ===")
print(f"r_neural_x:   {r_neural_x_mean:.4f}")
print(f"r_neural_y:   {r_neural_y_mean:.4f}")
print(f"r_neural: {r_neural:.4f}")
print(f"논문 기준:    r_neural = 0.9324")

# =========================================================
# 5. 시각화
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 그래프 1: 희소성 분포
ax1 = axes[0, 0]
spike_counts = X.sum(axis=0)          # Total spikes per neuron
ax1.hist(spike_counts, bins=30, color='steelblue', alpha=0.7)
ax1.set_xlabel('Total Spikes (Per Neuron)')
ax1.set_ylabel('Neuron Count')
ax1.set_title('Spike Distribution per Neuron\n(Sparse Coding Verification)')
ax1.axvline(np.mean(spike_counts), color='red',
            linestyle='--', label=f'Mean={np.mean(spike_counts):.0f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 그래프 2: 발화율 히트맵 (처음 100bins × 30뉴런)
ax2 = axes[0, 1]
im = ax2.imshow(X[:100, :30].T, aspect='auto',
                cmap='hot', interpolation='none')
ax2.set_xlabel('Time Bin (×10 ms)')
ax2.set_ylabel('Neuron Index')
ax2.set_title('Spike Raster Plot\n(First 1s × 30 Neurons)')
plt.colorbar(im, ax=ax2)

# 그래프 3: 실제 vs 예측 속도 (x축, 마지막 fold)
ax3 = axes[1, 0]
plot_range = min(200, len(Vx_test))
time_axis  = np.arange(plot_range) * 10  # ms
ax3.plot(time_axis, Vx_test[:plot_range],
         'b-', linewidth=1.5, label='Actual Velocity', alpha=0.8)
ax3.plot(time_axis, Vx_pred_smooth[:plot_range],
         'r--', linewidth=1.5, label='Predicted Velocity', alpha=0.8)
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('X Velocity (cm/s)')
ax3.set_title(f'Actual vs. Predicted Velocity (X-axis)\nr = {rx:.4f}')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 그래프 4: r_neural fold별 결과
ax4 = axes[1, 1]
folds = np.arange(1, 6)
ax4.plot(folds, r_neural_x, 'b-o', label='$r_x$', linewidth=2)
ax4.plot(folds, r_neural_y, 'r-o', label='$r_y$', linewidth=2)
ax4.axhline(y=0.9324, color='gray', linestyle='--',
            alpha=0.7, label='Baseline (0.9324)')
ax4.axhline(y=r_neural_x_mean, color='blue',
            linestyle=':', alpha=0.5,
            label=f'Mean $r_x$={r_neural_x_mean:.4f}')
ax4.set_xlabel('Fold')
ax4.set_ylabel('Correlation ($r$)')
ax4.set_title('5-Fold CV Decoding Performance')
ax4.set_ylim([0.5, 1.0])
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.suptitle('Phase 2: Baseline Decoding Performance (r_neural)\n'
             'pmd-1 Dataset (MM_S1_processed)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('phase2_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved: phase2_results.png")

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

