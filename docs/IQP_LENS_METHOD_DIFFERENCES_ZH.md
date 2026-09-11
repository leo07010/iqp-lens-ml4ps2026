# 我們的方法與原論文（arXiv:2603.06440）差在哪：資料、影像與協定的逐項說明

**用途**：給 <user> 與合作者的內部說明文件，不是論文。比論文詳細、比論文白話，但技術術語、表格 header、程式檔名一律用英文。
**對照的原論文**：Liu, Placidi, Brunner, Rinaldi（Quantinuum），*Toward Generative Quantum Utility via Correlation–Complexity Map*，arXiv:2603.06440 v2，2026-05-05（`docs/PAPER_2603.06440_SUMMARY.md:7-12`）。以下一律稱「原論文」。
**我們的投稿**：NeurIPS ML4PS 2026 workshop，稿件在 `paper/ml4ps2026/main.tex`。
**寫作日期**：2026-09-08。所有數字後面括號註明來源檔，數字不憑印象。

---

## 1. 一頁摘要

1. 我們把原論文的兩件事（correlation–complexity map、IQP 生成模型加 latent adaptation）從它的 2D 湍流速度場搬到 DeepLense 模擬強重力透鏡影像（`main.tex:103-110`）。
2. **最大的兩個改動是資料表示法與 latent 參數化**：原論文把每個樣本編成 18-bit `(x, y, intensity)` tuple 跑 18 qubits，我們改成把「質心對齊後的 32×32 crop 本身當成 1,024 格的 pmf」跑 10 qubits；原論文把 50 個 Walsh 相位當成固定的 latent block 依序調，我們改成學一組仿射方向 `θ = θ⁰ + Φᵀz` 並且對 N 張影像聯合訓練（`main.tex:158-172`）。兩個改動都是被量測逼出來的，不是品味選擇。
3. **結果**：學到方向之後，10-qubit IQP 在原論文自己的兩個指標上贏 RBM 與 DCGAN（PDF–JS 兩個 N 都贏、flux-aligned Conv2D-MMD² 兩個 N 都贏），參數少 2 到 39 倍，held-out 影像重建相關係數 0.968（`main.tex:293-300`；`facts.md §3`）。但 raw Conv2D-MMD² 在 N=100 仍輸 DCGAN（0.0861 對 0.0371，`main.tex:261-264`），而且原論文的 sequential 協定在影像上直接產出雜訊（`main.tex:212-218`）。我們不主張量子優勢。

---

## 2. 原論文方法回顧（讀者不必回去翻原文）

### 2.1 IQP 電路與兩個指標

| 項目 | 定義 | 出處 |
|---|---|---|
| IQP circuit | `U(θ) = H^⊗n exp(i Σ_s θ_s Z_s) H^⊗n`，量測分布 `q_θ(x) = |⟨x|U(θ)|0^n⟩|²` | `docs/PAPER_2603.06440_SUMMARY.md:50,64` |
| 為什麼用它 | 古典高效模擬 IQP 取樣會使 polynomial hierarchy 崩塌，而 MMD loss 需要的 `⟨Z_s⟩` 仍可古典計算，所以能 train-on-classical / deploy-on-quantum | `docs/PAPER_2603.06440_SUMMARY.md:34-35`；`docs/REF38_IQPOPT_METHOD_NOTES.md:24` |
| **QCLI** | 對經驗 pmf 做 Walsh–Hadamard 得 `P(s)`，依 `|s| = k` 聚合成 order spectrum `m_k`，與 i.i.d. 公平位元的 binomial baseline `b_k = C(n,k)/2ⁿ` 取 Jensen–Shannon divergence | `docs/PAPER_2603.06440_SUMMARY.md:71-81` |
| **CCI** | `I_CCI = 1 − I_TC^tree / I_TC`，`I_TC` 是 total correlation（multi-information），`I_TC^tree` 是最佳 pairwise Chow–Liu tree 抓到的那部分。值大 = 存在超越 pairwise 的多體相依 | `docs/PAPER_2603.06440_SUMMARY.md:98-103` |
| Proposition II.2 | 受限 IQP 架構的可調 Walsh 支撐集 `S_A` 之外的頻譜質量，會對 MMD loss 造成不可訓練消除的下界 `≥ C_A Σ_{S∉S_A} P(S)` | `docs/PAPER_2603.06440_SUMMARY.md:110` |
| 選題主張 | 高 QCLI 且高 CCI 的右上象限 = IQP 相容區，turbulence D1/D2 落在此區 | `docs/PAPER_2603.06440_SUMMARY.md:197` |

### 2.2 Algorithm 1（latent-parameter adaptation）

| 步驟 | 內容 | 出處 |
|---|---|---|
| 資料 | Khojasteh et al. 2022 的 D2，1000 個 2D snapshot，`(x, y, z)` 中 z 是 local velocity magnitude | `docs/IQP_REPLICATION_SPEC_2603.06440.md:31-34` |
| 編碼 | 每座標 N=6 bits，串接成 18-bit → **n = 18 qubits** | `docs/PAPER_2603.06440_SUMMARY.md:134` |
| 參數 | `θ = (θ_core, θ_lat)`，`d_lat = 50`，latent block = commuting Pauli string 依 Hamming weight 排序後的**前 50 個** | `docs/PAPER_2603.06440_SUMMARY.md:148`；`docs/IQP_REPLICATION_SPEC_2603.06440.md:54` |
| Stage 1 | 在 anchor `t=1` 上隨機初始化並**凍住** `θ_lat`，只訓 `θ_core`；每步用 **500 個 model samples** 估 MMD，共 `3×10⁴` 步，`N_t = 5×10⁴` 個訓練資料點 | `docs/PAPER_2603.06440_SUMMARY.md:142` |
| Stage 2 | 凍住 `θ_core`，對每個後續 anchor 只調 `θ_lat`，warm-start 自上一個 anchor，每個 snapshot **3000** 步 | `docs/PAPER_2603.06440_SUMMARY.md:144` |
| 插值 | 對 latent 軌跡做 piecewise-linear 內插到未見時間 τ | `docs/PAPER_2603.06440_SUMMARY.md:145` |
| 合成 | 每次抽 `10⁵` 個 bitstring，解碼回 tuple 後聚合回網格 | `docs/PAPER_2603.06440_SUMMARY.md:146` |
| 訓練集 | 11 個 anchor，`t ∈ {1,100,…,1000}`；評估用另外 200 個均勻抽樣的未見 τ | `docs/PAPER_2603.06440_SUMMARY.md:148` |
| 參數量 | Table 2 的 IQP 標 31k / 63k。逆向核對：n=18 時 `Σ_{k≤6} C(18,k) = 31,180`、`Σ_{k≤7} = 63,004`，高度疑似對應 `|s|≤6` / `|s|≤7` 截斷，但原文未明講 | `docs/IQP_REPLICATION_SPEC_2603.06440.md:235` |

### 2.3 目標函數、評估與 baseline

- **訓練用 MMD**：`L = Σ_s c_s (⟨Z_s⟩_θ − ⟨Z_s⟩_data)²`，Gaussian kernel；Ref [38] 對多個 bandwidth 的 MMD² 取**簡單算術平均**，大系統用三個 bandwidth（`docs/REF38_IQPOPT_METHOD_NOTES.md:90,107`）。原論文**從未寫出它用的 σ**（`docs/IQP_REPLICATION_SPEC_2603.06440.md:232`）。
- **評估指標**：PDF–JS（強度邊際直方圖的 JS divergence，B = 50 bins，`docs/IQP_REPLICATION_SPEC_2603.06440.md:119`）與 Conv2D-MMD²（三層 Conv2D–ReLU stride 2 + global average pooling 的固定隨機 encoder，median heuristic 選 σ，N = M = 200，`docs/IQP_REPLICATION_SPEC_2603.06440.md:129-133`）。
- **Baseline**：RBM（12k 參數，同樣的 18-bit 表示、同樣 11 個 snapshot）與 DCGAN（51k / 0.2M / 1.2M，吃量化後的 2D 場，11 或 100 個 snapshot）（`docs/IQP_REPLICATION_SPEC_2603.06440.md:83-104`）。
- **原文未給、我們必須自己決定的**：turbulence 訓練的 optimizer 與 learning rate、訓練 kernel 的 σ、random seeds、**取樣後聚合回網格的規則**（同一格點多筆 z 怎麼合併、空格點怎麼補）（`docs/IQP_REPLICATION_SPEC_2603.06440.md:225-234`）。最後一項在我們的復刻裡造成了實質差異，見 §5.4。

---

## 3. 我們的資料與它的特性

資料是 `<HOME>/mae-lensing/model_II.npz`（DeepLense Model II），三類 `axion` / `cdm` / `no_sub`，train 80,194 / val 8,910 張 64×64 float32（`paper/ml4ps2026/facts.md §1`）。以下每一條都是實測，全部出自 `docs/LENS_IMAGE_CHARACTERISTICS.md`。

| # | 量測 | 數字 | 出處 | 為什麼重要 |
|---|---|---|---|---|
| 1 | per-image min-max 正規化 | 8,910 張**每一張**都恰有 1 個像素 = 0、1 個像素 = 1；per-image 總 flux 125.7 ± 28.2 | `LENS:30` | 跨影像的絕對亮度沒有意義，亮度尺度必須另外處理 |
| 2 | raw 6-bit 強度量化的資訊量 | entropy 只有 **1.42 bits**（6 個可用 bit 中），83.75% 的像素落在 level 0；quantile 量化可到 6.00 bits | `LENS:45-48` | 原論文的 z-register 在這批影像上有 4.6 bits 是浪費的 Hilbert space |
| 3 | 影像不是稀疏的 | 有效支撐 `1/‖p‖²` ≈ **331 / 4,096 像素（8.1%）**，但每張只有 1 個嚴格為 0 的像素 | `LENS:60` | 任何「假設像素多半為零」的稀疏化技巧都會失效，能用的是「能量集中」 |
| 4 | 影像流形維度 | 前 6 個 PC = **89.4%** 變異；PC1/PC3（平移）合計 49.6%；PC2 = flux/大小；PC4/PC5 = ellipticity | `LENS:176,186-189,194` | 實質是 5 維：(x, y, flux, e1, e2)。prior 必須顯式建模平移 |
| 5 | class 訊號在哪 | PC6 之後才是子結構，只佔 **3.0%** 變異；nearest-centroid 分類只有 **38.3%**（chance 33.3%） | `LENS:190,205` | 以像素重建為目標的生成模型會系統性忽略掉唯一有科學價值的成分，所以我們不以 class-conditional 為主要目標 |
| 6 | 幾何 | Einstein 環半徑 r ≈ 8 px，rms 半徑 9.87–10.14 px，光心偏移 **3.36–3.42 ± 1.5–1.7 px**（p95 約 6.2） | `LENS:225-227` | 透鏡不置中，這正是最大的單一變異來源 |
| 7 | 方位角結構是噪音 | split-half 相關只有 r = 0.45 / 0.51 / 0.69 | `LENS:229` | 不要為方位角加任何偏好方向的 generator |
| 8 | 18-bit tuple 編碼的代價 | `|s|≤6` 需要 **31,180** 個 generator，只涵蓋 **79.9%** 的 Walsh 能量 | `LENS:275` | 這就是原論文 31k 參數那一格的實際覆蓋率 |
| 9 | image-as-pmf 的效率 | 12-bit `|s|≤6` 用 **2,510** 個 generator 涵蓋 **95.6%**；10-bit（32×32）`|s|≤6` 用 **848** 個涵蓋 **96.7%** | `LENS:279,285` | 參數少 12.4 倍、覆蓋反而更好。這是換編碼的直接依據 |
| 10 | bit 版面不重要 | Morton/Z-order 的 order spectrum 與 standard **逐點相同**（bit permutation 不改 popcount）；Gray code 是 GF(2)-linear 可逆映射，Walsh 能量 multiset 差異為 0.0 | `LENS:249-250` | 版面優化在數學上不可能有幫助（除非堅持用 Hamming 截斷） |
| 11 | 原論文的 latent 選法不是最好的 | Hamming-first-50 的 class acc（LDA）**0.488**；改成「`|s|≤2` 內依方差排序」是 **0.668**，`|s|≤4` 是 **0.717**（噪音門檻約 4 pp） | `LENS:136-139,144` | 「限制低階」是對的（一二階矩精確線性於 singleton/pair 係數），錯的是在低階裡用固定字典序 |

---

## 4. 逐項差異表

每一列：原論文做法 / 我們的做法 / 為什麼 / 證據。

| # | 面向 | 原論文 | 我們 | 為什麼改 | 證據 |
|---|---|---|---|---|---|
| 1 | **資料與任務** | 1000 個湍流 snapshot，有時間軸，任務是內插到未見時間 τ | DeepLense Model II 靜態影像，三類 pooled，任務是從 latent prior **無條件生成** | 影像沒有時間軸，Eq. 31 的 latent 內插無物可插 | `main.tex:159-162`；`facts.md §1` |
| 2 | **qubit 數與編碼** | 18-bit `(x, y, intensity)` tuple，n = 18 | 質心對齊 → 中心 32×32 crop → 1,024 格 pmf，`code = (kx<<5)|ky`，**n = 10** | raw 強度軸只帶 1.42 bits（`LENS:45`）；tuple 編碼把能量推高一個 Hamming 階，`|s|≤6` 要 31,180 個 generator 卻只涵蓋 79.9%，而 10-bit crop pmf 用 848 個就到 96.7% | `main.tex:162-166`；`LENS:275,285` |
| 3 | **generator 集合** | 依 `|s|` 截斷（推定 `|s|≤6` = 31,180；`|s|≤7` = 63,004） | **全部 1,024 個 Z-string**（n=10 的完整 Walsh basis），exact simulation | n=10 時完整基底才 1,024 個參數，比原論文的截斷版還少 30 倍，沒有截斷的必要；也因此 bit 版面對表達力完全無關 | `docs/IQP_LENS_ROUND3_PREREGISTRATION.md:14`；`codec10.py:13-15` |
| 4 | **latent 參數化** | 固定的 50 個 Walsh 相位（Hamming 順序前 50 個），只調它們 | 學到的仿射方向 `θ_i = θ⁰ + Φᵀz_i`，`Φ ∈ R^{M×1024}` 共用、`z_i ∈ R^M` 每張影像，`M ∈ {2,4,8}`；Φ 每列 L2 normalise 到 `√1024 = 32`，z 加 `10⁻³` 的相對 L2 penalty（M=8 arm 提高到 0.3） | 固定方向做出來的是 XOR-shift 干涉樣式，會 cross-fade 像素而不是移動環（§6）；換成學方向後 held-out projection corr 從 0.672 跳到 0.911–0.968 | `main.tex:129-143`；`facts.md §5`；`meta.json` `phi_row_norm` = 32.0、`zpen` = 0.001 |
| 5 | **訓練協定** | Sequential：先在單一 anchor 上訓 core，凍住後逐個 snapshot 調 latent | **Joint**：`θ⁰`、`Φ` 與全部 N 個 `z_i` 一起訓（Adam lr 0.02，4,000 步） | 用原協定時，除了第一張以外的每張訓練影像都只重建成雜訊（corr 0.47） | `main.tex:166-168`；`facts.md §5`；`R3PRE:107` |
| 6 | **目標函數** | 每步用 500 個 model sample 估 MMD | **閉式（exact）MMD**，另有三個變體：paper kernel（Ref [38] 三 bandwidth 規則在 n=10 重算，σ = 0.989 / 1.729 / 2.236，對應 mean Pauli weight 2.00 / 0.770 / 0.476）、**flat MMD**（每個 Walsh 係數等權，仍是 `Σ c_s(⟨Z_s⟩ 差)²`，可用硬體 shot 估）、exact KL（只當診斷） | 10-qubit pmf 只有 1,024 個振幅，精確可算，沒有理由引入抽樣噪音；flat MMD 是為了保住「硬體可估」這個性質而設計的對照 | `main.tex:140-143,168-170`；`outputs_lens_r3b/iqp_m4_auto3_resample_n100/meta.json` |
| 7 | **KL 的地位** | 不使用 | 只當**模擬專用診斷**，寫在 App. E，不進正文主表 | KL 需要完整的模擬分布，無法從硬體 shot 估計，放進主表會誤導硬體可行性 | `main.tex:427-431` |
| 8 | **prior 與生成方式** | 對 latent 軌跡做時間內插 | 影像沒有時間軸，所以 prior 是 (a) 對訓練 z 擬合的 full-covariance Gaussian 或 (b) **resample**：抽一個訓練 z 再加 0.2× 中位最近鄰間距的 Gaussian jitter | round-2 診斷顯示高維 Gaussian 在少量 anchor 上擬出來的質量幾乎全落在沒有真 latent 的區域（抽出的點離 anchor 的最近鄰間距 4.8 倍）；IQP 的相位空間離開訓練點就塌 | `R3PRE:104-116`；`R3RES:235` |
| 9 | **亮度尺度** | 隱含在 tuple 的 z 值裡 | 電路只輸出 `q(x,y)`，亮度是外部純量：`field = q × S`，`S = 124.92`（訓練 crop 的平均總亮度） | 影像做過 per-image min-max 正規化，絕對亮度不跨影像可比（`LENS:30`）。這是一個**我們自己標記的設計缺陷**，見 §9 | `main.tex:171-172`；`meta.json` `S` = 124.92079504305893 |
| 10 | **評估** | PDF–JS + Conv2D-MMD²（保留） | 同樣兩個指標，另加 (a) **held-out projection correlation**（凍住 `θ⁰`、`Φ`，只對 200 張 held-out 影像最佳化 z，看能重建到多好）、(b) **flux-aligned Conv2D-MMD²**（把生成影像重新縮放到 held-out 平均總亮度再算，標註 post-hoc） | 原論文用「合成未見時間」量泛化，靜態影像沒有時間軸可以 hold out；raw MMD² 有一半是在比亮度，因為 IQP 與 RBM 共用同一個 S 而 DCGAN 自己學尺度 | `main.tex:152-156,170-171`；`evaluate_r3.py:33-45` |
| 11 | **Baseline 對齊** | RBM 吃 18-bit tuple；DCGAN 吃量化的 2D 場 | RBM 改吃 **10-bit code**（10 visible / 1,000 hidden，PCD-1，11,010 參數）；DCGAN-32 是 32×32 crop 的 Radford 式 generator（199,837 參數） | 要公平，baseline 必須吃跟 IQP 一樣的表示法；DCGAN 則保持在像素空間（跟原論文一樣） | `main.tex:146-149`；`facts.md §3` |
| 12 | **Conditioning** | 時間 τ | 主結果全部 **pooled**（三類混合）；另做 per-class 版本（每類各自訓練與評分）當附錄 | 三類的差異只佔 3.0% 的像素變異、nearest-centroid 只有 38.3%，以 class 為主要成功指標不合理；但要回答「能不能單獨生成每一類」，所以補做 | `main.tex:302-305,449-457`；`LENS:190,205` |

---

## 5. 影像處理流程逐步說明

### 5.1 每一步做什麼、為什麼、對應程式

| 步驟 | 做什麼 | 為什麼 | 程式與參數 |
|---|---|---|---|
| 1. Centroid alignment | 算強度加權質心 `(cy, cx)`，取**整數**位移 `dy = clip(round(cy − (crop−1)/2), 0, 32)`，同樣算 dx；不做插值 | 光心偏移 3.4 ± 1.6 px 是最大的單一變異來源（`LENS:225-227`）；整數位移不會憑空造出或抹平任何光 | `data_lens_r3.py:72-95`（`align_crop`） |
| 2. 32×32 crop | 取以質心為中心的 32×32 視窗，記下角落 `(dy, dx)` | 對齊後的 crop 保留 **99.4%** 的光（最低 98.4%，量在 3,000 張 val 上）；32×32 幾乎不損失結構（`|s|≤6` 覆蓋 96.7% 比 64×64 的 95.6% 還高） | `R3PRE:9-11`；`LENS:285` |
| 3. pmf 正規化與編碼 | `p(kx,ky) = crop / sum(crop)`，`code = (kx<<5)|ky`，flat index = `kx*32 + ky`（所以是 `crop.T.ravel()`，不是 `crop.ravel()`） | 把影像本身當機率分布，避開 tuple 編碼的高階代價 | `codec10.py:9-10,19-21,54-68` |
| 4. 訓練 / held-out 分割 | 對 pooled train_x 做**一個**固定 permutation：前綴 N 張當訓練（所以 N=11 是 N=100 的子集），接下來 2,000 張當 latent 選擇池（與訓練集不相交），held-out = val_x 固定 permutation 的前 200 張。split seed 固定為 0，**與任何 arm 的 `--seed` 解耦** | 資料不可以隨訓練 seed 移動，否則不同 arm 不可比；latent 選擇不可以偷看訓練影像 | `data_lens_r3.py:17-23,47` |
| 5. 生成 | 從 prior 抽 z → `θ = θ⁰ + Φᵀz` → 三次 FWHT 得 exact `q` → `crop = q × S`，`S = 124.92` | S 是訓練 crop 的平均總亮度；對應的 held-out crop 平均是 123.20（1.014 倍） | `codec10.py:76-86`；`data_lens_r3.py:290`；`R3RES:27` |
| 6. 貼回 64×64 | 把 32×32 crop 貼進全黑的 64×64 框，角落 `(dy, dx)` 從**訓練影像的偏移分布中抽樣（有放回）** | 評分必須回到原始影像尺度；偏移分布是資料的真實性質（PC1+PC3 = 49.6% 變異），不能假設置中 | `codec10.py:123-146`（`paste`）；`data_lens_r3.py:315-320`（`sample_offsets`） |
| 7. 顯示 | 全部圖片用 `magma`、線性、`vmin = 0`、`vmax = 1` | 所有樣本被縮放到同一個 S，共用 0–1 尺度才誠實 | `figures/FIGURE_PROVENANCE.md:26,34` |
| 8. 顯示時的 clipping（已揭露） | 少數 IQP 峰值像素超過 1 被 `vmax = 1` 截掉：Fig. 1(b) 論文協定 3 / 12,288 px（max 1.146）、學到方向 13 / 12,288 px（max 1.491）；Fig. 2(b) paper kernel 15 px（max 1.602）、M=8 崩潰 arm 1 px。held-out、RBM、DCGAN 三列**沒有任何像素被 clip** | 這是圖面事實，寫進 figure provenance 而不是藏起來 | `figures/FIGURE_PROVENANCE.md:34-41` |

### 5.2 Round 1 為什麼失敗（18-bit tuple 編碼）

- 照 pre-registration 字面套用原論文編碼：`(kx, ky, kz)`，`kz = floor(64 × intensity)`。**跑前就量到 83.6–84.0% 的像素落在 intensity bin 0**，z-channel 幾乎被背景吃掉（`docs/IQP_LENS_PILOT_RESULTS.md:24`）。
- 結果：IQP 是所有沒崩潰的 arm 裡**最差的**，NN-holdout 5.8 / 7.5 / 7.7 對 CTRL-mean 的 3.6–3.8；而且輸給 198 個參數的 product-Bernoulli 控制組（`PILOT:12-13`）。
- **同時，這批資料的 QCLI 高得漂亮**：per-image `I_QCLI` = 0.4697 / 0.4693 / 0.4755，配對的 uniform null 只有 0.0077，高出 61 倍（`PILOT:16`）。按原論文的選題準則這是教科書級的「IQP-compatible」資料，IQP 卻是最差的 arm。**「高 QCLI ⇒ 適合 IQP」在這裡被直接證偽。**
- sqrt 量化確實有效（PDF–JS 從 0.29–0.42 掉到 0.050–0.060，機制清楚：bin-0 佔比從 7.4% 拉回 69.9%）但不夠，三個指標三個類別仍全輸 CTRL-mean（`PILOT:14`）。

### 5.3 Round 2 為什麼失敗（強度尺度假象）

- Round 2 換成 12-bit image-as-pmf。表面上 MMD² 從 0.66–1.27 掉到 0.42–0.50、NN-holdout 從 5.76–7.69 掉到 5.16–5.44（`docs/IQP_LENS_ROUND2_RESULTS.md:11`）。
- **但修好的是亮度尺度，不是空間結構**：pmf arm 與 class-mean 的 per-image 相關係數只有 −0.006 到 +0.041，弧/背景對比 **0.98–1.15**，而真影像是 17.6–19.0（`R2:11`）。三個 pmf arm、三個類別全部是**零弧結構**。
- 把每個 arm 的平均總亮度事後對齊 held-out 再重算，round-2 相對 round-1 的**全部 MMD² 增益消失**（round-1 IQP 掉到 0.48–0.53，pmf 停在 0.45–0.47）。改善的來源是 `field = q × S` 把平均總亮度從 1.49–2.31 倍鎖成 0.96–1.03 倍，而 S 是外部給的（`R2:11`）。這個教訓直接生出 round 3 的 **flux-aligned 次評分欄**。
- **協定與 generator 數在 12-bit 上完全不是槓桿**：凍 latent + 5×10⁴ 抽樣 pmf 對上全參數 + 精確 pmf，三類 N=100 的差距是 `|ΔJS| ≤ 0.010`、`|ΔMMD²| ≤ 0.011`、`|ΔNN| ≤ 0.027`，噪音等級（`R2:12`）。`meta.json` 的 `stage1_latent_shift` 給出機制：解凍後 `θ_lat` 在 stage 1 只移動了 **0.12–0.22 rad**，所以「解凍」數值上幾乎是 no-op（`R2:13`）。
- 這兩輪把 round 3 的三個設計決定逼出來：換 10-bit crop pmf、換 joint 協定、把亮度尺度的 confound 明確標出並加 aligned 次評分。

### 5.4 聚合規則（last-write vs mean）這件事為什麼重要

原論文從未定義「同一格點落到多筆樣本時 z 怎麼合併」（`docs/IQP_REPLICATION_SPEC_2603.06440.md:234`）。我們在 turbulence 復刻裡兩種都做了：primary 用 mean、secondary 用 last-write-wins。**在 last-write-wins 慣例下，原論文自己的 IQP PDF–JS 數字是對得上的**（0.0322 ± 0.0006 / 0.0476 ± 0.0004 對論文的 0.0238 / 0.0221），mean 慣例下就對不上（`docs/IQP_TURBULENCE_REPLICATION_RESULTS.md:13`）。這是一個未指定的實作細節足以改變復現判定的實例。lens 這條線因為改用 exact pmf 重建，完全沒有這個問題。

---

## 6. 為什麼原協定在影像上失效，而「學方向」有效

### 6.1 三個跑前做完的機制測試（因果鏈）

| 測試 | 結果 | 讀法 | 出處 |
|---|---|---|---|
| (i) 單張 crop 的表達力 | 一張對齊後的 crop 被 1,024 相位的 10-qubit IQP 幾乎精確擬合：KL corr **0.9999**、flat MMD **0.99997**、paper kernel **0.9962** | 表達力與 kernel 選擇**都不是**瓶頸 | `facts.md §5`；`NUMBERS_PROVENANCE.md` §3.2 表 |
| (ii) 固定基底的插值 | 沿論文的 Hamming latent 在兩張影像之間走，中途 `max(corrA, corrB)` 掉到 **0.233**、ringiness 掉到 0.353，而像素 cross-fade 對照在每個 t 上 corr 都是 1.0 | latent 空間裡兩個「好點」之間全是壞點，所以從 prior 抽樣得到斑塊 | `R3RES:207` |
| (ii-b) ancilla 能不能救 | ancilla 邊際被驗證為**恰好等權重**的兩個 hologram 非相干混合（每支 mass 0.500000000000，殘差 ≤ 2.6e-18） | 沒有「混合角」可調，ancilla 加不了東西 | `R3RES:207` |
| (iii) 學方向 | `θ = θ⁰ + Φᵀz` 訓 20 張 crop 後，held-out projection corr **0.911**（M=4）；換成隨機 Φ 只有 **0.672**、完全沒有 latent 是 0.671 | 增益全部來自「**學到**方向」，不是「多了自由度」 | `facts.md §5`；`mech_tests/mech_test2/diag2.json` |

**機制的一句話說明**：固定的 Walsh 方向在相位空間裡作用成 XOR-shift 干涉樣式，它會把兩張影像的像素 cross-fade 起來，做不到「把環移動 / 拉伸」這種幾何操作（`main.tex:216-218`）。用原論文的 sequential 協定時，core 被釘在單一 anchor 上，剩下 50 個固定的 latent 座標把**其他每一張訓練影像**都重建成 corr **0.47** 的雜訊（`facts.md §5`；`R3PRE:107`）。這不是我們實作有問題，這是協定本身在影像上的行為。

### 6.2 正式跑的數字（N = 100）

| 量 | 數字 | 出處 |
|---|---|---|
| 最佳 held-out projection corr（M=4，flat MMD） | **0.968**（min 0.851） | `meta.json` `projection_corr_mean` = 0.9683757 |
| 「從 100 張訓練影像挑最近那張」的參考上限 | **0.970** | `meta.json` `ref_projection_nearest_train_corr` = 0.96995 |
| 平均訓練影像的參考下限 | 0.779 | `meta.json` `ref_projection_mean_crop_corr` = 0.77886 |
| 隨機方向 Φ 對照（controlled N=20 sweep） | 0.672；同一實驗中學到的 Φ 是 **0.911** | `mech_tests/mech_test2/diag2.json` |
| paper kernel 的 projection | 只有 **0.74–0.79** | `results_r3b.jsonl`，N=100 |

### 6.3 M 的上限是相干半徑，不是參數量

| M | 行為 | 數字 | 出處 |
|---|---|---|---|
| 2 | 生成最好：100% 的 Gaussian 抽樣是環，rms 半徑 9.40 ± 0.21 px 對真值 9.21 ± 0.15，但 ellipticity 散布不足（e1 sd 0.048 對 0.099） | `‖z‖` = 0.015 | `R3PRE:168-170`；`R3RES:210` |
| 4 | 投影最好（0.968） | `‖z‖` = 0.020（KL）/ 0.029（auto3）；實際 run 的 `z_norm_max` = 0.0414、`z_norm_mean` = 0.0174 | `R3RES:210`；`meta.json` |
| 6 | 環的比例掉到 **0**，半徑跑到 12.84 | `‖z‖` 衝到 0.087 | `R3RES:210` |
| 8 | 崩潰：projection corr **0.636**、PDF–JS 0.0103、MMD² 0.244；加 z-penalty 0.3 也救不回 | 在 controlled N=20 sweep 中 M=8 掉到 **0.095**、M=16 為 0.147 | `NUMBERS_PROVENANCE.md` §3.3 表 |

**相干半徑約 0.03–0.05**（`facts.md §5`）。一旦 `‖z‖` 超過它，`exp(iθ)` 的非線性把重建打散。這是相位參數化的固有限制，不是可以靠更多參數解決的問題。

---

## 7. 結果對照表

### 7.1 我們的主表（Table 1，`main.tex:259-266`；欄位逐格出自 `results_r3.jsonl` / `results_r3b.jsonl`）

| Model | Params | N=11 PDF–JS | N=11 MMD² | N=11 MMD²(flux-al.) | N=100 PDF–JS | N=100 MMD² | N=100 MMD²(flux-al.) |
|---|---|---|---|---|---|---|---|
| IQP, paper protocol (Alg. 1) | 1,024 | 0.00609 | 0.2031 | 0.1837 | 0.00697 | 0.2140 | 0.2233 |
| IQP, fixed basis, joint | 1,024 | 0.00133 | 0.1416 | 0.1141 | 0.00099 | 0.1041 | 0.0617 |
| **IQP, learned dir. M=4 (flat MMD)** | 5,164 / 5,520 | 0.00196 | 0.1134 | **0.0279** | 0.00239 | 0.0861 | **0.0275** |
| **IQP, learned dir. M=4 (paper kernel)** | 5,164 / 5,520 | **0.00095** | 0.1168 | 0.0679 | **0.00092** | 0.1189 | 0.0876 |
| RBM | 11,010 | 0.00632 | **0.0741** | 0.1240 | 0.00195 | 0.0752 | 0.1144 |
| DCGAN | 199,837 | 0.00183 | 0.2165 | 0.1241 | 0.00763 | **0.0371** | 0.0694 |
| *Held-out split-half (floor)* | n/a | *0.00003* | *0.0040* | *0.0069* | *0.00003* | *0.0040* | *0.0069* |

讀法（`main.tex:293-300`）：M=4 paper-kernel arm 兩個 N 都贏 PDF–JS，M=4 flat-MMD arm 兩個 N 都贏 flux-aligned MMD²，參數比 RBM 少 2 倍、比 DCGAN 少 36–39 倍，**但沒有單一 arm 兩個指標都贏**。raw Conv2D-MMD² 上每個 IQP arm 在兩個 N 都輸，flux 對齊後排序反轉。

### 7.2 Per-class 生成（`main.tex:282-285`）

| Training set | PDF–JS | MMD² | MMD²(flux-al.) | proj. corr. |
|---|---|---|---|---|
| axion only | 0.002050 | 0.09012 | 0.02346 | 0.9720 |
| cdm only | 0.002275 | 0.09783 | 0.01848 | 0.9686 |
| no sub only | 0.002001 | 0.1034 | 0.02773 | 0.9703 |
| *pooled* | *0.002387* | *0.08607* | *0.02745* | *0.9684* |

三類都達到 pooled 水準；但**不可以**主張某一類比較容易，因為每類用的是不同的 held-out 集合、各自的 split-half 地板差 3 倍（MMD² 地板 0.0059 / 0.0179 / 0.0160，`facts.md` per-class addendum）。

### 7.3 與原論文 Table 2 的定性對照

| | 原論文（turbulence） | 我們（lens） |
|---|---|---|
| 小 N（11） | IQP 全面贏 RBM 與所有 DCGAN；DCGAN 在 11 筆資料下訓練崩潰（PDF–JS 0.245–0.367） | IQP 贏 PDF–JS 與 flux-aligned MMD²；**我們的 DCGAN 在 N=10 沒有崩**（PDF–JS 0.0048、MMD² 0.044） |
| 大 N（100） | 高容量 DCGAN 可在一兩個指標上打平或超越 IQP，但要 10 倍資料與 1.2M 參數 | DCGAN 在 raw MMD² 領先（0.0371 對最好的 IQP 0.0861），但 PDF–JS 反而變差（0.00183 → 0.00763） |
| Data scaling | 資料變多 DCGAN 才追上 | **不成立**：DCGAN 在 N=10 就已經領先（0.044 對 0.120），之後非單調起伏 3 倍（0.113 → 0.037）；IQP 在 N≥30 平坦（0.082–0.087），RBM 也平坦（0.066–0.089） |
| 絕對值 | PDF–JS 0.022–0.367，MMD² 0.896–1.472 | PDF–JS 0.0009–0.0076，MMD² 0.037–0.216 |

出處：`docs/PAPER_2603.06440_SUMMARY.md:161-178`；`docs/IQP_LENS_PAPER_SUITE_RESULTS.md:11-12,195`；`main.tex:298-300,422-425`。
**警告**：兩張表的絕對值不在同一個尺度上（不同資料集、不同影像大小與 channel 結構）。任何並排都只能比**排序**（IQP vs RBM vs DCGAN 誰贏哪個指標），不能比絕對數值（`facts.md §10`）。

---

## 8. 原論文的內部主張，我們重測的結果

### 8.1 Correlation–complexity map：permutation null（我們新增的方法學）

原論文只把每個資料集配一個位置，沒有配對照組。我們對透鏡與 D2 各加一個 **structure-matched permutation null**：保留每張影像/每個 snapshot 的完整強度直方圖，只把值在空間位置間隨機重排（`docs/IQP_LENS_MAP_RESULTS.md:25,52`）。

| 資料 | `I_QCLI` | `I_CCI` | permutation null QCLI | permutation null CCI | 讀法 |
|---|---|---|---|---|---|
| lens, 18-bit tuples, K=4,096 | 0.4715 | 0.8677 | **0.4154** | **0.8782** | 位置幾乎全部是編碼造成的；真正屬於空間結構的只有 QCLI +0.056、CCI −0.011 |
| turbulence D2, 18-bit, K=4,096 | 0.0232 | 0.9294 | 0.0113 | 0.9371 | 同一個問題命中原論文自己的選題依據 |
| lens, 10-bit crop pmf | 0.4731 | 0.4313 | **0.1782** | **0.9633** | **唯一兩個座標都明顯偏離對照的一列**，這是唯一站得住的「透鏡有 IQP-對齊結構」訊號 |

另外兩個方法學要點：plug-in CCI 的有限樣本 null 落在 **0.975–0.9998**，所以「高 CCI」本身沒有資訊，有資訊的只有「明顯低於 null」；QCLI 的 null 也不是 0 且隨樣本數 K 單調上升（n=18 在 K=4,096 是 0.008，K=4.1×10⁵ 已達 **0.405**），跨 K 直接比 QCLI 沒有意義（`docs/IQP_LENS_PAPER_SUITE_RESULTS.md:50`；`MAP:50`）。

**這個位置對生成結果沒有預測力**：地圖說透鏡比 D2 更靠右上，實測是互有勝負，round 3 的 pre-registered 判定是 failure（`MAP:21,76-78`）。

### 8.2 Fig. 3（support mismatch）：一個主張重現，一個方向相反

- 設定：n=16，140 個 ≤4-local 目標電路（exact QCLI 跨 0 到 0.408），用 ≤2-local、50 / 100 / 150 gate 的學習者擬合，exact MMD、Adam、5,000 步、lr 1e-4（`NUMBERS_PROVENANCE.md` App. C 表）。
- **主張 (b)「學習者 gate 越多，包絡整體下移」：重現。** 中位達成 MMD² `9.57e-3 → 2.15e-3 → 1.45e-3`（`NUMBERS_PROVENANCE.md` App. C 表）。
- **主張 (a)「目標 QCLI 越高，達成的 MMD 上包絡越低」：在可及範圍內方向相反。** 達成 MMD² 隨目標 exact QCLI **上升**，Spearman ρ = **+0.98 / +0.90 / +0.81**（`NUMBERS_PROVENANCE.md` App. C 表）。加了兩個控制（直接對 exact pmf 訓練、以及把 sample-trained 的 θ 拿去對 exact pmf 計分）給出幾乎相同的趨勢，排除有限樣本解釋（`docs/IQP_LENS_PAPER_SUITE_RESULTS.md:80`）。
- **但論文的前提在機制層面仍成立**：Proposition II.2 的 floor proxy 隨 QCLI 下降，ρ ≤ **−0.98**（`NUMBERS_PROVENANCE.md` App. C 表）。也就是說 support mismatch 這個機制是對的，只是在我們能構造的範圍內，達成的 MMD 被**可訓練性**主導而不是被 support mismatch 主導。
- **範圍限制要講清楚**：我們的目標族在 exact QCLI ≈ **0.41** 飽和（n=16、≤4-local 的天花板），而原論文畫到 ≈0.7，所以主張 (a) 被斷言的那一段我們根本構造不出來；用原論文自己的 10⁴-sample 估計量時，這 140 個目標全部落在 0–0.064，低於該估計量自己的 i.i.d.-uniform 地板 0.0696，**sample 軸在 n=16 完全不可測**（`NUMBERS_PROVENANCE.md` App. C 表）。

### 8.3 Fig. 4（QCLI 最大化 → CCI 跟著升）：低 QCLI 成立，高 QCLI 斷裂

- 低 QCLI 區（≤ 0.3）耦合明確重現，Spearman ρ = **+0.83 / +0.76 / +0.95**（n = 8 / 12 / 16）（`NUMBERS_PROVENANCE.md` App. D 表）。
- n=12、16 在 QCLI ≈ **0.38** 之後反轉，全域 ρ 變成 **−0.53 / −0.57**（`NUMBERS_PROVENANCE.md` App. D 表）。
- **反轉是定義逼出來的**：精確均勻分布的 QCLI 是 **0.982 / 0.998 / 1.000**（n=8/12/16）而 CCI 恰為 0（`NUMBERS_PROVENANCE.md` App. D 表）。原論文的圖停在 QCLI ≈ 0.7 以下，所以沒有畫到這一段。結論：**高 QCLI 未必代表可利用的結構，也可能只代表接近均勻分布**（`main.tex:404-407`）。

### 8.4 Turbulence 復現：未重現，且我們知道為什麼

| 事實 | 數字 | 出處 |
|---|---|---|
| Pre-registered 判定 | **未重現**：11-snapshot regime 下 RBM 與 DCGAN-1.2M 在**兩個指標上都**贏我們的 IQP | `TURB:12` |
| last-write-wins 慣例下的 PDF–JS | 我們 0.0322 ± 0.0006 / 0.0476 ± 0.0004 對論文 0.0238 / 0.0221（接近，但 RBM 在同一慣例下仍全面贏，判定不變） | `TURB:13-14` |
| 更深的問題 | 我們的 IQP 連自己的訓練 anchor 都只擬合到 corr **0.69**（抽樣場）/ **0.77**（exact 場） | `TURB:17` |
| 歸因（乾淨拆解） | stage 1 凍住 `θ_lat` 約 **−0.08**；5×10⁴ 抽樣 pmf 約 **−0.04**；把兩項都放開到 0.886，再換 kernel/K 最多 **0.892**，那是模型族加 closed-form MMD 目標的天花板 | `TURB:23-26` |
| Conv2D-MMD² 欄不可比 | 把 ground truth 在像素空間打散就得到 **0.704**，而論文 Table 2 整欄（0.896–1.472）都在這條線之上 | `TURB:20` |
| 評估設計無法支持論文主張 | held-out 的未見時間合成任務被**零參數線性內插**解到 corr **0.9998**，latent adaptation 的核心宣稱從頭到尾沒被它自己的指標量到 | `TURB:21` |

---

## 9. 誠實邊界與投稿時的措辭原則

| # | 邊界 | 我們怎麼寫 | 出處 |
|---|---|---|---|
| 1 | **exact classical simulation** | 全部用三次 FWHT 精確模擬，沒有 Monte-Carlo、沒有抽樣噪音；n=10 只有 1,024 個振幅。摘要與 Discussion 都明講 | `facts.md §9`；`iqp_exact.py:1-11`；`main.tex:74-75,310-312` |
| 2 | **single seed** | 所有 lens 與 turbulence 生成實驗都是 seed 0，沒有重複；用 DCGAN 在 scaling 上的 3 倍起伏當「這個尺度的噪音長什麼樣」的示範 | `main.tex:313-314`；`facts.md §9` |
| 3 | **不主張量子優勢** | 「hardness results 描述的是這個模型族在 scale 上的性質，不是我們的實例」。摘要最後一句就是這句 | `main.tex:74-75,310-312` |
| 4 | **只比 RBM 與 DCGAN** | 這是 <user> 指定的框架：只跟原論文自己跑過的 baseline 比，不引入原論文沒跑的古典模型 | `docs/IQP_LENS_PAPER_SUITE_RESULTS.md:4`；memory `feedback-paper-baselines-only.md` |
| 5 | **零學習與 PCA 控制在部分指標上更好** | 這件事**寫進 Discussion**，不藏：「Zero-learning and PCA controls match or exceed the IQP arms on some physical-quantity metrics」 | `main.tex:314-315`；完整數字在 `R3RES:18`（CTRL-ring physics 0.215 對最好的 IQP 0.513） |
| 6 | **KL 不進正文** | KL 需要完整模擬分布、無法從硬體 shot 估計，所以只放 App. E | `main.tex:427-428` |
| 7 | **flux-aligned 是 post-hoc** | 主表列出來但明確標註 post-hoc，因為它不在 round 3 的 pre-registration 裡 | `main.tex:154-155`；`docs/IQP_LENS_PAPER_SUITE_RESULTS.md:211` |
| 8 | **亮度尺度 S 是設計缺陷** | 所有 IQP arm 與 RBM 的 `field = q × S`，S 是外部純量不是學來的，導致這些 arm 的 `W1(flux)` 在 N=100 全部剛好是 0.808 ≈ √(2/π)（零變異但中心對的理論值） | `R3RES:19,216`；`docs/IQP_LENS_PAPER_SUITE_RESULTS.md:211` |
| 9 | **permutation null 削弱我們自己的賣點** | 摘要三個 caveat 之一就是「structure-matched permutation null 仍有 QCLI = 0.42」，主動寫在摘要而不是附錄 | `main.tex:72-74` |
| 10 | **兩套 pre-registered decision rule 都沒過** | round 3 判 failure、round 3b 判「不算有用」。論文正文不宣稱「IQP 是這份資料的好生成器」，只宣稱「學到的方向讓 IQP 第一次能生成並泛化」 | `R3RES:14-15,239-240` |
| 11 | **資料集引用（已確認 2026-09-08）** | DeepLense Model II（64×64、Euclid-like 觀測特徵、三類）的模擬方法出自 Alexander, Gleyzer, Reddy, Tidball, Toomey 2023, ApJ 954:28（arXiv:2112.12121）；Alexander et al. 2020（ApJ 893:15）是理想化的 Model I，只作任務起源引用；資料本身引 ML4SCI DeepLense GitHub repo。`refs.bib` 全部 18 條被引文獻已逐條上網核實（Codsi 為 Julien、Rajakumar 為 Joel、Alexander 2021 應為 2020）。 |

---

## 10. 檔案索引：每個結論對應到哪裡

### 10.1 文件

| 內容 | 檔案 |
|---|---|
| 原論文逐節摘要（含 Table 2 原始數字） | `docs/PAPER_2603.06440_SUMMARY.md` |
| 復刻規格書（每個設定的 `paper.txt:行號`，含「原文未給」清單） | `docs/IQP_REPLICATION_SPEC_2603.06440.md` |
| Ref [38] 方法筆記（MMD 閉式、三 bandwidth 規則） | `docs/REF38_IQPOPT_METHOD_NOTES.md` |
| 透鏡影像特性量測（換編碼與 latent 選法的全部依據） | `docs/LENS_IMAGE_CHARACTERISTICS.md` |
| Round 1 結果（18-bit tuple 失敗、QCLI 證偽） | `docs/IQP_LENS_PILOT_RESULTS.md` |
| Round 2 結果（強度尺度假象、協定不是槓桿） | `docs/IQP_LENS_ROUND2_RESULTS.md` |
| Round 3 / 3b pre-registration（協定、25 個 arm、P10–P23、decision rule） | `docs/IQP_LENS_ROUND3_PREREGISTRATION.md` |
| Round 3 / 3b 結果（合併主表、機制分析、限制） | `docs/IQP_LENS_ROUND3_RESULTS.md` |
| Correlation–complexity map 重算（15 列表 + permutation null） | `docs/IQP_LENS_MAP_RESULTS.md` |
| 論文級分析套件（Fig. 3 / Fig. 4 重測、scaling、pseudo-time、Q1–Q4） | `docs/IQP_LENS_PAPER_SUITE_RESULTS.md` |
| Turbulence 復現結果與 root-cause | `docs/IQP_TURBULENCE_REPLICATION_RESULTS.md`、`..._DIAGNOSIS.md` |
| 稿件、數字出處、圖片出處 | `paper/ml4ps2026/main.tex`、`facts.md`、`NUMBERS_PROVENANCE.md`、`figures/FIGURE_PROVENANCE.md` |

### 10.2 結果檔（全部在 `experiments/13_iqp_turbulence_replication/`）

| 內容 | 檔案 |
|---|---|
| Round 3 固定基底 14 arm × 2 N | `results_r3.jsonl` |
| Round 3b 學到方向 11 arm × 2 N | `results_r3b.jsonl` |
| Data scaling（N ∈ {10,30,50,70,90,100}） | `results_scaling.jsonl` |
| Per-class 生成（job 358382） | `results_class.jsonl` |
| Correlation–complexity map（15 列 + nulls） | `paper_suite/map/results_map.json` |
| Fig. 3 v2（140 targets + 420 fits，job 356221） | `outputs_controlled/results_fig3_v2.jsonl` |
| Fig. 4（60,016 列，job 356097） | `outputs_controlled/results_fig4.jsonl` |
| 機制測試（表達力、插值崩潰、學方向） | `mech_tests/mech_test/results.json`、`mech_tests/mech_test2/diag{2,3,4}.json` |
| 每個 run 的超參數與 S / Φ / z 統計 | `outputs_lens_r3b/<tag>/meta.json` |

### 10.3 程式

| 用途 | 檔案 |
|---|---|
| 18-bit tuple codec（round 1/2 與 turbulence 用） | `codec.py` |
| 10-bit crop pmf codec（round 3 用，含 `paste`） | `codec10.py` |
| 資料層：centroid alignment、crop、split 規則、S | `data_lens_r3.py` |
| Exact IQP 模擬器（三次 FWHT、kernel weights） | `iqp_exact.py` |
| 固定基底訓練（round 3） | `train_iqp_r3.py` |
| 學到方向訓練 `θ = θ⁰ + Φᵀz`、prior、z-penalty、held-out inference | `train_iqp_affine_r3.py` |
| 評分：physics、PDF–JS、Conv2D-MMD²、NN、flux-aligned 次評分 | `evaluate_r3.py`、`evaluate_r3b.py`、`metrics.py` |
| Baseline | `baseline_rbm_r3.py`、`baseline_dcgan_r3.py` |
| 控制組 | `controls_r3.py`、`controls_pca_r3.py` |
| Map 與 controlled study | `paper_suite/map/*.py`、`paper_suite/controlled/*.py` |
