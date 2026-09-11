# 我們的方法與原論文（arXiv:2603.06440）差在哪：資料、影像與協定的逐項說明（v12 最終版）

**用途**：給 Leo 與合作者的內部說明文件，不是論文。比論文詳細、比論文白話，但技術術語、表格 header、程式檔名一律用英文。
**對照的原論文**：Liu, Placidi, Brunner, Rinaldi（Quantinuum），*Toward Generative Quantum Utility via Correlation–Complexity Map*，arXiv:2603.06440 v2，2026-05-05（`docs/PAPER_2603.06440_SUMMARY.md:7-12`）。以下一律稱「原論文」。
**我們的投稿**：NeurIPS ML4PS 2026 workshop，稿件在 `paper/ml4ps2026/v12/main.tex`（**定稿，9 頁**：正文 4 頁、references、附錄 A–D）。中文全文譯稿在 `docs/IQP_LENS_PAPER_ZH_v12_full.md`。
**版本**：v12 最終版，2026-09-11。第 1 至 11 節沿用 v12 初版（`docs/IQP_LENS_METHOD_DIFFERENCES_ZH_v12.md`，2026-09-09，保留不動）；相對 v11 的差異列在第 11 節。
**這一版新增的四件事**（第 12 至 16 節）：論文最終主張收斂成什麼與為什麼；三輪獨立審查抓到並修正的 24 項問題；原論文 v1 與 v2 的逐項差異；52 筆參考文獻的線上核對結果。
**這一版的兩個系統性改動**：
1. **主 arm 從線性映射換成非線性潛變數映射** `θ_i = θ⁰ + g(z_i)`，`g` 是單隱藏層 tanh MLP，主 arm 為 `M = 2`、`H = 32`。來源：`seed_tables_f.md`（全套 M／H 掃描與消融）、`seed_tables_d.md`（與 12-qubit 線的對照）。
2. **生成時的總亮度改成通量抽樣（sampled flux）**：`S` 從 N 張訓練 crop 總亮度的 Gaussian KDE 抽出，不重訓任何模型。來源：`seed_tables_g.md`（兩種通量慣例下的全表）。

指標協定仍是 v11 的 `evaluate_v11.py`（同一批 200 張 held-out、同一個 `CONV_SEED`、同一組 bin 與 bandwidth），**沒有動過**，所以 v11 與 v12 的數字可以直接並排。

所有數字後面括號註明來源檔與表名，數字不憑印象。**單一數字沒有 seed 版本時會明講「seed 0，無 5-seed 版本」**。

---

## 1. 一頁摘要

1. 我們把原論文的兩件事（correlation–complexity map、IQP 生成模型加 latent adaptation）從它的 2D 湍流速度場搬到 DeepLense 模擬強重力透鏡影像。
2. **三個最大的改動是資料表示法、latent 參數化與亮度慣例**：原論文把每個樣本編成 18-bit `(x, y, intensity)` tuple 跑 18 qubits，我們改成把「質心對齊後的 32×32 crop 本身當成 1,024 格的 pmf」跑 10 qubits；原論文把 50 個 Walsh 相位當成固定的 latent block 依序調，v11 改成學一組線性方向 `θ = θ⁰ + Φᵀz`，**v12 再改成非線性映射 `θ = θ⁰ + g(z)`**；而生成時的總亮度 **v12 從單一訓練平均值改成從訓練通量分布抽樣**。每個改動都是被量測逼出來的。
3. **機制結論不變且被 5 seeds 支持**：固定基底的 sequential 協定在影像上把非 anchor 的訓練影像重建成 corr 0.4411 ± 0.0419（N=11）的雜訊（`seed_tables.md`「Section 3.2 diagnostics / Frozen-core reconstruction」），而換成非線性映射後 held-out projection corr 到 0.9773 ± 0.0006（M=2、N=100，`seed_tables_f.md` §1）。這一段是全文最穩的部分。
4. **生成品質在 v12 出現了第一個站得住的贏**：在通量抽樣之下，N=11 的 flux-aligned Conv2D-MMD² 由 MLP M=2 (flat MMD) 以 **0.02247 ± 0.001811** 領先，而且**沒有任何 arm 在 1 sd 內與它重疊**（`seed_tables_g.md` §1「N = 11 -- wins the seed spread does not support」）。raw MMD² 那一欄則是 DCGAN mean 最佳（0.05094 ± 0.01514），但五個 IQP／MLP arm 加 RBM 全部在 1 sd 內與它重疊，包括 MLP M=2 flat（0.05153 ± 0.01762）。
5. **N=100 時 DCGAN 仍然領先**：PDF-JS 與 raw MMD² 兩欄的領先連 1 sd 都推不翻，只有 flux-aligned 那欄由 MLP M=2 flat（0.01812 ± 0.00171）mean 最佳、與 DCGAN（0.02122 ± 0.004081）重疊（`seed_tables_g.md` §1）。
6. **我們不主張量子優勢，也不主張 IQP 全面贏**。可以主張的是：非線性潛變數映射讓 IQP 在影像上能生成並泛化；以及在 N=11、參數量約為 DCGAN 六分之一的條件下，flux-aligned 那一欄的領先是這份工作裡唯一撐得過 seed spread 的贏。

---

## 2. 原論文方法回顧（讀者不必回去翻原文）

### 2.1 IQP 電路與兩個指標

| 項目 | 定義 | 出處 |
|---|---|---|
| IQP circuit | `U(θ) = H^⊗n exp(i Σ_s θ_s Z_s) H^⊗n`，量測分布 `q_θ(x) = \|⟨x\|U(θ)\|0^n⟩\|²` | `docs/PAPER_2603.06440_SUMMARY.md:50,64` |
| 為什麼用它 | 古典高效模擬 IQP 取樣會使 polynomial hierarchy 崩塌，而 MMD loss 需要的 `⟨Z_s⟩` 仍可古典計算，所以能 train-on-classical / deploy-on-quantum | `docs/PAPER_2603.06440_SUMMARY.md:34-35`；`docs/REF38_IQPOPT_METHOD_NOTES.md:24` |
| **QCLI** | 對經驗 pmf 做 Walsh–Hadamard 得 `P(s)`，依 `\|s\| = k` 聚合成 order spectrum `m_k`，與 i.i.d. 公平位元的 binomial baseline `b_k = C(n,k)/2ⁿ` 取 Jensen–Shannon divergence | `docs/PAPER_2603.06440_SUMMARY.md:71-81` |
| **CCI** | `I_CCI = 1 − I_TC^tree / I_TC`，`I_TC` 是 total correlation，`I_TC^tree` 是最佳 pairwise Chow–Liu tree 抓到的那部分。值大 = 存在超越 pairwise 的多體相依 | `docs/PAPER_2603.06440_SUMMARY.md:98-103` |
| Proposition II.2 | 受限 IQP 架構的可調 Walsh 支撐集 `S_A` 之外的頻譜質量，會對 MMD loss 造成不可訓練消除的下界 `≥ C_A Σ_{S∉S_A} P(S)` | `docs/PAPER_2603.06440_SUMMARY.md:110` |
| 選題主張 | 高 QCLI 且高 CCI 的右上象限 = IQP 相容區，turbulence D1/D2 落在此區 | `docs/PAPER_2603.06440_SUMMARY.md:197` |

### 2.2 Algorithm 1（latent-parameter adaptation）

| 步驟 | 內容 | 出處 |
|---|---|---|
| 資料 | Khojasteh et al. 2022 的 D2，1000 個 2D snapshot，`(x, y, z)` 中 z 是 local velocity magnitude | `docs/IQP_REPLICATION_SPEC_2603.06440.md:31-34` |
| 編碼 | 每座標 N=6 bits，串接成 18-bit → **n = 18 qubits** | `docs/PAPER_2603.06440_SUMMARY.md:134` |
| 參數 | `θ = (θ_core, θ_lat)`，`d_lat = 50`，latent block = commuting Pauli string 依 Hamming weight 排序後的**前 50 個** | `docs/PAPER_2603.06440_SUMMARY.md:148`；`docs/IQP_REPLICATION_SPEC_2603.06440.md:54` |
| Stage 1 | 在 anchor `t=1` 上隨機初始化並**凍住** `θ_lat`，只訓 `θ_core`；每步用 500 個 model samples 估 MMD，共 `3×10⁴` 步，`N_t = 5×10⁴` 個訓練資料點 | `docs/PAPER_2603.06440_SUMMARY.md:142` |
| Stage 2 | 凍住 `θ_core`，對每個後續 anchor 只調 `θ_lat`，warm-start 自上一個 anchor，每個 snapshot 3000 步 | `docs/PAPER_2603.06440_SUMMARY.md:144` |
| 插值 | 對 latent 軌跡做 piecewise-linear 內插到未見時間 τ | `docs/PAPER_2603.06440_SUMMARY.md:145` |
| 合成 | 每次抽 `10⁵` 個 bitstring，解碼回 tuple 後聚合回網格 | `docs/PAPER_2603.06440_SUMMARY.md:146` |
| 訓練集 | 11 個 anchor，`t ∈ {1,100,…,1000}`；評估用另外 200 個均勻抽樣的未見 τ | `docs/PAPER_2603.06440_SUMMARY.md:148` |
| 參數量 | Table 2 的 IQP 標 31k / 63k。逆向核對：n=18 時 `Σ_{k≤6} C(18,k) = 31,180`、`Σ_{k≤7} = 63,004`，高度疑似對應 `\|s\|≤6` / `\|s\|≤7` 截斷，但原文未明講 | `docs/IQP_REPLICATION_SPEC_2603.06440.md:235`；`seed_tables_c.md` C6 |

### 2.3 目標函數、評估與 baseline

- **訓練用 MMD**：`L = Σ_s c_s (⟨Z_s⟩_θ − ⟨Z_s⟩_data)²`，Gaussian kernel。**原論文採用 IQPopt [38] 這個 library 的 kernel，但從未寫出它用的 σ**（`docs/IQP_REPLICATION_SPEC_2603.06440.md:232`）。我們稱為 paper kernel 的東西，實際上是 **IQPopt 的三 bandwidth 規則**在 n=10 重算的結果，`σ = {0.989, 1.729, 2.236}`（`v12/main.tex` App. E）。
- **評估指標**：PDF–JS（強度邊際直方圖的 JS divergence，B = 50 bins）與 Conv2D-MMD²（三層 Conv2D–ReLU stride 2 + global average pooling 的固定隨機 encoder，N = M = 200）（`docs/IQP_REPLICATION_SPEC_2603.06440.md:119,129-133`）。我們對這兩個指標的實作在 v11 有實質修改（metric protocol v11，見 §5.5），**v12 沒有再動**。
- **Baseline**：RBM（12k 參數，同樣的 18-bit 表示、同樣 11 個 snapshot）與 DCGAN（51k / 0.2M / 1.2M）（`docs/IQP_REPLICATION_SPEC_2603.06440.md:83-104`）。
- **原文未給、我們必須自己決定的**：turbulence 訓練的 optimizer 與 learning rate、訓練 kernel 的 σ、random seeds、**取樣後聚合回網格的規則**（`docs/IQP_REPLICATION_SPEC_2603.06440.md:225-234`）。最後一項在我們的復刻裡造成了實質差異，見 §5.4。

---

## 3. 我們的資料與它的特性

資料是 `<HOME>/mae-lensing/model_II.npz`（DeepLense Model II），三類 `axion` / `cdm` / `no_sub`，train 80,194 / val 8,910 張 64×64 float32。**本節全部是固定資料集上的一次性量測，不含隨機訓練，所以沒有 seed 版本**：split 由 `R3_SPLIT_SEED` 釘死且與任何 arm 的 `--seed` 解耦（`seed_tables.md` 開頭；`data_lens_r3.py:264-267`）。v12 沒有更動這一節的任何量測。

| # | 量測 | 數字 | 出處 | 為什麼重要 |
|---|---|---|---|---|
| 1 | per-image min-max 正規化 | 8,910 張**每一張**都恰有 1 個像素 = 0、1 個像素 = 1；per-image 總 flux 125.7 ± 28.2 | `LENS:30` | 跨影像的絕對亮度沒有意義，亮度尺度必須另外處理。這也是把 PDF–JS 的 bin 固定在 [0,1] 的依據（§5.5），也是 v12 要做通量抽樣的起點（§5.6） |
| 2 | raw 6-bit 強度量化的資訊量 | entropy 只有 1.42 bits，83.75% 的像素落在 level 0；quantile 量化可到 6.00 bits | `LENS:45-48` | 原論文的 z-register 在這批影像上有 4.6 bits 是浪費的 Hilbert space |
| 3 | 影像不是稀疏的 | 有效支撐 `1/‖p‖²` ≈ 331 / 4,096 像素（8.1%），但每張只有 1 個嚴格為 0 的像素 | `LENS:60` | 稀疏化技巧會失效，能用的是「能量集中」 |
| 4 | 影像流形維度 | 前 6 個 PC = 89.4% 變異；PC1/PC3（平移）合計 49.6%；PC2 = flux/大小；PC4/PC5 = ellipticity | `LENS:176,186-189,194` | 實質是 5 維：(x, y, flux, e1, e2)。**這是 M 的工作區間為什麼是 2 到 4 的資料側解釋**（§6.3） |
| 5 | class 訊號在哪 | PC6 之後才是子結構，只佔 3.0% 變異；nearest-centroid 分類只有 38.3%（chance 33.3%） | `LENS:190,205` | 以像素重建為目標的生成模型會系統性忽略掉唯一有科學價值的成分，所以我們不以 class-conditional 為主要目標 |
| 6 | 幾何 | Einstein 環半徑 r ≈ 8 px，rms 半徑 9.87–10.14 px，光心偏移 3.36–3.42 ± 1.5–1.7 px | `LENS:225-227` | 透鏡不置中，這是最大的單一變異來源 |
| 7 | 方位角結構是噪音 | split-half 相關只有 r = 0.45 / 0.51 / 0.69 | `LENS:229` | 不要為方位角加任何偏好方向的 generator |
| 8 | 18-bit tuple 編碼的代價 | `\|s\|≤6` 需要 31,180 個 generator，只涵蓋 79.9% 的 Walsh 能量 | `LENS:275` | 這就是原論文 31k 參數那一格的實際覆蓋率 |
| 9 | image-as-pmf 的效率 | 12-bit `\|s\|≤6` 用 2,510 個 generator 涵蓋 95.6%；10-bit（32×32）`\|s\|≤6` 用 848 個涵蓋 96.7% | `LENS:279,285` | 參數少 12.4 倍、覆蓋反而更好。這是換編碼的直接依據，也是 v12 實測 12-qubit 版本會更差的預期（§6.4） |
| 10 | bit 版面不重要 | Morton/Z-order 的 order spectrum 與 standard 逐點相同；Gray code 是 GF(2)-linear 可逆映射，Walsh 能量 multiset 差異為 0.0 | `LENS:249-250` | 版面優化在數學上不可能有幫助。**v12 實測 Gray code 確實是 null**（§6.4） |
| 11 | 原論文的 latent 選法不是最好的 | Hamming-first-50 的 class acc（LDA）0.488；「`\|s\|≤2` 內依方差排序」是 0.668，`\|s\|≤4` 是 0.717 | `LENS:136-139,144` | 「限制低階」是對的，錯的是在低階裡用固定字典序 |

---

## 4. 逐項差異表

每一列：原論文做法 / 我們的做法 / 為什麼 / 證據。標 **[v12 修正]** 的列是這一版改過的。

| # | 面向 | 原論文 | 我們 | 為什麼改 | 證據 |
|---|---|---|---|---|---|
| 1 | **資料與任務** | 1000 個湍流 snapshot，有時間軸，任務是內插到未見時間 τ | DeepLense Model II 靜態影像，三類 pooled，任務是從 latent prior 無條件生成 | 影像沒有時間軸，Eq. 31 的 latent 內插無物可插 | `v12/main.tex` §2 |
| 2 | **qubit 數與編碼** | 18-bit `(x, y, intensity)` tuple，n = 18 | 質心對齊 → 中心 32×32 crop → 1,024 格 pmf，`code = (kx<<5)\|ky`，**n = 10** | raw 強度軸只帶 1.42 bits；`\|s\|≤6` 在 18-bit 要 31,180 個 generator 卻只涵蓋 79.9%，10-bit crop pmf 用 848 個就到 96.7% | `LENS:275,285` |
| 3 | **generator 集合** | 依 `\|s\|` 截斷（推定 `\|s\|≤6` = 31,180；`\|s\|≤7` = 63,004） | 全部 1,024 個 Z-string（n=10 的完整 Walsh basis），exact simulation | n=10 時完整基底才 1,024 個參數，比原論文的截斷版還少 30 倍 | `codec10.py:13-15` |
| 4 | **latent 參數化** **[v12 修正]** | 固定的 50 個 Walsh 相位（Hamming 順序前 50 個） | **非線性映射 `θ_i = θ⁰ + g(z_i)`**，`g` 是 `M → H → 1,024` 的單隱藏層 tanh MLP（無輸出偏置），主 arm `M = 2`、`H = 32`；v11 的線性映射 `θ⁰ + Φᵀz` 保留為對照列 | 線性方向在 `‖z‖` 超過相干半徑後失效，M=6 就掉到 mean-image 線以下、M=8 崩潰；非線性映射把這個崩潰移除（§6.3） | `seed_tables_f.md` §1；`seed_tables_b.md` §2 |
| 5 | **訓練協定** | Sequential：先在單一 anchor 上訓 core，凍住後逐個 snapshot 調 latent | **Joint**：`θ⁰`、`g`（或 `Φ`）與全部 N 個 `z_i` 一起訓（Adam，固定 4,000 步） | 用原協定時，除了 anchor 以外的每張訓練影像只重建到 corr 0.4411 ± 0.0419（N=11）/ 0.4282 ± 0.0171（N=100），而 anchor 自己是 0.9985 ± 0.0004 | `seed_tables.md`「Frozen-core reconstruction」 |
| 6 | **目標函數** | 每步用 500 個 model sample 估 MMD | **閉式（exact）MMD**，兩個主變體：flat MMD（每個 Walsh 係數等權，可用硬體 shot 估）與 paper kernel（IQPopt 三 bandwidth 規則在 n=10 重算，σ = 0.989 / 1.729 / 2.236）；exact KL 只當診斷 | 10-qubit pmf 只有 1,024 個振幅，精確可算，沒有理由引入抽樣噪音 | `docs/REF38_IQPOPT_METHOD_NOTES.md:90,107` |
| 7 | **θ⁰ 的暖啟動** | 無此步驟 | `θ⁰` 先用 **1,000 步 exact-KL** 對平均 crop 暖啟動 | KL 不是硬體可估的，所以做了消融：換成 flat-MMD 暖啟動後**沒有一格差超過 1 sd**（§6.5） | `seed_tables_g.md` §5b |
| 8 | **KL 的地位** | 不使用 | 只當模擬專用診斷，不進正文主表 | KL 需要完整的模擬分布，無法從硬體 shot 估計 | `v12/main.tex` §2 |
| 9 | **prior 與生成方式** | 對 latent 軌跡做時間內插 | prior 為 resample：抽一個訓練 z 再加 Gaussian jitter | round-2 診斷顯示高維 Gaussian 在少量 anchor 上擬出來的質量幾乎全落在沒有真 latent 的區域 | `R3PRE:104-116`；`R3RES:235` |
| 10 | **亮度尺度 S** **[v12 修正]** | 隱含在 tuple 的 z 值裡 | 電路只輸出 `q(x,y)`，亮度是外部純量 `field = q × S`。**v12 起 `S` 逐張從 N 個訓練總亮度的 Gaussian KDE 抽出**（Silverman 頻寬），v11 的單一訓練平均值（N=11 為 112.3、N=100 為 124.9）改列到固定 S 對照附錄 | 固定 S 讓生成通量的變異數恰為 0，而真實 crop 的離散度約為平均值的四分之一；改成抽樣後 raw MMD² 降 22–39%，且不需重訓（§5.6） | `seed_tables_g.md` §1、§1c |
| 11 | **評估** | PDF–JS + Conv2D-MMD² | 同樣兩個指標但改用 **metric protocol v11**（固定 bin 網格、held-out 決定的 encoder 尺度與 bandwidth、200-vs-200 floor），另加 held-out projection correlation 與 flux-aligned Conv2D-MMD² | 舊協定讓每個 arm 用自己的 histogram 與 kernel 被評分，可被 generator 操弄（§5.5）。**v12 沒有再動這一層**，所以 v11 與 v12 的數字可以直接並排 | `v11/facts.md`「Addendum: metric protocol v11」 |
| 12 | **參數量的算法** | Table 2 直接標 31k / 63k | 用 **total-at-N**（shared + 每張影像的區塊）：**MLP 主 arm 33,910（N=11）/ 34,088（N=100）**；linear-map arm 5,164 / 5,520；fixed-basis 1,524 / 5,974；RBM 11,510 / 15,960；DCGAN generator 199,837 | v10 的 Params 欄混了三種慣例。**注意 v12 的主 arm 比 RBM 大約 3 倍**，見 §9 第 3 列 | `v12/main.tex` 表 2；`seed_tables_g.md` §1 |
| 13 | **Baseline 的訓練預算** | 固定步數 | **validation early stopping**：RBM 160k（N=11）/ 80k（N≥30）、DCGAN 5k–30k；IQP／MLP 仍固定 4,000 步 | 20k 步時 RBM 的 reconstruction MSE 還在降，固定 20k 等於讓 baseline 欠訓練；固定 4,000 步是對 IQP 側較嚴的慣例 | `seed_tables.md`「Baseline step counts chosen on the validation set」 |
| 14 | **Conditioning** | 時間 τ | 主結果全部 pooled（三類混合）；per-class 版本在附錄 | 三類的差異只佔 3.0% 的像素變異、nearest-centroid 只有 38.3% | `LENS:190,205` |

---

## 5. 影像處理流程逐步說明

### 5.1 每一步做什麼、為什麼、對應程式

| 步驟 | 做什麼 | 為什麼 | 程式與參數 |
|---|---|---|---|
| 1. Centroid alignment | 算強度加權質心 `(cy, cx)`，取整數位移，不做插值 | 光心偏移 3.4 ± 1.6 px 是最大的單一變異來源（`LENS:225-227`）；整數位移不會憑空造出或抹平任何光 | `data_lens_r3.py:72-95`（`align_crop`） |
| 2. 32×32 crop | 取以質心為中心的 32×32 視窗，記下角落 `(dy, dx)` | 對齊後的 crop 保留 99.4% 的光（最低 98.4%）；`\|s\|≤6` 覆蓋 96.7% 比 64×64 的 95.6% 還高 | `R3PRE:9-11`；`LENS:285` |
| 3. pmf 正規化與編碼 | `p(kx,ky) = crop / sum(crop)`，`code = (kx<<5)\|ky` | 把影像本身當機率分布，避開 tuple 編碼的高階代價 | `codec10.py:9-10,19-21,54-68` |
| 4. 訓練 / held-out 分割 | 對 pooled train_x 做一個固定 permutation：前綴 N 張當訓練（N=11 是 N=100 的子集），held-out = val_x 固定 permutation 的前 200 張。split seed 固定為 0，與任何 arm 的 `--seed` 解耦 | 資料不可以隨訓練 seed 移動；每個 seed 的 held-out 總亮度都是同一個值（123.2） | `data_lens_r3.py:264-267`；`seed_tables_g.md` §1d |
| 5. 生成 **[v12 修正]** | 從 prior 抽 z → `θ = θ⁰ + g(z)` → 三次 FWHT 得 exact `q` → **`S` 從訓練通量 KDE 抽樣** → `crop = q × S` | 見 §5.6。訓練總亮度 N=11 為 112.34 ± 26.96、N=100 為 124.92 ± 28.91，held-out 為 123.20 ± 27.80 | `codec10.py:76-86`；`evaluate_v12_flux.py`；`seed_tables_g.md` §1d |
| 6. 貼回 64×64 | 把 32×32 crop 貼進全黑的 64×64 框，角落 `(dy, dx)` 從訓練影像的偏移分布中抽樣（有放回） | 評分必須回到原始影像尺度；偏移分布是資料的真實性質（PC1+PC3 = 49.6% 變異） | `codec10.py:123-146`；`data_lens_r3.py:315-320` |
| 7. 顯示 | 全部圖片用 `magma`、線性、`vmin = 0`、`vmax = 1` | 共用 0–1 尺度才誠實 | `figures/FIGURE_PROVENANCE.md:26,34` |
| 8. 超出 [0,1] 的像素 | 兩邊都 clip 到資料範圍 [0,1]，並逐 run 記錄 `gen_clip_frac` 與 `gen_max` | v10 讓這些像素決定整個比較的 histogram 寬度與 encoder 輸入尺度。**通量抽樣會讓 `gen_max` 升高**（固定 S 時 MLP 的 `gen_max` 約 1.4–1.5，抽樣後約 2.0–2.2），clip fraction 仍在 1e-3 量級 | `seed_tables_e.md` §「N = 11」的 `gen_max` / `clip frac` 欄 |

### 5.2 Round 1 為什麼失敗（18-bit tuple 編碼）

- 照 pre-registration 字面套用原論文編碼：`(kx, ky, kz)`，`kz = floor(64 × intensity)`。跑前就量到 83.6–84.0% 的像素落在 intensity bin 0（`docs/IQP_LENS_PILOT_RESULTS.md:24`）。
- 結果：IQP 是所有沒崩潰的 arm 裡**最差的**，NN-holdout 5.8 / 7.5 / 7.7（`PILOT:12-13`）。
- **同時，這批資料的 QCLI 高得漂亮**：per-image `I_QCLI` = 0.4697 / 0.4693 / 0.4755，配對的 uniform null 只有 0.0077（`PILOT:16`）。**「高 QCLI ⇒ 適合 IQP」在這裡被直接證偽。**
- sqrt 量化確實有效（PDF–JS 從 0.29–0.42 掉到 0.050–0.060）但不夠（`PILOT:14`）。

### 5.3 Round 2 為什麼失敗（強度尺度假象）

- Round 2 換成 12-bit image-as-pmf。表面上 MMD² 從 0.66–1.27 掉到 0.42–0.50（`R2:11`）。
- **但修好的是亮度尺度，不是空間結構**：pmf arm 與 class-mean 的 per-image 相關係數只有 −0.006 到 +0.041，弧/背景對比 0.98–1.15，而真影像是 17.6–19.0（`R2:11`）。
- 把每個 arm 的平均總亮度事後對齊 held-out 再重算，round-2 相對 round-1 的全部 MMD² 增益消失。這個教訓直接生出 flux-aligned 次評分欄，**也是 v12 為什麼要把「通量」與「形狀」這兩件事分乾淨的原因**（§5.6、§9 第 1 列）。

### 5.4 聚合規則（last-write vs mean）這件事為什麼重要

原論文從未定義「同一格點落到多筆樣本時 z 怎麼合併」（`docs/IQP_REPLICATION_SPEC_2603.06440.md:234`）。我們在 turbulence 復刻裡兩種都做了：primary 用 mean、secondary 用 last-write-wins。**在 last-write-wins 慣例下，原論文自己的 IQP PDF–JS 數字是對得上的**（我們 0.0324 ± 0.0011 / 0.0473 ± 0.0005 對論文的 0.0238 / 0.0221），mean 慣例下就差一個數量級（`seed_tables_c.md` C1、C2）。lens 這條線因為改用 exact pmf 重建，完全沒有這個問題。

### 5.5 評估協定 v11（v11 的方法學改動，v12 沒有再動）

**問題**：v10 的兩個指標都讓被評分的 arm 部分決定自己的尺規。四個漏洞與其後果（`v11/facts.md`「What the v10 protocol let a generator do to its own yardstick」）：PDF–JS 的 50 個 bin 橫跨 real ∪ generated 的聯合 min/max（把 DCGAN 輸出的任一個像素改成 4.0，它的 PDF–JS 就除以約 13）；Conv2D encoder 的輸入正規化用同一組聯合 min/max；Conv2D kernel 的 σ 用混合特徵的 median heuristic（每個 arm 用不同的 kernel 被評分，排名可被翻轉）；floor 是 100-vs-100 而每個 model row 是 200-vs-200（floor 被高估約 2 倍）。

**v11 的定義**：PDF–JS 兩邊都 clip 到 [0,1]、50 個等寬 bin 固定在 [0,1]；Conv2D-MMD² 的 encoder 不變（三層 stride-2 conv 1→16→32→64 加 global average pooling，`CONV_SEED = 0`），但輸入範圍取自 held-out、Gaussian bandwidth 用 held-out 特徵的 median heuristic 算一次並由全表共用（pooled σ = 0.005547 raw、0.002558 flux-aligned），N = M = 200；floor 為 20 組 200-vs-200 的 real-vs-real 抽樣。

**v12 的立場**：這一層完全不動，`evaluate_v11.py` 是 `seed_tables_d/d2/e/f/g` 全部五份表的唯一評分器。這是 v11 與 v12 的數字可以逐格並排的前提（`seed_tables_g.md` §6 的 fixed-S 重算與 v11 published cell 最大相對差為 **0.00042**，純粹是四位有效數字的顯示捨入）。

### 5.6 通量抽樣（v12 最大的解碼側改動）

**規則**：`S_i` 從同樣那 N 個訓練 crop 總亮度的 Gaussian KDE 抽出，Silverman 頻寬 `h = 0.9 min(sd, IQR/1.34) N^(-1/5)`，clip 到正值。`h` 在 N=11 為 14.64、N=100 為 10.36（`seed_tables_g.md` §1d）。

**不需要重訓**：每個 arm 的 generator 輸出的就是正規化的 pmf 再乘 S，所以 `evaluate_v12_flux.py` 直接從每次 run 存下的 `crops/` 精確還原 pmf、重新給通量、再用該 run 自己存的 offset 重貼。它的 `--check` 驗證固定 S 那一路能重現已發表的 frame：**全部 325 次 run 的 `max |saved - rebuilt| = 0`**（`seed_tables_g.md` 開頭）。另外，part E 曾付 GPU 時間重訓八個 `--flux resample` 的 configuration，事後 KDE 重新給通量的結果與那八個重訓 arm **在 seed spread 內全部吻合**（`seed_tables_g.md` §5a-bis），這是「事後替換 = 同一個運算」的端到端證據。

**效果**：N=11 時每個 IQP／MLP arm 的 raw Conv2D-MMD² 降 **22–39%**（配對 delta，例如 MLP M=2 從 0.08508 ± 0.001855 到 0.05153 ± 0.01762，delta −0.03355 ± 0.01755）；N=100 降 **33–65%**（`seed_tables_g.md` §1c 的 delta 表與 §1 的兩欄相除）。

> **範圍的小警告**：`seed_tables_g.md` §1c 的警語方塊寫的是「At N = 11 it improves every IQP/MLP arm's raw MMD² by 29-37%」，那個 29–37% 不含 Alg. 1（−22%）與兩個 MLP row（−39% / −39%）。**涵蓋全部六個 arm 的正確範圍是 22–39%**，`v12/main.tex` §3.3 引的就是這一個。N=100 同理：v11 就有的四個 IQP arm 是 33–62%，加上兩個新的 MLP arm（−65% / −57%）之後是 33–65%。

**flux-aligned 欄恆等不變**：`MMD2_al` 在評分前就把每張 frame 縮放到 held-out 平均總亮度，等於把 per-image 通量除回去，所以通量慣例對它**在構造上**不可見，不是近似不可見：325 對配對 run 中最大的 `|resampled − fixed|` 是 **6.44e-09**（`seed_tables_g.md` 開頭的警語）。因此 raw 與 aligned 兩欄量的是「形狀加通量」與「形狀」，不是兩個獨立的贏。

**規則對 RBM 不中性**：同樣的 KDE、同樣的訓練總亮度，卻讓 RBM 在 N=11 的 raw MMD² **變差 20%**（0.06537 ± 0.008124 → 0.07866 ± 0.01932，配對 delta +0.0133 ± 0.01696），而在 N=100 改善 13%（−0.007603 ± 0.007703）。所以 N=11 那一格 MLP 對 RBM 的 raw 差距，有一部分來自 baseline 自己往下掉（`seed_tables_g.md` §1c 的警語）。這一點必須寫進論文，v12 正文 §3.3 已經寫了。

**兩個代價**：(a) 每次 run 多帶一個 200 次抽樣的 Monte-Carlo 通量樣本，raw MMD² 的 seed sd 從 0.0019 放大到 0.0176（MLP M=2、N=11）；(b) 抽樣只修好生成通量分布的**形狀**、修不好它的**位置**，因為 N=11 的十一張訓練影像平均比 held-out 暗 9%（112.34 對 123.20），這就是 raw 那一欄停在 DCGAN 旁邊而不是靠近 floor 的原因（`seed_tables_g.md` §1d）。

---

## 6. 為什麼原協定在影像上失效，而「非線性映射」有效

### 6.1 三個跑前做完的機制測試（因果鏈，全部 5 seeds）

| 測試 | 5-seed 結果 | 讀法 | 出處 |
|---|---|---|---|
| (i) 單張 crop 的表達力 | 一張對齊後的 crop 被 1,024 相位的 10-qubit IQP 幾乎精確擬合：KL corr **0.9999 ± 1.05e-05**、flat MMD **0.999966 ± 5.749e-06**、paper kernel **0.9964 ± 0.0008** | 表達力與 kernel 選擇都不是瓶頸 | `seed_tables.md`「Single-crop full-parameter fit」 |
| (ii) 固定基底的插值 | 沿論文的 Hamming latent 在兩張影像之間走，中途 `max(corrA, corrB)` 掉到 0.233、ringiness 掉到 0.353，而像素 cross-fade 對照在每個 t 上 corr 都是 1.0（seed 0，無 5-seed 版本） | latent 空間裡兩個「好點」之間全是壞點，所以從 prior 抽樣得到斑塊 | `R3RES:207` |
| (ii-b) ancilla 能不能救 | ancilla 邊際被驗證為恰好等權重的兩個 hologram 非相干混合（每支 mass 0.500000000000，殘差 ≤ 2.6e-18；解析結果，無 seed） | 沒有「混合角」可調，ancilla 加不了東西 | `R3RES:207` |
| (iii) 凍住 core 之後的其他訓練影像 | anchor 自己 **0.9985 ± 0.0004**，其他訓練影像 **0.4411 ± 0.0419**（N=11）/ **0.4282 ± 0.0171**（N=100） | 原協定不是實作問題，是協定本身在影像上的行為 | `seed_tables.md`「Frozen-core reconstruction」 |
| (iv) 學方向（線性） | 20 張 crop 的 controlled sweep：M=2 held-out projection corr **0.9060 ± 0.0021**、M=4 **0.8577 ± 0.0559**；配對的 random-direction 控制是 **0.7458 ± 0.0005**、不含 latent 的 M=0 是 **0.7439 ± 0.0000** | 增益來自「學到」方向，但線性映射的可用寬度只有 M ≤ 4 | `seed_tables_b.md` §2 |

**機制的一句話說明**：固定的 Walsh 方向在相位空間裡作用成 XOR-shift 干涉樣式，它把兩張影像的像素 cross-fade 起來，做不到「把環移動或拉伸」這種幾何操作。

### 6.2 為什麼要換成非線性映射：相干半徑 **[v12 核心論點]**

**直觀解釋**：這個電路的輸出是 `q_θ = |W e^{iθ}|² / 2ⁿ`，也就是把 1,024 個相位一次做完 Walsh–Hadamard 之後取模平方。它是一個**干涉映射**：影像的每一個像素值都是全部 1,024 個相位共同干涉的結果，沒有哪個相位只負責哪一塊。因此 `θ` 的一個小位移會同調地在整張影像上重新分配光，這是好事；但一旦位移大到讓 `e^{iθ}` 的相位繞過去，重建就不是「移動」而是「打散」，因為不同的相位分量開始以不同的速率繞圈。這個「還能同調」的半徑就是相干半徑。

**量測**：線性映射在 `‖z‖` 超過約 0.05 之後開始失效。實測的 held-out projection corr 對 M（20 張 crop 的 controlled sweep，`seed_tables_b.md` §2）：

| M | held-out proj. corr（5 seeds） | train corr | 讀法 |
|---|---|---|---|
| 0（無 latent） | 0.7439 ± 0.0000 | 0.7861 ± 0.0000 | 與 mean training image 同值 |
| **2** | **0.9060 ± 0.0021** | 0.9456 ± 0.0009 | 線性映射的最佳工作點 |
| 4 | 0.8577 ± 0.0559 | 0.9325 ± 0.0217 | mean 略低，sd 大一個數量級 |
| 6 | 0.7378 ± 0.0672 | 0.9237 ± 0.0201 | **掉到 mean-image 線（0.7439）以下** |
| 8 | 0.2184 ± 0.1213 | 0.9486 ± 0.0009 | 崩潰 |
| 16 | 0.1457 ± 0.0048 | 0.9965 ± 0.0002 | 崩潰（train corr 反而最高，是過擬合到相位空間的碎片） |

train corr 一路都很高、held-out corr 卻崩，這正是「相位繞過去之後只剩對訓練點的記憶」的指紋。

**非線性映射怎麼解掉它**：`g` 可以在 `z` 的不同區域用不同的方向與不同的步幅，所以它能在保持每一步都落在相干半徑內的前提下走得更遠。實測 MLP arm 的 `‖z‖` 反而比線性 arm 大（M=2、N=11：MLP 0.1010 ± 0.0044 對 affine M=4 的 0.0257 ± 0.0016，`seed_tables_d2.md` §3），而重建沒有崩，因為 `g(z)` 的實際位移不是 `‖z‖` 線性放大。非線性程度也確實被用上了：best-affine-fit 的相對殘差（`nonlin`）在 M=2、N=11 是 **0.6961 ± 0.0254**、N=100 是 **0.7322 ± 0.0315**（`seed_tables_f.md` §1），遠離 0，所以這不是一個塌回線性映射的 null test。

**結果**：非線性映射把 M=6 到 M=16 的崩潰整段移除。`seed_tables_f.md` §1 的 N=11、H=32 那一區塊，held-out proj corr 在 M=1..16 全程維持 0.8953 到 0.9076；N=100 全程維持 0.9120 到 0.9855。**但它沒有讓「有用的寬度」變大**：生成指標到 M=4 是平的，從 M=6 開始退化、M=16 徹底壞掉（見 §7.2 的 M 掃描表）。所以主 arm 報在 **M=2**。

**參數量核對**（避免日後對不上）：`params = 1,024（θ⁰） + [M×H + H]（輸入層） + H×1,024（輸出層，無偏置） + N×M（每張影像的 z）`。M=2、H=32、N=11 → 1024 + 64 + 32 + 32768 + 22 = **33,910** ✓；N=100 → 34,088 ✓。所以 `g` 的形狀是 `M → 32 → 1024`，主 arm 是 `2 → 32 → 1024`，`seed_tables_d2.md` 開頭寫的 `4 -> H -> 1024` 是 part D 那個 M=4 版本的形狀。

### 6.3 正式跑的數字（M=2、H=32、flat MMD）

| 量 | N=11 | N=100 | 出處 |
|---|---|---|---|
| held-out projection corr | 0.9016 ± 0.0012 | **0.9773 ± 0.0006** | `seed_tables_f.md` §1 |
| 同一格的 v11 線性 arm（M=4） | 0.9102 ± 0.0026 | 0.9684 ± 0.0004 | 同上 §0 |
| MLP M=4、H=32 | 0.9076 ± 0.0014 | **0.9840 ± 0.0003** | 同上 §1 |
| 「從訓練影像挑最近那張」的參考上限 | 0.9339（20 張 split） | 0.970 | `v12/main.tex` §3.2；`meta.json`（seed 0） |

讀法：N=100 時非線性映射把 projection corr 從 0.9684 推到 0.9773（M=2）或 0.9840（M=4），兩者都與線性 arm 分離超過 1 sd；N=11 時 M=2 反而略低於線性 arm 的 0.9102（分離超過 1 sd，方向對 MLP 不利），這一點要照實寫，論文正文報的是 M=2 的 0.9013 ± 0.0012（`v12/main.tex` §3.2，來自主表那一批 run 而非 20 張 controlled split）。

### 6.4 其他試過但沒有用的槓桿 **[v12 新增]**

這一節只給結論與數字，細節在 `seed_tables_d.md`（12-qubit）與 `seed_tables_e.md`／`seed_tables_g.md` §5a（其餘四項）。

| 嘗試 | 做了什麼 | 5-seed 結果 | 判定 | 出處 |
|---|---|---|---|---|
| **12-qubit 整張影像** | 不裁切，整張 64×64 降到 4,096 格 code，n = 12（`train_iqp_affine_n12.py`、`data_lens_n12.py`），線性 M=4 | N=11：PDF-JS 0.0433 ± 0.0300、raw MMD² 0.2328 ± 0.0879、aligned 0.1841 ± 0.1128、proj corr 0.8959 ± 0.0105。N=100：0.0921 ± 0.0711 / 0.2913 ± 0.1993 / 0.4303 ± 0.3084 / 0.7565 ± 0.1925 | **明顯更差，四個指標全部與 10-qubit 線性 arm 分離超過 1 sd**。而且 N=100 有 **2 of 5 seeds 整個塌掉**（train corr 0.004 與 0.008），所以那一格根本不該用 mean ± sd 讀。把 z 初始化改成 10-qubit arm 的尺度（0.05）更差（PDF-JS 0.1115 ± 0.0053）。與 §3 第 9 列的預期一致：格子變細只是把能量推到更高的 Walsh 階 | `seed_tables_d.md` §1、§2、§3b |
| **8-qubit（16×16）低解析度** | `lowres8`，把 crop 降到 16×16、n = 8 | N=11 最好的一組（`lowres8, mlpH8, M=2, spatial+flat, flux=resample`）aligned 0.04906 ± 0.002757，對照 10-qubit MLP M=2 的 0.02247 ± 0.001811 | **更差約 2 倍**，即使參數量只有 2,350。它的 encode→decode 天花板（真實 crop 進去再出來）只有 0.990，本身就把上限壓住了 | `seed_tables_e.md`「N = 11」 |
| **Gray code 位元版面** | `gray10`，把 `code = (kx<<5)\|ky` 換成 Gray code | raw MMD² 0.05278 → 0.05288（+0%）、aligned 0.02737 → 0.02793（+2%） | **null，在 1 sd 內**。與 §3 第 10 列的解析預期完全一致（Gray code 是 GF(2)-linear 可逆映射，Walsh 能量 multiset 差異為 0.0），所以這是一個成功的 sanity check 而不是失敗的實驗 | `seed_tables_g.md` §5a |
| **Walsh 低通懲罰** | `--mu 0.01 --walsh-k0 3`，對 `\|s\| > 3` 的 Walsh 能量加相對懲罰 | raw MMD² 0.05316 → 0.05525（+4%）、aligned 0.02825 → 0.02773（−2%） | **null，在 1 sd 內**。方向還互相矛盾（raw 變差、aligned 變好），所以不能宣稱任何效果 | `seed_tables_g.md` §5a |
| **像素空間 kernel** | `--objective spatial+flat`，在 flat MMD 之外加一個像素空間（而非 Walsh 空間）的 kernel 項，掃了權重 ×0.25 / ×1 / ×4 / ×16 | raw MMD² 0.05278 → 0.0508（−4%），但 aligned 0.02737 → 0.02921（**+7%**）。純 spatial（不加 flat）更糟：aligned 0.1229 ± 0.01652 | **是交換不是增益**：它買到 raw 那一欄，賣掉 aligned 那一欄。而 aligned 才是 v12 唯一站得住的贏，所以不採用 | `seed_tables_g.md` §5a；`seed_tables_e.md`「N = 11」 |

**一句話**：五項裡有三項（Gray code、Walsh 低通、8-qubit／12-qubit 換編碼）都證實了 §3 那些「跑前量測」的預測，一項（像素空間 kernel）是 trade-off，沒有一項提供新的槓桿。**唯一真正動得了數字的只有 latent 參數化與通量慣例這兩件事。**

### 6.5 無電路的 softmax 對照 **[v12 新增，僅內部，不進論文]**

**問題**：非線性 latent arm 贏過 v11 的線性 IQP arm，有多少是「非線性 latent」的功勞、有多少是「量子 generator」的功勞？把 generator 刪掉、其他全部不動，就能分開。

**設計**（`seed_tables_d2.md`、`train_mlpctrl.py`）：把 `q = |⟨x|U(θ)|0⟩|²` 換成 `q = softmax(θ)`，`θ = θ⁰ + g(z)` 完全相同，latent、z penalty、prior、projection、S、paste、步數、Adam 全部相同，目標函數同樣是從 pmf 的 Walsh 係數算的 flat MMD（另有一個 KL 變體）。評分同樣是 `evaluate_v11.py` 對同樣 200 張 held-out。

**結果**（N=11 / N=100，H=32 flat-MMD 對照組 vs 同 latent 的 `IQP-mlpH32-M4`）：

| 指標 | 對照（無電路） N=11 | 量子 N=11 | 對照 N=100 | 量子 N=100 |
|---|---|---|---|---|
| PDF-JS | 0.0018 ± 0.0001 | **0.0009 ± 0.0000** | 0.0017 ± 0.0002 | 0.0018 ± 0.0001 |
| Conv-MMD² | 0.1009 ± 0.0017 | **0.0866 ± 0.0024** | 0.0778 ± 0.0031 | **0.0700 ± 0.0031** |
| Conv-MMD²（flux-aligned） | 0.0328 ± 0.0018 | **0.0249 ± 0.0019** | 0.0372 ± 0.0038 | **0.0193 ± 0.0037** |
| held-out proj. corr | **0.9276 ± 0.0058** | 0.9076 ± 0.0015 | 0.9590 ± 0.0045 | **0.9840 ± 0.0003** |

**總計**：32 個（對照 arm、N、指標）比較裡，circuit-free 對照有 **3 個分離的贏**、**28 個分離的輸**、1 個在 1 sd 內無法區分（`seed_tables_d2.md` §4）。三個贏全部集中在 N=11 的 projection correlation（H=32 的 flat-MMD 與 KL 兩組），也就是「純粹把 held-out 影像壓進 latent 再重建」這一項；只要看生成指標，電路每一格都贏。

**但書（必須跟著這段一起講）**：**對照組的超參數沒有為它自己另外調過**。它沿用的是為 IQP arm 挑出來的 z penalty、`--g-input-scale 20`、學習率與 4,000 步。softmax 的輸出對 `θ` 的尺度敏感度與電路完全不同（電路是相位、對 `θ` 週期，softmax 是 logit、對 `θ` 單調），所以一組對電路合適的 `θ` 尺度未必對 softmax 合適；它的 `‖z‖`（N=11 為 0.7010 ± 0.0198）比電路 arm 大一個數量級就是這件事的跡象。**因此這只是一個內部 sanity check：它足以說明「非線性 latent 不是全部的功勞」，不足以宣稱 IQP generator 比一個調好的古典 softmax 生成器好。** 這也是它不進論文的原因；另一個原因是 memory `feedback-paper-baselines-only.md`：本文不與原論文未使用的古典模型比較。

---

## 7. 結果對照表

### 7.1 主表（v11 指標協定，通量抽樣，5 seeds，mean ± sd）

出處：`seed_tables_g.md` §1「Table 2 (v12)」的 resampled-S 欄。Params 為 `seed_tables_g.md` 的慣例（shared core 或 generator only）；論文表 2 用的是 total-at-N 慣例（IQP fixed-basis 1,524 / 5,974、RBM 11,510 / 15,960），兩者只差在「有沒有把每張影像的區塊算進去」。

**N = 11**

| Model | Params | PDF-JS | MMD² | MMD²(flux-al.) |
|---|---|---|---|---|
| IQP, paper protocol (Alg. 1) | 1,024 | 0.00655 ± 0.0007681 | 0.1477 ± 0.01552 | 0.2251 ± 0.02187 |
| IQP, fixed basis, joint | 1,024 | 0.001773 ± 0.0003711 | 0.08733 ± 0.01363 | 0.1253 ± 0.01326 |
| IQP, linear map M=4 (flat MMD) | 5,164 | 0.002834 ± 0.0001348 | 0.05643 ± 0.01939 | 0.02737 ± 0.002194 |
| IQP, linear map M=4 (paper kernel) | 5,164 | 0.001034 ± 4.555e-05 | 0.06266 ± 0.01543 | 0.07432 ± 0.006323 |
| **IQP, MLP map M=2 (flat MMD)** | 33,910 | 0.001176 ± 7.025e-05 | 0.05153 ± 0.01762 | **0.02247 ± 0.001811** |
| **IQP, MLP map M=2 (paper kernel)** | 33,910 | **0.0007499 ± 6.542e-05** | 0.05363 ± 0.01547 | 0.0514 ± 0.003457 |
| RBM | 11,010 | 0.001887 ± 0.0009503 | 0.07866 ± 0.01932 | 0.1106 ± 0.03881 |
| DCGAN *(learns own flux)* | 199,837 | 0.001106 ± 0.0005576 | 0.05094 ± 0.01514 | 0.03883 ± 0.0104 |
| *Floor (200 v 200, 20 draws)* | n/a | *4.079e-05 ± 2.057e-05* | *0.004806 ± 0.002783* | *0.003979 ± 0.00221* |

**N = 100**

| Model | Params | PDF-JS | MMD² | MMD²(flux-al.) |
|---|---|---|---|---|
| IQP, paper protocol (Alg. 1) | 1,024 | 0.007687 ± 0.0004039 | 0.1465 ± 0.003232 | 0.2659 ± 0.006753 |
| IQP, fixed basis, joint | 1,024 | 0.001113 ± 0.000154 | 0.05381 ± 0.003639 | 0.08521 ± 0.0149 |
| IQP, linear map M=4 (flat MMD) | 5,520 | 0.0026 ± 0.0002413 | 0.02752 ± 0.005375 | 0.02551 ± 0.004413 |
| IQP, linear map M=4 (paper kernel) | 5,520 | 0.001462 ± 0.000179 | 0.05894 ± 0.003623 | 0.09991 ± 0.007786 |
| **IQP, MLP map M=2 (flat MMD)** | 34,088 | 0.001705 ± 0.0001888 | 0.02421 ± 0.00496 | **0.01812 ± 0.00171** |
| **IQP, MLP map M=2 (paper kernel)** | 34,088 | 0.0009874 ± 3.601e-05 | 0.04215 ± 0.004002 | 0.06607 ± 0.005518 |
| RBM | 11,010 | 0.001105 ± 0.0001995 | 0.05105 ± 0.007834 | 0.08067 ± 0.02205 |
| DCGAN *(learns own flux)* | 199,837 | **0.0005534 ± 0.0001211** | **0.02225 ± 0.003118** | 0.02122 ± 0.004081 |
| *Floor (200 v 200, 20 draws)* | n/a | *4.079e-05 ± 2.057e-05* | *0.004806 ± 0.002783* | *0.003979 ± 0.00221* |

**哪些「贏」撐得住 seed spread**（`seed_tables_g.md` §1「wins the seed spread does not support」，resampled-S 區塊）：

| N | metric | mean 最好的 arm | 1 sd 內與它重疊的 |
|---|---|---|---|
| 11 | PDF-JS | IQP, MLP map M=2 (paper kernel) | DCGAN |
| 11 | MMD² | DCGAN | MLP M=2 (flat)、MLP M=2 (paper)、linear M=4 (flat)、linear M=4 (paper)、RBM |
| 11 | MMD²(flux-al.) | IQP, MLP map M=2 (flat MMD) | **無，領先成立** |
| 100 | PDF-JS | DCGAN | **無，領先成立** |
| 100 | MMD² | DCGAN | MLP M=2 (flat)、linear M=4 (flat) |
| 100 | MMD²(flux-al.) | IQP, MLP map M=2 (flat MMD) | DCGAN |

**與 DCGAN／RBM 差幾個 sd**（`v12/main.tex` §3.3；margin 以 baseline 自己的 sd 為單位，正號代表 MLP arm 較好）：

| N | 對象 | PDF-JS | raw MMD² | MMD²(flux-al.) |
|---|---|---|---|---|
| 11 | vs DCGAN | −0.13 | −0.04 | **+1.57** |
| 11 | vs RBM | +0.75 | +1.40 | +2.27 |
| 100 | vs DCGAN | 六個 gap 中最小者 −9.51（DCGAN 領先） | | |

讀法：**N=11 是唯一有站得住的贏的地方，而且只有一欄**（flux-aligned，領先 DCGAN 1.57 個 DCGAN sd，且沒有任何 arm 在 1 sd 內與它重疊）。PDF-JS 與 raw MMD² 在 N=11 與 DCGAN 打平（−0.13 與 −0.04 個 sd，方向上 DCGAN 略前但完全在噪音內）。對 RBM 則是三欄全贏且三欄的 gap 都超過 RBM 的 sd。N=100 時 DCGAN 在 PDF-JS 與 raw MMD² 兩欄的領先連 1 sd 都推不翻，flux-aligned 那欄 MLP mean 最佳但與 DCGAN 重疊。

**與 v11 的差別在於「贏」這個字**：v11 說「N=11 沒有任何一欄有站得住的贏家」；v12 因為換了 latent 參數化並把通量慣例改掉，flux-aligned 那一欄第一次出現「無人在 1 sd 內重疊」的贏。但這個贏只在一欄、只在 N=11，而且它的 caveat（aligned 欄對通量慣例恆等不變、抽樣規則對 RBM 不中性）必須跟著一起講，見 §9。

**驗證集選出的 checkpoint 不改變故事**（`seed_tables_g.md` §1e）：為了與 baseline 的 early stopping 對稱，MLP arm 也跑了 16,000 步、在 {1k, 2k, 4k, 8k, 16k} 中逐 seed 用 validation block 選 checkpoint。N=11 的 MLP M=2 (flat) 從 aligned 0.02247 ± 0.001811 變成 0.02232 ± 0.001686，N=100 從 0.01812 ± 0.00171 變成 0.01729 ± 0.001358。全部在 1 sd 內，所以主表保留較嚴的固定 4,000 步。

### 7.2 M 掃描（潛變數寬度） **[v12 新增]**

出處：`seed_tables_f.md` §1，H = 32、flat MMD、**固定 S**（v11 慣例，方便與 v11 逐格對照）；`nonlin` 是最佳線性擬合對學到的映射的相對殘差（0 = MLP 塌回線性映射）。

**N = 11**

| M | params | PDF-JS | MMD² | MMD²_al | proj corr | ‖z‖ | nonlin |
|---|---|---|---|---|---|---|---|
| 1 | 33,867 | 0.0009 ± 0.0001 | 0.0859 ± 0.0026 | 0.0237 ± 0.0046 | 0.9001 ± 0.0007 | 0.1777 ± 0.0148 | 0.7813 ± 0.0291 |
| **2** | 33,910 | 0.0010 ± 0.0001 | **0.0851 ± 0.0017** | **0.0225 ± 0.0017** | 0.9016 ± 0.0012 | 0.1010 ± 0.0044 | 0.6961 ± 0.0254 |
| 3 | 33,953 | 0.0009 ± 0.0001 | 0.0854 ± 0.0018 | 0.0240 ± 0.0021 | 0.9059 ± 0.0014 | 0.0829 ± 0.0019 | 0.5541 ± 0.0356 |
| 4 | 33,996 | 0.0009 ± 0.0000 | 0.0866 ± 0.0023 | 0.0249 ± 0.0018 | **0.9076 ± 0.0014** | 0.0802 ± 0.0083 | 0.4915 ± 0.0371 |
| 6 | 34,082 | 0.0013 ± 0.0006 | 0.0876 ± 0.0028 | 0.0284 ± 0.0079 | 0.9054 ± 0.0026 | 0.0823 ± 0.0041 | 0.3083 ± 0.0330 |
| 8 | 34,168 | 0.0029 ± 0.0014 | 0.0997 ± 0.0141 | 0.0503 ± 0.0283 | 0.8986 ± 0.0051 | 0.1202 ± 0.0095 | 0.1725 ± 0.0591 |
| 16 | 34,512 | 0.0085 ± 0.0009 | 0.1621 ± 0.0131 | 0.1732 ± 0.0251 | 0.8953 ± 0.0006 | 0.2245 ± 0.0110 | n/a |

**N = 100**

| M | params | PDF-JS | MMD² | MMD²_al | proj corr | ‖z‖ | nonlin |
|---|---|---|---|---|---|---|---|
| 1 | 33,956 | 0.0015 ± 0.0001 | 0.0694 ± 0.0014 | 0.0188 ± 0.0025 | 0.9624 ± 0.0016 | 0.3549 ± 0.0094 | 0.8326 ± 0.0183 |
| **2** | 34,088 | 0.0019 ± 0.0002 | **0.0684 ± 0.0014** | **0.0181 ± 0.0016** | 0.9773 ± 0.0006 | 0.1481 ± 0.0130 | 0.7322 ± 0.0315 |
| 3 | 34,220 | 0.0018 ± 0.0002 | 0.0692 ± 0.0031 | 0.0188 ± 0.0046 | 0.9807 ± 0.0004 | 0.1175 ± 0.0056 | 0.6498 ± 0.0318 |
| 4 | 34,352 | 0.0018 ± 0.0001 | 0.0700 ± 0.0029 | 0.0193 ± 0.0035 | 0.9840 ± 0.0003 | 0.0932 ± 0.0033 | 0.5596 ± 0.0201 |
| 6 | 34,616 | 0.0019 ± 0.0001 | 0.0706 ± 0.0021 | 0.0205 ± 0.0037 | **0.9855 ± 0.0011** | 0.0941 ± 0.0195 | 0.4363 ± 0.0762 |
| 8 | 34,880 | 0.0028 ± 0.0004 | 0.0684 ± 0.0004 | 0.0207 ± 0.0030 | 0.9752 ± 0.0023 | 0.2168 ± 0.0105 | 0.4909 ± 0.0304 |
| 16 | 35,936 | 0.0098 ± 0.0001 | 0.2083 ± 0.0075 | 0.2451 ± 0.0082 | 0.9120 ± 0.0013 | 0.4750 ± 0.0165 | 0.5417 ± 0.0108 |

讀法：**生成指標的最佳點是 M=2，且到 M=4 都是平的；M=6 開始退化，M=16 徹底壞掉**（N=100 的 aligned 從 0.0181 跳到 0.2451）。**projection corr 的最佳點卻在 M=4（N=11）或 M=6（N=100）**，方向與生成指標不同，所以「重建得好」與「生成得像」在這裡不是同一件事，論文報 M=2 是照生成指標選的。`nonlin` 隨 M 單調下降（M 越小，MLP 越需要用非線性去補足自由度），M=16 的 N=11 那格因為 `M ≥ N` 而無意義（標 n/a）。

### 7.3 H 掃描（隱藏層寬度） **[v12 新增]**

出處：`seed_tables_f.md` §2，M = 4、flat MMD、固定 S。最右欄是與 v11 線性 IQP arm 在 `MMD²_al` 上的 1-sd 判定。

| H | N | params | PDF-JS | MMD² | MMD²_al | proj corr | nonlin | vs 線性 arm |
|---|---|---|---|---|---|---|---|---|
| 4 | 11 | 5,184 | 0.0042 ± 0.0009 | 0.0955 ± 0.0061 | 0.0385 ± 0.0098 | 0.8938 ± 0.0117 | 0.3488 ± 0.0595 | 較差／重疊 |
| 8 | 11 | 9,300 | 0.0017 ± 0.0001 | 0.0881 ± 0.0033 | 0.0282 ± 0.0027 | 0.9087 ± 0.0052 | 0.4529 ± 0.0428 | 較差／重疊 |
| 16 | 11 | 17,532 | 0.0010 ± 0.0001 | 0.0866 ± 0.0023 | 0.0253 ± 0.0013 | 0.9093 ± 0.0028 | 0.5026 ± 0.0492 | 較好／重疊 |
| 32 | 11 | 33,996 | 0.0009 ± 0.0000 | 0.0866 ± 0.0023 | 0.0249 ± 0.0018 | 0.9076 ± 0.0014 | 0.4915 ± 0.0371 | 較好／重疊 |
| 4 | 100 | 5,540 | 0.0033 ± 0.0006 | 0.0722 ± 0.0046 | 0.0290 ± 0.0104 | 0.9625 ± 0.0123 | 0.4542 ± 0.0390 | 較差／重疊 |
| 8 | 100 | 9,656 | 0.0023 ± 0.0002 | 0.0706 ± 0.0029 | 0.0219 ± 0.0038 | 0.9760 ± 0.0008 | 0.4865 ± 0.0388 | 較好／重疊 |
| 16 | 100 | 17,888 | 0.0020 ± 0.0001 | 0.0701 ± 0.0034 | 0.0197 ± 0.0037 | 0.9826 ± 0.0004 | 0.5331 ± 0.0143 | 較好／重疊 |
| 32 | 100 | 34,352 | 0.0018 ± 0.0001 | 0.0700 ± 0.0029 | 0.0193 ± 0.0035 | 0.9840 ± 0.0003 | 0.5596 ± 0.0201 | 較好／重疊 |

讀法：**H 比 M 不重要，而且它在 H = 16 就飽和**（N=11 的 aligned 0.0253 → 0.0249、N=100 的 0.0197 → 0.0193，兩步都遠在 1 sd 內），但參數量從 17.5k 翻倍到 34k。只有 H=4 明顯不夠（N=11 的 aligned 0.0385 ± 0.0098，比線性 arm 還差）。論文正文只引 N=100、M=4 那四個數字（0.029 / 0.022 / 0.020 / 0.019）。**這也是一個減參數量的現成槓桿**：H=16 可以把主 arm 從 34k 砍到 17.5k 而不動任何結論，只是我們沒有為 M=2 重跑 H 掃描，所以沒有直接可引的 M=2、H=16 那一格。

### 7.4 消融 **[v12 新增]**

出處：`seed_tables_g.md` §5b（由 `results_seeds_mlp_suite.jsonl` 重算，不從 `seed_tables_f.md` 抄，避免兩份檔漂移）；`seed_tables_f.md` §6 是同一批的另一種排版。baseline = 主 arm（8 次 restart、KL 暖啟動、以相關度挑 restart），固定 S。

| 消融 | N | PDF-JS | MMD² | MMD²_al | proj corr | 判定（對 baseline） |
|---|---|---|---|---|---|---|
| baseline | 11 | 0.0009595 ± 9.236e-05 | 0.08508 ± 0.001855 | 0.02247 ± 0.001811 | 0.9016 ± 0.001305 | 基準 |
| **restart R1（只用零初始化那一次）** | 11 | 同上 | 同上 | 同上 | **0.8381 ± 0.002979** | proj corr 分離變差；四個生成指標完全不動 |
| restart R2（兩次） | 11 | 同上 | 同上 | 同上 | 0.901 ± 0.001314 | 1 sd 內，兩次即等同八次 |
| 以 loss 挑 restart（模型自己做得到的選法） | 11 | 同上 | 同上 | 同上 | 0.9013 ± 0.001178 | 1 sd 內，差 0.0003 |
| θ⁰ 改用 flat-MMD 暖啟動 | 11 | 0.0009484 ± 3.324e-05 | 0.08499 ± 0.001308 | 0.0231 ± 0.002219 | 0.9022 ± 0.001075 | 每一格都在 1 sd 內 |
| baseline | 100 | 0.001902 ± 0.0002152 | 0.06844 ± 0.001485 | 0.01812 ± 0.00171 | 0.9773 ± 0.0006473 | 基準 |
| **restart R1（只用零初始化那一次）** | 100 | 同上 | 同上 | 同上 | **0.8871 ± 0.01539** | proj corr 分離變差 |
| restart R2（兩次） | 100 | 同上 | 同上 | 同上 | 0.9772 ± 0.0007208 | 1 sd 內 |
| 以 loss 挑 restart | 100 | 同上 | 同上 | 同上 | 0.9773 ± 0.0006483 | 1 sd 內 |
| θ⁰ 改用 flat-MMD 暖啟動 | 100 | 0.001844 ± 6.061e-05 | 0.06872 ± 0.001554 | 0.01775 ± 0.001388 | 0.9773 ± 0.0004157 | 每一格都在 1 sd 內 |

三件事要一起讀：

1. **restart 只影響 projection correlation，完全不影響生成指標**。restart 是「凍住模型、只對 held-out 影像最佳化 z」的那一步，它不改變 generator，所以 PDF-JS / MMD² / MMD²_al 三欄逐格相同。**零初始化那一次單獨用只到 0.8381（N=11）與 0.8871（N=100）**，對照八次的 0.9016 與 0.9773。論文正文引的 0.838 / 0.887 對 0.901 / 0.977 就是這一組。
2. **以 loss 挑 restart（模型自己做得到的選法）與以 corr 挑（我們報的）幾乎沒差**，兩個 N 分別是 0.9013 對 0.9016、0.9773 對 0.9773。所以這個數字不是選擇偏差撐起來的。這一點在 v11 就已建立（差 ≤ 0.005），v12 的 MLP arm 上同樣成立。
3. **θ⁰ 的 KL 暖啟動不是硬體可估的**，所以做了消融：換成全程 shot-estimable 的 flat-MMD 暖啟動之後，兩個 N、四個指標**沒有一格差超過 1 sd**（N=11 的 aligned 0.0231 對 0.0225 是最大的一格）。這是「整條流程都可以在硬體上跑」這句話的支撐證據。

### 7.5 Data scaling 與 per-class（通量抽樣）

出處：`seed_tables_g.md` §3、§4。`MLP` = `IQP-mlpH32-M2-MMDflat-resample`。DCGAN 學自己的 per-image flux，所以 resampled-S 欄對它不適用，引的是已發表的曲線。

| N | MLP MMD² | MLP MMD²_al | linear MMD² | linear MMD²_al | RBM MMD² | RBM MMD²_al |
|---|---|---|---|---|---|---|
| 10 | 0.06171 ± 0.01554 | 0.02599 ± 0.004446 | 0.06396 ± 0.01635 | 0.02712 ± 0.003066 | 0.09117 ± 0.01745 | 0.07722 ± 0.01446 |
| 30 | 0.02846 ± 0.003613 | 0.02279 ± 0.002938 | 0.02906 ± 0.004492 | 0.02584 ± 0.004968 | 0.06454 ± 0.01035 | 0.103 ± 0.0218 |
| 50 | 0.02533 ± 0.005668 | 0.01982 ± 0.005044 | 0.02502 ± 0.004788 | 0.02479 ± 0.002751 | 0.06373 ± 0.01114 | 0.1149 ± 0.03428 |
| 70 | 0.02421 ± 0.003993 | 0.01605 ± 0.001457 | 0.02614 ± 0.001442 | 0.02262 ± 0.002206 | 0.07126 ± 0.01578 | 0.1272 ± 0.02807 |
| 90 | 0.02954 ± 0.005053 | 0.0188 ± 0.002804 | 0.02856 ± 0.005917 | 0.02296 ± 0.001042 | 0.05012 ± 0.0105 | 0.08166 ± 0.01382 |
| 100 | 0.02421 ± 0.00496 | 0.01812 ± 0.00171 | 0.02752 ± 0.005375 | 0.02551 ± 0.004413 | 0.05105 ± 0.007834 | 0.08067 ± 0.02205 |

讀法：通量抽樣把 IQP 側的 raw MMD² 曲線整條往下拉（v11 固定 S 時 IQP 對 N 是平的、停在 0.068–0.095；抽樣後從 N=10 的 0.062 降到 N=100 的 0.024），**所以 v11 說的「IQP 對 N 平坦」在通量抽樣之下不再成立**。flux-aligned 那一欄 MLP 在每個 N 都是最低 mean，但只有 N=10 與 N=70 的領先超過一個 seed sd（`seed_tables_f.md` §4 的 1-sd 判定）。

per-class（`seed_tables_g.md` §4，通量抽樣，各自對照自己的 200-vs-200 floor）：MLP 在 N=100 的 aligned 為 axion 0.0166 ± 0.002033、cdm 0.01462 ± 0.003116、no_sub 0.02058 ± 0.002953，對照的 floor 為 0.003068 ± 0.00162、0.003166 ± 0.001323、0.004463 ± 0.002394。**三類都達到 pooled 水準，但不可以主張某一類比較容易**，因為每類用的是不同的 held-out 集合。

### 7.6 與原論文 Table 2 的定性對照

| | 原論文（turbulence） | 我們（lens，v12） |
|---|---|---|
| 小 N（11） | IQP 全面贏 RBM 與所有 DCGAN；DCGAN 在 11 筆資料下訓練崩潰（PDF–JS 0.245–0.367） | **一欄站得住的贏**：flux-aligned 由 MLP M=2 領先且無人在 1 sd 內重疊；另外兩欄與 DCGAN 打平、三欄全贏 RBM。我們的 DCGAN 沒有崩 |
| 大 N（100） | 高容量 DCGAN 可在一兩個指標上打平或超越 IQP，但要 10 倍資料與 1.2M 參數 | DCGAN 在 PDF-JS 與 raw MMD² 兩欄穩定領先，只用 199,837 個 generator 參數 |
| Data scaling | 資料變多 DCGAN 才追上 | **方向一致但幅度不同**：DCGAN 在 N=10 的 raw MMD² 就已經最好；不過通量抽樣之後 IQP 側也開始隨 N 改善，不再像 v11 那樣完全平坦 |
| 絕對值 | PDF–JS 0.022–0.367，MMD² 0.896–1.472 | PDF–JS 0.00055–0.0084，MMD² 0.022–0.218 |

**警告**：兩張表的絕對值不在同一個尺度上（不同資料集、不同影像大小與 channel 結構）。任何並排都只能比排序，不能比絕對數值。

---

## 8. 原論文的內部主張，我們重測的結果

（v12 的 7 頁版把 §8.2 與 §8.3 這兩個附錄拿掉了，只留 §8.1 的 map 與 §8.4 的 turbulence。結論本身沒有改變，數字保留在此供內部參考與日後投長版時使用。）

### 8.1 Correlation–complexity map：permutation null（我們新增的方法學）

出處：`seed_tables_b.md` §1。我們對透鏡與 D2 各加一個 **structure-matched permutation null**：保留每張影像或每個 snapshot 的完整強度直方圖，只把值在空間位置間隨機重排。

| 資料與編碼 | K | `I_QCLI` | `I_CCI` |
|---|---|---|---|
| Lens, 18-bit tuples | 4,096 | 0.4715 ± 0.0000 | 0.8677 ± 0.0000 |
| &nbsp;&nbsp;permutation null | 4,096 | 0.4154 ± 0.0000 | 0.8783 ± 0.0001 |
| Turbulence D2, 18-bit tuples | 4,096 | 0.0232 ± 0.0002 | 0.9285 ± 0.0008 |
| &nbsp;&nbsp;permutation null | 4,096 | 0.0114 ± 0.0000 | 0.9371 ± 0.0009 |
| **Turbulence D2, 18-bit（原生 K）** | **50,000** | **0.0910 ± 0.0000** | **0.9085 ± 0.0000** |
| &nbsp;&nbsp;permutation null | 50,000 | **0.0981 ± 0.0007** | 0.8782 ± 0.0003 |
| Lens, 10-bit crop pmf | exact pmf | 0.4731 ± 0.0000 | 0.4313 ± 0.0000 |
| &nbsp;&nbsp;permutation null | exact pmf | 0.1782 ± 0.0000 | 0.9634 ± 0.0004 |

**沒有一格的 seed spread 超過 0.0009**：lens 18-bit 那一列的位置幾乎全部是編碼造成的（QCLI 只比自己的 null 高 0.056），只有 10-bit crop pmf 在兩個座標上都明顯偏離 null。在原論文自己的 `N_t = 5×10⁴` 上，turbulence 的 `I_QCLI`（0.0910）**低於自己的 permutation null**（0.0981），而在 subsample 到 K=4,096 時卻高於 null，所以一列只能對自己 K 的 null 讀。**這個位置對生成結果沒有預測力**（`MAP:21,76-78`）。

### 8.2 Fig. 3（support mismatch）：一個主張重現，一個方向相反 **[v12 已移出論文]**

出處：`seed_tables_b.md` §3。主張 (b)「學習者 gate 越多，包絡整體下移」**重現**（median MMD² 9.578e-03 ± 1.1e-05 → 2.147e-03 ± 6.8e-06 → 1.437e-03 ± 2.0e-05）。主張 (a)「目標 QCLI 越高，達成的 MMD 上包絡越低」**在可及範圍內方向相反**（Spearman +0.9786 ± 0.0010 / +0.9009 ± 0.0014 / +0.8091 ± 0.0039）。Proposition II.2 的 floor proxy 仍隨 QCLI 下降，但只有 exact-pmf proxy 滿足 ρ ≤ −0.98（最壞 −0.9876）；10⁴-sample proxy 在 `G_n = 150` 是 −0.9291。

### 8.3 Fig. 4（QCLI 最大化 → CCI 跟著升）：低 QCLI 成立，高 QCLI 斷裂 **[v12 已移出論文]**

出處：`seed_tables_b.md` §4。低 QCLI（≤ 0.3）的 pooled ρ 為 +0.7923 / +0.7938 / +0.9001（n = 8/12/16），global ρ 為 +0.0814 / −0.5673 / −0.6352，而精確均勻分布的 `I_QCLI` 為 0.9816 / 0.9984 / 0.9999 且 CCI 恰為 0。**global ρ 不是穩定的摘要統計量**：n=8 的 per-seed 值是 +0.994 / +0.959 / +0.808 / −0.889 / −0.604（sd ±0.92）。

### 8.4 Turbulence 復現：未重現，且我們知道為什麼

出處：`seed_tables_c.md` C1–C6。pre-registered 判定為**未重現**：mean 聚合、N=11 下 RBM-12k（0.2166 ± 0.0069）與 DCGAN-1.2M（0.0453 ± 0.0621）在兩個指標上都贏我們的 IQP-31k（0.2716 ± 0.0036）。last-write-wins 慣例下我們的 PDF–JS 落在已發表值附近（0.0324 ± 0.0011 / 0.0473 ± 0.0005 對 0.0238 / 0.0221），但 RBM 仍勝。協定歸因：凍住 `θ_lat` 值 **−0.1135 ± 0.0209**，5×10⁴ 抽樣 pmf 值 **−0.0010 ± 0.0222**（與 0 無法區分），stage 2 值 0.0000 ± 0.0004。模型族天花板 0.8921（seed-independent）。Conv2D-MMD² 欄不可比：ground truth 在像素空間打散就得到 0.70451 ± 0.00952，低於論文 Table 2 整欄。

---

## 9. 誠實邊界與投稿時的措辭原則

| # | 邊界 | 我們怎麼寫 | 出處 |
|---|---|---|---|
| 1 | **flux-aligned 欄對通量慣例恆等不變** **[v12 caveat 之一]** | 這是 v12 唯一站得住的贏所在的那一欄，所以必須主動說明：`MMD2_al` 在評分前把每張 frame 縮放到 held-out 平均總亮度，等於把 per-image 通量除回去，因此通量慣例對它**在構造上**不可見（325 對配對 run 最大差 6.44e-09）。raw 與 aligned 不是兩個獨立的贏，是「形狀加通量」與「形狀」。正文 §3.3 已寫 | `seed_tables_g.md` 開頭；`v12/main.tex` §3.3 |
| 2 | **通量抽樣規則對 arm 不中性** **[v12 caveat 之二]** | 同樣的 KDE 讓每個 IQP／MLP arm 的 N=11 raw MMD² 改善 22–39%，卻讓 RBM 變差 20%（N=100 又改善 13%）。所以 N=11 那一格對 RBM 的差距有一部分來自 baseline 自己往下掉。正文 §3.3 已寫 | `seed_tables_g.md` §1c |
| 3 | **參數量的方向要講清楚** **[v12 修正]** | 主 arm 是 33,910 / 34,088，**約為 DCGAN（199,837）的六分之一，但約為 RBM（11,510 / 15,960）的 3 倍**。`v12/main.tex` §3.3 原本寫「it leads the RBM on all three ... with six times fewer parameters」，讀起來像是「比 RBM 少六倍」，實際上六分之一是對 DCGAN 講的。**已於 2026-09-10 改為「at a sixth of the DCGAN's parameters and three times the RBM's」，中文譯稿同步更新** | `v12/main.tex` 表 2 與 §3.3 |
| 4 | **N=100 時 DCGAN 領先** | 直說：PDF-JS 與 raw MMD² 兩欄 DCGAN 領先且連 1 sd 都推不翻，flux-aligned 那欄 MLP mean 最佳但與 DCGAN 重疊。摘要與 Discussion 都寫了「the standing does not survive to 100 images」 | `seed_tables_g.md` §1；`v12/main.tex` §4 |
| 5 | **exact classical simulation** | 全部用三次 FWHT 精確模擬，沒有 Monte-Carlo、沒有抽樣噪音；n=10 只有 1,024 個振幅。摘要與 Discussion 都明講 | `iqp_exact.py:1-11` |
| 6 | **不主張量子優勢** | 「hardness results 描述的是這個模型族在 scale 上的性質，不是我們的實例」 | `v12/main.tex` §4 |
| 7 | **只比 RBM 與 DCGAN** | **本文不與原論文未使用的古典模型比較。** 這是 Leo 指定的框架。§6.5 的 softmax 對照是內部 sanity check，不進論文 | memory `feedback-paper-baselines-only.md` |
| 8 | **DCGAN 不可重現** | cuDNN 沒有 deterministic transposed-convolution kernel：同 config 同 seed 重跑，raw MMD² 差最大 57.0%，與 DCGAN 對 IQP 的差距同一量級。RBM 與所有 IQP／MLP config 重跑 bit-identical | `seed_tables.md`「Repeat runs of an identical configuration」 |
| 9 | **Baseline 有 early stopping、IQP 沒有** | RBM 160k/80k、DCGAN 5k–30k 由 validation 選出；IQP／MLP 固定 4,000 步。這只可能對 baseline 有利。而且我們也做了對稱檢查：MLP arm 在 16k 步 + validation-selected checkpoint 下沒有一格移動超過 1 sd（§7.1 末） | `seed_tables_g.md` §1e |
| 10 | **projection corr 的選擇偏差** | 論文報的是「八個 restart 取與 held-out 影像相關最高的那個」。附上「以 loss 選」欄（模型自己做得到的選法），兩者差 0.0003；單獨零初始化 restart 是 0.8381 / 0.8871 | `seed_tables_g.md` §5b |
| 11 | **flux-aligned 是 post-hoc** | 主表列出來但明確標註 post-hoc，因為它不在 round 3 的 pre-registration 裡 | `v12/main.tex` §2 |
| 12 | **θ⁰ 的 KL 暖啟動不是硬體可估的** | 已做消融：換成 flat-MMD 暖啟動後沒有一格差超過 1 sd | `seed_tables_g.md` §5b |
| 13 | **permutation null 削弱我們自己的賣點** | structure-matched permutation null 仍有 QCLI = 0.4154 ± 0.0000；v12 正文 §3.1 直接寫「a favourable position is necessary, not enough」 | `seed_tables_b.md` §1 |
| 14 | **兩套 pre-registered decision rule 都沒過** | round 3 判 failure、round 3b 判「不算有用」。正文只宣稱「非線性映射讓 IQP 能生成並泛化」與「N=11 的 flux-aligned 一欄領先」 | `R3RES:14-15,239-240` |
| 15 | **N=11 的生成影像系統性偏暗 9%** | 抽樣只修好通量分布的形狀、修不好位置：十一張訓練影像平均 112.34，held-out 是 123.20。正文 §3.3 直接寫出這是 raw 那欄停在 DCGAN 旁邊的原因 | `seed_tables_g.md` §1d |
| 16 | **12-qubit 那條線有 seed 塌陷** | N=100 有 2 of 5 seeds 的 train corr 掉到 0.004 / 0.008，那一格不可以用 mean ± sd 讀。內部文件寫出來，論文沒有引這條線 | `seed_tables_d.md` §3b |
| 17 | **資料集引用（2026-09-11 更正）** | v12 初版寫「Model II 的模擬方法出自 Alexander et al. 2023, ApJ 954:28」。引用審查讀了該文全文，它談的是 source/target 模擬的 domain adaptation，**沒有 Model 命名、也沒有 Euclid 設定**，不是 Model II 的來源。定稿改成：Model II 引 `ml4sci_deeplense`（repo README 明寫 Model 2 = Euclid observation characteristics、以 Lenstronomy 建成），`alexander_2020_deep` 與 `alexander_2021_decoding` 只作 DeepLense 這條線的出處引在引言，`alexander_2023_domain` 不再被引用 | `v12/CITATION_CHECK.md`；本文第 15 節 |

---

## 10. 檔案索引：每個結論對應到哪裡

### 10.1 文件

| 內容 | 檔案 |
|---|---|
| **v12 主表（兩種通量慣例、scaling、per-class、消融、v11→v12 對照）** | `paper/ml4ps2026/seed_tables_g.md` |
| **v12 非線性 MLP arm 全套（M 掃描、H 掃描、checkpoint 選擇、per-class、消融）** | `paper/ml4ps2026/seed_tables_f.md` |
| **v12 IQP 側槓桿掃描（multi 目標、Walsh 低通、spatial kernel、Gray code、lowres8）** | `paper/ml4ps2026/seed_tables_e.md` |
| **12-qubit 整張影像與 MLP 的初次對照** | `paper/ml4ps2026/seed_tables_d.md` |
| **無電路 softmax 對照（內部）** | `paper/ml4ps2026/seed_tables_d2.md` |
| 5-seed 統計（v11 主表、per-class、scaling、baseline 步數、θ⁰ 消融、診斷、per-restart、重跑差異） | `paper/ml4ps2026/seed_tables.md` |
| 5-seed 統計（map、線性 M 掃描、App. C、App. D） | `paper/ml4ps2026/seed_tables_b.md` |
| 5-seed 統計（turbulence 附錄） | `paper/ml4ps2026/seed_tables_c.md` |
| 指標協定 v11 的定義、floor 對照、排名變化、clipping | `paper/ml4ps2026/v11/facts.md`「Addendum: metric protocol v11」 |
| 原論文逐節摘要（含 Table 2 原始數字） | `docs/PAPER_2603.06440_SUMMARY.md` |
| 復刻規格書（含「原文未給」清單） | `docs/IQP_REPLICATION_SPEC_2603.06440.md` |
| Ref [38] 方法筆記（MMD 閉式、三 bandwidth 規則） | `docs/REF38_IQPOPT_METHOD_NOTES.md` |
| 透鏡影像特性量測 | `docs/LENS_IMAGE_CHARACTERISTICS.md` |
| Round 1 / 2 / 3 結果與 pre-registration | `docs/IQP_LENS_PILOT_RESULTS.md`、`docs/IQP_LENS_ROUND2_RESULTS.md`、`docs/IQP_LENS_ROUND3_{PREREGISTRATION,RESULTS}.md` |
| Correlation–complexity map 重算 | `docs/IQP_LENS_MAP_RESULTS.md` |
| Turbulence 復現結果與 root-cause | `docs/IQP_TURBULENCE_REPLICATION_RESULTS.md`、`..._DIAGNOSIS.md` |
| 稿件、數字出處、圖片出處 | `paper/ml4ps2026/v12/main.tex`、`v12/NUMBERS_PROVENANCE.md`、`v12/figures/FIGURE_PROVENANCE.md` |
| v11 版本的本文件與 v11 中文譯稿 | `docs/IQP_LENS_METHOD_DIFFERENCES_ZH_v11.md`、`docs/IQP_LENS_PAPER_ZH_v11.md` |
| v12 中文譯稿 | `docs/IQP_LENS_PAPER_ZH_v12.md` |

### 10.2 結果檔（全部在 `experiments/13_iqp_turbulence_replication/`）

| 內容 | 檔案 |
|---|---|
| **兩種通量慣例、每一次 run** | `results_v12_fluxres.jsonl` |
| **v12 IQP 側槓桿掃描** | `results_seeds_v12.jsonl` |
| **MLP 全套與其 checkpoint 選擇** | `results_seeds_mlp_suite.jsonl`、`results_seeds_mlp_suite_es.jsonl` |
| **12-qubit / MLP 初次網格與其 checkpoint 選擇** | `results_seeds_gridmlp.jsonl`、`results_seeds_gridmlp_es.jsonl` |
| **無電路 softmax 對照** | `results_seeds_mlpctrl.jsonl` |
| v11 主表、scaling、per-class 的 5-seed 原始記錄 | `results_seeds_table2_v11.jsonl`、`results_seeds_scaling_v11.jsonl`、`results_seeds_class_v11.jsonl` |
| v11 指標重新評分與 floor | `results_v11_all.jsonl`、`results_v11_floor.jsonl` |
| turbulence 5-seed | `results_d2_seeds.jsonl`、`results_d2_seeds_main.jsonl`、`results_d2_seeds_last.jsonl` |
| Correlation–complexity map（含 5 seeds） | `paper_suite/map/results_map_seeds.json` |
| Fig. 3 / Fig. 4（含 5 seeds） | `outputs_controlled/results_fig34_seeds.json` |
| 每個 run 的超參數與 S / g / z 統計 | `outputs_lens_r3b/<tag>/meta.json` |

### 10.3 程式

| 用途 | 檔案 |
|---|---|
| 10-bit crop pmf codec（含 `paste`） / 18-bit tuple codec | `codec10.py` / `codec.py` |
| 12-bit 整張影像 loader | `data_lens_n12.py` |
| 資料層：centroid alignment、crop、split 規則、S | `data_lens_r3.py` |
| Exact IQP 模擬器（三次 FWHT、kernel weights） | `iqp_exact.py` |
| 固定基底訓練 / 線性映射訓練 | `train_iqp_r3.py` / `train_iqp_affine_r3.py` |
| **非線性（MLP）映射訓練** | `train_iqp_affine_mlp.py` |
| **MLP 消融訓練** | `train_iqp_affine_mlp_abl.py` |
| **v12 槓桿訓練（multi 目標、Walsh 低通、spatial kernel、flux resample、Gray code、lowres8）** | `train_iqp_affine_v12.py` |
| **12-qubit 訓練** | `train_iqp_affine_n12.py` |
| **無電路 softmax 對照訓練** | `train_mlpctrl.py` |
| 評分（v11 協定，v12 沿用） | `metrics_v11.py`、`evaluate_v11.py` |
| **事後通量替換與重新評分（含 `--check`）** | `evaluate_v12_flux.py` |
| **checkpoint 選擇（validation block）** | `seeds/select_checkpoint.py` |
| Baseline / baseline 收斂掃描 | `baseline_rbm_r3.py`、`baseline_dcgan_r3.py`、`baseline_rbm_sweep.py`、`baseline_dcgan_sweep.py` |
| **5-seed 聚合（v12）** | `paper/ml4ps2026/make_seed_tables_g.py`、`aggregate_seeds.py`、`make_seed_tables_b.py` |
| SLURM wrapper（v12） | `seeds/submit_v12_fluxres.sbatch`、`seeds/submit_mlp_suite_{sweep,phase2,phase3}.sbatch`、`seeds/submit_seeds_grid_mlp.sbatch`、`seeds/submit_seeds_mlpctrl.sbatch` |

---

## 11. v11 → v12 變更摘要

共 18 條。標 **[數字]** 的是數值改變，標 **[修正]** 的是 v11 有錯或有誤導，標 **[新增]** 的是 v11 沒有的內容。

| # | 節 | 變更 | 類型 |
|---|---|---|---|
| 1 | §1、§4-4、§6.2 | **主 arm 從線性映射 `θ⁰ + Φᵀz` 換成非線性潛變數映射 `θ⁰ + g(z)`**，`g` 為 `M → 32 → 1,024` 的單隱藏層 tanh MLP（無輸出偏置），主 arm `M = 2`、`H = 32`。線性 arm 保留為對照列 | [數字][新增] |
| 2 | §6.2 | **給出「為什麼要非線性」的機制說明**：1,024 個相位的干涉映射有一個相干半徑（線性方向在 `‖z‖ > 0.05` 之後失相干），線性映射在 M=6 掉到 mean-image 線以下、M=8 崩潰；非線性映射可以在不同區域用不同方向與步幅，因此走得更遠而不失相干。附 `nonlin` 殘差 0.6961 ± 0.0254（M=2、N=11）證明非線性確實被用上 | [新增] |
| 3 | §1、§4-10、§5.6 | **生成時的總亮度改成通量抽樣**：`S` 從 N 個訓練總亮度的 Gaussian KDE 抽出（Silverman 頻寬 14.64 / 10.36），不重訓任何模型；`evaluate_v12_flux.py --check` 驗證固定 S 那一路對 325 次 run 的 `max \|saved - rebuilt\| = 0` | [數字][新增] |
| 4 | §5.6、§9-1 | **通量抽樣的三個必講事實**：raw MMD² 在 N=11 降 22–39%、N=100 降 33–65%；flux-aligned 欄在構造上恆等不變（325 對配對 run 最大差 6.44e-09）；規則對 RBM 不中性（N=11 讓 RBM 變差 20%、N=100 改善 13%） | [新增] |
| 5 | §7.1 | **主表全部換成 `seed_tables_g.md` 的 sampled-S 數字**（N=11 與 N=100 各一張），並新增兩行 MLP arm。**N=11 的 flux-aligned 第一次出現「無人在 1 sd 內重疊」的贏**（MLP M=2 flat 0.02247 ± 0.001811） | [數字][修正] |
| 6 | §7.1 | **新增「與 DCGAN／RBM 差幾個 sd」欄**：N=11 vs DCGAN 為 −0.13 / −0.04 / **+1.57**，vs RBM 為 +0.75 / +1.40 / +2.27；N=100 vs DCGAN 六個 gap 最小者 −9.51 | [新增] |
| 7 | §7.1 | **新增 validation-selected checkpoint 的對稱檢查**：MLP arm 在 16k 步 + 逐 seed 選 checkpoint 之下，N=11 aligned 0.02232 ± 0.001686、N=100 0.01729 ± 0.001358，全部在 1 sd 內，故主表保留固定 4,000 步 | [新增] |
| 8 | §7.2 | **新增 M 掃描表**（`seed_tables_f.md` §1，M = 1,2,3,4,6,8,16 × N = 11,100）。生成指標最佳點 M=2、到 M=4 平坦、M=16 徹底壞掉；projection corr 的最佳點卻在 M=4/M=6，兩者方向不同 | [新增] |
| 9 | §7.3 | **新增 H 掃描表**（`seed_tables_f.md` §2，H = 4,8,16,32 at M=4）。H 在 16 就飽和，H=4 明顯不夠；H=16 是把主 arm 從 34k 砍到 17.5k 的現成槓桿 | [新增] |
| 10 | §7.4 | **新增消融表**：restart 只影響 projection corr（零初始化單獨用 0.8381 / 0.8871，對照八次的 0.9016 / 0.9773），以 loss 選 restart 差 0.0003，θ⁰ 改 flat-MMD 暖啟動沒有一格差超過 1 sd | [新增] |
| 11 | §6.4 | **新增「試過但沒有用」一節**：12-qubit 整張影像（四指標全部分離變差，N=100 有 2/5 seeds 塌陷）、8-qubit lowres8（aligned 差約 2 倍，encode→decode 天花板只有 0.990）、Gray code（+0% / +2%，null）、Walsh 低通（+4% / −2%，null）、像素空間 kernel（raw −4% 但 aligned +7%，是交換不是增益） | [新增] |
| 12 | §6.5 | **新增無電路 softmax 對照**（`seed_tables_d2.md`）：32 個比較中對照組 3 勝 28 敗 1 平，三個勝全在 N=11 的 projection corr。**含但書：對照組的超參數沒有為它自己另外調過**，所以只是內部 sanity check、不進論文 | [新增] |
| 13 | §7.5 | **data scaling 的結論改變**：v11 說「IQP 對 N 平坦」（raw MMD² 0.068–0.095），通量抽樣之後 MLP 從 N=10 的 0.06171 降到 N=100 的 0.02421，**不再平坦** | [數字][修正] |
| 14 | §9-3 | **參數量方向的措辭修正**：主 arm 33,910 / 34,088 約為 DCGAN 的六分之一，但約為 **RBM 的 3 倍**；`v12/main.tex` §3.3 原句「with six times fewer parameters」讀起來像「比 RBM 少六倍」，**已於 2026-09-10 改為「at a sixth of the DCGAN's parameters and three times the RBM's」**，`docs/IQP_LENS_PAPER_ZH_v12.md` 同步改成「所用參數量是 DCGAN 的六分之一、RBM 的三倍」 | [修正] |
| 15 | §9 | **誠實邊界表加兩條 v12 caveat**（aligned 欄恆等不變、抽樣規則對 arm 不中性），並新增 N=11 偏暗 9%、12-qubit seed 塌陷兩條 | [新增] |
| 16 | §2.3、§5.5 | **指標協定沒有動**：v12 全部五份 seed table 都用 `evaluate_v11.py`，`seed_tables_g.md` §6 的 fixed-S 重算與 v11 published cell 最大相對差 0.00042（顯示捨入），這是 v11／v12 數字可逐格並排的前提 | [新增] |
| 17 | §8 | **§8.2 與 §8.3（support mismatch、QCLI–CCI 上限）已移出 7 頁論文**，數字保留在本文件供內部參考。§8.1 的 map 與 §8.4 的 turbulence 留在論文附錄 | [修正] |
| 18 | §3-4、§3-9、§3-10 | **把三條「跑前量測」與 v12 的實測結果接起來**：影像流形實質 5 維 ↔ M 的工作區間是 2–4；10-bit 比 12-bit 覆蓋更好 ↔ 12-qubit 實測更差；Gray code 是 GF(2)-linear ↔ 實測 null | [新增] |

**另外兩項編輯決定（不是數值變更）**：
- 涉及自訂控制組的段落仍然收斂成一句「**本文不與原論文未使用的古典模型比較**」（§9 第 7 列）。§6.5 的 softmax 對照是同一個 latent 架構把 generator 拿掉，屬於**消融**而不是古典 baseline，但它的超參數沒有為它自己調過，因此仍然只留在內部文件。
- 論文結構：v12 壓到 7 頁，附錄為 A 管線流程、B 固定 S 慣例、C 潛變數掃描與各類別、D turbulence 重現、E 訓練與評估細節；v11 的固定 20k 步基準表與兩個 controlled-study 附錄都拿掉了。

---

## 12. 論文最終的主張收斂成什麼，以及為什麼這樣寫

### 12.1 判準：1 sd 規則

所有「贏／輸／持平」的措辭都由同一條規則決定，而且這條規則寫進了論文本身。

| 名詞 | 定義 | 出處 |
|---|---|---|
| **以 sd 為單位的領先** | （基準模型 mean − 實驗組 mean）／**基準模型的種子 sd**，正值代表實驗組領先。這是描述性的，**不是顯著性檢定**，論文 §2 末句明講 | `v12/main.tex` §2；`v12/NUMBERS_PROVENANCE.md`「Section 3.3 sd margins」 |
| **持平（matches / level with）** | 差距落在 1 個基準模型 sd 之內 | `seed_tables_g.md` §1「wins the seed spread does not support」 |
| **表格粗體** | 該欄 mean 最佳，**且與其他每一個** arm 的差距都超過兩者 sd 之和（比「贏第二名」更嚴格） | `v12/main.tex` 表 2 caption |

### 12.2 最終主張逐格

**N = 11**（`seed_tables_g.md` §1，resampled S；sd margins 出自 `NUMBERS_PROVENANCE.md`）

| 對照 | PDF–JS | raw MMD² | flux-aligned MMD² | 論文怎麼寫 |
|---|---:|---:|---:|---|
| 非線性 M=2 vs **DCGAN** | −0.13 sd | −0.04 sd | **+1.57 sd** | 「matches the DCGAN on both generation metrics within the seed spread, is ahead of it on a flux-aligned variant」 |
| 非線性 M=2 vs **RBM** | +0.75 sd | +1.40 sd | +2.27 sd | 「ahead on the two MMD² columns … with PDF–JS inside one sd (0.75)」 |

**N = 100**（同上）

| 欄位 | mean 最佳者 | 1 sd 內有誰重疊 | 論文怎麼寫 |
|---|---|---|---|
| PDF–JS | DCGAN 0.0005534 ± 0.0001211 | 無，贏成立 | 「at 100 images the DCGAN is ahead」 |
| raw MMD² | DCGAN 0.02225 ± 0.003118 | 非線性 M=2 flat、線性 M=4 flat | 同上（論文只說 DCGAN ahead，未宣稱粗體） |
| flux-aligned | 非線性 M=2 flat 0.01812 ± 0.00171 | DCGAN 0.02122 ± 0.004081 | 「the flux-aligned 1.8 ± 0.2 sitting inside one sd of its 2.1 ± 0.4」 |

### 12.3 為什麼寫成這樣，而不是更強

| 不能寫的版本 | 為什麼不能 | 證據 |
|---|---|---|
| 「N=11 三欄全贏 RBM」 | PDF–JS 只領先 0.75 個 RBM sd，在 1 sd 內 | 審查 H1；`seed_tables_g.md` §1 overlap 表把 DCGAN 列為與最佳 arm 重疊 |
| 「IQP 打敗 DCGAN」 | 在兩個 N、兩種通量慣例之下，沒有任何一個 arm 在三欄上同時贏 DCGAN 與 RBM 且每個差距都超過基準 sd；12 列判定全是 `no` | `seed_tables_g.md` §1「which arms beat BOTH DCGAN and RBM on all three columns」 |
| 「flux-aligned 與 raw 是兩個獨立的贏」 | flux-aligned 在構造上對通量慣例恆等不變（325 對配對 run 最大差 6.44e-09），兩欄量的是「形狀」與「形狀加通量」 | `seed_tables_g.md` 開頭的 aligned-column note |
| 「raw 那欄贏 RBM 是模型的功勞」 | 通量抽樣讓每個 IQP arm 的 raw 改善 22–39%、卻讓 RBM 變差 20%；固定 S 之下 RBM 反而領先（6.5 ± 0.8 對 8.5 ± 0.2） | 審查 H5；`seed_tables_g.md` §1c、§1 fixed-S 欄 |
| 「透鏡是最適合 IQP 的資料集」 | 表 1 只比了兩個資料集加各自零模型；而且我們實際建模的 10-bit 編碼 I_CCI = 0.43，落在高 I_CCI 角落之外 | 審查 H2；`v12/main.tex` §3.1 末句 |
| 「固定的 latent block 造成失敗」 | `fixed basis, joint` 那一列（1.8 / 8.7 / 12.5）已經補回大部分差距，失敗的是「凍結的 core + 逐張循序調適」 | 審查 H4；`v12/main.tex` 表 2 |

---

## 13. 三輪獨立審查抓到並修正的問題清單

三輪都由乾淨 context 的獨立 agent 執行，寫稿的 agent 不驗自己的稿。合計 24 項（論理 10、引用 8、數字 6）。

### 13.1 論理審查（opus，2026-09-11，10 項 HIGH）

來源：`scratchpad/v12_review_logic.md`。另有 20 項 MEDIUM 與一串 LOW，也都套用了，此處只列 HIGH。

| # | 原本寫什麼 | 為什麼不對 | 改成什麼 |
|---|---|---|---|
| H1 | 摘要與 §3.3 寫「leads an RBM on all three」 | PDF–JS 的領先只有 0.75 個 RBM sd，在 1 sd 之內 | 摘要改「ahead … of an RBM on the two MMD² columns while level with it on the histogram metric」；§3.3 補上「with PDF–JS inside one sd (0.75)」 |
| H2 | 「most IQP-compatible position」用的是 18-bit 編碼的座標 | 我們實際建模的是 10-bit crop pmf，它的 I_CCI = 0.43，落在高 I_CCI 角落之外；而表 1 只比了兩個資料集 | §3.1 末句加「its favourable I_QCLI motivates the encoding, not the choice of an IQP over a pairwise model」；「most IQP-compatible dataset」降為「among the encodings we measured」 |
| H3 | 「a favourable position is necessary, not sufficient」 | 「必要條件」這個說法沒有證據支持，置換零模型證明的是另一件事 | 改成「the index is largely determined by the intensity histogram, so a favourable position is not by itself evidence of IQP-suitable structure」 |
| H4 | 「the fixed latent block yields only noise」 | 與 `fixed basis, joint` 那一列矛盾（1.8 / 8.7 / 12.5 對 Alg. 1 的 6.5 / 14.8 / 22.5）；而且我們跑的 Alg. 1 用的是自選低變異 block | 改成「the reference protocol's frozen core with sequential adaptation yields only noise, which joint training repairs」；§3.2 明說用的是自己的 block，且 block size 50 沒有掃描過 |
| H5 | §3.3 只講抽樣強度之下的名次 | 在固定強度之下，RBM 在 N=11 的 raw MMD² 反而領先非線性 arm | 加入「Under the fixed intensity the RBM leads the non-linear arm there (6.5 ± 0.8 against 8.5 ± 0.2), so the raw-column standing against the RBM is a property of the sampled decode and not of the model」 |
| H6 | 「reconstructs unseen images at 0.901」 | 「reconstruct」讀起來像模型本身的能力，實際是八次重啟中以擬合損失挑出最好的一次所做的投影 | 改成「projects unseen images at correlation 0.901」，並在 §2 說明「the best of eight latent restarts by the fit's own loss」，§3.2 補上 nearest-training-image（0.892 / 0.970）與單一零初始化重啟（0.838 / 0.887）兩條參考線 |
| H7 | 「trains classically and deploys on quantum」被當成本文已達成的事 | 有兩個步驟以目前跑法不是 shot-estimable；而且 n=10 把 2ⁿ 個相位全部參數化 | Discussion 加「Two steps are not shot-estimable as run, the exact-KL warm start … and the projection … at n=10 we parameterise all 2ⁿ Walsh phases where the scaling up would need them sparsified, untested here」 |
| H8 | 缺三個控制 | 沒有記憶檢查、沒有物理合理性、沒有標註缺上界控制 | 加入 nearest-training-image L2（4.21 ± 0.10 對真實的 2.90）、附錄 C 的 Wasserstein-1 三個量、限制段的「no destroyed image is scored as an upper reference」 |
| H9 | §3.2 出現內部程式名稱（mechanism-test driver / round-3 trainer） | 審稿人看不懂，而且沒說清楚兩條曲線為何可比 | 改成「the two sweeps come from separate drivers differing in optimiser schedule and initialisation, both scored on the same 20 held-out frames, so a small offset between them is not a model difference」 |
| H10 | 沒有定義 seed 到底變什麼 | 訓練影像跨 seed 固定（`R3_SPLIT_SEED`），所以回報的 sd 不含訓練抽樣的變異 | §2 與限制段都寫上「The N training images are fixed across seeds, so that spread is initialisation and sampling variance alone」 |

### 13.2 引用審查（opus，web 驗證，2026-09-11，8 項）

來源：`scratchpad/v12_review_citations.md`、`v12/CITATION_CHECK.md`。

| # | 原本寫什麼 | 為什麼不對 | 改成什麼 |
|---|---|---|---|
| C1 | Bremner 2011 與 2016 被混用在同一個「hard to sample」句 | 2011 是 worst-case、乘性（41%）誤差；2016 才是 average-case、加性誤差 | 拆成兩句：「Exact or multiplicatively approximate … would collapse the polynomial hierarchy [2011], and that hardness is conjectured to hold on average under additive error [2016]」 |
| C2 | 引 `codsi_2025_classically` 支持「特殊結構恢復古典可模擬」 | Codsi 給的是指數時間的改良演算法（約 50 → 70 qubits），不是恢復可模擬性 | 換成 `maslov_2024_fast`，句子改為「classical algorithms keep improving」；Codsi 的 bib 條目保留但不再被引用 |
| C3 | 「both demonstrated on 2D turbulence at n=18」 | 只有潛變數調適流程 (ii) 是湍流專屬，地圖 (i) 涵蓋數個資料集 | 改成「demonstrated at n=18 qubits on 2D turbulence, one of several datasets the map covers」 |
| C4 | 「freezes a random block of 50 of those phases」 | 原論文定義的是「Hamming 順序中最前面的 d_lat = 50 個」，隨機的是初始值不是選法 | 改成「freezes the first 50 of those phases in Hamming order」，並在附錄 C 說明我們用的是自己的低變異 block |
| C5 | Model II 影像引 `alexander_2023_domain` | 該文全文談 source/target 模擬的 domain adaptation，沒有 Model 命名、沒有 Euclid 設定，不是 Model II 的來源 | Model II 改引 `ml4sci_deeplense`；`alexander_2020_deep`、`alexander_2021_decoding` 只作 DeepLense 出處引在引言；`alexander_2023_domain` 不再引用 |
| C6 | `liu_2026_correlation` 只在引言引一次 | 後面五處實質依賴它（Algorithm 1、N_t、聚合規則、sparsification、Table 2），卻沒有引 | §3.1、附錄 C、附錄 D 與 Discussion 各補上引用，共六處 |
| C7 | 「at the paper's own K = 5×10⁴」 | K 是我們自己定義的「每張影像編碼數」，原論文那個數字是 N_t（每張快照的訓練樣本數） | 改成「at the protocol's own N_t = 5×10⁴ samples per snapshot [Liu 等 2026a]」 |
| C8 | 三頻寬核寫成「IQPopt 的，而該論文一個都沒公布」 | IQPopt（`recio_2025_iqpopt`）用的是**單一**中位數啟發法頻寬；三頻寬規則的逐字出處是 `recio_2025_train` | §2 與附錄 C 都改引 `recio_2025_train`，並寫出規則本身（σ₁ 使平均 Pauli weight 為 2、σ₃ 為中位數啟發法的平方根、σ₂ 為兩者均方根）；`recio_2025_iqpopt` 只在引言引一次 |

另外補進去的引用：`walsh_1923_closed`（Walsh–Hadamard 變換，原本在 bib 裡卻沒被引用）、`silverman_1986_density`（Silverman 頻寬）、以及第 15 節所列的八筆新增文獻。

### 13.3 數字審查（獨立 agent 從 seed tables 重新推導，2026-09-11，6 項）

來源：`v12/NUMBERS_PROVENANCE.md`「Mechanical check」「Independent review, and what it changed」「2026-09-11 revision」。機械比對的結論是**表 2、3、4 全部格子 0 個不符**，以下是文字與慣例層面的六項修正。

| # | 原本寫什麼 | 為什麼不對 | 改成什麼 |
|---|---|---|---|
| N1 | 表 2 caption 的粗體規則寫成「領先 **runner-up**」 | 實際挑粗體用的是更嚴格的「領先**其他每一個** arm」 | caption 改寫成 every-other-arm 的說法；沒有任何一格粗體因此改變 |
| N2 | 參數量欄取自 `seed_tables_g.md` 的 Params 欄（例如 RBM 11,010） | 那一欄是舊的「單張影像」慣例，與附錄 C 宣告的慣例不一致 | 全部改為「共用參數（含 θ⁰）＋ N 組 per-image latent」：RBM = 10,960 + 50N = 11,510 / 15,960、fixed-basis = 974 + 50N = 1,524 / 5,974、learned-direction = 5,120 + MN、DCGAN 只算 generator（199,837） |
| N3 | §3.2 的 0.9013 / 0.9773 無法追溯到哪一種重啟挑法 | 消融表有「以 loss 挑」與「以 correlation 挑」兩個版本（0.9013 對 0.9016），論文沒說用哪個 | §2 明寫「the best of eight latent restarts by the fit's own loss」，數字對應 `seed_tables_g.md` §5b 的「restart picked by loss」列 |
| N4 | 抽樣對 raw MMD² 的改善寫成「29 to 37%」 | 逐 arm 重算是 22 到 39% | 改成「22 to 39%」（此項在審查回報之前已自行更正） |
| N5 | 表 2 的 reference-protocol N=11 PDF–JS 在 provenance 檔記成 6.6 | 原始值 0.00655 ± 0.0007681 進位為 6.5，`main.tex` 本來就是 6.5，錯的是 provenance 檔 | provenance 檔更正為 6.5 ± 0.8 |
| N6 | 附錄 C 的 DCGAN 重跑差異下界寫 12% | 只讀了七列中的前五列 | 改成 raw MMD² 5 到 57%、PDF–JS 2 到 74%（七次重跑的完整值見 `docs/IQP_LENS_PAPER_ZH_v12_full.md` §S9） |

另有一項被回報但查證後不成立：§3.4 說「非線性 arm 領先 DCGAN 超過一個 seed sd 只在 N=10 與 N=70」，審查者拿去對比的是 affine arm，而該敘述的來源 `seed_tables_f.md` §4 的 `MLP vs DCGAN (MMD2_al)` 判定欄確實只在這兩個 N 讀作 SEPARATED，原文不改。

---

## 14. 原論文 v1 與 v2 的差異（我們打的是哪一版）

來源：`docs/PAPER_2603.06440_V1_V2_DIFF.md`（2026-09-11 逐項比對，v1 = 2026-03-06、v2 = 2026-05-05）。

**先講結論：我們用的一直就是 v2，沒有任何改動會推翻我們的複現或否定結論。** 本機快取 `paper.html` 與 `arxiv.org/html/2603.06440v2` 的 md5 完全相同；兩項獨立佐證是我們引 Proposition II.2（v2 編號，v1 是 Theorem 2.2）、以及我們的 `--steps-lat` 預設 3000（這個數字只存在於 v2）。

| 項目 | v1 | v2 | 對我們的影響 |
|---|---|---|---|
| **Algorithm 1** | **不存在**。流程只有 eq.26–28 加 Fig. 6 的四步文字敘述 | 新增 12 行虛擬碼「Latent-parameter adaptation for turbulence synthesis」 | 我們論文寫「Algorithm 1 of that paper's **v2**」，這句只在 v2 成立，所以版本必須鎖住（bib 的 `note` 已標 v2） |
| **stage-2（latent 適配）步數** | **未給**，全文無 "3000" 字串 | 主文 eq.(29) 後與 Appendix J Table 6 都寫「3000 optimization steps per adapted snapshot」 | 對我們有利：`--steps-lat 3000` 直接取自 v2，若只有 v1 得用猜的 |
| **聚合規則** | 只說 aggregate on the discretized grid，**沒有定義同格點多筆怎麼合併** | 仍然**沒有定義**，只是把「aggregate」這個動詞多寫了三次 | 我們附錄 C 那句「a last-write-wins aggregation, never specified in that paper」**兩版都成立**，不需要改 |
| **訓練 MMD 的 kernel 頻寬** | Appendix G 只說「a Gaussian kernel」，**沒有 σ、沒有 median heuristic、沒有 multi-bandwidth** | **逐字相同**，Appendix J 也沒補 | 我們附錄 C 那句「the protocol we replicate publishes none for this encoding」兩版都成立（三頻寬規則的正確出處是 `recio_2025_train`，見第 13.2 節 C8） |
| **Table 2 數值** | RBM 0.168770 / 1.472481；IQP(31k) 0.023784 / 0.952420；IQP(63k) 0.022084 / 0.942267；GAN(1.2M) 0.012204 / 0.964874 | **逐位元相同**（`diff` 兩版表格數字列 0 差異） | 我們拿來對照的 0.0238 / 0.0221 依然是官方數字 |
| **定理降級** | **Theorem 2.2**「Lower Bounding the MMD Loss by I_QCLI」，結論直接是 `min ℒ_MMD ≥ C′(1 − I_QCLI(p))` | 改為 **Proposition II.2**「Walsh-support mismatch induces an MMD floor」，結論改成 `min ℒ_MMD ≥ C_A Σ_{S∉S_A} P(S)`；與 I_QCLI 的連結被移出定理，變成 **Assumption II.3 之下的經驗假說**（原文自稱 empirical hypothesis） | 這是 v2 最大的科學性改動：原作者自己把「QCLI 能預測 MMD floor」從定理降為條件性假設 |
| **baseline 自陳侷限** | 無 | III.2 新增「They are **not exhaustive**. In particular, diffusion models, autoregressive models, tensor-network distributions, coordinate-based neural fields, and reduced-order physical models could possibly provide stronger baselines」 | 原作者自承 baseline 不窮盡，我們附錄 C 直接引這一句 |
| **metric 自陳侷限** | 無 | III.2 新增「it is **not a substitute for physics-specific turbulence diagnostics such as energy spectra or structure functions**」 | 與我們「Conv2D-MMD² 不會把打散的 ground truth 排在訓練過的模型之後（0.7045 ± 0.0095）」的批評同方向，我們附錄 C 也引了 |
| Abstract 與 Limitations | Abstract 有「We provide **theoretical support** … higher QCLI implies a smaller irreducible approximation floor」 | 該句整段刪除；新增獨立 Limitations 節，寫「does not certify quantum advantage」「low-data competitiveness against the tested baselines, not a hardware sampling advantage」 | 主張全面弱化，與我們「no advantage claim」的立場一致 |
| 其他 | 章節 1/2/3、author–year 引用、46 篇參考文獻 | 章節 I/II/III（REVTeX 風）、數字引用、49 篇；Sec II.2 多插一條 display equation 使式號整體 +1 | **式號要小心**：v1 的 eq.N = v2 的 eq.N+1（自 (4) 起）。無新實驗、無新 baseline、無新圖 |

不需要改的敘述（已逐句對照兩版）：11 張 anchor 快照、n=18、D2、stage-1 的 500 samples / 3×10⁴ steps / N_t = 5×10⁴、d_lat = 50、18-bit tuple 編碼、Fig. 3 與 Fig. 4 的全部複現設定與否證結論。

---

## 15. 參考文獻的核對結果

來源：`v12/REFS_VERIFICATION.md`（全 52 條逐條對 arXiv API 與 Crossref）、`v12/CITATION_CHECK.md`（20 條被引用文獻逐條對權威來源，並讀 abstract 判斷「這句話有沒有被這篇支持」）。

### 15.1 總計

| 項目 | 數量 |
|---|---|
| `refs.bib` 條目總數 | 52 |
| **至少有一個事實錯誤或缺欄位的條目** | **26** |
| 需人為判斷、未自動改的 | 2（`recio_2025_iqpopt`、`ml4sci_deeplense`） |
| key 的年份標籤與 `year` 欄不一致但無害、依指示不改 key | 1（`alexander_2021_decoding`） |
| NOT FOUND（查不到來源） | **0**，每一個 arXiv ID 與 DOI 都解析成功 |

修正後的檔是 `v12/refs_verified.bib`（bib key 不變，只改欄位值）。

### 15.2 錯在哪幾類

| 類別 | 說明 | 例子 |
|---|---|---|
| **作者名錯誤（最嚴重）** | 12 條。給定名整個被吃掉、或名字張冠李戴 | `kurkin_2025`：Andrey → **Andrii**、Yudai → **Kevin**、Suzanne → **Susanne**、bare Wang → **Wang, Hao**；`lerch_2026`：Sabine → **Sacha**；`banks_2026`：Harry → **Robert J.**、Crippa, Leonardo → **Crippa, Arianna**；`kremer_2026`：Marius → **David**、Frédéric → **Nicolas**；`gharibyan_2025` 被截成「and others」，補回完整 6 位作者 |
| **出版狀態過時** | 預印本已正式出版 | `rajakumar_2024_polynomial` 已是 SODA 2025，pp. 1037–1056，doi 10.1137/1.9781611978322.30 |
| **姓氏形式** | Crossref/Elsevier 記的是複姓 | `khojasteh_2022_lagrangian` 第一作者為 **Rahimi Khojasteh, Ali**，不是 `Khojasteh, Ali R.` |
| **缺欄位** | 缺 DOI 或 eprint | 補上 `bremner_2016_average`、`chow_1968_approximating`、`walsh_1923_closed`、`alexander_2020_deep` 的 DOI，補 `rajakumar` 的 eprint |
| **分類欄錯** | `liu_2026_correlation` 的 `primaryclass` 寫 quant-ph | arXiv 實為 **cs.LG** |
| **venue 名稱不完整** | `tieleman_2008_training` 的 booktitle | 補成 Proceedings of the 25th International Conference on Machine Learning (ICML) |

`REFS_VERIFICATION.md` 對這個錯誤分布的判讀是：幾乎所有作者名錯誤都集中在 2025–2026 那一批 IQP 生成模型預印本，看起來是原本的 bib 由抓取 abstract 頁 HTML 建成，而名字切分器在多字名與帶重音的名字上靜默失敗。

### 15.3 一個會被審稿人抓到的實質誤植

`CITATION_CHECK.md` 對 20 條被引用文獻逐條判 metadata 與 claim support，其中 19 條 SUPPORTED，唯一一條 **NOT SUPPORTED** 是：

> 三頻寬高斯 MMD 核被歸給 IQPopt（`recio_2025_iqpopt`, arXiv:2501.04776）。全文檢索該文 23 頁，只找得到**單一**中位數啟發法頻寬（`sigma = median_heuristic(X)`），「three bandwidths」「average Pauli weight … bandwidth」與均方根組合規則都不出現。逐字寫出這條規則的是 **`recio_2025_train`（arXiv:2503.02934）**：「σ₁ is such that the average Pauli weight of Z_a is 2, σ₃ is the square root of the median heuristic, and σ₂ = √((σ₁²+σ₃²)/2)」，並附 Table 1 的逐實驗 σ 值。

定稿已把 §2 與附錄 C 的兩處都改引 `recio_2025_train`，`recio_2025_iqpopt` 只在引言引一次（作為實作 train-classically-deploy-on-quantum 框架的軟體）。另有一項 PARTIAL：`liu_2026_correlation` 論證 sparsification 的理由是硬體連通性、資源與可學習性，不是「維持古典困難的區間」，因此 Discussion 已改成中性的「scaling up would need them sparsified」。

### 15.4 Chen-Yu Liu 這一組的四篇 IQP 論文（全部引到）

| bib key | arXiv | 引在哪 | 那句話主張什麼 |
|---|---|---|---|
| `liu_2026_correlation` | 2603.06440 | 引言、§3.1、Discussion、附錄 C、附錄 D | 我們所複現的地圖與潛變數調適流程；`primaryclass` 更正為 cs.LG |
| `liu_2026_physics` | 2607.28866 | 引言 | 後續工作用物理知情的代理模型來初始化同一個固定 Walsh latent block，而我們是把方向學出來，是同一個瓶頸的兩種解法。原本在 bib 裡卻從未被引用 |
| `placidi_2026_connectivity` | 2604.12635 | Discussion | 在「不宣稱量子優勢」那一句旁邊，說明任何部署上的餘裕仍受量子位元連通性與雜訊門檻限制 |
| `lizziobosco_2026_pauli` | 2607.20409 | 引言 | 與 `recio_2025_iqpopt` 並列，作為同一種「期望值可古典計算」結構在生成模型之外的用法 |

刻意不引的：同名但不同人的 Chen-Yu Liu 中子物理論文，以及 Quantum-Train／QLSTM／tensor-network 那條線（不是 IQP 工作）。

### 15.5 新增的八筆文獻（每一筆寫進 bib 之前都線上查過）

| key | 文獻 | 引在哪 |
|---|---|---|
| `liu_2018_differentiable` | Differentiable learning of quantum circuit Born machines, Phys. Rev. A 98, 062324 | 引言，MMD 訓練 Born machine 的譜系 |
| `benedetti_2019_generative` | npj QI 5, 45 | 同一句 |
| `coyle_2020_born` | The Born supremacy, npj QI 6, 60 | 同一句（該文自稱 Ising-Hamiltonian-derived 而非 IQP，所以我們只主張 Born machine 譜系） |
| `cerezo_2025_does` | Does provable absence of barren plateaus imply classical simulability? Nat. Commun. 16, 7907 | 引言，「trainability itself pulls towards simulability」 |
| `binkowski_2018_demystifying` | Demystifying MMD GANs, ICLR 2018 | §2 指標定義，附錄 C 說明我們的 Conv2D-MMD² 就是 KID 但編碼器未預訓練 |
| `smolensky_1986_information` | Parallel Distributed Processing vol. 1, pp. 194–281 | §2，RBM 的起源 |
| `mustafa_2019_cosmogan` | CosmoGAN, Comput. Astrophys. Cosmol. 6, 1 | 附錄 C，古典方法生成宇宙學影像已成熟 |
| `gili_2024_generalization` | Phys. Rev. Applied 21, 044032 | 附錄 C，存在為此打造的泛化指標而我們沒有用 |

查過但沒用的兩筆：`gili_2023_doqcbms`（2207.13645）與 `hibatallah_2024_framework`（2303.15626）是套用該泛化框架而非提出它，所以引原始提案。

---

## 16. v12 初版 → v12 最終版 變更摘要

共 12 條。標 **[主張]** 的改變了論文宣稱什麼，標 **[引用]** 的改了引用，標 **[數字]** 的改了數值或慣例，標 **[結構]** 的只動排版。

| # | 變更 | 類型 |
|---|---|---|
| 1 | **對 RBM 的主張從「三欄全贏」收斂成「兩個 MMD² 欄領先（1.40 / 2.27 sd）、PDF–JS 在 1 sd 內（0.75）」**，摘要與 §3.3 同步 | [主張] |
| 2 | **失效原因從「固定的 latent block」改成「凍結的 core 加上逐張循序調適」**，並明說 `fixed basis, joint` 已補回大部分差距、我們跑的 Alg. 1 用的是自選低變異 block、block size 50 未掃描 | [主張] |
| 3 | **地圖的結論從「有利位置是必要非充分條件」改成「指標大致由強度直方圖決定，位置有利本身不構成證據」**，並加上 10-bit 編碼 I_CCI = 0.43 落在高 I_CCI 角落之外的但書 | [主張] |
| 4 | **新增三個控制**：記憶檢查（生成框到最近訓練影像的 L2 為 4.21 ± 0.10，真實留出框為 2.90）、物理量 Wasserstein-1（環半徑與兩個橢率分量）、以及「本文沒有把被破壞的影像當上界控制」這條限制 | [數字][主張] |
| 5 | **投影的措辭從 reconstruct 改成 project**，並公開三條參考線（最近鄰訓練影像 0.892 / 0.970、平均裁切影像 0.765 / 0.779、單一零初始化重啟 0.838 / 0.887）與挑重啟的規則（八次取擬合損失最低） | [主張][數字] |
| 6 | **加上兩條 shot-estimability 的誠實邊界**：精確 KL 暖啟動與投影都不是 shot-estimable；n=10 把 2ⁿ 個相位全部參數化，往上擴展需要稀疏化而本文未測 | [主張] |
| 7 | **加上 seed 定義**：訓練影像跨 seed 固定，所以 sd 只含初始化與取樣變異；以及「以 sd 為單位的領先」是描述性而非顯著性檢定 | [主張] |
| 8 | **三頻寬核的出處從 IQPopt 改成 `recio_2025_train`**，並寫出規則本身（σ₁ 使平均 Pauli weight 為 2、σ₃ 為中位數啟發法平方根、σ₂ 為兩者均方根） | [引用] |
| 9 | **Model II 的資料集引用從 `alexander_2023_domain` 改成 `ml4sci_deeplense`**；Bremner 2011／2016 拆成 worst-case 乘性與 average-case 加性兩句；`codsi_2025` 換成 `maslov_2024_fast`；補引 `walsh_1923_closed` 與 `silverman_1986_density`；新增八筆文獻；Liu 這一組四篇全部引到 | [引用] |
| 10 | **參數量欄改用單一慣例**（共用參數含 θ⁰ 加上 N 組 per-image latent），RBM 從 11,010 改為 11,510 / 15,960、fixed-basis 為 1,524 / 5,974、learned-direction 為 5,120 + MN、DCGAN 只算 generator | [數字] |
| 11 | **表 2 caption 的粗體規則從「領先 runner-up」改成「領先其他每一個 arm」**（沒有任何一格粗體因此改變）；抽樣改善幅度從「29–37%」改為「22–39%」；DCGAN 重跑差異下界從 12% 改為 5%／2% | [數字] |
| 12 | **節數由 7 併成 4**（引言、實驗設定、結果、討論與限制），附錄由 A–E 併成 A–D；`fig2_v12_msweep`（潛變數寬度）與 `figA_scaling_v12`（資料量縮放）兩張圖、固定 S 全表、M／H 掃描表、資料量縮放表為了守住三頁附錄上限而移出論文，**全部在中文完整版 `docs/IQP_LENS_PAPER_ZH_v12_full.md` 的「英文版為頁數所捨去的內容」一節放回來** | [結構] |
