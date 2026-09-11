# 我們的方法與原論文（arXiv:2603.06440）差在哪：資料、影像與協定的逐項說明（v11）

**用途**：給 <user> 與合作者的內部說明文件，不是論文。比論文詳細、比論文白話，但技術術語、表格 header、程式檔名一律用英文。
**對照的原論文**：Liu, Placidi, Brunner, Rinaldi（Quantinuum），*Toward Generative Quantum Utility via Correlation–Complexity Map*，arXiv:2603.06440 v2，2026-05-05（`docs/PAPER_2603.06440_SUMMARY.md:7-12`）。以下一律稱「原論文」。
**我們的投稿**：NeurIPS ML4PS 2026 workshop，稿件在 `paper/ml4ps2026/v11/main.tex`。
**版本**：v11，2026-09-09。相對 v10（`docs/IQP_LENS_METHOD_DIFFERENCES_ZH.md`，2026-09-08）的差異列在文末第 11 節。
**這一版的兩個系統性改動**：
1. **所有生成實驗的數字改成 5 seeds 的 mean ± sd**（seed 0–4，sample sd，ddof = 1），來源是 `paper/ml4ps2026/seed_tables.md`（主表、per-class、scaling、baseline 步數、θ⁰ 消融、main.tex §3.2 診斷、per-restart、重跑差異）、`seed_tables_b.md`（map、M 掃描、App. C／App. D）、`seed_tables_c.md`（turbulence 附錄）。
2. **評估指標換成 metric protocol v11**（`paper/ml4ps2026/v11/facts.md` §「Addendum: metric protocol v11」），舊協定讓每個 arm 用自己的 histogram 網格與自己的 kernel 被評分，v11 把兩者固定住。細節見 §5.5。

所有數字後面括號註明來源檔與表名，數字不憑印象。**單一數字沒有 seed 版本時會明講「seed 0，無 5-seed 版本」**。

---

## 1. 一頁摘要

1. 我們把原論文的兩件事（correlation–complexity map、IQP 生成模型加 latent adaptation）從它的 2D 湍流速度場搬到 DeepLense 模擬強重力透鏡影像。
2. **最大的兩個改動是資料表示法與 latent 參數化**：原論文把每個樣本編成 18-bit `(x, y, intensity)` tuple 跑 18 qubits，我們改成把「質心對齊後的 32×32 crop 本身當成 1,024 格的 pmf」跑 10 qubits；原論文把 50 個 Walsh 相位當成固定的 latent block 依序調，我們改成學一組仿射方向 `θ = θ⁰ + Φᵀz` 並且對 N 張影像聯合訓練。兩個改動都是被量測逼出來的，不是品味選擇。
3. **機制結論不變且被 5 seeds 支持**：固定基底的 sequential 協定在影像上把非 anchor 的訓練影像重建成 corr 0.4411 ± 0.0419（N=11）的雜訊（`seed_tables.md`「Section 3.2 diagnostics / Frozen-core reconstruction」），而學到方向後 held-out projection corr 到 0.9684 ± 0.0004（N=100，`seed_tables.md`「Held-out projection correlation, by restart」）。這一段是全文最穩的部分。
4. **生成品質的結論在 v11 變弱了**。N=11 時 mean 最好的 IQP arm 在 PDF–JS（0.0008062 ± 5.176e-05）與 flux-aligned MMD²（0.02737 ± 0.002194）兩欄都是全表最佳 mean，但**兩欄的領先都被 seed spread 吃掉**：PDF–JS 與 RBM、DCGAN 在 1 sd 內重疊，flux-aligned 與 DCGAN 重疊；raw MMD² 則是 baseline 領先（DCGAN 0.05094 ± 0.01514）（`seed_tables.md`「Table 2 (v11) / wins that the seed spread does not support」）。N=100 時 DCGAN 四欄 mean 全部最佳，其中 PDF–JS、MMD²、MMD²_u 三欄的領先連 1 sd 都推不翻（同表）。
5. **我們不主張量子優勢，也不主張 IQP 在這份資料上贏**。可以主張的是：學到的方向讓 IQP 第一次能在影像上生成並泛化；以及在參數量小 2 到 39 倍的條件下，flux-aligned 那一欄仍與 DCGAN 打平（N=100 IQP 0.02551 ± 0.004413 對 DCGAN 0.02122 ± 0.004081，1 sd 內重疊）。

---

## 2. 原論文方法回顧（讀者不必回去翻原文）

### 2.1 IQP 電路與兩個指標

| 項目 | 定義 | 出處 |
|---|---|---|
| IQP circuit | `U(θ) = H^⊗n exp(i Σ_s θ_s Z_s) H^⊗n`，量測分布 `q_θ(x) = |⟨x|U(θ)|0^n⟩|²` | `docs/PAPER_2603.06440_SUMMARY.md:50,64` |
| 為什麼用它 | 古典高效模擬 IQP 取樣會使 polynomial hierarchy 崩塌，而 MMD loss 需要的 `⟨Z_s⟩` 仍可古典計算，所以能 train-on-classical / deploy-on-quantum | `docs/PAPER_2603.06440_SUMMARY.md:34-35`；`docs/REF38_IQPOPT_METHOD_NOTES.md:24` |
| **QCLI** | 對經驗 pmf 做 Walsh–Hadamard 得 `P(s)`，依 `|s| = k` 聚合成 order spectrum `m_k`，與 i.i.d. 公平位元的 binomial baseline `b_k = C(n,k)/2ⁿ` 取 Jensen–Shannon divergence | `docs/PAPER_2603.06440_SUMMARY.md:71-81` |
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
| 參數量 | Table 2 的 IQP 標 31k / 63k。逆向核對：n=18 時 `Σ_{k≤6} C(18,k) = 31,180`、`Σ_{k≤7} = 63,004`，高度疑似對應 `|s|≤6` / `|s|≤7` 截斷，但原文未明講 | `docs/IQP_REPLICATION_SPEC_2603.06440.md:235`；`seed_tables_c.md` C6 |

### 2.3 目標函數、評估與 baseline

- **訓練用 MMD**：`L = Σ_s c_s (⟨Z_s⟩_θ − ⟨Z_s⟩_data)²`，Gaussian kernel。**原論文採用 IQPopt [38] 這個 library 的 kernel，但從未寫出它用的 σ**（`docs/IQP_REPLICATION_SPEC_2603.06440.md:232`）。我們稱為 paper kernel 的東西，實際上是 **IQPopt 的三 bandwidth 規則**在 n=10 重算的結果，`σ = {0.989, 1.729, 2.236}`，對應 mean Pauli weight 2.00 / 0.77 / 0.48（`v11/main.tex:444-447`）。這在 v11 是一個措辭修正：它不是原論文公布的 kernel，是我們照 library 規則補出來的。
- **評估指標**：PDF–JS（強度邊際直方圖的 JS divergence，B = 50 bins）與 Conv2D-MMD²（三層 Conv2D–ReLU stride 2 + global average pooling 的固定隨機 encoder，N = M = 200）（`docs/IQP_REPLICATION_SPEC_2603.06440.md:119,129-133`）。我們對這兩個指標的實作在 v11 有實質修改，見 §5.5。
- **Baseline**：RBM（12k 參數，同樣的 18-bit 表示、同樣 11 個 snapshot）與 DCGAN（51k / 0.2M / 1.2M）（`docs/IQP_REPLICATION_SPEC_2603.06440.md:83-104`）。
- **原文未給、我們必須自己決定的**：turbulence 訓練的 optimizer 與 learning rate、訓練 kernel 的 σ、random seeds、**取樣後聚合回網格的規則**（`docs/IQP_REPLICATION_SPEC_2603.06440.md:225-234`）。最後一項在我們的復刻裡造成了實質差異，見 §5.4。

---

## 3. 我們的資料與它的特性

資料是 `<HOME>/mae-lensing/model_II.npz`（DeepLense Model II），三類 `axion` / `cdm` / `no_sub`，train 80,194 / val 8,910 張 64×64 float32（`v11/facts.md §1`）。**本節全部是固定資料集上的一次性量測，不含隨機訓練，所以沒有 seed 版本**：split 由 `R3_SPLIT_SEED` 釘死且與任何 arm 的 `--seed` 解耦（`seed_tables.md` 開頭；`data_lens_r3.py:264-267`）。

| # | 量測 | 數字 | 出處 | 為什麼重要 |
|---|---|---|---|---|
| 1 | per-image min-max 正規化 | 8,910 張**每一張**都恰有 1 個像素 = 0、1 個像素 = 1；per-image 總 flux 125.7 ± 28.2 | `LENS:30` | 跨影像的絕對亮度沒有意義，亮度尺度必須另外處理。這也是 v11 把 PDF–JS 的 bin 固定在 [0,1] 的依據（§5.5） |
| 2 | raw 6-bit 強度量化的資訊量 | entropy 只有 1.42 bits，83.75% 的像素落在 level 0；quantile 量化可到 6.00 bits | `LENS:45-48` | 原論文的 z-register 在這批影像上有 4.6 bits 是浪費的 Hilbert space |
| 3 | 影像不是稀疏的 | 有效支撐 `1/‖p‖²` ≈ 331 / 4,096 像素（8.1%），但每張只有 1 個嚴格為 0 的像素 | `LENS:60` | 稀疏化技巧會失效，能用的是「能量集中」 |
| 4 | 影像流形維度 | 前 6 個 PC = 89.4% 變異；PC1/PC3（平移）合計 49.6%；PC2 = flux/大小；PC4/PC5 = ellipticity | `LENS:176,186-189,194` | 實質是 5 維：(x, y, flux, e1, e2)。prior 必須顯式建模平移 |
| 5 | class 訊號在哪 | PC6 之後才是子結構，只佔 3.0% 變異；nearest-centroid 分類只有 38.3%（chance 33.3%） | `LENS:190,205` | 以像素重建為目標的生成模型會系統性忽略掉唯一有科學價值的成分，所以我們不以 class-conditional 為主要目標 |
| 6 | 幾何 | Einstein 環半徑 r ≈ 8 px，rms 半徑 9.87–10.14 px，光心偏移 3.36–3.42 ± 1.5–1.7 px | `LENS:225-227` | 透鏡不置中，這是最大的單一變異來源 |
| 7 | 方位角結構是噪音 | split-half 相關只有 r = 0.45 / 0.51 / 0.69 | `LENS:229` | 不要為方位角加任何偏好方向的 generator |
| 8 | 18-bit tuple 編碼的代價 | `|s|≤6` 需要 31,180 個 generator，只涵蓋 79.9% 的 Walsh 能量 | `LENS:275` | 這就是原論文 31k 參數那一格的實際覆蓋率 |
| 9 | image-as-pmf 的效率 | 12-bit `|s|≤6` 用 2,510 個 generator 涵蓋 95.6%；10-bit（32×32）`|s|≤6` 用 848 個涵蓋 96.7% | `LENS:279,285` | 參數少 12.4 倍、覆蓋反而更好。這是換編碼的直接依據 |
| 10 | bit 版面不重要 | Morton/Z-order 的 order spectrum 與 standard 逐點相同；Gray code 是 GF(2)-linear 可逆映射，Walsh 能量 multiset 差異為 0.0 | `LENS:249-250` | 版面優化在數學上不可能有幫助 |
| 11 | 原論文的 latent 選法不是最好的 | Hamming-first-50 的 class acc（LDA）0.488；「`|s|≤2` 內依方差排序」是 0.668，`|s|≤4` 是 0.717 | `LENS:136-139,144` | 「限制低階」是對的，錯的是在低階裡用固定字典序 |

---

## 4. 逐項差異表

每一列：原論文做法 / 我們的做法 / 為什麼 / 證據。標 **[v11 修正]** 的列是這一版改過的。

| # | 面向 | 原論文 | 我們 | 為什麼改 | 證據 |
|---|---|---|---|---|---|
| 1 | **資料與任務** | 1000 個湍流 snapshot，有時間軸，任務是內插到未見時間 τ | DeepLense Model II 靜態影像，三類 pooled，任務是從 latent prior 無條件生成 | 影像沒有時間軸，Eq. 31 的 latent 內插無物可插 | `v11/main.tex:113-171`；`v11/facts.md §1` |
| 2 | **qubit 數與編碼** | 18-bit `(x, y, intensity)` tuple，n = 18 | 質心對齊 → 中心 32×32 crop → 1,024 格 pmf，`code = (kx<<5)|ky`，**n = 10** | raw 強度軸只帶 1.42 bits；tuple 編碼把能量推高一個 Hamming 階，`|s|≤6` 要 31,180 個 generator 卻只涵蓋 79.9%，10-bit crop pmf 用 848 個就到 96.7% | `LENS:275,285` |
| 3 | **generator 集合** | 依 `|s|` 截斷（推定 `|s|≤6` = 31,180；`|s|≤7` = 63,004） | 全部 1,024 個 Z-string（n=10 的完整 Walsh basis），exact simulation | n=10 時完整基底才 1,024 個參數，比原論文的截斷版還少 30 倍 | `docs/IQP_LENS_ROUND3_PREREGISTRATION.md:14`；`codec10.py:13-15` |
| 4 | **latent 參數化** | 固定的 50 個 Walsh 相位（Hamming 順序前 50 個） | 學到的仿射方向 `θ_i = θ⁰ + Φᵀz_i`，`Φ ∈ R^{M×1024}` 共用、`z_i ∈ R^M` 每張影像，`M ∈ {2,4,8}`；Φ 每列 L2 normalise 到 32，z 加 `10⁻³` 相對 L2 penalty | 固定方向做出來的是 XOR-shift 干涉樣式（§6）；換成學方向後 held-out projection corr 從 mean-image 水準 0.7439 ± 0.0000 提到 0.9060 ± 0.0021（M=2）與 0.8577 ± 0.0559（M=4） | `seed_tables_b.md` §2「Fig. 2(a) held-out projection corr vs M」 |
| 5 | **訓練協定** | Sequential：先在單一 anchor 上訓 core，凍住後逐個 snapshot 調 latent | **Joint**：`θ⁰`、`Φ` 與全部 N 個 `z_i` 一起訓（Adam lr 0.02，固定 4,000 步） | 用原協定時，除了 anchor 以外的每張訓練影像只重建到 corr 0.4411 ± 0.0419（N=11）/ 0.4282 ± 0.0171（N=100），而 anchor 自己是 0.9985 ± 0.0004 | `seed_tables.md`「Section 3.2 diagnostics / Frozen-core reconstruction」 |
| 6 | **目標函數** **[v11 修正]** | 每步用 500 個 model sample 估 MMD | **閉式（exact）MMD**，三個變體：paper kernel（**IQPopt [38] 的三 bandwidth 規則**在 n=10 重算，σ = 0.989 / 1.729 / 2.236，mean Pauli weight 2.00 / 0.77 / 0.48。原論文採用該 kernel 但未公布 bandwidth，數值是照 library 規則補的）、flat MMD（每個 Walsh 係數等權，可用硬體 shot 估）、exact KL（只當診斷） | 10-qubit pmf 只有 1,024 個振幅，精確可算，沒有理由引入抽樣噪音 | `v11/main.tex:444-447`；`docs/REF38_IQPOPT_METHOD_NOTES.md:90,107` |
| 7 | **θ⁰ 的暖啟動** **[v11 新增]** | 無此步驟 | `θ⁰` 先用 **1,000 步 exact-KL** 對平均 crop 暖啟動 | KL 不是硬體可估的，所以做了消融：換成 flat-MMD 暖啟動（全程 shot-estimable）後，兩個 N、四個指標**沒有一格差超過 1 sd** | `seed_tables.md`「theta0 warm-start ablation」與其 wins 表；`v11/main.tex:449-451` |
| 8 | **KL 的地位** | 不使用 | 只當模擬專用診斷，寫在附錄，不進正文主表 | KL 需要完整的模擬分布，無法從硬體 shot 估計 | `v11/main.tex:427-431` |
| 9 | **prior 與生成方式** | 對 latent 軌跡做時間內插 | prior 是 (a) 對訓練 z 擬合的 full-covariance Gaussian 或 (b) resample：抽一個訓練 z 再加 0.2× 中位最近鄰間距的 Gaussian jitter | round-2 診斷顯示高維 Gaussian 在少量 anchor 上擬出來的質量幾乎全落在沒有真 latent 的區域 | `R3PRE:104-116`；`R3RES:235` |
| 10 | **亮度尺度 S** **[v11 修正]** | 隱含在 tuple 的 z 值裡 | 電路只輸出 `q(x,y)`，亮度是外部純量 `field = q × S`，**S 是訓練集平均總亮度，隨 N 變**：N=11 為 **112.3**、N=100 為 **124.9**（held-out 為 123.9） | 影像做過 per-image min-max 正規化，絕對亮度不跨影像可比。v10 只寫了 124.92 一個值，等於漏掉 N=11 那一格 | `seed_tables.md`「Parameter counts and the flux scale S」；`v11/main.tex:168` |
| 11 | **評估** **[v11 修正]** | PDF–JS + Conv2D-MMD² | 同樣兩個指標但改用 **metric protocol v11**（固定 bin 網格、held-out 決定的 encoder 尺度與 bandwidth、200-vs-200 floor、同時報 biased 與 unbiased 估計量），另加 held-out projection correlation 與 flux-aligned Conv2D-MMD² | 舊協定讓每個 arm 用自己的 histogram 與 kernel 被評分，可被 generator 操弄（§5.5） | `v11/facts.md`「Addendum: metric protocol v11」 |
| 12 | **參數量的算法** **[v11 修正]** | Table 2 直接標 31k / 63k | 改用 **total-at-N**（shared + 每張影像的區塊）：fixed-basis IQP 1,524（N=11）/ 5,974（N=100）；learned-direction 5,164 / 5,520；**RBM 11,510 / 15,960**（joint RBM 每張影像 50 個 bias）；DCGAN generator 199,837 加上 training-only discriminator 53,497 | v10 的 Params 欄混了三種慣例（單張影像的參數向量、含 N 的總量、只算 generator），讀成總量之後「N=100 最小的模型」不是原本那一列 | `seed_tables.md`「Parameter counts and the flux scale S」的警告段與表 |
| 13 | **Baseline 的訓練預算** **[v11 修正]** | 固定步數 | **改成 validation early stopping**：RBM 選到 160k（N=11）/ 80k（N≥30）、DCGAN 選到 5k–30k；IQP 仍固定 4,000 步 | 20k 步時 RBM 的 reconstruction MSE 還在降（N=100：20k 是 0.0662，160k 是 0.0041），固定 20k 等於讓 baseline 欠訓練 | `seed_tables.md`「Baseline step counts chosen on the validation set」；`v11/facts.md`「baseline convergence sweep」 |
| 14 | **Conditioning** | 時間 τ | 主結果全部 pooled（三類混合）；per-class 版本移到 **附錄 F** | 三類的差異只佔 3.0% 的像素變異、nearest-centroid 只有 38.3% | `v11/main.tex:483`（`\section{Per-class samples}`）；`LENS:190,205` |

---

## 5. 影像處理流程逐步說明

### 5.1 每一步做什麼、為什麼、對應程式

| 步驟 | 做什麼 | 為什麼 | 程式與參數 |
|---|---|---|---|
| 1. Centroid alignment | 算強度加權質心 `(cy, cx)`，取整數位移，不做插值 | 光心偏移 3.4 ± 1.6 px 是最大的單一變異來源（`LENS:225-227`）；整數位移不會憑空造出或抹平任何光 | `data_lens_r3.py:72-95`（`align_crop`） |
| 2. 32×32 crop | 取以質心為中心的 32×32 視窗，記下角落 `(dy, dx)` | 對齊後的 crop 保留 99.4% 的光（最低 98.4%）；`|s|≤6` 覆蓋 96.7% 比 64×64 的 95.6% 還高 | `R3PRE:9-11`；`LENS:285` |
| 3. pmf 正規化與編碼 | `p(kx,ky) = crop / sum(crop)`，`code = (kx<<5)|ky` | 把影像本身當機率分布，避開 tuple 編碼的高階代價 | `codec10.py:9-10,19-21,54-68` |
| 4. 訓練 / held-out 分割 | 對 pooled train_x 做一個固定 permutation：前綴 N 張當訓練（N=11 是 N=100 的子集），接下來 2,000 張當 latent 選擇池，held-out = val_x 固定 permutation 的前 200 張。split seed 固定為 0，與任何 arm 的 `--seed` 解耦 | 資料不可以隨訓練 seed 移動。**這一條在 5-seed 版本被逐次 assert**：每個 seed 的 held-out 總亮度都是同一個值（n_holdout=200 為 123.9） | `data_lens_r3.py:264-267`；`seed_tables.md` 開頭 |
| 5. 生成 | 從 prior 抽 z → `θ = θ⁰ + Φᵀz` → 三次 FWHT 得 exact `q` → `crop = q × S` | S 是訓練 crop 的平均總亮度：N=11 為 112.3、N=100 為 124.9，對應的 held-out 平均是 123.9（所以 N=11 的生成影像系統性偏暗 9%，N=100 偏亮 0.8%） | `codec10.py:76-86`；`seed_tables.md`「Parameter counts and the flux scale S」 |
| 6. 貼回 64×64 | 把 32×32 crop 貼進全黑的 64×64 框，角落 `(dy, dx)` 從訓練影像的偏移分布中抽樣（有放回） | 評分必須回到原始影像尺度；偏移分布是資料的真實性質（PC1+PC3 = 49.6% 變異） | `codec10.py:123-146`；`data_lens_r3.py:315-320` |
| 7. 顯示 | 全部圖片用 `magma`、線性、`vmin = 0`、`vmax = 1` | 所有樣本被縮放到同一個 S，共用 0–1 尺度才誠實 | `figures/FIGURE_PROVENANCE.md:26,34` |
| 8. 超出 [0,1] 的像素 **[v11 修正]** | v10 只在畫圖時被 `vmax = 1` 截掉並揭露。**v11 把 clip 變成評分協定的一部分**：兩邊都 clip 到資料範圍 [0,1]，並逐 run 記錄 `gen_clip_frac` 與 `gen_max` | 106 個 seed-0 run 裡有 **64 個**有像素超過 1，比例很小（中位 5.4e-04、最大 6.2e-03，`rbm_n30`），但在 v10 這些像素在決定整個比較的 histogram 寬度與 encoder 輸入尺度。DCGAN 是唯一從不超過 1 的 arm（tanh generator 後接 `clip((x+1)/2, 0, None)`），所以它是唯一不受 v10 bin-widening 影響的 arm | `v11/facts.md`「Clipping (which arms v10 was flattering)」 |

### 5.2 Round 1 為什麼失敗（18-bit tuple 編碼）

- 照 pre-registration 字面套用原論文編碼：`(kx, ky, kz)`，`kz = floor(64 × intensity)`。跑前就量到 83.6–84.0% 的像素落在 intensity bin 0，z-channel 幾乎被背景吃掉（`docs/IQP_LENS_PILOT_RESULTS.md:24`）。
- 結果：IQP 是所有沒崩潰的 arm 裡**最差的**，NN-holdout 5.8 / 7.5 / 7.7（`PILOT:12-13`）。
- **同時，這批資料的 QCLI 高得漂亮**：per-image `I_QCLI` = 0.4697 / 0.4693 / 0.4755，配對的 uniform null 只有 0.0077（`PILOT:16`）。按原論文的選題準則這是教科書級的「IQP-compatible」資料，IQP 卻是最差的 arm。**「高 QCLI ⇒ 適合 IQP」在這裡被直接證偽。**
- sqrt 量化確實有效（PDF–JS 從 0.29–0.42 掉到 0.050–0.060，機制清楚：bin-0 佔比從 7.4% 拉回 69.9%）但不夠（`PILOT:14`）。

### 5.3 Round 2 為什麼失敗（強度尺度假象）

- Round 2 換成 12-bit image-as-pmf。表面上 MMD² 從 0.66–1.27 掉到 0.42–0.50（`docs/IQP_LENS_ROUND2_RESULTS.md:11`）。
- **但修好的是亮度尺度，不是空間結構**：pmf arm 與 class-mean 的 per-image 相關係數只有 −0.006 到 +0.041，弧/背景對比 0.98–1.15，而真影像是 17.6–19.0（`R2:11`）。三個 pmf arm、三個類別全部是零弧結構。
- 把每個 arm 的平均總亮度事後對齊 held-out 再重算，round-2 相對 round-1 的全部 MMD² 增益消失。改善的來源是 `field = q × S` 把平均總亮度鎖住，而 S 是外部給的（`R2:11`）。這個教訓直接生出 round 3 的 flux-aligned 次評分欄。
- **協定與 generator 數在 12-bit 上完全不是槓桿**：凍 latent + 5×10⁴ 抽樣 pmf 對上全參數 + 精確 pmf，三類 N=100 的差距是 `|ΔJS| ≤ 0.010`、`|ΔMMD²| ≤ 0.011`、`|ΔNN| ≤ 0.027`（`R2:12`）。**這個「協定不是槓桿」的結論在 turbulence 那條線上被 5 seeds 部分推翻**（凍 latent 其實值 −0.11，見 §8.4），差別在編碼：12-bit pmf 的 `θ_lat` 幾乎不動，18-bit tuple 的會。

### 5.4 聚合規則（last-write vs mean）這件事為什麼重要

原論文從未定義「同一格點落到多筆樣本時 z 怎麼合併」（`docs/IQP_REPLICATION_SPEC_2603.06440.md:234`）。我們在 turbulence 復刻裡兩種都做了：primary 用 mean、secondary 用 last-write-wins。**在 last-write-wins 慣例下，原論文自己的 IQP PDF–JS 數字是對得上的**（我們 0.0324 ± 0.0011 / 0.0473 ± 0.0005 對論文的 0.0238 / 0.0221），mean 慣例下就差一個數量級（0.2716 ± 0.0036 / 0.2360 ± 0.0013）（`seed_tables_c.md` C1、C2）。這是一個未指定的實作細節足以改變復現判定的實例。lens 這條線因為改用 exact pmf 重建，完全沒有這個問題。

### 5.5 評估協定 v11（這一版最大的方法學改動）

**問題**：v10 的兩個指標都讓被評分的 arm 部分決定自己的尺規（`v11/facts.md`「What the v10 protocol let a generator do to its own yardstick」）。

| # | v10 的漏洞 | 後果（可量測） |
|---|---|---|
| 1 | PDF–JS 的 50 個 bin 橫跨 real ∪ generated 的聯合 min/max（`metrics.py:32-36`） | 亮度峰值高的 arm 被用比較粗的 histogram 評分。**把 DCGAN 輸出的任一個像素改成 4.0，它的 PDF–JS 就除以約 13** |
| 2 | Conv2D encoder 的輸入正規化用同一組聯合 min/max（`metrics.py:94-95`） | 每個 arm 用自己的尺度進特徵萃取器 |
| 3 | Conv2D kernel 的 σ 用 real+generated 混合特徵的 median heuristic（`metrics.py:112-116`） | **每個 arm 用不同的 Gaussian kernel 被評分**，排名可被 kernel 選擇翻轉 |
| 4 | floor 是 100-vs-100，而每個 model row 是 200-vs-200 | biased MMD² 的 bias 約 1/n，floor 被高估約 2 倍 |

**v11 的定義**（`v11/facts.md`「v11 definitions」）：

- **PDF–JS**：兩邊都 clip 到資料範圍 [0,1]（Model II 做過 per-image min-max，0 與 1 就是資料的精確端點），50 個等寬 bin 固定在 [0,1]，eps 平滑，JS 以 bit 計。bin 網格不依賴任何一邊的輸入；被 clip 移動的質量逐 run 記錄成 `gen_clip_frac` / `gen_max`。
- **Conv2D-MMD²**：encoder 與 v10 相同（三層 stride-2 conv 1→16→32→64 加 global average pooling，`CONV_SEED = 0`），但**輸入範圍取自 held-out 集**、**Gaussian bandwidth 用 held-out 特徵的 median heuristic 算一次並由全表共用**（pooled σ = 0.005547 raw、0.002558 flux-aligned）。N = M = 200。同時存 biased V-statistic（`conv_mmd2`，主表引用，保持與 v10 同一個欄位語意）與 unbiased U-statistic（`conv_mmd2u`）。
- **Floor**：20 組 200-vs-200 的 real-vs-real 抽樣（取自 held-out split 剩下的 validation 影像），跟每個 model row 用同一個 encoder context 評分，報 mean ± sd。

**協定換掉之後 floor 的變化**（`v11/facts.md`「Floor, v10 against v11」）：

| kind | v10（100 v 100，單一 split） | v11 n = 100（20 draws） | v11 n = 200（20 draws） | v11 n = 200 unbiased |
|---|---|---|---|---|
| pooled PDF–JS | 3.41e-05 | 7.50e-05 ± 3.6e-05 | 4.08e-05 ± 2.1e-05 | n/a |
| pooled MMD² | 0.004006 | 0.00931 ± 0.0057 | 0.00481 ± 0.0028 | 0.00033 ± 0.0028 |
| axion MMD² | 0.005915 | 0.00947 ± 0.0060 | 0.00360 ± 0.0017 | −0.00074 |
| cdm MMD² | 0.017943 | 0.00804 ± 0.0044 | 0.00302 ± 0.0017 | −0.00121 |
| no_sub MMD² | 0.016027 | 0.00933 ± 0.0076 | 0.00446 ± 0.0027 | 0.00019 |

讀法：biased floor 在 n 加倍時減半（0.00931 → 0.00481，比值 1.94），unbiased floor 對 n 平坦且與 0 相容，這是 U-statistic 該有的行為。**v10 的 per-class floor 高了 2 到 6 倍**；v10 的 pooled floor 剛好落在正確值附近純屬運氣（單一 100-vs-100 split 是一個 sd 超過自身 mean 一半的分布的抽樣）。

**協定換掉之後排名的變化**（`v11/facts.md`「Table 2 under v11」，seed 0）：只有兩個 winner 移動。N=100 的 PDF–JS winner 從 learned M=4 paper kernel 換成 fixed-basis joint（兩者原本只差 7e-05，而 gen_max 較高的那個在加寬自己的 bin）；N=11 的 raw MMD² winner 從 RBM 換成 learned M=4 flat MMD（RBM 那個 run 的 clip fraction 是全表最大的 4.8e-03）。其餘欄位的 winner 不變。

**為什麼這件事非做不可**：上面兩個機制各自都能在不改變模型的情況下改變論文的結論欄位，一個是 13 倍的 PDF–JS，一個是 RBM 的排名。這不是精修，是把可被操弄的尺規換成不可被操弄的。

---

## 6. 為什麼原協定在影像上失效，而「學方向」有效

### 6.1 三個跑前做完的機制測試（因果鏈，全部 5 seeds）

| 測試 | 5-seed 結果 | 讀法 | 出處 |
|---|---|---|---|
| (i) 單張 crop 的表達力 | 一張對齊後的 crop 被 1,024 相位的 10-qubit IQP 幾乎精確擬合：KL corr **0.9999 ± 1.05e-05**、flat MMD **0.999966 ± 5.749e-06**、paper kernel（auto3）**0.9964 ± 0.0007963** | 表達力與 kernel 選擇都不是瓶頸。sd 在第五位小數，這是全文最穩的量 | `seed_tables.md`「Section 3.2 diagnostics / Single-crop full-parameter fit」 |
| (ii) 固定基底的插值 | 沿論文的 Hamming latent 在兩張影像之間走，中途 `max(corrA, corrB)` 掉到 0.233、ringiness 掉到 0.353，而像素 cross-fade 對照在每個 t 上 corr 都是 1.0（seed 0，無 5-seed 版本） | latent 空間裡兩個「好點」之間全是壞點，所以從 prior 抽樣得到斑塊 | `R3RES:207` |
| (ii-b) ancilla 能不能救 | ancilla 邊際被驗證為恰好等權重的兩個 hologram 非相干混合（每支 mass 0.500000000000，殘差 ≤ 2.6e-18；解析結果，無 seed） | 沒有「混合角」可調，ancilla 加不了東西 | `R3RES:207` |
| (iii) 凍住 core 之後的其他訓練影像 | anchor 自己 **0.9985 ± 0.0004356**，其他訓練影像 **0.4411 ± 0.04186**（N=11）/ **0.4282 ± 0.01714**（N=100） | 原協定不是實作問題，是協定本身在影像上的行為。v10 引的 0.47 是 round-2 codec 的對應值，round-3 codec 下是 0.44 | `seed_tables.md`「Frozen-core reconstruction of the other training images」 |
| (iv) 學方向 | 20 張 crop 的 controlled sweep：M=4 held-out projection corr **0.8577 ± 0.0559**、M=2 **0.9060 ± 0.0021**；**配對的 random-direction 控制是 0.7458 ± 0.0005**、不含 latent 的 M=0 是 **0.7439 ± 0.0000** | 增益仍然來自「學到」方向，但幅度比 v10 說的小：要贏的是 0.746 這條線，不是 v10 引的 0.672 | `seed_tables_b.md` §2 的 M 表與 reference level 表 |

**關於 v10 引的 0.672**：那是 mismatched 配對（把未訓練的 Φ 接在**已訓練的 M=4 θ⁰** 上），5 seeds 為 0.6468 ± 0.0484。正確的配對控制是「未訓練的 Φ 接在 latent-free 的 M=0 θ⁰ 上」，5 seeds 為 0.7458 ± 0.0005，**這是比較難的一條線**，v11 改用它（`seed_tables_b.md` §2 reference level 表）。

**機制的一句話說明**：固定的 Walsh 方向在相位空間裡作用成 XOR-shift 干涉樣式，它會把兩張影像的像素 cross-fade 起來，做不到「把環移動或拉伸」這種幾何操作。

### 6.2 正式跑的數字（N = 100）

| 量 | 數字 | 出處 |
|---|---|---|
| 最佳 held-out projection corr（M=4，flat MMD，N=100） | **0.9684 ± 0.0003872** | `seed_tables.md`「Held-out projection correlation, by restart」 |
| 同一個 arm 在 N=11 | 0.9102 ± 0.002598 | 同上 |
| paper-kernel arm 的 projection | 0.7921 ± 0.005793（N=11）/ 0.8047 ± 0.01163（N=100） | 同上 |
| **只用零初始化那一個 restart** | 0.8991 ± 0.0176（N=11）/ **0.9684 ± 0.0003872**（N=100） | 同上 |
| **以 loss 選 restart（模型自己做得到的選法）對上以 corr 選（論文報的）** | 四列的差距分別是 0.0000 / 0.0000 / 0.0041 / 0.0012，**最大 0.0041，全部 ≤ 0.005** | 同上，「best by LOSS」對「best by CORR」兩欄 |
| 「從 100 張訓練影像挑最近那張」的參考上限 | 0.970（seed 0，無 5-seed 版本） | `outputs_lens_r3b/.../meta.json` |
| 平均訓練影像的參考下限 | 0.779（seed 0，無 5-seed 版本） | 同上 |

**這一格是 v11 加固最多的地方**：v10 只報了「八個 restart 取最好」的 0.968，而那個選法是拿被評分的指標本身去選（select on the reported metric）。v11 把它拆開，結果是**拆開後幾乎不掉**：改用 loss 選最多掉 0.005，單獨用零初始化那個 restart（完全不用訓練集資訊）在 N=100 就已經是 0.9684。所以這個數字不是選擇偏差撐起來的。

### 6.3 M 的上限是相干半徑，不是參數量 **[v11 結論修正]**

| M | held-out projection corr（5 seeds） | train corr | 讀法 |
|---|---|---|---|
| 0（無 latent） | 0.7439 ± 0.0000 | 0.7861 ± 0.0000 | 與 mean training image 同值；init-independent |
| **2** | **0.9060 ± 0.0021** | 0.9456 ± 0.0009 | **平均最高且最穩（sd 0.0021）** |
| 4 | 0.8577 ± 0.0559 | 0.9325 ± 0.0217 | mean 略低於 M=2，sd 大一個數量級 |
| 6 | 0.7378 ± 0.0672 | 0.9237 ± 0.0201 | 掉回 mean-image 水準（0.7439）以下 |
| 8 | 0.2184 ± 0.1213 | 0.9486 ± 0.0009 | 崩潰 |
| 16 | 0.1457 ± 0.0048 | 0.9965 ± 0.0002 | 崩潰（train corr 反而最高，是過擬合到相位空間的碎片） |

出處：`seed_tables_b.md` §2「Fig. 2(a)」表。

**這是 5 seeds 改掉一個結論的地方**（`seed_tables_b.md` §2 的 "This is the block where seeds change a conclusion" 段）：seed 0 時 M=4 是 0.9113，是這五個 M=4 run 裡**最好的一個**（per-seed：0.9113 / 0.8979 / 0.8839 / 0.8066 / 0.7889）。所以 v10 的「M=4 是工作點」站不住；誠實的說法是 **M=2 與 M=4 是工作區間，M=2 較可靠**。M=6 在 v10 被說成「還在參考線之上」，5 seeds 之後它落到 mean-image 線（0.7439）之下。

**M=8、M=16 的崩潰在每個 seed 都成立**，這一半的結論不變。機制不變：相干半徑約 0.03–0.05，一旦 `‖z‖` 超過它，`exp(iθ)` 的非線性把重建打散。這是相位參數化的固有限制，不是可以靠更多參數解決的問題。

**執行細節（會影響重現）**：`affine_latent.py` 設 `torch.set_num_threads(4)`。在 2 threads 下 M ≥ 6 的 joint Adam 落到略不同的最佳點（`z @ Φ` 的 matmul 歸約順序改變），seed 0 會偏離發表值約 7e-3，而 M = 0/2/4 仍對到 1e-10（`seed_tables_b.md` §2「Thread count is part of the protocol」）。

---

## 7. 結果對照表

### 7.1 主表（v11 指標協定，5 seeds，mean ± sd）

參數量用 total-at-N（見 §4 第 12 列）。出處：`seed_tables.md` Part 1「Table 2 (v11)」與「Parameter counts and the flux scale S」。

| Model | Params @N | N | PDF–JS | MMD² | MMD²_u | MMD²(flux-al.) |
|---|---|---|---|---|---|---|
| IQP, paper protocol (Alg. 1) | 1,524 | 11 | 0.007199 ± 0.0006771 | 0.1883 ± 0.0103 | 0.1849 ± 0.01027 | 0.2251 ± 0.02187 |
| IQP, paper protocol (Alg. 1) | 5,974 | 100 | 0.008383 ± 0.0004439 | 0.2176 ± 0.006072 | 0.2139 ± 0.006075 | 0.2659 ± 0.006753 |
| IQP, fixed basis, joint | 1,524 | 11 | 0.001629 ± 0.0004399 | 0.1227 ± 0.003883 | 0.1194 ± 0.003849 | 0.1253 ± 0.01326 |
| IQP, fixed basis, joint | 5,974 | 100 | 0.001042 ± 0.0001092 | 0.102 ± 0.007668 | 0.0983 ± 0.007662 | 0.08521 ± 0.0149 |
| IQP, learned dir. M=4 (flat MMD) | 5,164 | 11 | 0.002979 ± 0.0001767 | 0.08914 ± 0.002443 | 0.08597 ± 0.002479 | **0.02737 ± 0.002194** |
| IQP, learned dir. M=4 (flat MMD) | 5,520 | 100 | 0.002912 ± 0.0002788 | 0.07155 ± 0.002939 | 0.06811 ± 0.002961 | 0.02551 ± 0.004413 |
| IQP, learned dir. M=4 (paper kernel) | 5,164 | 11 | **0.0008062 ± 5.176e-05** | 0.09632 ± 0.002696 | 0.09304 ± 0.002712 | 0.07432 ± 0.006323 |
| IQP, learned dir. M=4 (paper kernel) | 5,520 | 100 | 0.001363 ± 0.0001801 | 0.1132 ± 0.004813 | 0.1095 ± 0.004808 | 0.09991 ± 0.007786 |
| RBM | 11,510 | 11 | 0.001669 ± 0.0008203 | 0.06537 ± 0.008124 | 0.06097 ± 0.008183 | 0.1106 ± 0.03881 |
| RBM | 15,960 | 100 | 0.0009171 ± 0.000132 | 0.05865 ± 0.004363 | 0.05425 ± 0.004423 | 0.08067 ± 0.02205 |
| DCGAN | 199,837 (+53,497 D) | 11 | 0.001106 ± 0.0005576 | **0.05094 ± 0.01514** | **0.04704 ± 0.01527** | 0.03883 ± 0.0104 |
| DCGAN | 199,837 (+53,497 D) | 100 | **0.0005534 ± 0.0001211** | **0.02225 ± 0.003118** | **0.01752 ± 0.003006** | **0.02122 ± 0.004081** |
| *Floor (200 v 200, 20 draws)* | n/a | n/a | *4.079e-05 ± 2.057e-05* | *0.004806 ± 0.002783* | *0.0003347 ± 0.002762* | *0.003979 ± 0.00221* |

**哪些「贏」撐得住 seed spread**（`seed_tables.md`「Table 2 (v11) / wins that the seed spread does not support」）：

| N | metric | mean 最好的 arm | 1 sd 內與它重疊的 |
|---|---|---|---|
| 11 | PDF–JS | IQP learned M=4 (paper kernel) | RBM、DCGAN |
| 11 | MMD² | DCGAN | RBM |
| 11 | MMD²_u | DCGAN | RBM |
| 11 | MMD²(flux-al.) | IQP learned M=4 (flat MMD) | DCGAN |
| 100 | PDF–JS | DCGAN | **無，領先成立** |
| 100 | MMD² | DCGAN | **無，領先成立** |
| 100 | MMD²_u | DCGAN | **無，領先成立** |
| 100 | MMD²(flux-al.) | DCGAN | IQP learned M=4 (flat MMD) |

**讀法（v11 版）**：N=11 時沒有任何一欄有站得住的贏家，四欄裡有兩欄的 mean 最佳者是 IQP arm，但都與 baseline 重疊；N=100 時 DCGAN 四欄 mean 全最佳，三欄的領先穩固，只有 flux-aligned 那欄與 IQP 重疊。**與 v10 的差別在於「贏」這個字**：v10 說 IQP 在兩個 N 都贏 PDF–JS 與 flux-aligned MMD²，5 seeds + v11 協定之後只剩下「N=11 mean 最佳但重疊」與「N=100 flux-aligned 打平」。

**兩邊都做 early stopping 之後誰贏**（`seed_tables.md`「With both sides early-stopped, who wins Table 2?」）：N=11 的 PDF–JS 與 flux-aligned 是 1 sd 內平手，其餘六格 DCGAN 贏。IQP 這邊即使被允許 early stopping，v11 指標下**沒有一格移動超過 1 sd**（`seed_tables.md`「Does early stopping change the IQP numbers by more than 1 sd?」），而且 validation 選出的步數在 1k–16k 之間毫無規律，這本身就是結論：IQP 的 validation 曲線是平的，4,000 步已經在 plateau 上。

### 7.2 Baseline 的訓練預算（v11 新增，會改變主表怎麼被讀）

| arm | N | validation 選到的步數（per seed） | median |
|---|---|---|---|
| RBM-10bit | 11 | 160000 ×5 | 160000 |
| RBM-10bit | 30–100 | 80000 ×5（每個 N） | 80000 |
| DCGAN-32 | 11 | 30000, 30000, 5000, 15000, 5000 | 15000 |
| DCGAN-32 | 30–100 | 5000 ×5（除 N=50 有一個 10000） | 5000 |

出處：`seed_tables.md`「Baseline step counts chosen on the validation set」。步數是在與 held-out 不相交的 200 張 validation 影像上，用 PDF–JS 與 raw MMD² 的名次平均選的；held-out 不參與任何選擇。**IQP 不做 early stopping**，所以這個程序只可能對 baseline 有利。

固定 20k 對上選出來的預算差多少（`seed_tables.md`「Appendix -- the same baselines at a fixed 20,000 steps」，v10 指標）：RBM N=11 的 PDF–JS 從 0.004252 ± 0.001401 降到 0.001591 ± 0.0002874；DCGAN N=11 從 0.02766 ± 0.03362 降到 0.001106 ± 0.0005576（超過一個數量級）。**v10 主表的 baseline 是欠訓練的**，這是 v11 必須換掉的原因。

**但 DCGAN 這個 arm 的數字本身有硬體噪音**（`seed_tables.md`「Repeat runs of an identical configuration」）：同一個 config、同一個 seed、只是換 job 名字重跑，raw MMD² 差距最大 **57.0%**（N=50）、PDF–JS 最大 **74.1%**（N=30）。cuDNN 沒有 deterministic 的 transposed-convolution kernel。RBM 與所有 IQP config 重跑都是 bit-identical。**所以 DCGAN 的每個數字在改 seed 之前就已經是一次抽樣**，而這個抽樣尺度與 DCGAN 對 IQP 的差距同一個量級。

### 7.3 Per-class 生成（v11 移到附錄 F）

出處：`seed_tables.md` Part 1「Table 3 (v11)」。

| Training set | N | PDF–JS | MMD² | MMD²_u | MMD²(flux-al.) |
|---|---|---|---|---|---|
| axion only | 11 | 0.003487 ± 0.0002465 | 0.08574 ± 0.002982 | 0.08236 ± 0.003012 | 0.02823 ± 0.005261 |
| axion only | 100 | 0.002841 ± 8.37e-05 | 0.07113 ± 0.002232 | 0.06762 ± 0.002247 | 0.02081 ± 0.002261 |
| cdm only | 11 | 0.002514 ± 0.0001139 | 0.08458 ± 0.002662 | 0.08133 ± 0.002683 | 0.02569 ± 0.004515 |
| cdm only | 100 | 0.002767 ± 0.0002411 | 0.07848 ± 0.001893 | 0.0751 ± 0.001909 | 0.01835 ± 0.002811 |
| no sub only | 11 | 0.002817 ± 0.0002377 | 0.06893 ± 0.001651 | 0.0654 ± 0.001689 | 0.02606 ± 0.00103 |
| no sub only | 100 | 0.002953 ± 0.0003167 | 0.07987 ± 0.001285 | 0.07654 ± 0.001279 | 0.02939 ± 0.002721 |
| *pooled* | 11 | *0.002979 ± 0.0001767* | *0.08914 ± 0.002443* | *0.08597 ± 0.002479* | *0.02737 ± 0.002194* |
| *pooled* | 100 | *0.002912 ± 0.0002788* | *0.07155 ± 0.002939* | *0.06811 ± 0.002961* | *0.02551 ± 0.004413* |

projection corr 是普通的影像相關係數，不受 v11 指標協定影響，只報一次：axion 0.9718 ± 0.0003707、cdm 0.9688 ± 0.0004435、no sub 0.9701 ± 0.0001667、pooled 0.9684 ± 0.0003872（N=100，`seed_tables.md` Part 2「Table 3」）。

三類都達到 pooled 水準；**不可以主張某一類比較容易**，理由有二：(a) 每類用的是不同的 held-out 集合；(b) 5 seeds 之後多數欄位的類間差距落在 1 sd 內（N=100 的 PDF–JS 四列全部互相重疊，`seed_tables.md`「Table 3 (v11) / wins that the seed spread does not support」）。此外 v11 把 per-class floor 修低了 2 到 6 倍（§5.5 floor 表），所以「離 floor 多近」這句話在 v11 要重寫。

### 7.4 Data scaling（Figure 5，v11 指標，5 seeds）

出處：`seed_tables.md` Part 1「Figure 5 (v11)」與其 wins 表。

| N | IQP M=4 MMD² | RBM MMD² | DCGAN MMD² | IQP M=4 MMD²(flux-al.) | DCGAN MMD²(flux-al.) |
|---|---|---|---|---|---|
| 10 | 0.0952 ± 0.001527 | 0.07649 ± 0.01655 | 0.04577 ± 0.01733 | **0.02712 ± 0.003066** | 0.04523 ± 0.008452 |
| 30 | 0.06775 ± 0.001577 | 0.06108 ± 0.005951 | 0.02399 ± 0.004494 | 0.02584 ± 0.004968 | 0.03241 ± 0.007044 |
| 50 | 0.06793 ± 0.00183 | 0.06269 ± 0.008308 | 0.02403 ± 0.007969 | 0.02479 ± 0.002751 | 0.02437 ± 0.009352 |
| 70 | 0.07242 ± 0.0008608 | 0.06742 ± 0.007281 | 0.02347 ± 0.006126 | 0.02262 ± 0.002206 | 0.02136 ± 0.003667 |
| 90 | 0.06985 ± 0.003029 | 0.05469 ± 0.002378 | 0.02262 ± 0.006715 | 0.02296 ± 0.001042 | 0.0191 ± 0.001832 |
| 100 | 0.07155 ± 0.002939 | 0.05865 ± 0.004363 | 0.02225 ± 0.003118 | 0.02551 ± 0.004413 | 0.02122 ± 0.004081 |

讀法：**IQP 與 RBM 都對 N 平坦**（IQP raw MMD² 0.0678–0.0952、RBM 0.0547–0.0765），DCGAN 從 N=10 的 0.04577 降到 0.02225 並在 N≥30 之後穩定領先 raw MMD²。flux-aligned 那一欄只有 **N=10 的 IQP 領先撐得住 seed spread**，N=30 mean 最佳但與 DCGAN 重疊，N≥50 換 DCGAN mean 最佳（N=50、70、100 與 IQP 重疊，N=90 不重疊）。**v10 說的「DCGAN 在 N=10 就已經領先」在 raw MMD² 上仍成立**，但 v10 引的「非單調起伏 3 倍」在 5 seeds 之後應歸因於 §7.2 的硬體噪音而不是資料量效應。

### 7.5 與原論文 Table 2 的定性對照

| | 原論文（turbulence） | 我們（lens，v11） |
|---|---|---|
| 小 N（11） | IQP 全面贏 RBM 與所有 DCGAN；DCGAN 在 11 筆資料下訓練崩潰（PDF–JS 0.245–0.367） | **沒有站得住的贏家**：兩欄 mean 最佳是 IQP 但都與 baseline 重疊；我們的 DCGAN 在 N=10 沒有崩（PDF–JS 0.003182 ± 0.001651、MMD² 0.04577 ± 0.01733） |
| 大 N（100） | 高容量 DCGAN 可在一兩個指標上打平或超越 IQP，但要 10 倍資料與 1.2M 參數 | DCGAN 在三欄穩定領先，只用 199,837 個 generator 參數 |
| Data scaling | 資料變多 DCGAN 才追上 | **不成立**：DCGAN 在 N=10 的 raw MMD² 就已經最好；IQP 與 RBM 對 N 平坦 |
| 絕對值 | PDF–JS 0.022–0.367，MMD² 0.896–1.472 | PDF–JS 0.00055–0.0084，MMD² 0.022–0.218 |

出處：`docs/PAPER_2603.06440_SUMMARY.md:161-178`；`seed_tables.md` Part 1「Table 2 (v11)」「Figure 5 (v11)」。
**警告**：兩張表的絕對值不在同一個尺度上（不同資料集、不同影像大小與 channel 結構）。任何並排都只能比排序，不能比絕對數值。

---

## 8. 原論文的內部主張，我們重測的結果

### 8.1 Correlation–complexity map：permutation null（我們新增的方法學）

原論文只把每個資料集配一個位置，沒有配對照。我們對透鏡與 D2 各加一個 **structure-matched permutation null**：保留每張影像或每個 snapshot 的完整強度直方圖，只把值在空間位置間隨機重排。

出處：`seed_tables_b.md` §1「Table 1 correlation–complexity map (5 seeds)」。

| 資料與編碼 | K | K 怎麼來的 | `I_QCLI` | `I_CCI` |
|---|---|---|---|---|
| Lens, 18-bit tuples | 4,096 | 窮舉：64×64 影像的每個 (kx,ky) 各一次 | 0.4715 ± 0.0000 | 0.8677 ± 0.0000 |
| &nbsp;&nbsp;permutation null | 4,096 | 同 K | 0.4154 ± 0.0000 | 0.8783 ± 0.0001 |
| Turbulence D2, 18-bit tuples | 4,096 | 從原生網格 subsample 到與 lens 同 K | 0.0232 ± 0.0002 | 0.9285 ± 0.0008 |
| &nbsp;&nbsp;permutation null | 4,096 | 同 K | 0.0114 ± 0.0000 | 0.9371 ± 0.0009 |
| **Turbulence D2, 18-bit（原生 K）** **[v11 新增]** | **50,000** | **原論文自己的 `N_t = 5×10⁴`** | **0.0910 ± 0.0000** | **0.9085 ± 0.0000** |
| &nbsp;&nbsp;permutation null | 50,000 | 同 K | **0.0981 ± 0.0007** | 0.8782 ± 0.0003 |
| Lens, 10-bit crop pmf | exact pmf | 不抽樣，32×32 crop 的精確母體 pmf | 0.4731 ± 0.0000 | 0.4313 ± 0.0000 |
| &nbsp;&nbsp;permutation null | exact pmf | 同 K | 0.1782 ± 0.0000 | 0.9634 ± 0.0004 |

**沒有一格的 seed spread 超過 0.0009**，所以這張表的結論完全不動：lens 18-bit 那一列的位置幾乎全部是編碼造成的（QCLI 只比自己的 null 高 0.056），只有 10-bit crop pmf 在兩個座標上都明顯偏離 null。

**原生 K 這一列是 v11 新加的，而且它讓原論文的選題依據更難成立**：在原論文自己的 `N_t = 5×10⁴` 上，turbulence 的 `I_QCLI`（0.0910）**低於自己的 permutation null**（0.0981），而在 subsample 到 K=4,096 時卻高於 null。兩個指數都是強烈依賴 K 的有限樣本統計量，所以一列只能對自己 K 的 null 讀，這正是每列各配一個 null 的理由。

另外兩個方法學要點（seed 0，性質是解析或掃描，非隨機訓練）：plug-in CCI 的有限樣本 null 落在 0.975–0.9998，所以「高 CCI」本身沒有資訊，有資訊的只有「明顯低於 null」；QCLI 的 null 也不是 0 且隨 K 單調上升（n=18 在 K=4,096 是 0.008，K=4.1×10⁵ 已達 0.405）（`MAP:50`）。

**這個位置對生成結果沒有預測力**：地圖說透鏡比 D2 更靠右上，實測是互有勝負，round 3 的 pre-registered 判定是 failure（`MAP:21,76-78`）。

### 8.2 Fig. 3（support mismatch）：一個主張重現，一個方向相反

出處：`seed_tables_b.md` §3「App. C（reference paper Fig. 3）5 learner seeds」。seed 動的是 learner 的初始角度；140 個 target circuit、它們的 10⁴-sample 資料集與 kernel bandwidth 都固定。

| G_n | median MMD² | Spearman(MMD², target `I_QCLI`) | median MMD²（exact-pmf-trained 控制） | Spearman（控制） |
|---|---|---|---|---|
| 50 | 9.578e-03 ± 1.1e-05 | **+0.9786 ± 0.0010** | 9.453e-03 ± 1.9e-05 | +0.9812 ± 0.0012 |
| 100 | 2.147e-03 ± 6.8e-06 | **+0.9009 ± 0.0014** | 2.109e-03 ± 1.3e-05 | +0.9019 ± 0.0016 |
| 150 | 1.437e-03 ± 2.0e-05 | **+0.8091 ± 0.0039** | 1.374e-03 ± 1.4e-05 | +0.8057 ± 0.0048 |

- **主張 (b)「學習者 gate 越多，包絡整體下移」：重現**，且 5 seeds 之間 median 移動不超過 2.0e-05。
- **主張 (a)「目標 QCLI 越高，達成的 MMD 上包絡越低」：在可及範圍內方向相反**，Spearman 為正且 5 seeds 之間移動不超過 0.004。加上兩個控制之後趨勢幾乎相同，排除有限樣本解釋。
- **Proposition II.2 的 floor proxy 仍隨 QCLI 下降，但 v11 必須指名是哪一個 proxy**：exact-pmf proxy 為 −0.9876 / −0.9883 / −0.9956（最壞 −0.9876，符合 v10 寫的 ρ ≤ −0.98）；**10⁴-sample proxy 在 G_n = 150 是 −0.9291，不滿足 ≤ −0.98**（`seed_tables_b.md` §3「One wording issue this surfaces」）。兩個 proxy 的 sd 都恰為 0，因為它們只是資料頻譜與學習者固定 Pauli 支撐的函數，不可能隨 learner 初始角度移動。
- **範圍限制**：我們的目標族在 exact QCLI ≈ 0.41 飽和（n=16、≤4-local 的天花板），而原論文畫到 ≈0.7，所以主張 (a) 被斷言的那一段我們構造不出來。

### 8.3 Fig. 4（QCLI 最大化 → CCI 跟著升）：低 QCLI 成立，高 QCLI 斷裂 **[v11 數字更新]**

出處：`seed_tables_b.md` §4「App. D（reference paper Fig. 4）5 SPSA seeds」。seed 動的是整條 SPSA 軌跡（mask 集合、初始角度、擾動流）。

| n | global ρ（per-seed mean ± sd） | global ρ（5 seeds pooled） | ρ for `I_QCLI` ≤ 0.3（pooled） | 精確均勻分布的 `I_QCLI` |
|---|---|---|---|---|
| 8 | +0.2534 ± 0.9210 | **+0.0814** | +0.7923 | 0.9816 |
| 12 | −0.5240 ± 0.3621 | **−0.5673** | +0.7938 | 0.9984 |
| 16 | −0.6338 ± 0.0985 | **−0.6352** | +0.9001 | 0.9999 |

**v10 的五個數字全部要改**：低 QCLI 區的 ρ 從 +0.83 / +0.76 / +0.95（3 pooled seeds）變成 **+0.79 / +0.79 / +0.90**（5 pooled seeds）；n=12、16 的 global ρ 從 −0.53 / −0.57 變成 **−0.57 / −0.64**。

**質性結論存活，而且其中一個變強**：低 QCLI 的耦合在每個 seed、每個 size 都是正的（per-seed 最小 +0.750）。**反轉現在在 n=8 也出現**：最高 QCLI bin 的 median CCI 從 3 seeds 的 1.000 掉到 5 seeds 的 0.074，n=8 也塌到自己的均勻校準點（0.9816, 0.0000），pooled global ρ 從 +0.91 掉到 +0.08。n=12、16 的同一個 bin 幾乎沒動（0.462→0.450、0.472→0.529），所以這是 n=8 追上另外兩個，不是新效應。

**但 global ρ 不是穩定的摘要統計量**：n=8 的 per-seed 值是 +0.994 / +0.959 / +0.808 / −0.889 / −0.604（5 個裡 3 個為正），n=12 是 −0.815 / +0.077 / −0.775 / −0.471 / −0.636（5 個裡 1 個為正）。一條軌跡在 3,333 步 SPSA 內會不會走到那個退化的均勻天花板，本身就是 seed 的性質。**±0.92 的 sd 應該報出來，而不是報一個 pooled 值。** 機制不變：精確均勻分布最大化 QCLI 而 CCI 恰為 0，所以高 QCLI 未必代表可利用的結構。

### 8.4 Turbulence 復現：未重現，且我們知道為什麼 **[v11 三處修正]**

出處：`seed_tables_c.md` C1–C6。

| 事實 | 5-seed 數字 | 出處 |
|---|---|---|
| Pre-registered 判定 | **未重現**：mean 聚合、N=11 下，RBM-12k（0.2166 ± 0.0069 / 0.7186 ± 0.0189）與 DCGAN-1.2M（0.0453 ± 0.0621 / 0.3966 ± 0.3829）在兩個指標上都贏我們的 IQP-31k（0.2716 ± 0.0036 / 0.8187 ± 0.0105） | `seed_tables_c.md` C1、C5b |
| last-write-wins 慣例下的 PDF–JS | 我們 IQP-31k 0.0324 ± 0.0011、IQP-63k 0.0473 ± 0.0005，對論文的 0.0238 / 0.0221（接近）；但 RBM 在同一慣例下仍以 0.0224 ± 0.0019 全面贏，判定不變 | `seed_tables_c.md` C2 |
| **anchor-fit 的 provenance 修正** | v10 引的「0.69 抽樣 / 0.77 exact」其實是 **IQP-63k** 的 seed-0 值（0.6913 / 0.7755）。**IQP-31k 的 5-seed 值是 0.6510 ± 0.0103（抽樣）/ 0.7394 ± 0.0100（exact）** | `seed_tables_c.md` C3「Provenance correction」 |
| **協定歸因的重新拆解** | 凍住 `θ_lat`：**−0.1135 ± 0.0209**（v10 寫 −0.08）；5×10⁴ 抽樣 pmf：**−0.0010 ± 0.0222**（v10 寫 −0.04，5 seeds 之後與 0 無法區分）；stage 2 latent adaptation：0.0000 ± 0.0004 | `seed_tables_c.md` C4 |
| 模型族天花板 | **0.8921，seed-independent**（跑 seed 0–4 回傳 bit-identical `corr`，因為最佳化過程不消耗 RNG）。v10 引的 0.892 數值正確，但它是**跨 config 的範圍，不是 seed spread** | `seed_tables_c.md` C4 |
| Conv2D-MMD² 欄不可比 | 把 ground truth 在像素空間打散就得到 0.70451 ± 0.00952，而論文 Table 2 整欄（0.896–1.472）都在這條線之上 | `seed_tables_c.md` C5 |
| 評估設計無法支持論文主張 | held-out 的未見時間合成任務被零參數線性內插解到 corr 0.99978（deterministic） | `seed_tables_c.md` C5 |

**為什麼 seed-0 的歸因是假的**（`seed_tables_c.md` C4「This block is where 5 seeds change the story」）：`paper_stage1` 這一格把 `θ_lat` 凍在一個隨機 N(0, 0.01) 抽樣上，是全表最敏感的格；seed 0 剛好落在 0.8074，而 5-seed mean 是 0.7646 ± 0.0261。凍結項因此被低估、抽樣項被高估。**誠實的 5-seed 說法是：stage 1 凍住 `θ_lat` 幾乎就是全部的協定代價，而 5×10⁴ 的抽樣 pmf 沒有可量測的成本。** 兩個凍結對比（confounded −0.1135 ± 0.0209 與 unconfounded −0.1145 ± 0.0024）現在一致，正因為分開它們的那個因子是零。

**sd 慣例**：附錄 B 目前引的 ± 來自 `evaluate.py` 的 `numpy.std`（ddof = 0），在 n=5 時比 sample sd 小 0.894 倍。`seed_tables_c.md` 兩種都給，論文要挑一種並全篇一致。

---

## 9. 誠實邊界與投稿時的措辭原則

| # | 邊界 | 我們怎麼寫 | 出處 |
|---|---|---|---|
| 1 | **exact classical simulation** | 全部用三次 FWHT 精確模擬，沒有 Monte-Carlo、沒有抽樣噪音；n=10 只有 1,024 個振幅。摘要與 Discussion 都明講 | `v11/facts.md §9`；`iqp_exact.py:1-11` |
| 2 | **seed 數** **[v11 修正]** | v10 是 single seed。**v11 全部生成實驗改成 5 seeds 的 mean ± sd（ddof=1）**，且每一個「贏」都附一句「1 sd 內誰與它重疊」。不再需要拿 DCGAN 的起伏當噪音示範 | `seed_tables.md` 開頭與各 wins 表 |
| 3 | **不主張量子優勢** | 「hardness results 描述的是這個模型族在 scale 上的性質，不是我們的實例」 | `v11/main.tex:74-75,310-312` |
| 4 | **只比 RBM 與 DCGAN** **[v11 措辭統一]** | **本文不與原論文未使用的古典模型比較。** 這是 <user> 指定的框架，v11 把文件裡所有涉及自訂控制組的段落收斂成這一句 | memory `feedback-paper-baselines-only.md` |
| 5 | **DCGAN 不可重現** **[v11 新增]** | cuDNN 沒有 deterministic transposed-convolution kernel：同 config 同 seed 重跑，raw MMD² 差最大 57.0%、PDF–JS 最大 74.1%，與 DCGAN 對 IQP 的差距同一量級。RBM 與所有 IQP config 重跑 bit-identical | `seed_tables.md`「Repeat runs of an identical configuration」與「Protocol notes」 |
| 6 | **Baseline 曾經被欠訓練** **[v11 新增]** | v10 的固定 20k 步讓 RBM 的 reconstruction MSE 還在降（N=100：0.0662 對 160k 的 0.0041）。v11 改成 validation early stopping，這只可能對 baseline 有利；IQP 仍固定 4,000 步，且被證明給它同樣的規則也不會動超過 1 sd | `v11/facts.md`「baseline convergence sweep」；`seed_tables.md`「Does early stopping change the IQP numbers by more than 1 sd?」 |
| 7 | **projection corr 的選擇偏差** **[v11 新增]** | 論文報的是「八個 restart 取與 held-out 影像相關最高的那個」，也就是拿被報的指標本身去選。v11 附上「以 loss 選」欄（模型自己做得到的選法），兩者差距 ≤ 0.005；單獨用零初始化 restart 在 N=100 是 0.9684 ± 0.0004 | `seed_tables.md`「Held-out projection correlation, by restart」 |
| 8 | **flux-aligned 是 post-hoc** | 主表列出來但明確標註 post-hoc，因為它不在 round 3 的 pre-registration 裡 | `v11/main.tex:154-155` |
| 9 | **亮度尺度 S 是設計缺陷** **[v11 補正]** | 所有 IQP arm 與 RBM 的 `field = q × S`，S 是外部純量不是學來的；而且**它隨 N 變**（112.3 / 124.9 對 held-out 的 123.9），所以 N=11 的所有 arm 系統性偏暗約 9% | `seed_tables.md`「Parameter counts and the flux scale S」 |
| 10 | **θ⁰ 的 KL 暖啟動不是硬體可估的** **[v11 新增]** | 已做消融：換成 flat-MMD 暖啟動（全程 shot-estimable）後沒有一格差超過 1 sd | `seed_tables.md`「theta0 warm-start ablation」 |
| 11 | **舊指標協定可被 generator 操弄** **[v11 新增]** | 直說：v10 協定下把 DCGAN 的一個像素設成 4.0 就能讓它的 PDF–JS 除以 13，而 per-arm 的 kernel bandwidth 會翻轉 RBM 的排名。v11 把兩者固定住並重算全表 | `v11/facts.md`「Addendum: metric protocol v11」 |
| 12 | **permutation null 削弱我們自己的賣點** | 摘要三個 caveat 之一就是「structure-matched permutation null 仍有 QCLI = 0.4154 ± 0.0000」，主動寫在摘要而不是附錄 | `seed_tables_b.md` §1 |
| 13 | **兩套 pre-registered decision rule 都沒過** | round 3 判 failure、round 3b 判「不算有用」。正文只宣稱「學到的方向讓 IQP 第一次能生成並泛化」 | `R3RES:14-15,239-240` |
| 14 | **資料集引用（已確認 2026-09-08）** | DeepLense Model II 的模擬方法出自 Alexander, Gleyzer, Reddy, Tidball, Toomey 2023, ApJ 954:28（arXiv:2112.12121）；Alexander et al. 2020（ApJ 893:15）是理想化的 Model I，只作任務起源引用；資料本身引 ML4SCI DeepLense GitHub repo | `v11/refs.bib` |

**論文結構相關的兩個決定（v11）**：Table 3（per-class）移到**附錄 F**（`v11/main.tex:483`）；附錄總長約 3.6 頁；preamble 不做任何模板修改。

---

## 10. 檔案索引：每個結論對應到哪裡

### 10.1 文件

| 內容 | 檔案 |
|---|---|
| **5-seed 統計（主表、per-class、scaling、baseline 步數、θ⁰ 消融、main.tex §3.2 診斷、per-restart、重跑差異）** | `paper/ml4ps2026/seed_tables.md` |
| **5-seed 統計（map、M 掃描、App. C、App. D）** | `paper/ml4ps2026/seed_tables_b.md` |
| **5-seed 統計（turbulence 附錄 B）** | `paper/ml4ps2026/seed_tables_c.md` |
| **指標協定 v11 的定義、floor 對照、排名變化、clipping** | `paper/ml4ps2026/v11/facts.md`「Addendum: metric protocol v11」 |
| Baseline convergence sweep 與建議步數 | `paper/ml4ps2026/v11/facts.md`「Addendum: baseline convergence sweep」 |
| 原論文逐節摘要（含 Table 2 原始數字） | `docs/PAPER_2603.06440_SUMMARY.md` |
| 復刻規格書（含「原文未給」清單） | `docs/IQP_REPLICATION_SPEC_2603.06440.md` |
| Ref [38] 方法筆記（MMD 閉式、三 bandwidth 規則） | `docs/REF38_IQPOPT_METHOD_NOTES.md` |
| 透鏡影像特性量測 | `docs/LENS_IMAGE_CHARACTERISTICS.md` |
| Round 1 / 2 / 3 結果與 pre-registration | `docs/IQP_LENS_PILOT_RESULTS.md`、`docs/IQP_LENS_ROUND2_RESULTS.md`、`docs/IQP_LENS_ROUND3_{PREREGISTRATION,RESULTS}.md` |
| Correlation–complexity map 重算 | `docs/IQP_LENS_MAP_RESULTS.md` |
| 論文級分析套件 | `docs/IQP_LENS_PAPER_SUITE_RESULTS.md` |
| Turbulence 復現結果與 root-cause | `docs/IQP_TURBULENCE_REPLICATION_RESULTS.md`、`..._DIAGNOSIS.md` |
| 稿件、數字出處、圖片出處 | `paper/ml4ps2026/v11/main.tex`、`v11/facts.md`、`v11/NUMBERS_PROVENANCE.md`、`figures/FIGURE_PROVENANCE.md` |
| v10 版本的本文件 | `docs/IQP_LENS_METHOD_DIFFERENCES_ZH.md` |

### 10.2 結果檔（全部在 `experiments/13_iqp_turbulence_replication/`）

| 內容 | 檔案 |
|---|---|
| Round 3 固定基底 / Round 3b 學到方向 / scaling / per-class | `results_r3.jsonl`、`results_r3b.jsonl`、`results_scaling.jsonl`、`results_class.jsonl` |
| **5-seed 原始記錄（主表、scaling、per-class、baseline early stopping、θ⁰、診斷、restart）** | `results_seeds_table2.jsonl`、`results_seeds_scaling.jsonl`、`results_seeds_class.jsonl`、`results_seeds_baselines_es.jsonl`、`results_seeds_theta0.jsonl`、`results_seeds_diag.jsonl`、`results_seeds_restarts.jsonl` |
| **v11 指標重新評分與 floor** | `results_v11_all.jsonl`、`results_v11_seed0.jsonl`、`results_v11_floor.jsonl`、`results_v11_floor_disjoint.jsonl`、`results_v11_floor_n100.jsonl` |
| Baseline convergence sweep | `results_convergence.jsonl`、`outputs_convergence/` |
| **turbulence 5-seed** | `results_d2_seeds.jsonl`、`results_d2_seeds_main.jsonl`、`results_d2_seeds_last.jsonl` |
| Correlation–complexity map（含 5 seeds） | `paper_suite/map/results_map.json`、`results_map_seed{0..4}.json`、`results_map_seeds.json` |
| Fig. 3 / Fig. 4（含 5 seeds） | `outputs_controlled/results_fig3_v2.jsonl`、`results_fig3_v2_ls{0..4}.jsonl`、`results_fig4.jsonl`、`results_fig4_seeds34.jsonl`、`results_fig34_seeds.json` |
| 機制測試（表達力、插值崩潰、學方向、M 掃描 5 seeds） | `mech_tests/mech_test/results.json`、`mech_tests/mech_test2/diag{2,3,4}.json`、`mech_seeds_s{0..4}.json`、`results_seeds.json` |
| 每個 run 的超參數與 S / Φ / z 統計 | `outputs_lens_r3b/<tag>/meta.json` |

### 10.3 程式

| 用途 | 檔案 |
|---|---|
| 10-bit crop pmf codec（含 `paste`） / 18-bit tuple codec | `codec10.py` / `codec.py` |
| 資料層：centroid alignment、crop、split 規則、S | `data_lens_r3.py` |
| Exact IQP 模擬器（三次 FWHT、kernel weights） | `iqp_exact.py` |
| 固定基底訓練 / 學到方向訓練 | `train_iqp_r3.py` / `train_iqp_affine_r3.py` |
| **θ⁰ 目標函數消融** | `seeds/train_iqp_affine_r3_theta0obj.py` |
| 評分（v10 協定） | `evaluate_r3.py`、`evaluate_r3b.py`、`evaluate_class.py`、`metrics.py` |
| **評分（v11 協定，新檔，不動舊檔）** | `metrics_v11.py`、`evaluate_v11.py` |
| Baseline / baseline 收斂掃描 | `baseline_rbm_r3.py`、`baseline_dcgan_r3.py`、`baseline_rbm_sweep.py`、`baseline_dcgan_sweep.py`、`aggregate_convergence.py` |
| **5-seed 聚合** | `paper/ml4ps2026/aggregate_seeds.py`、`make_seed_tables_b.py`、`turb_aggregate_seeds.py`、`turb_seed_extras.py` |
| **5-seed 執行（map / M 掃描 / App. C / App. D）** | `paper_suite/map/{agg_map_seeds.py,submit_map_seeds.sbatch}`、`mech_tests/mech_test2/{mech_seeds.py,agg_mech_seeds.py}`、`paper_suite/controlled/{fig3_learner_seeds.py,agg_fig34_seeds.py}` |
| Map 與 controlled study | `paper_suite/map/*.py`、`paper_suite/controlled/*.py` |

---

## 11. v10 → v11 變更摘要

共 22 條。標 **[數字]** 的是數值改變，標 **[修正]** 的是 v10 有錯或有誤導，標 **[新增]** 的是 v10 沒有的內容。

| # | 節 | 變更 | 類型 |
|---|---|---|---|
| 1 | 全文 | 所有生成實驗數字改成 **5 seeds 的 mean ± sd（ddof=1）**；沒有 5-seed 版本的單一數字一律標「seed 0，無 5-seed 版本」 | [數字][新增] |
| 2 | §1、§7.1 | **主表換成 metric protocol v11**：N=11 沒有站得住的贏家（兩欄 mean 最佳是 IQP 但與 baseline 在 1 sd 內重疊、raw MMD² 由 DCGAN 領先）；N=100 DCGAN 四欄 mean 全最佳，三欄領先穩固 | [數字][修正] |
| 3 | §1 | 摘要不再寫「IQP 在原論文自己的兩個指標上贏 RBM 與 DCGAN」，改成「mean 最佳但重疊」與「flux-aligned 在 N=100 與 DCGAN 打平」 | [修正] |
| 4 | §2.3、§4-6 | **paper kernel 的來源講清楚**：它是 IQPopt [38] 的三 bandwidth 規則（σ = 0.989 / 1.729 / 2.236），原論文採用該 kernel 但從未公布 bandwidth | [修正] |
| 5 | §4-10、§5.1、§9-9 | **S 改成隨 N 的兩個值**：N=11 為 112.3、N=100 為 124.9，held-out 為 123.9（v10 只寫 124.92） | [數字][修正] |
| 6 | §4-12、§7.1 | **參數量改成 total-at-N**：fixed-basis 1,524 / 5,974；learned-direction 5,164 / 5,520；RBM 11,510 / 15,960（joint RBM 每張 50 個 bias）；DCGAN generator 199,837 加 discriminator 53,497。v10 的 Params 欄混了三種慣例 | [數字][修正] |
| 7 | §4-13、§7.2 | **Baseline 改成 validation early stopping**（RBM 160k / 80k、DCGAN 5k–30k），因為 20k 時 RBM 的 reconstruction MSE 還在降（0.0662 對 160k 的 0.0041） | [數字][新增] |
| 8 | §4-7、§9-10 | **新增 θ⁰ 的 1,000 步 exact-KL 暖啟動與其消融**：換成 flat-MMD 暖啟動後沒有一格差超過 1 sd | [新增] |
| 9 | §4-11、§5.5 | **新增評估協定 v11 一整節**：固定 [0,1] 的 50 bins 加 clip、Conv2D 的尺度與 σ 只由 held-out 決定且全表共用、200-vs-200 的 20 組 floor、同時報 biased 與 unbiased | [新增] |
| 10 | §5.5 | **寫出舊協定為什麼非換不可**：把 DCGAN 的一個像素設成 4.0 可讓 PDF–JS 除以約 13；per-arm 的 σ 讓 RBM 在 N=11 的 raw MMD² 排名從第 1 掉到第 3 | [新增] |
| 11 | §5.5 | **floor 修正**：v10 的 per-class floor 高了 2 到 6 倍；pooled floor 從 0.004006（100v100 單一 split）改成 0.004806 ± 0.002783（200v200，20 draws） | [數字][修正] |
| 12 | §5.1-8 | **clipping 從「圖面揭露」升級成「評分協定的一部分」**：106 個 seed-0 run 有 64 個像素超過 1（中位 5.4e-04、最大 6.2e-03），DCGAN 是唯一從不超過 1 的 arm | [修正][新增] |
| 13 | §6.1 | 機制測試全部換成 5 seeds：單張 crop 擬合 KL 0.9999 ± 1.05e-05、flat 0.999966 ± 5.7e-06、auto3 0.9964 ± 0.0008；凍 core 後其他訓練影像 0.4411 ± 0.0419（v10 引 0.47） | [數字] |
| 14 | §6.1 | **random-direction 控制改用配對版本 0.7458 ± 0.0005**；v10 引的 0.672 是 mismatched 配對（5 seeds 為 0.6468 ± 0.0484），配對版是比較難的一條線 | [修正] |
| 15 | §6.2 | **新增 per-restart 拆解**：以 loss 選（模型做得到的選法）對上以 corr 選（論文報的）最大差 0.0041（全部 ≤ 0.005）；單獨零初始化 restart 在 N=100 已是 0.9684 ± 0.0004 | [新增] |
| 16 | §6.3 | **M 的工作點結論改變**：M=2 為 0.9060 ± 0.0021、M=4 為 0.8577 ± 0.0559，seed 0 是五個 M=4 run 裡最好的一個。改寫成「M=2 與 M=4 是工作區間，M=2 較可靠」。M=6 落到 mean-image 線之下 | [數字][修正] |
| 17 | §7.3 | **Table 3（per-class）移到附錄 F**，並加註 N=100 的 PDF–JS 四列在 1 sd 內全部重疊，不可主張某一類較容易 | [數字][修正] |
| 18 | §7.2、§9-5 | **新增 DCGAN 的硬體不可重現性**：同 config 同 seed 重跑，raw MMD² 差最大 57.0%、PDF–JS 最大 74.1%，與被量測的差距同量級 | [新增] |
| 19 | §8.1 | **map 新增原生 K = 5×10⁴ 那一列**：turbulence 的 QCLI（0.0910 ± 0.0000）低於自己的 permutation null（0.0981 ± 0.0007），與 subsample 到 K=4,096 時的方向相反 | [新增] |
| 20 | §8.2 | **App. C 的 floor-proxy 措辭修正**：ρ ≤ −0.98 只對 exact-pmf proxy 成立（最壞 −0.9876）；10⁴-sample proxy 在 G_n = 150 是 −0.9291 | [修正] |
| 21 | §8.3 | **App. D 五個 ρ 全部更新**：低 QCLI 從 +0.83 / +0.76 / +0.95 變成 +0.79 / +0.79 / +0.90；n=12、16 的 global ρ 從 −0.53 / −0.57 變成 −0.57 / −0.64；反轉現在在 n=8 也出現，且 n=8 的 per-seed sd 是 ±0.92 | [數字][修正] |
| 22 | §8.4 | **turbulence 三處修正**：anchor-fit 的 0.69/0.77 其實是 IQP-63k seed 0，IQP-31k 是 0.6510 ± 0.0103 / 0.7394 ± 0.0100；協定歸因從 −0.08 / −0.04 變成凍結 −0.1135 ± 0.0209、抽樣 pmf −0.0010 ± 0.0222（與 0 無法區分）；天花板 0.8921 是 seed-independent 的跨 config 範圍而非 seed spread | [數字][修正] |

**另外兩項編輯決定（不是數值變更）**：
- 涉及自訂控制組（product-Bernoulli、PCA、zero-learning、CTRL-ring 等）的段落全部收斂成一句「**本文不與原論文未使用的古典模型比較**」（§9 第 4 列），只討論生成模型之間的效果。permutation null、i.i.d.-uniform null、random-direction 參考屬於方法學 null 而非古典 baseline，予以保留。
- 論文結構：Table 3 移到附錄 F，附錄總長約 3.6 頁，preamble 不做模板修改（§9 末）。
