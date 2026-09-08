# IQP Turbulence Generative Modeling — 復刻規格書

> Source: arXiv 2603.06440 (Liu, Placidi, Brunner, Rinaldi, Quantinuum, "Toward Generative Quantum Utility via Correlation–Complexity Map", v2 2026-05-05)
> 文字檔：`<WORK>/data/papers/2603.06440/paper.txt`（3117 行，內文從 316 行「Abstract」開始）
> HTML 檔：`<WORK>/data/papers/2603.06440/paper.html`（用於核對 MathML alttext，確認 LaTeX 原始碼）
> 本文件目的：把 Section III（Turbulence IQP 生成模型）復刻所需的每個具體數字/設定抄錄，並附精確 `paper.txt:行號` 出處。所有行號皆已用 `grep -n` 逐一核對，非憑印象。

---

## 0. 重要更正說明（先讀這段）

任務指示中給的幾個「不確定」項目，經逐一查證後，**全部正確**，只有一處需要澄清（Appendix C 的 eq 編號重複）：

| 原本不確定的項目 | 查證結果 |
|---|---|
| `N_t=5×10^4` 這個記號是否存在 | **存在**，主文 `paper.txt:1129` 附近與 Appendix J Table 6（`paper.txt:2929`）都明確寫 `N_t=5×10^4`。不是筆記錯誤。 |
| anchors 是否剛好是 `t∈{1,100,...,1000}` | **正確**，共 **11** 個值：`{1,100,200,300,400,500,600,700,800,900,1000}`。出處：Fig. 7 caption (`paper.txt:1324`) 與 Appendix J Table 6 (`paper.txt:2867`) 皆逐字給出此集合。 |
| eq.53 是否落在 Appendix C | **正確**，eq.53 在 `paper.txt:2473`，落在 Appendix C 範圍 (2399–2487)。 |
| eq.54–56 是否落在 Appendix F | **幾乎正確，且有 bonus**：eq.54 (`2547`)、eq.55 (`2559`)、eq.56 (`2568`) 確實在 Appendix F (2537–2586)；另外 **eq.57 (`2580`)** 也在 Appendix F 範圍內（是 (x,y,z)→bitstring 串接公式），建議一併納入。 |
| eq.58–59 (MMD) 是否落在 Appendix G | **正確**，eq.58 (`2602`)、eq.59 (`2612`) 都在 Appendix G (2586–2642)。 |
| eq.60–63 (評估指標) 是否落在 Appendix I | **正確**，eq.60 (`2715`)、eq.61 (`2724`)、eq.62 (`2737`)、eq.63 (`2747`) 都在 Appendix I (2699–2754)。 |

**發現的一個原文內部不一致**：Appendix C 在 `paper.txt:2412` 與 `paper.txt:2421` 重新印出了 `P(s)` 與 `m_k` 的定義式，但標號分別是 **(4)** 與 **(5)**——這與主文 Section II.2 的 eq.5 (`paper.txt:485`, P(s)) 與 eq.6 (`paper.txt:496`, m_k) 內容相同但編號**對不上**（差一號）。這很可能是 arXiv LaTeXML HTML 轉換時 Appendix 區塊重啟計數器造成的錯位，不影響數學內容，但寫程式時應以主文 eq.5/eq.6 為準，Appendix C 這兩個「(4)/(5)」只是重複展示。

---

## 1. 資料（Turbulence Dataset）

| 項目 | 數值 | 出處 |
|---|---|---|
| 資料集 | D2 dataset，來自 Ref. [25] = A. R. Khojasteh, S. Laizet, D. Heitz, Y. Yang (2022), *"Lagrangian and Eulerian dataset of the wake downstream of a smooth cylinder at a Reynolds number equal to 3900"*, Data in Brief 40 | `paper.txt:1069`（引用文字）、`paper.txt:1798-1808`（bibliography entry） |
| Snapshot 總數 | 1000 | `paper.txt:1069`："total 1000 turbulence snapshots" |
| 時間標籤 | t ∈ {1, 2, …, 1000} | `paper.txt:1069` |
| 基本資料單元 | 座標/數值 tuple (x, y, z) ∈ ℝ³，x,y 是 2D 空間位置，**z 是 local velocity magnitude（不是 intensity，也不是 vorticity）** | `paper.txt:2545`（Appendix F 原句："Here, x and y denote 2D spatial positions, while z corresponds to the local velocity magnitude."） |
| 量化 bit 數 | N=6 bits/coordinate | `paper.txt:1076`, Table 5 (`paper.txt:2869`) |
| Bitstring 長度 | 3N = 18 bits/tuple → n=18 qubits | `paper.txt:1076`, Table 5 (`paper.txt:2871`) |
| Anchor snapshots（IQP/RBM 訓練用） | 11 個，t ∈ {1,100,200,300,400,500,600,700,800,900,1000} | `paper.txt:1341-1345`（Section III.2 正文）、Fig. 7 caption `paper.txt:1324`、Table 6 `paper.txt:2867` |
| Unseen τ（測試集） | 200 個，從 [1,1000] 均勻抽樣，**排除**訓練點 | `paper.txt:1349`："we uniformly sample 200 additional time indices from [1,1000] (excluding the training points)" |
| N_t（每個 anchor 的訓練資料點數） | N_t = 5×10⁴ | `paper.txt:1129`（Section III.1 主文）、Table 6 `paper.txt:2929` |
| 量化公式 | eq.54–57（見下方 Appendix F 摘要） | `paper.txt:2547, 2559, 2568, 2580` |

**⚠ Section III.3 的一處疑似內部矛盾**：`paper.txt:1496` 寫「our turbulence experiment, each three-dimensional snapshot has spatial resolution 64×64×64=262,144 grid points」，暗示是 3D volumetric field。但 Appendix F 明確說 x,y 是 2D 空間位置、z 是純量的 velocity magnitude（即一個 2D scalar field，不是 3D volumetric field），且 Appendix I (`paper.txt:2703`) 也說 "each snapshot is a 2D field"（用 HTML 核對過，`paper.txt` 顯示的 "ZZ-channel" 其實是單一大寫 Z 被文字抽取重複輸出，原始 LaTeX 只有一個 `Z`，見下方 Appendix I 段落的核對記錄）。**建議復刻時以 Appendix F/I 為準（2D 場 + 座標/速度量值 tuple），把 III.3 的「64×64×64／262,144 qubits」理解為作者為了強調『若直接對整個高維 field 逐 pixel 建模需要多少 qubit』所做的說明性/上界示意計算，而非精確描述 D2 資料實際維度**（D2 原始資料集本身依你既有的 memory 記錄，也是 2D velocity snapshot：`2D_velocity_snapshot_sub_domain_2.zip`）。實作時 grid 解析度（例如是否真的是 64×64）本篇論文並未在別處給出精確數字，需要另外去查 Ref. [25] 或既有的 D2 檔案本身確認。

---

## 2. IQP 訓練（Section III.1 + Algorithm 1 + Appendix G）

| 項目 | 數值 | 出處 |
|---|---|---|
| 電路 qubit 數 | n = 18 | `paper.txt:1076` |
| 電路形式 | U(θ) = H^⊗n exp(i Σ_s θ_s Z_s) H^⊗n，測量得 x ~ q_θ(x) = \|⟨x\|U(θ)\|0^n⟩\|² | eq.25, `paper.txt:1104` |
| 參數切分 | θ = (θ_core, θ_lat)，dim(θ_lat)=d_lat ≪ dim(θ_core) | eq.26, `paper.txt:1113` |
| Latent dimension | d_lat = 50 | `paper.txt:1355`（Section III.2 正文）、Fig.7 caption `paper.txt:1324`、Table 6 `paper.txt:2913` |
| Latent block 定義 | θ_lat = canonical IQP 參數向量（依 commuting Pauli string 的 **Hamming weight 由小到大排序**：{∅},{0},{1},…,{n-1},{0,1},{0,2},…）的**前 50 個元素** | `paper.txt:1357-1362` |
| Algorithm 1 Step A（t=1 初始化） | θ_lat^(1) 隨機初始化並固定；只訓練 θ_core，最小化 L_MMD(q_(θcore,θlat(1)), p̂_1) | eq.27, `paper.txt:1129`；Algorithm 1 (`paper.txt:1140` 起) |
| Step A 訓練規模 | 每步用 **500** 個 model samples 估計 L_MMD；共 **3×10⁴** 訓練步；N_t=5×10⁴ 個訓練資料點 | `paper.txt:1129`；Table 6 `paper.txt:2921, 2925, 2929` |
| Algorithm 1 Step B（t≥2 序列 latent 適配） | 凍結 θ_core^(1)（對所有 t 共用固定）；只對 θ_lat 做 warm-started 優化，初始值取自上一個 anchor 的 θ_lat^(t-1) | eq.28-29, `paper.txt:1251, 1264` |
| Step B 訓練規模 | 每個 anchor **3000** 個優化步 | `paper.txt:1267`；Table 6 `paper.txt:2933` |
| 插值方式 | Piecewise-linear：k=⌊τ⌋, α=τ-k, θ̃_lat(τ)=(1-α)θ_lat^(k)+α θ_lat^(k+1)；一般化為凸組合 Σ w_t(τ)θ_lat^(t)，允許非凸權重做外插 | eq.30-31, `paper.txt:1281, 1293` |
| 生成時取樣數 | 主文「例如 10⁵ samples」；Table 6 給更精確範圍 **約 10⁵–10⁶ samples** | 主文 `paper.txt:1275`；Table 6 `paper.txt:2939` |
| MMD loss（訓練用，可classical evaluate 版本） | L_MMD(θ) = Σ_s c_s (⟨Z_s⟩_θ - ⟨Z_s⟩_data)² | eq.58-59, `paper.txt:2602, 2612` |
| Train-on-classical/deploy-on-quantum 框架 | 引用 [38,14,6]；核心是 IQP 的 ⟨Z_s⟩_θ 可用 [38] eq.15 高效古典計算，訓練全程在古典硬體完成，只有部署階段的取樣在量子硬體上做 | `paper.txt:1078`（正文引用）、Appendix G 全文 `paper.txt:2586-2641` |
| Appendix G 內容摘要 | (1) MMD loss 用 Gaussian kernel k(b,b')；(2) 依照 [38] Appendix A.2.1/A.2.2/A.3 的推導把每一項 MMD 改寫成 Pauli-Z_s 期望值的封閉形式 (eq.59)；(3) θ* = argmin_θ L_MMD(θ) 全程可在古典硬體上跑；(4) 部署階段把 θ* 燒到量子硬體做 projective measurement 取樣，再用 float-to-bitstring 的反函數解碼回物理場；(5) 因編碼可逆，解碼後分布的取樣困難度由 Appendix H 的 reduction 論證繼承自底層 bitstring 分布；(6) 此框架可擴展到上千 qubit（引 [38]），資料相依的參數初始化策略有助避開 barren plateau | `paper.txt:2586-2641` |

---

## 3. Snapshot 重建（取樣聚合回 grid）

原文對「怎麼把取樣到的 tuple 聚合回一張 snapshot」只有**流程性**描述，沒有給出精確的聚合演算法細節（例如空格子怎麼處理、是否加權平均、是否用直方圖計數）：

- 主文：「A synthetic snapshot is reconstructed by drawing many such tuples from the time-conditioned IQP distribution and aggregating them on the discretized grid.」(`paper.txt:1074`)
- Algorithm 1 Step 12：「Decode sampled bitstrings into coordinate/value tuples and aggregate them on the discretized grid to obtain ỹ(τ).」(`paper.txt:1238` 附近)
- Table 6：「Output reconstruction — Decode and aggregate samples into a synthetic turbulence snapshot」(`paper.txt:2942`)

→ **聚合規則本身的精確定義（bin 內多筆 (x,y) 相同格點時 z 值怎麼合併、空格點怎麼填補）不在原文任何地方給出**，已列入第 11 節「原文未給」清單。

---

## 4. RBM Baseline（Table 7 逐項）

| Item | Setting | 出處 |
|---|---|---|
| Training snapshots | 與 IQP 相同的 11 個 snapshots | `paper.txt:2964-2965` |
| Data representation | 與 IQP 相同的 18-bit coordinate/value tuple representation | `paper.txt:2967` |
| Training domain | Bitstring domain（直接在 binarized 表示上訓練，而非連續域） | `paper.txt:2969`；亦見主文 `paper.txt:1352` |
| Latent dimension | d_lat = 50 | `paper.txt:2972` |
| Latent block | 固定 flatten 後的 RBM 參數（weights+biases）排序中的**前 50 個元素** | `paper.txt:2975`；主文 `paper.txt:1370` |
| Fixed parameters | 其餘 RBM 參數在 snapshot adaptation 時保持固定（做法對齊 IQP 的 core/latent 切分邏輯） | `paper.txt:2978` |
| Table 2 中的模型規模 | RBM (12k params) | `paper.txt:1409` |

**Table 7 沒有給、且我已核對 Appendix J 全文確認查無**：hidden units 數量、是否為 Bernoulli-Bernoulli RBM、層數（隱含只有 1 層，因為稱為 "RBM" 而非 DBM，但未明說）、CD-k / PCD 訓練演算法、learning rate、optimizer、epoch 數/steps 數。

---

## 5. DCGAN Baseline（Table 8 逐項）

| Item | Setting | 出處 |
|---|---|---|
| Data representation | Quantized 2D turbulence fields（**不是** bitstring representation） | `paper.txt:2989` |
| Data-matched regime | 11 個訓練 snapshots（與 IQP/RBM 對齊） | `paper.txt:2995-2998` |
| Data-rich regime | 100 個訓練 snapshots | `paper.txt:3001` |
| Model sizes | 約 51k、0.2M、1.2M 參數（三個容量設定） | `paper.txt:3004`；Table 2 對應列 `paper.txt:1432-1445` |
| Latent dimension | 128（選定理由：與 IQP 的「有效 latent 容量」相當） | `paper.txt:3007`；主文 `paper.txt:1390` |
| Data-scaling study | N_train ∈ {10,30,50,70,90,100} snapshots（對應 Fig. 8） | `paper.txt:3010`；主文 `paper.txt:1481` |
| Evaluation protocol | 與其他 turbulence 實驗相同的 held-out protocol | `paper.txt:3013` |

**Table 8 沒有給、且已核對 Appendix J 全文確認查無**：DCGAN 具體層數、每層 channel 數、kernel size、stride、是否用 BatchNorm/LeakyReLU（DCGAN 論文 [37] 的標準配置未在此文中重申或修改說明）、optimizer（Adam 是 DCGAN 慣例但本文未明講）、learning rate、β1/β2、epoch 數、mini-batch size。

---

## 6. 評估指標（Appendix I，`paper.txt:2699-2754`）

原文只評估 **z-channel（velocity magnitude）**：「each snapshot is a 2D field (in our implementation we evaluate the Z-channel only)」(`paper.txt:2703`)。**核對記錄**：`paper.txt` 抽取文字顯示成 "ZZ-channel"，但用 python 讀取 `paper.html` 原始 MathML 確認，LaTeX 原文只有單一 `Z`（`<math id="A9.p1.m3" ... alttext="Z">`），"ZZ" 是文字抽取把可見文字與 annotation 重複印出造成的假象，**應理解為 "the Z-channel"（即 Appendix F 定義的 z = velocity magnitude 這個分量）**。

### (i) PDF–JS divergence（marginal intensity）

| 項目 | 數值 | 出處 |
|---|---|---|
| Histogram bins B | 50 | `paper.txt:2709`："a number of bins B (we use B=50)" |
| Histogram range [a,b] | 由 real 與 generated 兩組資料的 joint min/max 決定 | `paper.txt:2709` |
| 公式 | p̂_k, q̂_k 正規化直方圖機率 (eq.60)；D_JS(p̂‖q̂) = ½D_KL(p̂‖m) + ½D_KL(q̂‖m), m=½(p̂+q̂) (eq.61) | eq.60 `paper.txt:2715`；eq.61 `paper.txt:2724` |
| 小 ε 處理 | 加一個小 ε 避免零值格子，再重新正規化 | `paper.txt:2713` |
| **log base** | **原文未明講**是 log2 還是 ln（eq.60-61 只寫 D_KL，沒有像 QCLI eq.53 那樣明確標 log₂）| 已核對，`paper.txt:2699-2754` 全文查無 log base 說明 → 列入第 11 節 |

### (ii) Feature-space MMD（random Conv2D encoder）

| 項目 | 數值 | 出處 |
|---|---|---|
| Encoder 架構 | 固定、隨機初始化的卷積網路 φ(·)：**3 個 Conv2D–ReLU blocks，stride 2**，接 global average pooling | `paper.txt:2730` |
| Kernel | Gaussian kernel k_σ(u,v)=exp(-‖u-v‖²₂/(2σ²)) (eq.62) | eq.62 `paper.txt:2737` |
| σ 選取 | **Median heuristic**：對 pooled set {f_i}∪{g_j} 的 pairwise feature distances 取中位數 | `paper.txt:2739` |
| MMD 估計式 | Biased 版本，eq.63 | eq.63 `paper.txt:2747` |
| 樣本數 | N = M = 200（無放回 subsample） | `paper.txt:2749` |
| 兩組比較樣本 | 200 真實 snapshots vs. 200 生成 samples（各模型皆同） | 主文 `paper.txt:1397-1399` |

**Appendix I 沒有給、且已核對確認查無**：Conv2D encoder 每層的 channel 數/kernel size/是否有 padding；權重初始化分布（例如是否為標準常態）；random encoder 是否對每次評估重新隨機初始化或固定同一組權重（跨模型比較時通常需要固定同一 encoder 才公平，但原文未明講此點）。

---

## 7. QCLI（Section II.3, eq.7–11；Appendix A, eq.34–45；Appendix C, eq.51–53）

| 項目 | 內容 | 出處 |
|---|---|---|
| 定義 | I_QCLI(p) = D_JS(m‖b) = ½D_KL(m‖M) + ½D_KL(b‖M), M=½(m+b)，其中 m 是 empirical order spectrum、b_k=C(n,k)/2ⁿ 是 binomial baseline | eq.7, `paper.txt:522` |
| 二階近似 | D_JS(m‖b) ≈ (1/8ln2) Σ_k (m_k-b_k)²/b_k | eq.8, `paper.txt:562` |
| Total variation 界 | 經 Cauchy-Schwarz 推導 (eq.9) 得 TV(m,b) ≤ ½√(Σδ²/b) ≈ √(2ln2·QCLI) (eq.10)，最終 TV(m,b) ⪅ O(√QCLI) (eq.11) | eq.9-11, `paper.txt:579, 590, 599` |
| Appendix A 完整推導 | p(z) 從 eq.34→38 一步步展開到 Walsh-Fourier 形式；A.1 節用二階展開證明 D_JS ≈ 加權 ℓ2 距離（eq.39-45，含 H(m)、H(M) 的展開） | `paper.txt:2057-2185` |
| **Monte Carlo 估計協定（n 大時）**（Appendix C） | Subset budget L；equal-per-order allocation（各 order k 抽 m_k^MC 個 subset，Σ m_k^MC=L）；importance reweighting 用 C(n,k)/m_k^MC；估計式 Â_k, Â (eq.51)；m̂_k=Â_k/Â (eq.52) | eq.51-52, `paper.txt:2448, 2457` |
| 小 n 時的處理 | 「for small n (up to a user-specified threshold)」直接窮舉所有 subset 精確計算；未指明門檻值是多少 | `paper.txt:2478` |
| **默認 subset budget** | L = 2×10⁴（原文標「default」，暗示可調） | `paper.txt:2480` |
| I_QCLI 最終計算式（Appendix C 版） | I_QCLI = D_JS(m̂‖b) = ½Σ m̂_k log₂(m̂_k/(M_mix)_k) + ½Σ b_k log₂(b_k/(M_mix)_k)，**log 底數明確是 log₂** | eq.53, `paper.txt:2473` |
| n=18（turbulence）情境下究竟用精確枚舉還是 MC | **原文未明講**：2¹⁸=262144 屬於「可能可以窮舉」但也可能超過該文設定的「small n threshold」，門檻值本身未給 | 已查 Appendix C 全文 (`paper.txt:2399-2487`) 確認無此細節 → 列入第 11 節 |

**注意**：Appendix C 在 `paper.txt:2412`（標號誤植為(4)）與 `paper.txt:2421`（標號誤植為(5)）重複列出 P(s) 與 m_k 定義（內容同主文 eq.5/eq.6），純粹是重申定義，不是新公式。

---

## 8. CCI（Section II.4, eq.12–20）

| 項目 | 內容 | 出處 |
|---|---|---|
| 定義 | I_CCI = 1 - I_TC^tree / I_TC，I_TC = Σ_i H(X_i) - H(X_1,...,X_n)（total correlation / multi-information，源自 Watanabe [46]，Schneidman et al. [39] 延伸） | eq.12, `paper.txt:633` |
| Tree 模型部分 | I_TC^tree = Σ_(i,j)∈T* I(X_i;X_j)，T* 是用 pairwise mutual information 當權重的 maximum-weight spanning tree（Chow-Liu [13]） | eq.13-15, `paper.txt:645, 664, 676` |
| Pairwise MI | I(X_i;X_j) = Σ p_ij log(p_ij/(p_i p_j)) | eq.14, `paper.txt:664` |
| Tree 分布 | p_T(x) = Π p_i(x_i) Π_(i,j)∈T* [p_ij(x_i,x_j)/(p_i(x_i)p_j(x_j))] | eq.16, `paper.txt:687` |
| 資訊理論性質 | T* 同時最大化 tree likelihood 也等價於最小化 D_KL(p‖p_T) (eq.19)；且 D_KL(p‖p_T) = I_TC - I_TC^tree (eq.20) | eq.19-20, `paper.txt:723, 733` |
| 值域 | 0 ≤ I_CCI ≤ 1（因 D_KL≥0 ⟹ I_TC≥I_TC^tree） | `paper.txt:709-715` |

**CCI 計算協定的實作細節（例如 pairwise marginal 的 estimator 是否加平滑/pseudo-count、n=18 情況下 C(18,2)=153 條邊窮舉是否可行）原文未特別著墨**，但因 n=18 遠小於 QCLI 需要窮舉 2¹⁸ 個 subset 的規模，CCI 只需 O(n²) 對的 pairwise MI + MST，計算量不是問題，這點不必列入「未給」清單（屬於顯然可行、非缺失資訊）。

---

## 9. Fig. 3 / Fig. 4 設定（Table 3、Table 4 逐欄位）

### Table 3 — Controlled QCLI–MMD support-mismatch probe（對應 Fig. 3）(`paper.txt:2766-2789`)

| Item | Setting |
|---|---|
| Target datasets | 10⁴ samples from n=16 IQP circuits |
| Data-generating circuits | Locality ≤4，G_n^data ∈ {140,280,420,560,700,840,1050} |
| Target diversity | SPSA-optimized parameters spanning a range of I_QCLI |
| Learner circuits | Restricted IQP, locality ≤2, G_n ∈ {50,100,150} |
| Training | Train-on-classical/deploy-on-quantum；**Adam, 5000 iterations, lr 10⁻⁴** |
| Evaluation | Achieved MMD loss after training |
| Envelope estimation | Binning-and-spline 程序（詳 Appendix D，bin width Δ=0.09，upper envelope 用 quadratic spline k=2、lower envelope 用較強平滑） |

> **注意**：此 Adam/lr=10⁻⁴/5000-iterations 設定是**這個 controlled probe 專屬**（n=16 電路），**不是** turbulence 實驗（n=18）用的優化器/學習率——turbulence 訓練的 optimizer/lr 原文完全未給（見第 11 節）。

### Table 4 — High-I_QCLI IQP outputs and CCI（對應 Fig. 4）(`paper.txt:2809-2831`)

| Item | Setting |
|---|---|
| System sizes | n ∈ {8,12,16} qubits |
| Circuit architecture | IQP circuits with 150 randomly selected 2-local parameterized gates |
| Initialization/optimizer | Random initialization; SPSA optimization |
| Objective | Maximize I_QCLI only；I_CCI 完全不進入 objective（只是輔助診斷） |
| Sampling along optimization | 20000 IQP circuits per system size |
| Evaluation | 對每個 sampled circuit 計算 I_QCLI 與 I_CCI |

---

## 10. Appendix 行號地圖（已逐一用 grep -n 核對）

| Appendix | 起始行 | 結束行（下一個 Appendix 開始前） | 內容 |
|---|---|---|---|
| A | 2057 | 2185 | Derivation detail：p(z) 展開 (eq.34-38)；JS≈ℓ2 距離證明 (eq.39-45) |
| B | 2186 | 2398 | Proof of Proposition II.2：MMD 在 parity/Walsh basis 分解 (eq.46-50) |
| C | 2399 | 2486 | Empirical Estimation of QCLI：MC subset sampling (eq.51-52)，I_QCLI 計算式 (eq.53) |
| D | 2487 | 2515 | Estimating upper/lower envelopes（Fig.3 用，binning+spline，Δ=0.09） |
| E | 2516 | 2536 | Constructing empirical IQP envelope（Fig.5 用，binwise min + moving average） |
| F | 2537 | 2585 | Float-to-Bitstring Representation：quantization (eq.54)，quantization map (eq.55)，binary encode (eq.56)，tuple 串接 (eq.57) |
| G | 2586 | 2641 | Train on Classical, Deploy on Quantum：MMD loss (eq.58)，Pauli-Z_s 封閉形式 (eq.59) |
| H | 2642 | 2698 | Hardness Preservation Under Efficient Invertible Decoding（Theorem H.1 及證明） |
| I | 2699 | 2753 | Evaluation metrics：PDF-JS (eq.60-61)，Conv2D feature MMD (eq.62-63) |
| J | 2754 | 3117（檔案結尾，含 HTML footer 雜訊） | Reproducibility Summary：Table 3 (2766)、Table 4 (2809)、Table 5 (2851)、Table 6 (2891)、Table 7 (2955)、Table 8 (2983) |

**主文各 eq. 標號 → 行號**（逐一 grep 核對，供交叉查驗）：

eq.1=407, eq.2=438, eq.3=454, eq.4=474, eq.5=485, eq.6=496, eq.7=522, eq.8=562, eq.9=579, eq.10=590, eq.11=599, eq.12=633, eq.13=645, eq.14=664, eq.15=676, eq.16=687, eq.17=696, eq.18=705, eq.19=723, eq.20=733, eq.21=803, eq.22=826, eq.23=845, eq.24=858, eq.25=1104, eq.26=1113, eq.27=1129, eq.28=1251, eq.29=1264, eq.30=1281, eq.31=1293, eq.32=1305, eq.33=1314, eq.34=2069, eq.35=2076, eq.36=2083, eq.37=2090, eq.38=2097, eq.39=2111, eq.40=2122, eq.41=2131, eq.42=2144, eq.43=2155, eq.44=2170, eq.45=2179, eq.46=2205, eq.47=2265, eq.48=2298, eq.49=2304, eq.50=2318, [duplicate (4)=2412, (5)=2421 in Appendix C — 見第0節說明], eq.51=2448, eq.52=2457, eq.53=2473, eq.54=2547, eq.55=2559, eq.56=2568, eq.57=2580, eq.58=2602, eq.59=2612, eq.60=2715, eq.61=2724, eq.62=2737, eq.63=2747。

---

## 11. 「原文未給」清單（逐條已先查 Appendix J Table 3-8 確認真的找不到）

1. **Turbulence IQP 訓練的 optimizer / learning rate**：Table 6（Appendix J，`paper.txt:2891-2942`）完全沒有 optimizer 或 lr 欄位。Table 3 給的 "Adam, 5000 iterations, lr 1e-4" 是**另一個實驗**（n=16 controlled probe，Fig.3 用），不能直接套用到 turbulence（n=18, Algorithm 1）。
2. **RBM 訓練的 optimizer / learning rate / 訓練演算法（CD-k 或 PCD）**：Table 7（`paper.txt:2955-2980`）沒有這些欄位。
3. **DCGAN 訓練的 optimizer / learning rate / β1,β2 / epoch 數 / batch size**：Table 8（`paper.txt:2983-3013`）沒有這些欄位。
4. **DCGAN 完整架構**（conv/transposed-conv 層數、每層 channel 數、kernel size、stride、是否用 BatchNorm/LeakyReLU）：Table 8 只給 latent dim=128 與三個總參數量（51k/0.2M/1.2M），無層級細節。
5. **RBM 完整架構**（hidden units 數量、是否 Bernoulli-Bernoulli）：Table 7 只給 latent block 定義（前 50 個 flatten 參數），Table 2 只給總參數量 12k，無 visible/hidden 拆分。
6. **PDF–JS 的 log base（log2 或 ln）**：Appendix I eq.60-61（`paper.txt:2715, 2724`）只寫 D_KL 沒指定底數；對照 QCLI 的 eq.53 明確用 log₂，PDF-JS 這裡沒有對應說明。
7. **QCLI 計算 n=18 turbulence 資料時，是用精確枚舉還是 Monte Carlo 估計**：Appendix C（`paper.txt:2399-2487`）只說「for small n (up to a user-specified threshold)」用精確枚舉，未給該門檻值，也未指明 turbulence（n=18）落在哪一邊；default MC budget L=2×10⁴ 是否適用於此情境不明。
8. **訓練用 MMD loss（eq.58-59）Gaussian kernel 的 bandwidth/σ**：Appendix G（`paper.txt:2586-2641`）只說是 "a Gaussian kernel"，未給 σ 或 median-heuristic 之類的選取方式（對照 Appendix I 的評估用 Conv2D-feature MMD 明確給了 median heuristic，eq.62 附近，`paper.txt:2739`）。訓練用的這個 kernel 完全沒有 σ 說明。
9. **隨機種子（random seeds）**：全文（`grep -n -i seed`）**零命中**，任何一個實驗（QCLI-MMD probe、CCI 實驗、turbulence IQP/RBM/DCGAN 訓練）都沒有給 seed。
10. **Snapshot 聚合規則的精確定義**：只有流程性描述「aggregate them on the discretized grid」（`paper.txt:1074`, `1238`, `2942`），沒有給空格點處理、多筆落在同格點時如何合併（平均/眾數/最後一筆覆蓋）等演算法細節。
11. **Table 2 的 IQP (31k) / IQP (63k) 對應的具體電路 locality/gate 數**：Table 2（`paper.txt:1400-1463`）只給參數量標籤 "31k"/"63k"，正文沒有解釋這兩個電路架構的差異來源。**已做逆向工程驗證**（非原文陳述，屬推論）：n=18 時，Hamming-weight-ordered canonical IQP 參數若截到 order-6（Σ_{k=1}^{6} C(18,k) = 18+153+816+3060+8568+18564 = **31,179**）與 order-7（再加 C(18,7)=31824 → **63,003**）分別非常接近 "31k" 與 "63k"，高度疑似對應 order-6 / order-7 的 restricted IQP 電路，但**這只是數字巧合推論，原文從未明講**，實作前建議自行用相同的 order 截斷跑一次驗證是否真的落在 31k/63k 附近。
12. **哪 100 個 snapshot 用於 DCGAN data-rich regime，以及 N_train∈{10,...,90} 各自是哪些 snapshot 的子集**：只說「100 turbulence snapshots」（`paper.txt:1381`）與 Data-scaling 用 N_train∈{10,30,50,70,90,100}（`paper.txt:1481`, Table 8 `paper.txt:3010`），未給具體時間索引清單或抽樣規則（隨機/等間隔/前綴）。
13. **200 個 held-out τ 的具體索引清單**：只說「uniformly sample 200 additional time indices... excluding the training points」（`paper.txt:1349`），沒有給出這 200 個確切數值，也沒給對應的 seed，無法逐一復現同一組 held-out 集合（但抽樣規則本身——均勻、排除訓練點——是給了的）。
14. **Table 2 中 dagger (†) 標記的意義**：`paper.txt:1429-1430` 的 IQP(63k) 兩個數字後面有 "†" 符號，但全文搜尋（`grep -n "†"`）找不到對應的註腳文字說明其含義（推測是原始 PDF 表格中標示「最佳分數」的粗體/註記，在 HTML 抽取過程中遺失了說明文字）。

---

## 檔案位置

- 本規格書：`<HOME>/gsoc2026-deeplense-qml/docs/IQP_REPLICATION_SPEC_2603.06440.md`
- 原文純文字：`<WORK>/data/papers/2603.06440/paper.txt`
- 原文 HTML（含 MathML alttext）：`<WORK>/data/papers/2603.06440/paper.html`
