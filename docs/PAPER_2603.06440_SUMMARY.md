# 論文摘要：Toward Generative Quantum Utility via Correlation–Complexity Map

## 1. Metadata

| 欄位 | 內容 | 出處 |
|---|---|---|
| 標題 | *Toward Generative Quantum Utility via Correlation–Complexity Map* | `paper.txt:280` |
| 作者 | Chen-Yu Liu、Leonardo Placidi、Eric Brunner、Enrico Rinaldi（全部隸屬 Quantinuum） | `paper.txt:283-306` |
| 機構 | Quantinuum（Liu、Brunner、Rinaldi：London, UK；Placidi：Tokyo, Japan） | `paper.txt:288-306` |
| arXiv ID | 2603.06440 | `paper.txt:64,282` |
| 版本 | v2 | `paper.txt:282`（HTML 亦見 `/abs/2603.06440v2`、`/pdf/2603.06440v2`） |
| 日期 | 2026-05-05（原文：`05 May 2026`） | `paper.txt:282` |

**核對結果**：版本（v2）與日期（2026-05-05）與 Leo 提供的資訊完全一致，無差異。

---

## 2. TL;DR

1. 本文問一個訓練前（pre-training）就能回答的問題：給定一組古典資料集，能否在投入昂貴的量子生成模型訓練之前，就判斷它是否「適合」用 IQP（instantaneous quantum polynomial-time）電路來生成？
2. 提出 **Correlation–Complexity Map**：由兩個可直接從資料樣本計算的指標構成——量子相關相似指標 **QCLI**（頻譜面向，衡量資料的 Walsh–Fourier correlation-order 頻譜偏離古典 i.i.d. binomial baseline 的程度）與古典相關複雜指標 **CCI**（結構面向，衡量資料中無法被 Chow–Liu tree 這種 pairwise 模型解釋的殘餘相依性比例）。
3. 理論上證明（Proposition II.2）：若目標分布在受限 IQP 架構「可調 Walsh 支撐集合」之外仍有頻譜質量，則此質量會對 MMD loss 造成不可訓練消除的下界（irreducible floor）；並提出 Assumption II.3 將此下界與 QCLI 連結。
4. 用這套地圖掃描多種資料集，發現 **turbulence（紊流）資料**落在「高 QCLI、高 CCI」的 IQP 相容象限。
5. 針對紊流資料，提出一個 float-to-bitstring 編碼（每座標 N=6 bits，(x,y,z) 共 18 bits → 18-qubit IQP 電路）與一個「latent-parameter adaptation」機制（Algorithm 1），只用 11 個訓練 snapshot 就在 PDF–JS 與 feature-space MMD 兩個指標上打平甚至超越參數量大 10–40 倍、訓練資料多 10 倍的 classical GAN／RBM baseline。
6. 作者明確聲明：這是「朝向 utility 的一步」，**不是**端到端的量子優勢（quantum advantage）證明；QCLI/CCI 是診斷性（diagnostic）而非認證性（certifying）指標。

---

## 3. 問題設定

原文 Introduction 指出（`paper.txt:326-345`）：

- 量子計算被認為能表示古典方法難以複製的機率分布；量子機器學習（QML）中，**生成式建模（generative modeling）**特別有前景，因為採樣（sampling）本身就是量子硬體的原生操作。
- **IQP 電路**是被廣泛研究的一類生成模型：其輸出分布被相信在古典上難以取樣——因為若能有效古典模擬 IQP 取樣，會導致多項式階層（polynomial hierarchy, PH）崩塌 [9]，這是複雜度理論上支持 IQP 取樣古典難解的論據。
- 已有工作提出「train on classical, deploy on quantum」的可擴展 workflow [38]：在古典硬體上用可古典計算的量（例如可交換 observable 的期望值）優化訓練目標，避免硬體迴路訓練的成本與不穩定，同時保留取樣階段古典難模擬的可能性。
- 但仍有兩個尚待解決的方向（`paper.txt:332`）：
  1. 從「純 bitstring 原生」資料集擴展到能自然支援浮點數／整數值資料的場景；
  2. 改善「尋找與 IQP 這類古典難解量子模型的 inductive bias 相符的真實資料分布」——把重點從證明最壞情況（worst-case）難度，轉向「在投入大量研究資源前，診斷一個真實世界資料集是否結構上與 IQP 生成模型相容」。
- 本文的核心貢獻（`paper.txt:350`）：(i) 兩個互補、可計算的指標 QCLI、ICCI，用來刻劃資料集的「IQP 相容性」與「超越 pairwise 的古典相依程度」；(ii) 一個引導尋找 IQP 相容真實資料的 correlation–complexity map；(iii) 透過 turbulence 資料集上的新型 latent adaptation 方法所做的實證案例研究。

---

## 4. 方法

### 4.1 IQP 電路（一般定義，Eq. 1；套用到 turbulence 的版本，Eq. 25）

一般 $n$-qubit IQP 電路定義為（`paper.txt:410`，Eq. (1)）：

$$
U=\exp\!\Bigl(i\sum_{s\subseteq[n]}\theta_{s}X_{s}\Bigr)=H^{\otimes n}\Bigl(\exp\!\bigl(i\sum_{s\subseteq[n]}\theta_{s}Z_{s}\bigr)\Bigr)H^{\otimes n},
$$

其中 $X_s := \prod_{j\in s}X_j$、$Z_s := \prod_{j\in s}Z_j$ 為作用在子集合 $s\subseteq[n]$ 上的多體 Pauli string。因所有 $X_s$ 彼此交換（commute），此電路族在 $X$-basis 下同時可對角化。實作上通常只實現一、二體項的受限子集（`paper.txt:411-419`）。

量測分布（computational/$Z$ basis）即 phase function $f(x)=e^{i\phi(x)}$ 的 Walsh–Hadamard transform（WHT）平方模（`paper.txt:426-455`，Eq. (2)(3)）：

$$
p(z)=\bigl|2^{-n}\sum_{x}(-1)^{z\cdot x}e^{i\phi(x)}\bigr|^{2},\quad \phi(x)=\sum_{s}\theta_{s}\chi_{s}(x).
$$

在 Turbulence 應用中（Section III.1，`paper.txt:1102-1108`，Eq. (25)），電路寫成受限版本，用來對量化後的 coordinate/value tuple 分布建模：

$$
U(\theta)=H^{\otimes n}\exp\!\Bigl(i\sum_{s\subseteq[n]}\theta_{s}Z_{s}\Bigr)H^{\otimes n},\qquad x\sim q_\theta(x):=|\langle x|U(\theta)|0^n\rangle|^2.
$$

參數被拆成共享的 core 區塊與 snapshot 專屬的 latent 區塊：$\theta=(\theta_{\text{core}},\theta_{\text{lat}})$，$\dim(\theta_{\text{lat}})=d_{\text{lat}} \ll \dim(\theta_{\text{core}})$（`paper.txt:1112`，Eq. (26)）。

### 4.2 QCLI（Quantum Correlation–Likeness Indicator，Section II.2–II.3，`paper.txt:461-577`）

先定義 correlation-order 分解：對 $M$ 個 $n$-bit 樣本的經驗 pmf $p(x)$ 做 WHT 並平方，得到 Walsh–Fourier power $P(s)\in[0,1]$（Eq. (5)）；再依 subset 大小 $|s|=k$ 聚合成 order-$k$ power fraction（Eq. (6)）：

$$
m_k=\frac{\sum_{|s|=k}P(s)}{\sum_{s\subseteq[n]}P(s)},\quad k=0,\ldots,n,\qquad \sum_k m_k=1.
$$

古典 i.i.d. 公平位元的參考基準是 binomial 分布 $b_k = \binom{n}{k}/2^n$。QCLI 定義為 $m$ 與 $b$ 的 Jensen–Shannon divergence（`paper.txt:513-518`，Eq. (7)）：

$$
D_{\mathrm{JS}}(m\|b)=\tfrac12 D_{\mathrm{KL}}(m\|M)+\tfrac12 D_{\mathrm{KL}}(b\|M) := I_{\text{QCLI}}(p),\quad 0\le I_{\text{QCLI}}(p)\le 1,\qquad M=\tfrac12(m+b).
$$

四個良好性質（`paper.txt:520-540`）：null 行為正確（無結構偏差時 $\to 0$）、置換不變性（只看 order，不看是哪些 bit）、有界（$[0,1]$）、跨 $n$ scale-free（可比較不同位元長度的資料集）。

二階近似顯示 $D_{\mathrm{JS}}$ 表現如加權 $\ell_2$ 距離（Appendix A.1，`paper.txt:547-549`，Eq. (8)），並可推導出 total variation 上界（`paper.txt:551-577`，Eq. (9)-(11)）：

$$
\mathrm{TV}(m,b)\lesssim O(\sqrt{I_{\text{QCLI}}}).
$$

作者特別澄清（`paper.txt:579`）：「likeness」是操作性用語——QCLI 量測與 binomial null 的偏離，動機來自 IQP parity 干涉，**不應解讀為「此分布是量子產生」的證書**。

### 4.3 CCI（Classical Correlation–Complexity Indicator，Section II.4，`paper.txt:616-745`）

定義（Eq. (18)）：

$$
I_{\text{CCI}} = 1 - \frac{I_{\mathrm{TC}}^{\text{tree}}}{I_{\mathrm{TC}}}, \qquad I_{\mathrm{TC}} = \sum_{i=1}^n \mathcal{H}(X_i) - \mathcal{H}(X_1,\ldots,X_n),
$$

其中 $I_{\mathrm{TC}}$ 是 **total correlation**（multi-information，源自 Watanabe [46]，後由 Schneidman et al. [39] 用於研究二元神經群體的集體相關性），$I_{\mathrm{TC}}^{\text{tree}} = \sum_{(i,j)\in T^*} I(X_i;X_j)$ 是由最大似然 **Chow–Liu tree**（`paper.txt:642`，[13]）捕捉到的那部分依賴性；$T^*$ 是使 $\sum_{(i,j)\in T} I(X_i;X_j)$ 最大化的 spanning tree（Eq. (15)）。

Chow-Liu 定理保證 tree 分布 $p_T(x)$ 在所有 tree-structured 模型中最大化 likelihood，等價於最小化 $D_{\mathrm{KL}}(p\|p_T) = I_{\mathrm{TC}} - I_{\mathrm{TC}}^{\text{tree}}$（`paper.txt:695-705`，Eq. (19)(20)）。由此 $0\le I_{\text{CCI}}\le 1$：值小代表資料被局部 pairwise 相依性充分解釋，值大代表存在複雜的非局部（non-local）多體相互作用（`paper.txt:719-724`）。

### 4.4 Proposition II.2：Walsh-support mismatch 導致 MMD 下界

**命題陳述**（`paper.txt:792-813`，Eq. (21)(22)）：設 $p$ 為 $\{0,1\}^n$ 上的二元分布，$\mathcal{F}_{\mathrm{IQP}}$ 為可調 Walsh–Fourier 支撐集合為 $S_A \subseteq 2^{[n]}$ 的受限 IQP 架構，$\mathcal{L}_{\mathrm{MMD}}(p,q)$ 為正核（positive kernel）下的 MMD loss。令 $P(S):=|2^{-n}\sum_x p(x)\chi_S(x)|^2$ 為 $p$ 在頻率 $S$ 上的 Walsh–Fourier power。則存在常數 $C_A>0$（僅依賴 kernel 正規化與架構）使得

$$
\min_{q\in\mathcal{F}_{\mathrm{IQP}}} \mathcal{L}_{\mathrm{MMD}}(p,q) \ge C_A \sum_{S\notin S_A} P(S).
$$

**Assumption II.3（QCLI–support alignment，`paper.txt:822-832`，Eq. (23)）**：假設存在常數 $\kappa>0$ 使 $\sum_{S\notin S_A} P(S) \ge \kappa(1-I_{\mathrm{QCLI}}(p))$，即架構外的頻譜質量與 order-spectrum 偏離 binomial baseline 的程度負相關。

**結合結果（`paper.txt:849-853`，Eq. (24)）**：

$$
\min_{q\in\mathcal{F}_{\mathrm{IQP}}} \mathcal{L}_{\mathrm{MMD}}(p,q) \ge C'(1 - I_{\text{QCLI}}(p)), \qquad C' = C_A\kappa > 0.
$$

**證明梗概**（Appendix B，`paper.txt:2186-2320`）：把 MMD loss 在 parity/Walsh basis 下展開為 $\mathcal{L}_{\mathrm{MMD}}(p,q_\theta)=\sum_S c_S(\mathbb{E}_p[\chi_S]-\mathbb{E}_{q_\theta}[\chi_S])^2$（Eq. (46)）；對 $S\notin S_A$，架構無法調整 $\mathbb{E}_{q_\theta}[\chi_S]$（設為 0），因此該項與 $\theta$ 無關，是不可訓練消除的「irreducible error」（Eq. (47)）；用 $(\mathbb{E}_p[\chi_S])^2=4^n P(S)$ 及 $c_{\min}=\min_S c_S>0$ 導出下界 $C_0\sum_{S\notin S_A}P(S)$（Eq. (48)-(50)），即證得 $C_A=C_0$。

**適用範圍**（`paper.txt:866-874`）：此命題僅在**受限**（truncated）情形 $S_A \subsetneq 2^{[n]}$ 下有意義；全支撐情形 $S_A=2^{[n]}$ 時外部質量為零，命題變得空洞（vacuous），但**不代表** IQP 能完美表示該分布——只是此特定的 support-mismatch 機制不再是誤差來源，有限預算優化、有限採樣、kernel mismatch 等其他誤差仍可能存在。

### 4.5 Turbulence 的 float-to-bitstring 編碼（Appendix F，`paper.txt:2537-2560`）

給定 turbulence 樣本 $(x,y,z)\in\mathbb{R}^3$（$x,y$ 為 2D 空間座標，$z$ 為局部速度大小），每個座標落在已知區間 $[a,b]$：

- 用 $N$ qubit 每座標，量化解析度 $\Delta = (b-a)/2^N$（Eq. (54)）。
- 量化映射：$k(v) := \lfloor 2^N \frac{v-a}{b-a}\rfloor + 1 \in \{1,\ldots,2^N\}$（Eq. (55)）。
- 標準 $N$-bit 二進位編碼：$b(v):=\text{BinaryEncode}_N(k(v))\in\{0,1\}^N$（Eq. (56)）。
- 三座標串接：$(x,y,z)\mapsto [b(x)\|b(y)\|b(z)]\in\{0,1\}^{3N}$（Eq. (57)），整個資料集變成形狀 $(M,3N)$ 的二元矩陣。

**關鍵事實**：實驗中選 $N=6$ bits/coordinate，故每個 $(x,y,z)$ tuple 編碼成 **18-bit** 向量，即 IQP 電路作用在 **$n=18$ qubits**（`paper.txt:2560`；亦見 `paper.txt:1449-1451`）。此編碼與 [10] 中的編碼概念相似（`paper.txt:2540`）。此編碼是純古典、確定性（deterministic）且可逆的（efficiently invertible），生成的 bitstring 可反解回連續座標。

### 4.6 Algorithm 1：Latent-parameter adaptation for turbulence synthesis（Section III.1 / Appendix G，`paper.txt:1081-1250`）

流程（`paper.txt:1134-1246`）：

1. **Input**: anchor snapshots $\{\hat{p}_t\}_{t\in\mathcal{T}_{\text{train}}}$、IQP 模型 $q_{(\theta_{\text{core}},\theta_{\text{lat}})}$、latent 維度 $d_{\text{lat}}$。**Output**：未見時間 $\tau$ 的合成 snapshot $\tilde{y}(\tau)$。
2. 選一個初始 anchor time $t_1$（實驗中 $t=1$），隨機初始化 $\theta_{\text{lat}}^{(t_1)}$ 並固定，只訓練 core 參數：
   $\theta_{\text{core}}^{(1)} \leftarrow \arg\min_{\theta_{\text{core}}} \mathcal{L}_{\mathrm{MMD}}(q_{(\theta_{\text{core}},\theta_{\text{lat}}^{(t_1)})}, \hat{p}_{t_1})$（Eq. (27)）。訓練用每步 500 個模型樣本估計 $\mathcal{L}_{\mathrm{MMD}}$，共 $3\times10^4$ 訓練步，$N_t=5\times10^4$ 訓練資料點。
3. 凍結 $\theta_{\text{core}}^{(1)}$；對每個後續 anchor time $t_j$，從前一個 anchor 的 latent 初始化，只調整 latent 區塊：
   $\theta_{\text{lat}}^{(t_j)} \leftarrow \arg\min_{\theta_{\text{lat}}} \mathcal{L}_{\mathrm{MMD}}(q_{(\theta_{\text{core}}^{(1)},\theta_{\text{lat}})}, \hat{p}_{t_j})$（Eq. (29)），每個 snapshot 3000 個優化步。
4. 對學到的 latent 軌跡 $\{\theta_{\text{lat}}^{(t)}\}$ 做**內插**（piecewise-linear，Eq. (31)：$\tilde{\theta}_{\text{lat}}(\tau)=(1-\alpha)\theta_{\text{lat}}^{(k)}+\alpha\theta_{\text{lat}}^{(k+1)}$，$k=\lfloor\tau\rfloor$，也支援非凸權重外插）取得未見時間 $\tau$ 的 $\tilde{\theta}_{\text{lat}}(\tau)$。
5. 從 $q_{(\theta_{\text{core}}^{(1)},\tilde{\theta}_{\text{lat}}(\tau))}$ 採樣 bitstring（每次合成抽 $10^5$ 樣本），並用 Appendix F 編碼的反函數解碼回連續座標並聚合，得到 $\tilde{y}(\tau)$。

實驗設定（Section III.2，`paper.txt:1354-1385`）：latent 維度 $d_{\text{lat}}=50$（即 $\theta_{\text{lat}}\in\mathbb{R}^{50}$），依 commuting Pauli string 的 Hamming weight 遞增排序取前 50 個作為 latent block；訓練用 11 個 anchor snapshots，$t\in\{1,100,200,\ldots,1000\}$（共 11 點），評估用另外均勻抽樣的 200 個未見時間點。

---

## 5. 主要結果

### 5.1 Table 2：turbulence 生成模型定量比較（出處 `paper.txt:1400-1463`）

逐格核對結果：**與 Leo 提供的數字完全一致，無差異。**

| Model (Params.)(# TD.) | PDF–JS ↓ | MMD² (Conv2D) ↓ |
|---|---|---|
| Real (—)(—) | – | – |
| RBM (12k)(11) | 0.168770 | 1.472481 |
| IQP (31k)(11) | 0.023784 | 0.952420 |
| **IQP (63k)(11)** | **0.022084†** | **0.942267†** |
| GAN (51k)(100) | 0.135584 | 1.327760 |
| GAN (0.2M)(100) | 0.027695 | 0.895912 |
| GAN (1.2M)(100) | 0.012204 | 0.964874 |
| GAN (51k)(11) | 0.366758 | 1.397969 |
| GAN (0.2M)(11) | 0.328273 | 1.407441 |
| GAN (1.2M)(11) | 0.244852 | 1.446292 |

（# TD. = 訓練資料筆數 / number of training data；PDF–JS 與 MMD²(Conv2D) 定義見 Appendix I，`paper.txt:2699-2739`：PDF–JS 是像素強度邊際分布的 Jensen–Shannon divergence（$B=50$ bins，Eq. (61)）；MMD²(Conv2D) 是用固定隨機初始化的三層 Conv2D–ReLU + global average pooling encoder $\phi(\cdot)$ 抽特徵後，以 median-heuristic 選 $\sigma$ 的 Gaussian kernel 計算的 biased MMD² 估計量（$N=M=200$，Eq. (63)）。）

**關於 † 符號**：已在 HTML（`paper.html` 中 `S3.T2.2.5.3.1.1` 與 `S3.T2.2.5.4.1.1` 兩個 `<sup class="ltx_sup">†</sup>`）與該表格 caption 全文（`paper.txt:1463`：「Table 2: Qualitative samples (center) with corresponding PDF–JS and feature-space MMD² scores (right). Lower scores indicate better alignment with the real turbulence distribution. Models are shown with their parameter counts in parentheses.」）中搜尋，**原文未找到 † 的圖例說明**。這兩格同時也是該欄唯一以粗體（`ltx_font_bold`）呈現的儲存格，推測 † 標記可能代表「該欄最佳整體成績」，但論文正文與 HTML 皆無明確的 footnote 文字佐證，故此為推測、非原文陳述。

**文字結論**（`paper.txt:1465-1490`）：
- **資料匹配（11 snapshots）情形**：DCGAN 在僅 11 筆訓練資料下訓練失敗，樣本出現嚴重 artifact（training instability 的表現），PDF–JS 與 feature MMD 都明顯劣於 IQP；IQP（31k、63k 參數）保持穩定，產生視覺上連貫的 turbulence-like 場，PDF–JS 與 MMD 都優於 RBM 與 DCGAN。
- **資料豐富（100 snapshots）情形**：DCGAN 顯著進步，高容量變體可在一或兩個指標上打平甚至超越 IQP，但需要約 10 倍訓練資料、且最強變體需要高達 1.2M 參數；相對地 IQP 只用 11 筆 snapshot、固定緊湊電路加 latent adaptation 就達到可競爭表現。
- 整體呈現「regime-dependent trade-off」：DCGAN 在資料/容量充足時表現很好，但在極低資料量下脆弱；IQP latent-adaptation 在 turbulence 上維持有效且樣本效率高，與 turbulence 落在「高 QCLI/高 CCI」IQP 相容區域的假設一致。

### 5.2 Figure 3（`paper.txt:920-928`）：QCLI–MMD 支撐失配機制的實證探測

- 對一族 16-qubit（gate locality ≤4）IQP 電路生成的目標資料集（$G_n^{\text{data}}\in\{140,\ldots,1050\}$，透過 SPSA 優化以擴大 QCLI 範圍至約 0 到 0.7），用受限學習者（locality ≤2，$G_n\in\{50,100,150\}$）擬合，記錄達成的 MMD loss。
- 結論：雖然單次訓練有變異性，但 MMD 的**上包絡（upper envelope）隨 QCLI 上升而系統性下降**；增加學習者的 gate count $G_n$ 會使上、下包絡整體下移，與 support-mismatch 機制一致（更大的可調支撐 $S_A$ 降低架構外殘餘頻譜質量）。
- 行為分區（`paper.txt:930-934`）：QCLI ≲0.1 時目標幾乎無可及 parity 結構，容易被擬合到低 MMD；QCLI≈0.15 附近出現最大失配（目標開始有結構但與受限學習者的 $S_A$ 對不齊）；QCLI 更高時上包絡持續下降，下包絡則可能飽和甚至輕微下降（alignment 帶來的好處開始抵銷優化難度提升）。

### 5.3 Figure 4（`paper.txt:944`）：高 QCLI IQP 輸出呈現伴生的高 CCI

- 針對 8Q/12Q/16Q（2-local, 150 random gates）IQP 電路，僅以最大化 QCLI 為目標優化參數（**未對 CCI 施加任何約束或懲罰**），每個設定沿優化過程採樣 20000 個電路計算 QCLI 與 CCI。
- 結論：儘管目標函數從未包含 CCI，隨著 QCLI 被最大化，**CCI 系統性地同步升高**，在所有系統大小下皆觀察到此「emergent coupling」，顯示 IQP 相容性與 beyond-pairwise 相依性並非獨立現象。

### 5.4 Figure 5（`paper.txt:988-990`）：Correlation–Complexity Map

- 涵蓋標準結構化 benchmark（MNIST、synthetic blobs）、量子硬體/取樣資料（D-Wave 退火樣本、Google RCS 資料 [3]）、動力系統（Lorenz attractor [43]、turbulence D1/D2 [25]）。
- 低 QCLI、低 CCI 區域（原點附近）：MNIST、簡單幾何 blobs 等「classically expressible」資料；短退火時間的 D-Wave 100 qubit 樣本也落於此區。
- 沿 CCI 軸上移：非局部、多體相依性升高的資料，如 Lorenz attractor，無法被 tree-based 模型忠實捕捉。
- 沿 QCLI 軸移動：Fourier–Walsh 特徵含量子式干涉結構的資料，如 Google RCS 資料。
- **右上象限（高 QCLI、高 CCI）＝ IQP 相容區域**：**turbulence 資料集（D1、D2，[25]）落在此區**，是本文後續實驗聚焦於 turbulence 的動機來源。
- 補充說明：目前所觀察到的真實/benchmark 資料集最大 QCLI 值約低於 0.2，遠低於合成 IQP-generated targets 可達到的 ~0.7，顯示強 IQP-aligned 的真實資料相對罕見，此地圖的用途正是讓這種搜尋變得可操作（`paper.txt:986`）。

---

## 6. 與 Ref [38] 的比較（Table 1，出處 `paper.txt:995-1080`）

Ref [38]（Recio-Armengol, Ahmed, Bowles, 2025, *Train on classical, deploy on quantum*）在多個資料集上已有 IQP 生成建模的結果，本文用來說明為何不需要在 Fig. 5 中對每個資料集重新訓練 IQP（`paper.txt:995`），並藉此聚焦於全新（高 QCLI/高 CCI）的 turbulence regime。

| Dataset (size) | IQP setup | Baselines | Key outcome |
|---|---|---|---|
| BinaryBlobs (16 bits) | Local IQP gates (up to 6-body), 14892 params | EBM (1700 params), RBM, Bitflip | EBM 的 log-likelihood 優於 IQP（EBM $-5.34$ vs. IQP $-6.35$）；Bitflip 表現差；MMD 顯示對 mode imbalance 敏感。 |
| D-Wave annealer samples (484 qubits, 100 μs) | All 1- and 2-qubit gates on 484 qubits, $>10^5$ params | RBM, EBM, Bitflip | 沒有模型完美擬合，但 IQP 在比較模型中表現最好；RBM/EBM 在 test MMD 上有時比 random 還差；EBM 在此設定下失敗。 |
| Binarized MNIST (784 bits) | All-to-all 2-qubit gates, 307,720 params | Bitflip only | IQP 捕捉到大量結構；covariance 大致吻合整體模式，但較弱；test MMD 與 noisy-data reference（約 $p\approx0.3$ 噪聲水準）相當；Bitflip 學不到可比較的結構。 |

**核對結果**：與 Leo 提供的摘要在事實內容上一致，本次核對原文用詞後在「Key outcome」欄補上原文的精確措辭（log-likelihood 數值 $-5.34$ / $-6.35$、$>10^5$ params、noise level $p\approx0.3$ 等），無實質差異。

原文額外評論（`paper.txt:1005-1010`）：這些結果為 correlation–complexity map 提供外部一致性檢驗——在 beyond-pairwise 依賴性高的 regime，classical 訓練不穩定性（RBM 的 mode collapse/imbalance、EBM 的 MCMC mixing 不足）可能主導表現，而 IQP 訓練在相同計算預算下相對穩健。但作者也**明確提醒**（引用 [38] 自身的說明）：他們測試的 classical baseline 並非 state-of-the-art，更強的超參數/初始化/模型選擇可能得到更好的 classical 結果——此提醒對 D-Wave 那個資料點特別重要。

---

## 7. 作者自承限制（Section IV「Limitations」與 Section V「Conclusion」，`paper.txt:1548-1569`）

- **指標是診斷性、非認證性（diagnostic rather than certifying）**：高 (QCLI, CCI) 只是暗示一個「promising regime」，**不能**證明量子優勢、不能保證高 QCLI 資料集一定能被 IQP 模型學好、也不能排除存在更強的 classical baseline、改良的訓練 heuristic 或替代資料表示法會改變資料集在地圖上的位置。
- **低 QCLI 不代表 IQP 學不會**：只代表本文強調的這種 parity-structured support-mismatch 機制不太可能是（若成功的話）成功的主要解釋。
- **Turbulence 實驗的定位**：應被解讀為「在所測試 baseline 下低資料量情境的競爭力證據」，**不是**硬體取樣優勢的證明，也不是 turbulence 本質上古典難以取樣的宣稱。
- **Baseline 不夠全面**：RBM 與 DCGAN 只是兩個對照組，diffusion models、autoregressive models、tensor-network 分布、coordinate-based neural field、reduced-order physical model 都可能是更強的 baseline（`paper.txt:1332`）。
- **評估指標的侷限**：PDF–JS 與 random Conv2D feature MMD 是分布層級的 proxy，**不能取代物理層級的紊流診斷**（如 energy spectra 或 structure functions），完整物理層級驗證留待未來工作（`paper.txt:1394-1396`）。
- **未來方向**（Section V Conclusion，`paper.txt:1557-1568`）：(1) 更廣泛/嚴謹的 classical baseline suite（更強的 graphical models、diffusion/score-based 生成器、representation-learning pipeline）；(2) 對指標本身做 stress-test（量化/二值化選擇的敏感度、直接優化 (QCLI, CCI) 的表示法搜尋、conditional/multiscale 變體）；(3) 加入 tensor-network 導向的古典可追蹤性軸，區分「對目前 baseline 難」與「對已知最佳結構化古典模擬器難」；(4) 最終需納入硬體限制（有限 shots、noise、connectivity）並在真實量子裝置上驗證預測的 IQP 相容 regime 是否仍成立。

---

## 8. 關鍵參考文獻

（HTML 書目解析陷阱已驗證：`<li id="bib.bibN">` 的 N **不等於**正文引用編號 [M]，須以每筆條目內 `ltx_role_refnum` span 標示的 `[M]` 為準。以下 9 筆逐一核對，皆找到對應內容，無缺漏。）

| 正文編號 | HTML `id` | 內容 |
|---|---|---|
| **[3]** | `bib.bib6` | F. Arute, K. Arya, R. Babbush, D. Bacon, J. C. Bardin, R. Barends, R. Biswas, S. Boixo, F. G. Brandao, D. A. Buell, et al. (2019). *Quantum supremacy using a programmable superconducting processor*. Nature **574**(7779), pp. 505–510. （被引用於 §I、§II.8） |
| **[4]** | `bib.bib40` | O. Balló-Gimbernat, M. Arroyo-Sánchez, P. García-Molina, A. Garriga, F. Vilariño (2026). *Shallow instantaneous quantum polynomial-time circuits for generative modeling on noisy intermediate-scale quantum hardware*. Physical Review A **113**(4), pp. 042617. （被引用於 §I） |
| **[9]** | `bib.bib10` | M. J. Bremner, R. Jozsa, D. J. Shepherd (2011). *Classical simulation of commuting quantum computations implies collapse of the polynomial hierarchy*. Proc. R. Soc. A **467**(2126), pp. 459–472. （被引用於 §I、§II.1 — 即本文 IQP 古典難模擬論證的核心引用） |
| **[11]** | `bib.bib26` | H. Buhrman, N. Galke, K. Meichanetzidis (2025). *Formal framework for quantum advantage*. arXiv:2510.01953. （被引用於 §I、§II） |
| **[22]** | `bib.bib9` | H. Huang, M. Broughton, N. Eassa, H. Neven, R. Babbush, J. R. McClean (2025). *Generative quantum advantage for classical and quantum problems*. arXiv:2509.09033. （被引用於 §I ×4） |
| **[25]** | `bib.bib17` | A. R. Khojasteh, S. Laizet, D. Heitz, Y. Yang (2022). *Lagrangian and Eulerian dataset of the wake downstream of a smooth cylinder at a Reynolds number equal to 3900*. Data in Brief **40**, pp. 107725. （被引用於 Table 5、§I、§II.8、§III — **本文 turbulence D1/D2 資料集的來源**，與 memory 中 `turbulence-dataset-khojasteh2022.md` 記錄的 DOI 10.15454/GLNRHK 對應同一資料集） |
| **[38]** | `bib.bib4` | E. Recio-Armengol, S. Ahmed, J. Bowles (2025). *Train on classical, deploy on quantum: scaling generative quantum machine learning to a thousand qubits*. arXiv:2503.02934. （被引用次數最多：Appendix B、Appendix G ×4、§I ×3、§II.6 ×2、§II.8 ×3、Table 1、§III.3 ×2、§III ×2、§IV — 本文方法論的直接基礎） |
| **[39]** | `bib.bib2` | E. Schneidman, S. Still, W. Bialek, M. J. Berry (2003). *Network information and connected correlations*. Physical Review Letters **91**(23), pp. 238701. （被引用於 §II.4，CCI 的 total correlation 概念發展） |
| **[46]** | `bib.bib1` | S. Watanabe (1960). *Information theoretical analysis of multivariate correlation*. IBM Journal of Research and Development **4**(1), pp. 66–82. （被引用於 §II.4，total correlation/multi-information 的原始出處） |

---

*文件產生依據：`<WORK>/data/papers/2603.06440/paper.txt`（3117 行）與 `paper.html`（928,593 bytes），2026-09-05 讀取。所有數字、公式與引文均附行號或 HTML 定位出處；找不到的項目（Table 2 † 符號的圖例說明）已在對應章節明確標註「原文未找到」。*
