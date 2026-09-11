# arXiv:2603.06440 v1 vs v2 逐項比對（針對我們的複現與投稿敘述）

> 目的：確認我們的複現規格與 ML4PS 2026 投稿中的否定結論，是否打在過時的版本上。
> 日期：2026-09-11。比對者：Claude（Opus 5）。
> 論文：Liu, Placidi, Brunner, Rinaldi, *Toward Generative Quantum Utility via Correlation–Complexity Map*。
> 版本：**v1** = 2026-03-06 16:25 UTC（3,443 KB）；**v2** = 2026-05-05 14:45 UTC（3,421 KB）。arXiv 主分類為 **cs.LG**（非 quant-ph）。

## 結論（先講）

**沒有任何改動會推翻我們的複現或否定結論。而且方向和原本的擔心相反：我們用的一直就是 v2。**

1. 本機快取 `<WORK>/data/papers/2603.06440/paper.html` 與 `arxiv.org/html/2603.06440v2` **md5 完全相同**
   （`0142011108b84ec3788a95c59b5f464b`），`paper.txt:282` 自帶 `arXiv:2603.06440v2 [cs.LG] 05 May 2026` 標記；
   `docs/IQP_REPLICATION_SPEC_2603.06440.md:3` 也已標註「v2 2026-05-05」。快取抓取時間 2026-09-05，晚於 v2 發布。
2. 兩項獨立佐證我們讀的是 v2：(i) 我們論文 `main.tex:393` 引用 **Proposition II.2**，這是 v2 的編號（v1 為 Theorem 2.2）；
   (ii) 我們的複現程式 `experiments/13_iqp_turbulence_replication/train_iqp_r3.py:348` 用 `--steps-lat` 預設 **3000**，
   這個數字**只存在於 v2**（v1 全文無 "3000" 字串）。
3. v2 相對 v1 是「期刊化改寫 + 新增可複現性附錄 + 主張弱化」，**所有數值設定與 Table 2 結果完全不變**。
   v2 只有一個方向的實質變化與我們有關：它**補上了我們原本得靠推斷的兩個超參數**（stage-2 步數 3000、以及把 stage-1 的
   N_t / 500 samples / 3×10⁴ steps 集中到 Appendix J Table 6），而我們已經照 v2 做了。

---

## 1. 逐項比對表（依派工清單）

所有 v1 行號指 `<WORK>/data/papers/2603.06440/paper_v1.txt`，v2 行號指 `paper_v2_h2t.txt`（同一支轉檔器 `h2t.py`，可對照）。
原始快取 `paper.txt` 也是 v2，但用的是另一支轉檔器，行號與 `paper_v2_h2t.txt` 不同。

| # | 項目 | v1 怎麼寫 | v2 怎麼寫 | 是否改變 / 對我們的影響 |
|---|------|-----------|-----------|--------------------------|
| 1 | **Algorithm 1（虛擬碼）** | **不存在。** v1 沒有任何 algorithm 環境，流程只用 eq.26–28 + Fig. 6 四步驟文字敘述（`paper_v1.txt:713-737`） | **v2 新增 Algorithm 1「Latent-parameter adaptation for turbulence synthesis」**（12 行虛擬碼，`paper_v2_h2t.txt:748-780`）：step 3 隨機初始化 θ_lat^(t1)、step 4 固定 θ_lat 只訓 θ_core、step 5 Freeze θ_core^(1)、steps 6-9 逐 anchor 只調 θ_lat（warm start 前一個 anchor）、step 10 內插、steps 11-12 取樣解碼並 aggregate 到 grid | **v2 新增。**我們論文 `main.tex:97` 寫「Algorithm~1 of the reference paper」——這句**只在 v2 成立**，v1 讀者會找不到 Algorithm 1。→ 需在 bib/文中鎖版本（見 §3） |
| 2 | stage 1 訓練什麼、θ_lat 是否凍結 | 「we draw θ_lat^(1) from a random initialization and **keep it fixed during the first-stage training**. We then fit only the core parameters θ_core by minimizing the MMD-based loss」（`paper_v1.txt:719`） | 主文同一句逐字不變（`paper_v2_h2t.txt:745`）；另在 Algorithm 1 step 4 與 Appendix J Table 6「First snapshot training: θ_lat^(1) is randomly initialized and fixed; θ_core is trained by minimizing MMD」重述 | **無變化**（v2 只是多講一次）。我們 appendix 講的「frozen θ_lat in stage 1（≈ −0.08 correlation）」依據仍然成立 |
| 3 | stage 1 步數 / 每步樣本數 / N_t | 「using **500 samples** … at each optimization step, with a total of **3×10⁴ training steps** and **N_t = 5×10⁴ training datapoints**」（`paper_v1.txt:719`） | 主文逐字相同（`paper_v2_h2t.txt:745`）；Appendix J Table 6 再列一次（`paper_v2_h2t.txt:2153-2159`） | **無變化**。我們 `--steps-core 30000`、5×10⁴-sample pmf 的設定兩版皆對 |
| 4 | stage 2（latent 適配）步數 | **未給。** v1 只說 warm start 自 θ_lat^(t-1)，**全文無 "3000"**（`grep -c 3000 paper_v1.txt` = 0） | **v2 新增**：主文 eq.(29) 後「**with 3000 optimization steps per adapted snapshot**」（`paper_v2_h2t.txt:824`）＋ Table 6「Latent adaptation optimization steps: 3000 per adapted snapshot」（`:2165`） | **v2 新增資訊，對我們有利。**我們 `--steps-lat 3000` 直接取自 v2；若只有 v1，這個數字得用猜的 |
| 5 | latent block 定義（d_lat=50、Hamming 序、初始化） | 「we set the latent dimension to **d_lat = 50** … we define θ_lat as the **first d_lat elements of the canonical IQP parameter vector obtained by ordering the commuting Pauli strings by increasing Hamming weight**, {∅},{0},{1},…,{n−1},{0,1},{0,2},…」（`paper_v1.txt:789-793`）；後續 anchor「initialized at θ_lat^(t−1)」 | **逐字相同**（`paper_v2_h2t.txt:883-887`），Table 6 另列「Latent dimension d_lat = 50」「Latent initialization for later snapshots: Initialize θ_lat from the previous snapshot」 | **無變化** |
| 6 | 資料編碼（18-bit tuple、量化位元） | 「each (x,y,z) ∈ ℝ³ sample is mapped to a bitstring in {0,1}^{3N} by quantizing each coordinate to one of 2^N bins … we set **N = 6**, yielding an **18-bit** dataset, so the IQP generator acts on **n = 18** qubits」（`paper_v1.txt:709`）；第三維語意在**主文**（`:695`）與 **Appendix F**（`:1727`）各講一次：「x and y denote the spatial coordinates and **z denotes the velocity magnitude**」 | 編碼句逐字相同（`paper_v2_h2t.txt:707`）；Appendix F 的語意句也**逐字保留**（`:1850`「z corresponds to the local velocity magnitude」）；但 **v2 刪掉了主文 Sec III.1 的那句重述**，主文改由 Table 5「Basic data unit: Coordinate/value tuple (x,y,z) ∈ ℝ³」承接 | **實質無變化**（語意仍在 Appendix F）。我們 (x, y, intensity) 的讀法在兩版都有明文依據，無須改敘述 |
| 7 | 每張快照樣本數 N_t = 5×10⁴ | 見 #3，v1 主文已給 | v2 主文＋Table 6 兩處 | **無變化** |
| 8 | 生成時的取樣數 | 「By drawing a number of samples (**we use 10⁵–10⁶ shots**)…」（`paper_v1.txt:695`）＋「(e.g., **10⁵ samples**)」（`:731`） | 主文那句被刪，改放 Appendix J Table 6「Samples per generated time: Approximately 10⁵–10⁶ samples」（`paper_v2_h2t.txt:2171`）；`:757` 附近的 e.g. 10⁵ 保留 | **數值無變化**，只是搬家 |
| 9 | **重建時的聚合規則** | 「aggregate them according to the same probability-aggregation/decoding pipeline used during training」＋「reconstruct the snapshot over the discretized grid」——**沒有定義同格點多筆時怎麼合併** | 仍然**沒有定義**：主文「A synthetic snapshot is reconstructed by drawing many such tuples … and **aggregating them on the discretized grid**」（`paper_v2_h2t.txt:711`）、Algorithm 1 step 12「Decode … and **aggregate them on the discretized grid**」、Table 5「Snapshot reconstruction: Sample many tuples and **aggregate them on the discretized grid**」、Table 6「Output reconstruction: Decode and aggregate samples」 | **兩版都未定義。**我們 appendix 那句「a last-write-wins aggregation convention, **never specified in the paper**」**依然正確**，不需要改（v2 只是把「aggregate」這個動詞多寫了三次，仍沒給規則：空格點怎麼辦、同格點取平均 / 眾數 / 覆蓋都沒說） |
| 10 | **訓練 MMD 的 kernel 與頻寬 σ** | Appendix G：「Here, k(b,b′) is **a Gaussian kernel**…」——**沒有 σ、沒有 median heuristic、沒有 multi-bandwidth**（`paper_v1.txt:1769`） | Appendix G 逐字相同（`paper_v2_h2t.txt:1892`）；Appendix J **也沒有補 σ** | **兩版都未給。**我們「頻寬未定義、沿用 IQPopt(Ref.[38]) 多頻寬慣例」的處理不需要改 |
| 11 | 評估用 Conv2D-MMD 的 kernel/σ | Appendix I：三層 Conv2D–ReLU(stride 2)+GAP、Gaussian kernel、「σ is chosen by the **median heuristic**」、N=M=200 subsample without replacement（`paper_v1.txt:1849-1858`） | **逐字相同**（`paper_v2_h2t.txt:1972-1981`） | **無變化** |
| 12 | 每步 500 樣本 | 見 #3 | 同 | **無變化** |
| 13 | 生成模型參數量（31k / 63k，|s| ≤ 6 / 7） | Table 2 列「IQP (31k)(11)」「IQP (63k)(11)」（`paper_v1.txt:834,840`）；**兩版都沒說 interaction order cutoff**，|s| ≤ 6 / 7 是我們從 Σ_{k≤6} C(18,k)=31,180、Σ_{k≤7}=63,004 反推 | Table 2 完全相同（`paper_v2_h2t.txt:930,936`） | **無變化**；「order cutoff 未載明」在 v2 仍成立 |
| 14 | **Table 2 數值** | RBM(12k)(11) 0.168770 / 1.472481；IQP(31k)(11) **0.023784 / 0.952420**；IQP(63k)(11) **0.022084† / 0.942267†**；GAN(51k)(100) 0.135584 / 1.327760；GAN(0.2M)(100) 0.027695 / 0.895912；GAN(1.2M)(100) 0.012204 / 0.964874 | **逐位元相同**（`diff` v1 `809-895` vs v2 `903-989` 只出現新增的兩句 metric caveat 散文，表格數字列 0 差異） | **無變化。**我們拿來對照的 0.0238 / 0.0221 兩個 PDF–JS 值依然是 v2 的官方數字 |
| 15 | QCLI / CCI 定義式（Eq. 5–7、12–18） | m_k、b_k=C(n,k)/2ⁿ、I_QCLI := D_JS(m‖b) ∈[0,1]；I_CCI = 1 − I_TC^tree / I_TC，I_TC = Σℋ(X_i) − ℋ(X₁…X_n) | **數學完全相同**。差異只有：(i) v2 在 Sec II.2 多插一條 display equation，導致 **Eq. 編號整體 +1**（v1 的 eq.N = v2 的 eq.N+1，自 (4) 起）；(ii) 寫成 I_QCLI(p) 並加註「to emphasize that the indicator quantifies a property of the underlying distribution p」；(iii) I_CCI 由一行拆成兩行 display | **無實質變化**，但**式號要小心**（我們若引用式號，須註明 v2） |
| 16 | Appendix C（Monte-Carlo 估計 L = 2×10⁴） | 「For larger n, we use the Monte Carlo estimator … with a default subset budget (e.g., **L = 2×10⁴**）…（bitmask for n ≤ 64）」（`paper_v1.txt:1695`） | **逐字相同**（`paper_v2_h2t.txt:1818`，只有 equation 編號 51→52） | **無變化** |
| 17 | **Fig. 3（support mismatch）設定** | targets：**10⁴ bitstrings**、n=16、locality ≤ 4、G_n^data ∈ {140,280,420,560,700,840,1050}、SPSA 調參；learner：locality ≤ 2、G_n ∈ {50,100,150}；訓練：train-on-classical、**Adam 5000 iterations、lr 1e-4**（`paper_v1.txt:576-590`） | **逐字相同**（`paper_v2_h2t.txt:586-600`）；v2 另在 Appendix J Table 3 重列同一組數字 | **無變化。**我們 App. `fig3` 的複現設定（140 targets、≤4-local、≤2-local learners 50/100/150 gates、Adam 5000 steps）對兩版都成立 |
| 18 | **Fig. 3 的結論敘述** | 「While **Theorem 2.2** provides a lower bound … the joint dependence on I_QCLI and G_n observed in Fig. 3 is consistent with its central mechanism」 | 同樣的結論，但改口為「consistent with the support-mismatch mechanism underlying **Eq. (24)**」，並刪掉「While Theorem 2.2 provides a lower bound on irreducible MMD rather than a guarantee…」的開頭 | **結論方向不變**（仍主張 upper envelope 隨 QCLI 下降、G_n 越大整體下移），只是措辭從「定理」退到「機制」。我們「claim (a) 反向、claim (b) 重現」的否證**照打不誤** |
| 19 | **Fig. 4（QCLI–CCI，8/12/16 qubits）設定與結論** | 8Q/12Q/16Q、**150 random 2-local gates**、SPSA 只最大化 QCLI、**每個 size 取樣 20000 個電路** | **逐字相同**（只有 "locality-2"→"2-local"、"complexity measures"→"dependence measures" 等字面）；Appendix J Table 4 重列 | **無變化。**我們「低 QCLI 正相關、QCLI≳0.38 後反轉」的結果對兩版都適用 |
| 20 | **新增/刪除的 baseline 或實驗** | baseline：RBM + DCGAN（data-rich 100 snapshots、data-matched 11 snapshots）；DCGAN data-scaling（51k/0.2M/1.2M × N_train ∈ {10,30,50,70,90,100}，Fig. 8）**v1 已有** | **完全相同**。v2 沒有新增任何實驗、圖（兩版都是 Fig. 1–8），只新增 **Appendix J 的 Table 3–8**（純設定彙整，無新數據） | **無新實驗、無新 baseline。**我們「原文只比 RBM/DCGAN」的框架成立 |

---

## 2. v2 其他實質改動（不在清單內，但值得知道）

| 類別 | v1 | v2 | 說明 |
|------|----|----|------|
| 排版體例 | 章節 1/2/3、author–year 引用（46 篇） | 章節 I/II/III（REVTeX 風）、**數字引用 [1]–[49]**、"equation N" → "Eq. (N)" | 顯示 v2 是投期刊的改寫版 |
| **定理降級** | **Theorem 2.2**「Lower Bounding the MMD Loss by I_QCLI」，結論直接是 `min ℒ_MMD ≥ C′(1 − I_QCLI(p))`（eq.22） | **Proposition II.2**「Walsh-support mismatch induces an MMD floor」，結論改成乾淨的 `min ℒ_MMD ≥ C_A Σ_{S∉S_A} P(S)`（eq.22）；與 QCLI 的連結被**移出定理**，變成 Assumption II.3 之下的推論 **Eq. (24)** | 這是 v2 最大的科學性改動：**把「QCLI 能預測 MMD floor」從定理降為條件性假設**，並明說「We treat this alignment as an **empirical hypothesis** and assess it in Sec. II.6」 |
| 節名 | 2.5 "as a **Proxy** for Architecture–Data Support Alignment" | II.5 "as a **Conditional Proxy** …" | 同上 |
| Abstract | 有一句「We provide **theoretical support** … higher QCLI implies a smaller irreducible approximation floor」 | **整段重寫並刪掉這句**；改成「observe **competitive performance against classical baselines in a low-data, low-parameter regime**」 | 主張全面弱化 |
| 新增「Limitations」節 | 標題為 "Caveats and responsible interpretation."，段落較短 | 改為獨立 **Limitations** 節，新增「does not certify quantum advantage」「The turbulence experiments should be interpreted as evidence of **low-data competitiveness against the tested baselines, not as a hardware sampling advantage**」 | 與我們論文「no advantage claim」的立場一致 |
| 新增 baseline 侷限自陳 | 無 | III.2 新增：「They are **not exhaustive**. In particular, diffusion models, autoregressive models, tensor-network distributions, coordinate-based neural fields, and reduced-order physical models could possibly provide stronger baselines」 | 值得注意：**原作者自己承認 baseline 不完整**，可在我們 rebuttal 引用 |
| 新增 metric 侷限自陳 | 無 | III.2 新增：「it is **not a substitute for physics-specific turbulence diagnostics such as energy spectra or structure functions** … leave a full physics-level turbulence validation to future work」 | 與我們「Conv2D-MMD² 已飽和（shuffled ground truth 得 0.704）」的批評方向一致 |
| 新增 tuple vs. full-field 澄清 | 無 | III 新增整段：「the IQP circuit generates a **distribution over quantized coordinate/value tuples rather than a full field** represented as a single bitstring as in [38] … the qubit count scales with the number of bits used to represent a tuple, not with the number of grid points … **should not be confused with direct generation of an entire high-dimensional field vector in one circuit shot**」 | v2 主動澄清了 v1 讀者容易誤解之處；我們的 per-image / per-tuple 編碼理解與之一致 |
| 新增 QCLI 語意但書 | 無 | II.3 新增：「We use "likeness" **operationally** … It should **not be interpreted as a certificate that a distribution is quantum-generated**」 | |
| 新增 map 語意但書 | 「the map is not a hardness certificate」 | 擴寫為「**decision-support diagnostic** … the shaded IQP region is a **protocol-specific** reference … Points in the high-QCLI/high-CCI region are therefore **not guaranteed to exhibit quantum advantage**」 | |
| 參考文獻 | 46 篇 | 49 篇，新增 3 篇、未刪任何一篇：[4] Balló-Gimbernat et al. 2026 *Shallow IQP circuits for generative modeling*；[5] Belis et al. 2026 *Spectral methods: crucial for ML, natural for quantum computers?*；[27] Leontica & Amaro 2024 *Exploring the neighborhood of 1-layer QAOA with IQP circuits* | [4][5] 是 2026 新作，可能值得我們 related work 一併引用 |
| 致謝 | 無 | 新增 "We are grateful to Jacob Swain, Marco Ballarin, Stephen Clark, and Harry Buhrman" | |
| Appendix A–I | — | **全部只有 equation 編號位移與字面潤飾**，無數學或設定改動（Appendix B 證明結構不變，只是把結論改述為 Proposition II.2 + Assumption II.3） | |

---

## 3. 我們論文要改的句子（`paper/ml4ps2026/main.tex`，與 release 副本 `release/iqp-lens-ml4ps2026/paper/main.tex` 同步；兩檔 md5 相同）

沒有任何實驗結論需要改。**只有「版本鎖定」層級的三處需要動**，全部是為了讓審稿人查得到我們引的東西。

| # | 位置 | 現況 | 建議改成 | 理由 |
|---|------|------|----------|------|
| 1 | `refs.bib` 的 `liu_2026_correlation`（`refs.bib:10-18`） | `eprint = {2603.06440}`、`primaryclass = {quant-ph}`、`note = {arXiv:2603.06440}` | `eprint = {2603.06440v2}`、`primaryclass = {cs.LG}`、`note = {arXiv:2603.06440v2 (v2, 5 May 2026); all section, equation, proposition and algorithm numbers in this paper refer to v2}` | (a) v1 沒有 Algorithm 1、Proposition 編號也不同，不鎖版本審稿人會對不上；(b) arXiv 主分類實為 **cs.LG**，我們寫 quant-ph 是錯的 |
| 2 | `main.tex:97` | 「A latent-adaptation protocol (Algorithm~1 of the reference paper, which trains a shared phase core on one anchor sample, …)」 | 「A latent-adaptation protocol (Algorithm~1 of the reference paper, **v2**, which trains …)」（或若採用 #1 的 bib note，可只保留一處版本提示） | **Algorithm 1 只在 v2 存在**，v1 讀者找不到 |
| 3 | `main.tex:393` | 「(\rho \le -0.98 for the Proposition~II.2 floor proxy)」 | 「… for the **v2** Proposition~II.2 floor proxy」（或靠 #1 的 bib note 統一交代） | v1 對應物是 Theorem 2.2，且 v1 的敘述形式是 `C′(1 − I_QCLI)`、v2 才是 `C_A Σ_{S∉S_A} P(S)`；我們的 floor proxy 用的正是 v2 形式 |

**確認不需要改的句子**（已逐句對照兩版）：

- `main.tex:355-356`「Under a last-write-wins aggregation convention, **never specified in the paper**」→ v1、v2 **都**沒有指定聚合規則，敘述正確。若想更保險可寫「never specified in either arXiv version」。
- `main.tex:361`「Our anchor-fit ceiling … traces to a frozen $\theta_{\mathrm{lat}}$ in stage~1 … a $5\times10^{4}$-sample pmf」→ 兩版皆明文如此。
- `main.tex:351-352`「11 anchor snapshots、$n=18$ qubits、D2」→ 兩版相同。
- `main.tex:141`「the reference paper's multi-bandwidth Gaussian MMD kernel」→ 兩版的 Appendix G 都只說 "a Gaussian kernel"，多頻寬是 Ref.[38]/IQPopt 的慣例。這句把 kernel 歸給 reference paper 稍嫌寬鬆，但**與版本無關**，如要更精確可改為「the reference paper's MMD objective under the IQPopt multi-bandwidth Gaussian kernel」。
- App.`fig3`、App.`fig4` 的所有複現設定與否證結論（140 targets、≤4/≤2-local、5,000 steps、20,000 logged points、$\rho$ 值）→ 兩版設定完全相同。
- 我們引用的 Table 2 數字（0.0238、0.0221、0.90–1.47 區間）→ 兩版逐位元相同。

---

## 4. 佐證與可重跑指令

```bash
# 取得兩版（已存檔）
curl -sSL https://arxiv.org/html/2603.06440v1 -o paper_v1.html   # 636,702 B
curl -sSL https://arxiv.org/html/2603.06440v2 -o paper_v2.html   # 928,593 B
md5sum paper_v2.html <WORK>/data/papers/2603.06440/paper.html   # 兩者皆 0142011108b84ec3788a95c59b5f464b

# 轉純文字後比對（轉檔器已存檔）
python3 <WORK>/data/papers/2603.06440/h2t.py paper_v1.html v1.txt
python3 <WORK>/data/papers/2603.06440/h2t.py paper_v2.html v2.txt
diff -u v1.txt v2.txt | wc -l        # 2319 行，50 個 hunk，+833 / -445
grep -c 3000 v1.txt                   # 0  ← stage-2 步數只在 v2
diff <(sed -n '809,895p' v1.txt) <(sed -n '903,989p' v2.txt)   # Table 2 數字零差異
```

存檔位置：
- `<WORK>/data/papers/2603.06440/paper.html` / `paper.txt` — **v2**（2026-09-05 抓，spec 與 summary 的行號基準）
- `<WORK>/data/papers/2603.06440/paper_v1.html` / `paper_v1.txt` — **v1**（2026-09-11 新增）
- `<WORK>/data/papers/2603.06440/paper_v2_h2t.txt` — v2 以同一轉檔器產生，供與 `paper_v1.txt` 對行號
- `<WORK>/data/papers/2603.06440/h2t.py` — 轉檔器

**覆蓋範圍**：v1 與 v2 皆取得 arXiv LaTeXML 全文 HTML（正文＋全部附錄＋參考文獻），非僅 abstract。唯一未比對的是 PDF 中的圖形像素內容（Fig. 1–8 的圖檔本身）；兩版圖數與圖說文字已逐句比對，圖說僅有字面潤飾。
