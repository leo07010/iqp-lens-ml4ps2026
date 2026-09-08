# IQP (Instantaneous Quantum Polynomial-time) 電路文獻調查

> 目的：為復刻 arXiv:2603.06440（Quantinuum, "Toward Generative Quantum Utility via Correlation-Complexity Map"）提供背景文獻整理。
> 方法：所有列入的 arXiv ID 均已透過 WebFetch 實際核對其 abs 頁面的 title / authors / submission date，非僅憑搜尋摘要臆測。
> 日期：2026-09-05。共收錄 **26 篇**論文（含 5 篇 2023 年前的奠基性文獻，21 篇 2023–2026 主體文獻）。

---

## 1. IQP 的定義

**電路形式：**

$$
C = H^{\otimes n} \cdot \exp\!\Big(i \sum_{S} \theta_S Z_S\Big) \cdot H^{\otimes n}
$$

其中 $Z_S = \bigotimes_{j \in S} Z_j$ 是作用在子集合 $S \subseteq \{1,\dots,n\}$ 上的 Pauli-$Z$ 張量積（對應 diagonal gate，例如單體 $R_Z$、雙體 $ZZ$、甚至高階 $ZZZ$ 交互項），$\theta_S$ 為可調角度。整個電路在計算基底 $|0\rangle^{\otimes n}$ 上初始化，中間層全部由**互相對易（commuting）的 diagonal gate** 構成，兩側各夾一層 Hadamard $H^{\otimes n}$。

**為何稱為「instantaneous」：** 因為中間所有 gate 彼此對易，執行順序、甚至「同時發生」都不影響結果——量子計算不需要時間上的因果排序（"temporally unstructured quantum computation"，這正是 Shepherd 與 Bremner 最早使用的說法）。這與一般需要嚴格 gate 排序的通用量子電路（如 BQP 電路）形成對比：IQP 只需要「準備、施加一堆互相對易的 phase、測量」三步驟，沒有中間的量子態演化順序問題。

**與 X-program 的關係：** Dan Shepherd 在更早的工作中，用二元矩陣（binary matroid）理論刻畫「所有 gate 皆對易」電路所能產生的機率分布，稱為 **X-program**（因為對易 gate 集合可用一個 GF(2) 上的線性系統描述，等價於 hidden linear function / stabilizer 結構）。IQP 電路正是 X-program 在量子計算脈絡下的具體實現：$\theta_S$ 的支撐集合 $S$ 對應矩陣的列，電路輸出的機率分布與該二元矩陣的 **Tutte 多項式（Tutte polynomial）**、以及對應線性碼的 **weight enumerator polynomial** 直接相關。這條矩陣理論的連結，是後來許多「哪些 IQP 家族可古典模擬」結果（例如 sparse/稀疏交互作圖）的數學基礎。

---

## 2. 複雜度理論：抽樣難度、古典模擬邊界、驗證

### 2.1 抽樣難度（sampling hardness）

- **精確 / 乘法誤差（multiplicative error）**：Bremner–Jozsa–Shepherd（1005.1407, 2010）證明，若 IQP 電路族的輸出分布可被古典電腦以**乘法誤差 41%** 以內有效抽樣，則多項式階層（polynomial hierarchy, PH）會坍縮到第三層——這在複雜度理論中被視為極不可能發生，因此構成「IQP sampling 難」的強證據。他們同時證明 **post-IQP = PP**（加上 postselection 後與 PP 等價），並指出若只測量 $O(\log n)$ 條輸出線，則可古典有效模擬。
- **平均情況 / 加法誤差（average-case, additive error）**：真正貼近實驗現實的是**加法誤差**下的難度。Bremner–Montanaro–Shepherd（1504.07999, PRL 2016——**注意：使用者原先標註為 PRX 是誤植，正確期刊是 Physical Review Letters 117, 080501**）證明，在兩個合理的平均情況困難假設下（估計隨機 Ising 模型的複溫度配分函數、或估計隨機低次多項式的零點數），IQP 在加法誤差下仍難以古典模擬；其中一個假設（IQP 版本的 permanent anticoncentration conjecture）已被證明成立。
- **稀疏（sparse）電路與抗噪（noise robustness）**：同一組作者的後續論文（1610.01808, Quantum 1:8, 2017）展示存在可在 2D 方格上、深度 $O(\sqrt{n}\log n)$ 實現的稀疏 IQP 電路家族仍然難以古典模擬；但同時指出一個關鍵弱點——**只要每個 qubit 尾端加上任意小的常數雜訊，且輸出分布充分反集中（anticoncentrated），就存在多項式時間古典演算法可模擬抽樣**。這是後來「雜訊會摧毀 IQP 優勢」整條研究線的起點，作者提出可用古典糾錯碼緩解，但代價是電路複雜度上升。

### 2.2 何時可古典模擬：expectation value vs. sampling 的關鍵區分

這是對 Leo 專案最重要的一組結果：**估計 expectation value（尤其是 Pauli-$Z$ 型觀測量）與「抽樣整個輸出分布」是兩件複雜度完全不同的任務。**

- IQPopt 系列論文（2501.04776, 2503.02934）明確指出：IQP 電路的 **Pauli-$Z$ observable 期望值可以被古典演算法有效估計**（因為對易 diagonal 結構讓期望值化簡為對隨機變數取平均，可用抽樣/解析方式在古典端算出），但**抽樣整個輸出分布仍被信為古典困難**。這使得訓練（只需要 loss 的梯度，梯度是期望值的函數）可以完全在古典硬體、甚至 GPU 上完成，只有「部署後去抽樣」這一步才需要量子硬體，即所謂 **"train on classical, deploy on quantum"** 範式。
- 雜訊會讓「抽樣也變得可古典模擬」：Rajakumar–Watson–Liu（2403.14607, 2024）證明，只要 dephasing 或 depolarizing 雜訊使電路深度超過某個 $O(1)$ 臨界深度，輸出分布本身也能被古典多項式時間抽樣——且**不需要**假設電路架構或 anti-concentration 性質，機制是雜訊在 IQP 的 diagonal gate 區段中累積，造成大規模糾纏崩潰。
- 針對特定電路家族的具體古典模擬演算法：
  - Codsi–van de Wetering（2212.08609, PRA 2025）用 random graph 方法，對稀疏電路（每 qubit 只與 $O(\log n)$ 個其他 qubit 交互）達到 subexponential scaling，對稠密電路用 stabilizer 分解，實測可在筆電上數分鐘內模擬 50-qubit 電路。
  - Maslov–Bravyi–Tripier–Maksymov–Latone（2402.03211, 2024）針對 Harvard/QuEra 的 48-qubit IQP 硬體宣稱，提出比原論文快約 1000 倍的振幅估計演算法（0.0026 秒／振幅），並推估 96、甚至 192-qubit 規模在特化硬體（如 TPU）上可行——直接反駁了該實驗的「古典難以模擬」宣稱。
  - Takahashi–Miyamoto–Kunihiro（2603.05920, 2026）給出「量子電路 + 稀疏古典後處理」何時可整體古典模擬的充分必要條件，證明 IQP 電路、Clifford Magic 電路、Simon 演算法的量子部分，只要接上稀疏（低 Fourier degree）後處理函數，即使各自單獨難模擬，組合起來仍可古典模擬。

### 2.3 驗證與 Peaked circuit：從樂觀宣稱到被反駁

「怎麼在不用指數古典資源的前提下，驗證一個量子裝置真的做了古典難做的事」是 IQP-based 量子優勢實驗的核心痛點，近年出現了一條「驗證方案 → 被攻破 → 修補 → 新宣稱 → 再被反駁」的完整脈絡：

1. **量子貨幣式驗證方案被攻破再修補**：早期以 quadratic-residue code（QRC）為基礎的 IQP 可驗證方案曾被 Kahanamoku-Meyer 的攻擊破解；Bremner–Cheng–Ji（2308.07152, PRX Quantum 2025）提出 stabilizer scheme，改用 Hidden Structured Code（HSC）問題作為安全性基礎，論證只要參數選擇得當即可修復漏洞。
2. **Peaked circuit 作為新的可驗證優勢路徑**：Aaronson–Zhang（2404.14493, 2024）提出用「peaked circuits」——刻意讓輸出分布集中在單一（或少數）bitstring 上的電路——來繞開「驗證需要指數古典資源」的困境：因為驗證只需檢查是否真的量到那個尖峰，不需要重建整個分布。
3. **理論下限確立**：Zhang（2510.00132, 2025）證明隨機 peaked circuit 幾乎必然具有 $\tilde{\Omega}(nk)$ 的電路複雜度（不可壓縮），且估計電路的「尖峰程度（peakedness）」在平均情況下是 #P-hard，解答了 Aaronson 2022 年提出的開放問題。
4. **硬體宣稱**：Gharibyan et al.（2510.25838, 2025）在 Quantinuum System Model H2（56-qubit，全連接拓撲，2000 個雙 qubit gate 的最大實例）上實現 heuristic peaked circuit，宣稱硬體在 2 小時內完成、但用張量網路加 belief propagation 及 Pauli path 模擬器外推，等效古典模擬需要 exascale 系統跑「數年」，構成潛在指數級分離；並證明「判斷一個電路是否為 peaked（在輸入輸出 bitstring 皆未知時）」是 QCMA-complete。
5. **被反駁**：Kremer–Dupuis（2604.21908, 2026）針對性地設計張量網路收縮法，利用該電路的鏡像結構將前後兩半迭代收縮成一個 Matrix Product Operator（MPO），在單張 GPU 上約 1 小時內完整收縮出最大實例的 peak，時間與硬體的約 2 小時打平甚至更快，直接推翻了 Gharibyan et al. 的指數分離宣稱。

**教訓（對評估任何新 IQP 硬體優勢宣稱都適用）**：這條脈絡顯示 IQP／peaked-circuit 量子優勢宣稱的壽命普遍很短（本例從宣稱到被駁不到半年），評估任何新論文的「量子優勢」用語時要保持懷疑，優先檢查是否有同期的古典模擬反駁論文。

---

## 3. IQP 在 QML（量子機器學習）

### 3.1 Feature map / kernel embedding

IQP 電路作為 feature map 的用法可追溯到 Havlíček et al.（1804.11326, Nature 2019）：將經典特徵 $x$ 編碼進 diagonal 角度 $\theta_S(x)$，核函數定義為 $K(x,y) = |\langle 0^n | H^{\otimes n} U_{\Phi(y)}^\dagger U_{\Phi(x)} H^{\otimes n} | 0^n \rangle|^2$，對應 IQP 電路的輸出機率。此後 PennyLane 的 `qml.IQPEmbedding` 即直接實作此架構（Hadamard 層 + 單體 $R_Z$/雙體 $ZZ$ diagonal 層，可設定 `n_repeats` 增加重複次數以提升表達力，也可指定 entangler pattern）。**限制**：`IQPEmbedding` 對輸入特徵不可微分（features 經過非平凡古典前處理），梯度只能對電路參數算，不能對輸入特徵算。

這條「IQP 當 embedding」路線與「power of data」的結果（Huang et al., 2011.01938, Nat Commun 2021）放在一起看格外重要：即使產生 kernel 的量子電路本身古典難模擬，**只要有訓練資料，古典模型在多數實際場景下仍可能追上量子 kernel 的表現**——因為分類任務不是憑空估計某個複雜度理論量，而是「從資料學習」，資料本身提供了古典模型可用的額外資訊捷徑。這對「IQP feature map 用於影像分類是否有優勢」是一記警鐘。

### 3.2 Generative model（IQP Born machine / IQPopt 系列）

這是 2023–2026 IQP-QML 文獻中最活躍的子領域，核心賣點是「訓練可在古典硬體完成、抽樣（部署）才需要量子硬體」：

- **工具鏈起點**：IQPopt（2501.04776）用 JAX 實作大規模（千 qubit、百萬 gate 級）IQP 電路的古典最佳化，內建以 Maximum Mean Discrepancy（MMD）為 loss 的生成模型訓練模組，支援 GPU。
- **規模化驗證**：Recio-Armengol–Ahmed–Bowles（2503.02934）把此範式推到千 qubit 規模，提出「data-dependent parameter initialisation」以避免 barren plateau，並在多個資料集上證明生成品質可與古典 energy-based model 競爭。
- **表達力上限與修補**：基礎 IQP 生成模型**不是 universal**——無法表示任意分布。Kurkin et al.（2510.08476, 2025）證明對 $n$-qubit IQP generator 加上 $n+1$ 個 hidden qubit 即可達成 universality，並提出 kernel-adaptive training（讓 MMD 的 kernel 本身被對抗式訓練）以緩解固定 Gaussian kernel 造成的梯度消失，資料維度愈高效果差距愈明顯。
- **可訓練性（trainability）理論群**：這是 2026 年出現的密集小群體，彼此高度相關：
  - Shen–Pielawa–Dunjko–Wang（2602.11042）推導 loss 梯度變異數的解析式，證明 barren plateau 與「生成元集合＋所選 kernel 的頻譜」有關，稀疏 IQP 變體可在保持初始化可訓練的同時維持古典抽樣困難。
  - Lerch–Bowles–Puig–Armengol–Holmes–Thanasilp（2603.14576）針對不同初始化策略證明存在 barren plateau，並提出「data-dependent 初始化」——讓初始分布貼近目標分布——以加速收斂；此框架可推廣到一般 QML 最佳化問題。
  - De Luca（2606.10179）用 Stein's lemma 與 Lipschitz concentration bound，分析 **Gaussian（而非均勻）初始化**下的梯度變異數下界，同時給出「如何刻意觸發或避免」exponential concentration 的條件。
  - Nguyen（2606.24264，**已撤回**）原宣稱發現 IQP 電路連通度（connectivity）與可訓練性的 trade-off，作者以「梯度變異數（Fig. 2）數據可能有誤」為由撤回，2026-06-23 投稿、隔天修訂、隨即撤回——**引用時務必標註撤回，不可採用其數據結論**。
- **泛化能力**：Coyle 等人的較早工作已提出 QCBM 的「generalization」評估框架（強調生成新穎、高品質樣本的能力而非只複現訓練分布），是評估任何 IQP 生成模型時應對齊的評估口徑，本次調查未將其列入主表（早於 2023 年主體範圍，且非嚴格 IQP-only），僅在此提及作為評估方法論參照。
- **架構推廣**：
  - Spectral Born machines（Huang–Maxwell–Belis–Peters–Pye–Jahangiri–Bowles, 2607.06675）用 group Fourier analysis 把 IQP Born machine 推廣到一般「spectral」架構，用於整數結構資料，展示 190-qubit、百萬參數規模訓練 RNA 序列分布的可行性，且觀察到 overparameterization 不易 overfit。
  - Qudit 版本（Banks–Crippa–Traube–Unger–Ertler–Lechner, 2606.28236）把 parameterized IQP 電路推廣到 qudit，直接編碼整數值特徵，避免 binary 展開破壞原始度量結構，在 CLIC 探測器量能器的真實整數值資料上驗證。
- **物理／應用導向的初始化與部署**：
  - Correlation-Complexity Map（Liu–Placidi–Brunner–Rinaldi, 2603.06440，**Leo 目標復刻論文**）提出 QCLI（Quantum Correlation-Likeness Indicator，比較資料集的 correlation-order 頻譜與 IQP 電路自然產生的頻譜有多接近）與 CCI（Classical Correlation-Complexity Indicator，衡量最優 Chow-Liu tree 近似無法捕捉的 correlation 比例），作為「這個資料集適不適合上 IQP 生成模型」的**訓練前診斷工具**；並用一個緊湊 IQP 電路搭配 latent-parameter 內插的方式，跨時間序列重複使用同一電路，在 turbulence（紊流）資料的低資料量、低參數量場景中對比古典基準表現有競爭力。
  - 同一組作者的後續工作（Liu–Placidi–Ballarin–Rinaldi, 2607.28866）用古典 physics-informed neural network（PINN）求解 Burgers' equation 的表示，作為 IQP 生成模型的 latent 初始化起點，取代隨機初始化，在多個黏滯係數設定下一致優於隨機初始化。

### 3.3 硬體上的 shallow IQP 生成模型

Balló-Gimbernat–Arroyo-Sánchez–García-Molina–Garriga–Vilariño（2511.05267, PRA 2026）是目前規模最大的 IQP 生成模型硬體展示：用 shallow IQP 電路做 graph generation，在真實硬體上跑到 **28–153 qubit**（模擬對照組固定在 28 qubit），發現全域結構特徵在超過 91 qubit 後明顯劣化，但**局部 correlation 即使到 153 qubit 仍能高精度重現**——顯示 shallow IQP 在當前 NISQ 硬體上，局部統計量比全域統計量更 robust 的具體證據。

---

## 4. 論文總表（26 篇，依 arXiv ID／時間排序；2023–2026 為主體，另附 5 篇奠基文獻）

| Year | arXiv ID | Title | First author et al. | 一句話貢獻 | 類別 |
|------|----------|-------|---------------------|-----------|------|
| 2010 | 1005.1407 | Classical simulation of commuting quantum computations implies collapse of the polynomial hierarchy | Bremner, Jozsa, Shepherd | 乘法誤差下 IQP sampling 難 → PH 坍縮；post-IQP=PP | 複雜度-基礎 |
| 2015 (PRL 2016) | 1504.07999 | Average-case complexity versus approximate simulation of commuting quantum computations | Bremner, Montanaro, Shepherd | 加法誤差下平均情況困難假設；證明 IQP 版 anti-concentration | 複雜度-基礎 |
| 2016 (Quantum 2017) | 1610.01808 | Achieving quantum supremacy with sparse and noisy commuting quantum computations | Bremner, Montanaro, Shepherd | 稀疏 IQP 家族仍難模擬；但雜訊+反集中 → 可古典模擬 | 複雜度-基礎/雜訊 |
| 2018 (Nature 2019) | 1804.11326 | Supervised learning with quantum-enhanced feature spaces | Havlíček et al. | IQP 型 kernel/feature map 分類器，硬體示範 | QML-Embedding |
| 2020 (Nat Commun 2021) | 2011.01938 | Power of data in quantum machine learning | Huang et al. | 有訓練資料時，古典模型多數情況可追上「難」量子 kernel | QML-評估方法論 |
| 2022 (PRA 2025) | 2212.08609 | Classically Simulating Quantum Supremacy IQP Circuits through a Random Graph Approach | Codsi, van de Wetering | Random graph 法，筆電數分鐘模擬 50-qubit IQP | 複雜度-古典模擬 |
| 2023 (PRXQ 2025) | 2308.07152 | Instantaneous Quantum Polynomial-Time Sampling and Verifiable Quantum Advantage: Stabilizer Scheme and Classical Security | Bremner, Cheng, Ji | 修補被攻破的 QRC 驗證方案，提出 stabilizer scheme + HSC 難題 | 驗證 |
| 2024 | 2402.03211 | Fast classical simulation of Harvard/QuEra IQP circuits | Maslov, Bravyi, Tripier, Maksymov, Latone | 反駁 QuEra 48-qubit 硬體宣稱，快 1000 倍 | 複雜度-古典模擬/硬體反駁 |
| 2024 | 2403.14607 | Polynomial-Time Classical Simulation of Noisy IQP Circuits with Constant Depth | Rajakumar, Watson, Liu | 雜訊超過 O(1) 臨界深度 → 抽樣本身也可古典模擬 | 複雜度-古典模擬/雜訊 |
| 2024 | 2404.14493 | On verifiable quantum advantage with peaked circuit sampling | Aaronson, Zhang | 提出 peaked circuit 作為可驗證優勢新路徑 | 驗證/Peaked |
| 2025 | 2501.04776 | IQPopt: Fast optimization of instantaneous quantum polynomial circuits in JAX | Recio-Armengol, Bowles | GPU/JAX 千 qubit 級 IQP 古典訓練工具（開源） | QML-工具 |
| 2025 | 2503.02934 | Train on classical, deploy on quantum: scaling generative quantum machine learning to a thousand qubits | Recio-Armengol, Ahmed, Bowles | Data-dependent 初始化避免 barren plateau，千 qubit 規模示範 | QML-Generative |
| 2025 | 2510.00132 | Complexity and hardness of random peaked circuits | Zhang | 隨機 peaked circuit 複雜度下界 + peakedness 估計 #P-hard | 驗證/Peaked |
| 2025 | 2510.08476 | Universality and kernel-adaptive training for classically trained, quantum-deployed generative models | Kurkin, Shen, Pielawa, Wang, Dunjko | n+1 隱藏 qubit 使 IQP 生成模型 universal；kernel 對抗式訓練 | QML-Generative-理論 |
| 2025 | 2510.25838 | Heuristic Quantum Advantage with Peaked Circuits | Gharibyan et al. | Quantinuum H2 (56q) peaked circuit 宣稱指數優勢 | 硬體/驗證（**已被下一篇反駁**） |
| 2025 (PRA 2026) | 2511.05267 | Shallow instantaneous quantum polynomial-time circuits for generative modeling on NISQ hardware | Balló-Gimbernat, Arroyo-Sánchez, García-Molina, Garriga, Vilariño | 28–153 qubit 硬體示範 graph generation；局部 correlation 高度 robust | 硬體/QML-Generative |
| 2026 | 2602.11042 | Characterizing Trainability of Instantaneous Quantum Polynomial Circuit Born Machines | Shen, Pielawa, Dunjko, Wang | 梯度變異數解析界；稀疏 IQP 可訓練且維持難模擬 | QML-Generative-Trainability |
| 2026 | 2603.05920 | Classical simulability of quantum circuits followed by sparse classical post-processing | Takahashi, Miyamoto, Kunihiro | IQP+稀疏後處理的古典可模擬充要條件 | 複雜度-古典模擬 |
| 2026 | 2603.06440 | Toward Generative Quantum Utility via Correlation-Complexity Map | Liu, Placidi, Brunner, Rinaldi | QCLI/CCI 診斷工具；IQP 生成模型應用於 turbulence（**Leo 目標復刻論文**） | QML-Generative-應用 |
| 2026 | 2603.14576 | IQP Born Machines under Data-dependent and Agnostic Initialization Strategies | Lerch, Bowles, Puig, Armengol, Holmes, Thanasilp | 證明 barren plateau，提出 data-dependent 初始化 | QML-Generative-Trainability |
| 2026 | 2604.21908 | Efficient Classical Simulation of Heuristic Peaked Quantum Circuits | Kremer, Dupuis | 張量網路收縮，1 GPU ~1 小時反駁 2510.25838 的指數優勢宣稱 | 驗證/Peaked（反駁） |
| 2026 | 2606.10179 | Trainability of IQP Quantum Circuit Born Machines Under Gaussian Initialization | De Luca | Stein's lemma 分析 Gaussian 初始化下梯度變異數界 | QML-Generative-Trainability |
| 2026 | ~~2606.24264~~ | Discovery of connectivity-trainability trade-off of IQP Circuits for Hamiltonian Optimization | Nguyen | **已撤回**（作者稱梯度變異數數據可能有誤） | 撤回 |
| 2026 | 2606.28236 | Qudit extension of parameterized IQP circuits: A generative quantum machine learning approach to integer data | Banks, Crippa, Traube, Unger, Ertler, Lechner | Qudit 化 IQP，直接編碼整數特徵，CLIC 量能器真實資料驗證 | QML-Generative-應用 |
| 2026 | 2607.06675 | Spectral Born machines: classically trainable quantum generative models for discrete data | Huang, Maxwell, Belis, Peters, Pye, Jahangiri, Bowles | 用 group Fourier analysis 推廣 IQP Born machine，190-qubit RNA 示範 | QML-Generative-理論/應用 |
| 2026 | 2607.28866 | Generative IQP Circuit Learning with Physics-Informed Latent Initialization | Liu, Placidi, Ballarin, Rinaldi | PINN 求解 Burgers' equation 作為 latent 初始化，優於隨機初始化 | QML-Generative-應用 |

**未列入主表但相關（範圍外或非嚴格 IQP-only），供延伸閱讀：**
- 2509.09033「Generative quantum advantage for classical and quantum problems」（68-qubit 超導處理器，一般難模擬生成模型家族，經 WebFetch 摘要確認**非 IQP-only 架構**，故未列入表格）。
- Coyle et al. 的 QCBM generalization 評估框架（2207.13645，早於 2023 年主體範圍）。

---

## 5. 硬體實驗總覽

| 硬體平台 | qubit 數 | 論文 | 結果 |
|---------|---------|------|------|
| 超導處理器（IBM，Havlíček et al.） | 少量（2-qubit 級demo） | 1804.11326 | IQP kernel 分類器最早硬體示範 |
| QuEra/Harvard 中性原子陣列 | 48（宣稱難模擬） | （原始宣稱論文非本次核對對象；反駁見 2402.03211） | 反駁論文證明可快 1000 倍古典模擬 |
| Quantinuum System Model H2 | 56（全連接） | 2510.25838（宣稱）／2604.21908（反駁） | 宣稱 peaked circuit 指數優勢，數月內被張量網路法反駁 |
| 未指名硬體平台（Balló-Gimbernat et al.） | 28–153 | 2511.05267 | 目前規模最大的 IQP 生成模型硬體示範；局部 correlation 在 153 qubit 仍精準，全域結構 >91 qubit 後劣化 |

**觀察**：目前為止，每一個「IQP 硬體優勢」宣稱（QuEra 48q、Quantinuum H2 56q peaked circuit）都在數月內被同行的古典模擬論文正面反駁。截至本次調查（2026-09），**沒有一個 IQP 硬體實驗的量子優勢宣稱撐過同行審查後的古典模擬複驗**。Balló-Gimbernat 等人的 153-qubit 生成模型實驗則不主打「優勢」宣稱，而是誠實報告「局部統計量準、全域統計量隨規模劣化」，是目前最貼近實用評估的硬體結果。

---

## 6. 開源實作清單

| 專案 | 連結 | 說明 |
|------|------|------|
| IQPopt | https://github.com/XanaduAI/iqpopt | Xanadu 官方，JAX 實作，支援千 qubit 級 IQP 電路的期望值估計、MMD loss 生成模型訓練、GPU 加速。對應論文 2501.04776。有 PennyLane demo：`tutorial_iqp_circuit_optimization_jax` |
| PennyLane `qml.IQPEmbedding` | https://docs.pennylane.ai/en/stable/code/api/pennylane.IQPEmbedding.html | 標準 IQP feature map 模板（Havlíček et al. 架構），可設定 `n_repeats`、自訂 entangler pattern；**features 對輸入不可微分**，僅電路參數可微分 |
| PennyLane `qml.IQPEmbedding` 原始碼 | https://github.com/PennyLaneAI/qml/blob/master/demonstrations/tutorial_iqp_circuit_optimization_jax.py | IQPopt 官方 demo script |

**未找到公開程式碼的論文**：Quantinuum 的 Correlation-Complexity Map（2603.06440，Leo 目標復刻論文）與其後續 latent 初始化論文（2607.28866）在 WebSearch 中**沒有找到對應 GitHub repo**——復刻時需要完全依論文描述的 QCLI/CCI 指標定義與 latent-parameter 內插方案自行實作，這點請 Leo 特別注意（無官方程式碼可對照除錯）。

---

## 7. 對 Leo 專案的評估

專案背景：Leo 正在復刻 2603.06440（IQP 生成模型 + Correlation-Complexity Map，應用於 turbulence 資料），gsoc2026-deeplense-qml repo 過去也做過 QML 用於重力透鏡影像分類（DeepLense）的多個實驗（QVF、QOVT、QPE 等）。以下評估同時涵蓋「IQP 用於分類」與「IQP 用於生成（本次復刻對象）」兩條應用線：

1. **「訓練可在 GPU 古典完成」是 IQP 生成模型最實際的優勢，也是本次復刻可行的原因。** IQPopt 系列（2501.04776, 2503.02934）與 Correlation-Complexity Map 論文本身都依賴「Pauli-$Z$ expectation value 可古典估計」這個性質——這代表 Leo 復刻整個訓練 pipeline（含 MMD loss 梯度）不需要排隊等真實量子硬體或跑指數級 statevector 模擬，可以直接在 NANO4 的 GPU partition 上用 JAX/PyTorch 跑完，**只有最終「抽樣」這步在概念上需要量子硬體或難以規模化的古典抽樣模擬**——這與 repo 過去的 QVF/QOVT 系列用 statevector 模擬器硬跑的做法有本質差異，值得在復刻的 sbatch/README 中明確寫清楚「這裡在做的是古典可行的訓練，不是量子優勢展示」。

2. **「Expectation value 可古典估計」同時是復刻的可行性保證，也是任何「量子優勢」敘事的天花板——寫論文/報告時要避免過度宣稱。** 正因為訓練端本質上是古典演算法（只是恰好被設計成「未來可搬到量子硬體抽樣」的形式），Leo 若要在復刻報告中討論「量子優勢」，必須清楚區分「這個 IQP 模型的抽樣分布被信為古典難模擬」（複雜度理論的猜想）與「我們的訓練/推論流程有沒有實際跑在量子硬體上」（工程事實）。參照 §2.3 的教訓——過去兩個 IQP 硬體優勢宣稱（QuEra 48q、Quantinuum H2 56q）都在數月內被反駁——任何「IQP 對 turbulence/影像資料有優勢」的結論都應該保守表述為「同參數量下表現具競爭力」而非「證明量子優勢」，並且要對照 judgment-rubric 的「單一指標差 <3% 當噪音」原則來看待與古典基準的比較。

3. **雜訊敏感性是復刻若涉及真實硬體部署時的關鍵風險**：1610.01808 與 2403.14607 都指出，一旦雜訊使電路深度超過某臨界值（或反集中程度足夠），IQP 電路連「抽樣」本身都會變成可古典模擬——換言之，在真實 NISQ 硬體上部署 IQP 生成模型，若電路夠深或雜訊夠大，聲稱的「量子難模擬」性質可能在部署當下就已經名存實亡。若 Leo 的復刻計畫包含在真實硬體（或 noisy simulator）上測試，應先估算目標電路深度與雜訊率是否落在「仍然難模擬」的理論安全區，否則跑硬體實驗的意義會被削弱。

4. **QCLI/CCI 診斷工具沒有公開程式碼，是復刻的最大工程風險，而非理論風險。** 目標論文 2603.06440 的核心貢獻——用 correlation-order 頻譜比對（QCLI）與 Chow-Liu tree 殘差（CCI）預先判斷資料集適不適合上 IQP 生成模型——完全沒有找到開源實作。復刻時這兩個指標的具體計算細節（例如 Walsh-Hadamard 頻譜的 binning 方式、Jensen-Shannon divergence 的具體 reference 分布參數）需要從論文文字重建，建議先寫一個小型單元測試（例如對已知的簡單合成分布手算 QCLI/CCI 驗證實作對不對），再套用到真正的 turbulence 資料，避免指標算錯導致後續生成模型比較全部失焦。

5. **對 DeepLense／影像分類這條線的啟示（如果 Leo 之後想把 IQP embedding 用回分類任務）**：Havlíček 型 IQP feature map（1804.11326, PennyLane `IQPEmbedding`）與 Huang et al. 的「power of data」結果（2011.01938）放在一起看，意味著單純把 IQP 當作影像分類的 feature map，若沒有搭配資料集本身在複雜度理論意義上「貼近 IQP 電路自然分布」（也就是 §3.2 提到的 QCLI 概念，但用在分類而非生成場景），很可能被一個參數量相當的古典模型追平——這與 repo memory 中 `feedback_scope_vit_family_only.md` 記錄的「量子方法要跟同量級古典方法比較」原則完全一致，建議在任何未來的 IQP-分類實驗中，比照 QOVT ablation 的作法設計 param-matched 古典基準，而不是只跟 plain baseline 比。

---

## 附註：與使用者原始假設清單的核對結果

- Bremner–Montanaro–Shepherd 2016：**期刊應為 PRL（Physical Review Letters 117, 080501），非 PRX**。arXiv ID 為 1504.07999（已核對）。
- 2606.24264：核對後確認**已撤回**，題目為《Discovery of connectivity-trainability trade-off of IQP Circuits for Hamiltonian Optimization》，作者 Quoc Chuong Nguyen，撤回原因為「梯度變異數數據可能有誤」。
- 其餘使用者提供的 arXiv ID（1005.1407, 2501.04776, 2503.02934, 2403.14607, 2603.14576, 2604.21908, 1804.11326, 2603.06440）**皆已透過 WebFetch 核對，title/authors/year 與使用者原始描述一致**。
