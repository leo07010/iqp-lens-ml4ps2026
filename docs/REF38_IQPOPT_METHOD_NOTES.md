# Ref[38] 方法筆記 — Train on classical, deploy on quantum (arXiv:2503.02934)

## 1. Metadata

- **標題**：Train on classical, deploy on quantum: scaling generative quantum machine learning to a thousand qubits
- **作者**：Erik Recio-Armengol (Xanadu / ICFO / Eurecat), Shahnawaz Ahmed, Jonathan Bowles（皆 Xanadu）
- **arXiv ID**：2503.02934（抓取版本 v2，2026-09-05 確認 abs 頁面有 v1/v2 兩版）
- **本檔依據**：`<WORK>/data/papers/2503.02934/paper.html`（718396 bytes，抓取成功，非錯誤頁）＋自轉純文字 `paper.txt`（1334 行，用於 grep 行號引用）
- **GitHub repo**：`XanaduAI/iqpopt`（抓取成功，見 §10 附錄）；正確路徑是 `src/iqpopt/`，不是 README 猜測的 `iqpopt/`
- **本文件用途**：這是 2603.06440（Liu et al., turbulence IQP 生成模型論文，我們正在復刻的目標）Ref [38]，其訓練方法（MMD loss、data-dependent init、bandwidth 排程）完全建立於此篇之上。2603.06440 §II 也直接引用了本篇 Table 1 的三個實驗數字作為 external consistency check。
- **抓取日期**：2026-09-05

---

## 2. TL;DR

這篇論文提出「Train on classical, deploy on quantum」框架：用 **parameterised IQP circuits**（生成器只有 X-type Pauli 旋轉閘）搭配 **MMD（maximum mean discrepancy）loss**，證明訓練階段可以完全在**古典硬體**上高效完成（多項式時間），理由是：

1. den Nest (2010) 的結果：IQP 電路的 Pauli-Z expectation value ⟨Z_a⟩ 可以用古典演算法高效估計（多項式時間/空間），因為 Hadamard 共軛後電路變成對角么正，⟨Z_a⟩ 化簡成對均勻分布 z 的一個 cosine 期望值（見 §3）。
2. Rudolph et al. (2024) 的觀察：MMD² 可以寫成這些 ⟨Z_a⟩ expectation value 的一個機率混合（見 §4）。

把兩者結合，就得到一個完全古典可算、對 θ 可微分（配合 JAX autodiff）、且對 MMD² 及其梯度都是**無偏估計**的訓練演算法，實作在 Xanadu 開源套件 **IQPopt** 裡（訓練時完全用不到量子硬體），訓練完的參數之後才部署到量子硬體上取樣（因為取樣 IQP 電路本身被認為是古典難以模擬的）。論文用這套方法把 IQP 生成模型的規模一路推到 484~1000 qubits（6 個資料集：2D Ising, Binary Blobs, D-Wave, MNIST, Scale free, Genomic），並用一個「stochastic bitflip circuit」（IQP 的完全去相干古典版本）當作對照，探討 coherence 的角色。

**與 2603.06440 的關係**：2603.06440 直接套用本篇的「train on classical, deploy on quantum」框架（§III「To train IQP-based generative models without relying on quantum-hardware optimization, we adopt the train-on-classical, deploy-on-quantum framework [38, 14, 6]」），把資料從固定長度 bitstring 換成 turbulence 的量化座標/強度 tuple（18-qubit 編碼），MMD loss、data-dependent init、multi-bandwidth 排程的具體公式都應該照搬本篇。

---

## 3. eq. 15 與 ⟨Z_a⟩ 估計（Prop. 1, den Nest 2010）

出處：`paper.txt:156-180`（Sec. 4.1）

- 定義：⟨Z_a⟩_{q_θ} = ⟨0|U†(θ) Z_a U(θ)|0⟩，其中 Z_a = ∏ᵢ (Zᵢ)^{aᵢ}，a∈{0,1}ⁿ 指定哪些 qubit 上有 Z（eq. 7-8）。
- **推導核心**（`paper.txt:165-178`）：插入 H⊗ⁿ 恆等式把 U 變成對角么正 D(θ)=∏ⱼ exp(iθⱼ Z_{gⱼ})（eq. 9-10）。D(θ) 的本徵值 λ_z = exp(i Σⱼ θⱼ (-1)^{gⱼ·z})（eq. 11）。代入後：
  ⟨Z_a⟩_{q_θ} = (1/2ⁿ) Σ_z Re[λ*_{z⊕a} λ_z]（eq. 12）
  再展開 λ 得到最終、可以直接寫成對均勻分布 z~U 的期望值形式（**eq. 13**）：
  ⟨Z_a⟩_{q_θ} = E_{z∼U}[ cos( Σⱼ θⱼ (-1)^{gⱼ·z} (1-(-1)^{gⱼ·a}) ) ]
- **樣本估計式（eq. 14）**：⟨Ẑ_a⟩_{q_θ} = (1/|Z|) Σᵢ cos(Σⱼ θⱼ (-1)^{gⱼ·zᵢ}(1-(-1)^{gⱼ·a}))，其中 z_i 從均勻分布抽樣。
- **eq. 15** 正是這個估計式無偏性的陳述：E_Z[⟨Ẑ_a⟩_{q_θ}] = ⟨Z_a⟩_{q_θ}（`paper.txt:180`），因為 eq.14 就是對 Z 的元素取樣本均值。由於 ⟨Z_a⟩ 是 [-1,1] 有界隨機變數的期望值，樣本均值的變異數 ≤ 1/|Z|；取 |Z| ≥ 1/ε² 就得到標準差 < ε 的多項式時間估計器（Prop. 1）。
- **對應 repo 實作**：`iqp_optimizer.py`（未逐行核對，但 `gen_qml/iqp_methods.py` 的 `f(a,z,θ)=cos(...)` 與 eq.14/22 的 f 函數完全一致，見下方 §7）。

---

## 4. MMD 定義：expectation value 表示、Gaussian kernel、Appendix A.2.1/A.2.2/A.3

### 4.1 MMD 與 Gaussian kernel（Def. 3.2/3.3，`paper.txt:126-145`）

- MMD(p,q_θ) = ‖E_{x∼p}[φ(x)] − E_{y∼q_θ}[φ(y)]‖（eq. 1），MMD²(p,q_θ) = E_{x,y∼p}[k(x,y)] − 2E_{x∼p,y∼q_θ}[k(x,y)] + E_{x,y∼q_θ}[k(x,y)]（**eq. 2**，Def 3.2）。
- **Gaussian kernel（eq. 3）**：k(x,y) = exp(−‖x−y‖² / 2σ²)。σ 稱為 bandwidth。
- **重要注意（footnote 5，`paper.txt:141`）**：本篇明確聲明 σ 是「(unnormalized) Gaussian distribution 的標準差」，但其他文獻（例如 Rudolph et al. 2024）的 σ 代表的是**變異數**（variance）而非標準差 —— 兩篇 σ 的定義不同，換算/比對數字時要小心。
- **Median heuristic（Def. 3.3，eq. 4）**：σ̂ = med({‖xᵢ−xⱼ‖}ᵢ,ⱼ)，即所有點對距離的中位數。
- 標準（有 sample access 的兩邊）MMD² 無偏估計式（eq. 5-6）：即 Gretton et al. (2012) 的 U-statistic 形式。

### 4.2 MMD² 化為 expectation value 的機率混合（Prop. 2，Rudolph et al. 2024，`paper.txt:189-206`）

**eq. 16**：MMD²(p, q_θ) = E_{a∼P_σ(a)}[ (⟨Z_a⟩_p − ⟨Z_a⟩_{q_θ})² ]

**eq. 17（product-Bernoulli 分布 P_σ(a)）**：
P_σ(a) = (1−p_σ)^{n−|a|} p_σ^{|a|}；p_σ = (1 − exp(−1/(2σ²))) / 2

其中 |a| 是 a 的 Hamming weight。⟨Z_a⟩_p = E_{x∼p}[(−1)^{x·a}]（eq. 18），可用資料樣本均值 ⟨Ẑ_a⟩_p =(1/|X|)Σᵢ(−1)^{xᵢ·a}（eq. 19）估計。

**核對結果**：paper 中確實**沒有出現「Walsh coefficient」這個詞**（我在 paper.txt 全文 grep "Walsh" 無結果）。P_σ(a) 就是直接被稱為「a product of Bernoulli distributions」，論文本身沒有借用 Walsh 分析的術語；你上次印象中的「Walsh 係數」說法應該是外部/口語化的稱呼（P_σ(a) 在傅立葉/Walsh-Hadamard 分析裡確實對應 Walsh 係數的抽樣分布，但原文並未使用此詞彙）。

### 4.3 Appendix A.1/A.2.1/A.2.2/A.3/A.4（無偏 MMD² 估計器完整推導，`paper.txt:1078-1188`）

- **A.1**（`paper.txt:1082-1101`）：兩邊都有 sample access 時的標準 U-statistic 無偏估計（eq. 51-57），對應正文 eq. 5-6。
- **A.2**（`paper.txt:1102-1112`）：其中一邊是 IQP 電路時，把 k(x,y) 的三項期望值都改寫成 E_{a∼P(a)} E_z∼U[f_{p_k}(a,z)] 的形式（eq. 58-60），f_{p_k} 對量子模型是 cos(...)（eq. 59，與 eq.14 相同），對可抽樣的古典分布是 ⟨Z_a⟩_p（eq. 60）。
- **A.2.1「The last term」**（`paper.txt:1113-1132`）：處理 E_{x,y∼q_θ}[k(x,y)]（兩邊都是量子模型的情況，eq. 61），估計式是 eq. 62（需要 |Z|(|Z|-1) 分母排除 k≠j 自相關項），推導出它的期望值等於 eq. 58 給出的理論值（eq. 63-66）。
- **A.2.2「The second term」**（`paper.txt:1133-1152`）：處理 E_{x∼p,y∼q_θ}[k(x,y)]（一邊資料一邊量子模型的交叉項，eq. 67-68），估計式 eq. 69，證明無偏（eq. 70-72）。
- **A.3「The first term」**（`paper.txt:1153-1172`）：處理 E_{x,y∼p}[k(x,y)]（兩邊都是資料，eq. 73 是標準 U-statistic 形式；另外也給出用 a∼P_σ 抽樣的替代估計式 eq. 74-78，這是因為要跟另外兩項共用同一批 A 樣本以降低方差）。
- **A.4「The full expression」**（`paper.txt:1173-1188`）：把三項合併成最終無偏估計器 **eq. 21**（正文 Prop. 3，`paper.txt:220-230`）：

  MMD²_u(X,A,Z,θ) = [1/(|A||Z|(|Z|-1))] Σ_{i,j,k≠j} f(aᵢ,zⱼ,θ) f(aᵢ,zₖ,θ)
  − [2/(|A||Z||X|)] Σ_{i,j,k} f(aᵢ,zⱼ,θ)(−1)^{xₖ·aᵢ}
  + [1/(|A||X|(|X|-1))] Σ_{i,j,k≠j} (−1)^{xⱼ·aᵢ}(−1)^{xₖ·aᵢ}

  其中 f(a,z,θ) = cos(Σⱼ θⱼ(-1)^{gⱼ·z}(1-(-1)^{gⱼ·a}))（eq. 22）。訓練時用 automatic differentiation 對 θ 微分這個表達式即可得到無偏梯度估計（eq. 23-24，Sec 4.4）。

- **對照 repo 實作**：`iqpopt/gen_qml/iqp_methods.py::loss_estimate_iqp` 的 `tr_iqp_samples`（對應 f(a,z,θ) 的 batch 抽樣）、`tr_train`（對應 (−1)^{x·a} 項）、以及非 bitflip 分支的 `res = (tr_iqp*tr_iqp-correction)*n_samples/(n_samples-1) - 2*tr_iqp*tr_train + (tr_train*tr_train*m-1)/(m-1)` 正是 eq. 21 的 JAX 向量化實作（`correction` 項是為了讓 tr_iqp² 這個二次項在有限樣本下仍無偏，對應 A.2.1 的 |Z|(|Z|-1) 分母技巧）。`iqp_circuit.bitflip` 分支則省略 correction，因為 bitflip 模型的 f(a,z,θ) 不依賴 z（見 §6）。

---

## 5. 多個 σ（多 bandwidth）的合併方式 —— 核對結果：**簡單算術平均，確認屬實**

出處：`paper.txt:388-391`（Sec. 8.1.1，eq. 45），repo 程式碼 `iqpopt/gen_qml/iqp_methods.py::mmd_loss_iqp` 最後一行。

**eq. 45**：
L = (1/L) Σᵢ₌₁^L MMD²_{σᵢ}(P_{X_train}, q_θ)

即對 L 個不同 bandwidth σ₁,...,σ_L 算出的 MMD² 值，取**簡單算術平均**當作總 loss。原文用詞是「We use the **average** of a number of MMD² values for different values of the bandwidth σ」，沒有加權、沒有取和不正規化。

**Repo 程式碼逐行確認**（`iqp_methods.py`）：
```python
losses.append(loss(...))   # 對每個 sigma 算一次 loss
...
return sum(losses)/len(losses)   # 簡單算術平均
```
與 eq. 45 完全一致，**確認：<user> 的「簡單取平均」印象正確**。

---

## 6. Bandwidth（σ）選法規則 —— 核對結果：**16-qubit {0.6,0.3} 與 Table 1 數字有明確矛盾，需以 Table 1 為準**

### 6.1 通則（正文，`paper.txt:390-391`，Sec. 8.1.1）

- 對 16-qubit 案例（2D Ising、Binary Blobs 兩個實驗），正文明講：「the specific choice **{σ₁,σ₂} = {0.6, 0.3}** was used, which corresponds to sampling observables Z_a ... with an average Pauli weight of **2 and 6** respectively」。
- 對其餘（更大）系統，用三個 bandwidth {σ₁,σ₂,σ₃}：
  - **σ₁**：使 Z_a 的平均 Pauli weight = 2
  - **σ₃**：sqrt(median heuristic)（即 σ̂ from Def 3.3 開根號）
  - **σ₂ = sqrt((σ₁² + σ₃²) / 2)**（即 σ₁,σ₃ 的均方根平均 RMS，不是算術平均！）

### 6.2 ⚠️ 核對出的矛盾：正文 {0.6,0.3} 與 Table 1 實際數字不一致

`paper.txt:398-406`（**Table 1**）列出的 2D Ising / Binary Blobs（n=16）實際數字是：
| | σ₁ | σ₂ |
|---|---|---|
| 2D Ising (n=16) | **1.3** (weight 2) | **0.6** (weight 6) |
| Binary Blobs (n=16) | **1.3** (weight 2) | **0.6** (weight 6) |

這與正文明講的「{σ₁,σ₂}={0.6,0.3}」**不一致**（正文用 0.6/0.3，Table 1 用 1.3/0.6）。我用原文附的 p_σ 公式（eq. 17）反推驗證：對 n=16、目標平均 Pauli weight=2，p_σ=2/16=0.125，解 σ=sqrt(−1/(2·ln(1−2p_σ)))=sqrt(−1/(2·ln0.75))≈**1.318**——與 Table 1 的 1.3 吻合，而非正文的 0.6。同樣目標 weight=6 時 p_σ=6/16=0.375，解得 σ≈**0.601**——與 Table 1 的 0.6 吻合，而非正文的 0.3。

我也用 GitHub repo `iqpopt/gen_qml/utils.py::sigma_heuristic` 裡對「weight=2」的公式做交叉驗證：
```python
sigma_2 = np.sqrt(-1/(2*np.log(1-4/X.shape[-1])))  # mean operator weight of 2
```
代入 n=16 得 sigma_2 = sqrt(-1/(2*ln(1-4/16))) ≈ **1.318**，與 Table 1 的 1.3 一致，佐證 **Table 1 才是與公式自洽、實際被使用的數字**。

**結論（供實作依據）**：這是原文本身的一個不一致（v2 版本，2026-09-05 確認），**應以 Table 1 的 {1.3, 0.6} 為準**，正文敘述處的「{0.6,0.3}」很可能是筆誤或版本間殘留的舊數字。**<user> 上次印象中的 {0.6,0.3} 出自正文，但與可驗證的公式和 Table 1 不符，不建議照抄，請改用 Table 1 的 {1.3, 0.6}。**

### 6.3 更大系統的三 bandwidth 數字（Table 1，`paper.txt:398-428`）

| Dataset | n (qubits) | σ₁ (weight) | σ₂ (weight) | σ₃ (weight) |
|---|---|---|---|---|
| 2D Ising | 16 | 1.3 (2) | 0.6 (6) | n/a |
| Binary Blobs | 16 | 1.3 (2) | 0.6 (6) | n/a |
| D-Wave | 484 | 7.8 (2) | 6.1 (4) | 3.9 (8) |
| MNIST | 784 | 9.9 (2) | 7.4 (4) | 3.4 (17) |
| Scale free | 1000 | 11.2 (2) | 8.3 (4) | 3.8 (17) |
| Genomic | 805 | 10.0 (2) | 7.7 (4) | 4.2 (11) |

我用 σ₂=sqrt((σ₁²+σ₃²)/2) 核對 D-Wave 一列：sqrt((7.8²+3.9²)/2)=sqrt((60.84+15.21)/2)=sqrt(38.03)≈**6.17**，與表列 6.1 吻合（四捨五入誤差內）。**確認：三 bandwidth 情況下 σ₂ 公式（RMS 平均）與 Table 1 自洽。**

**⚠️ 另一個發現（repo vs 論文的落差）**：GitHub repo 目前的 `sigma_heuristic(X, n_sigmas)` 函式用的是 `np.linspace(sigma_2, med, n_sigmas)`，n_sigmas=3 時等同**算術平均** (σ₁+σ₃)/2，而非論文正文與 Table 1 用的**均方根平均** sqrt((σ₁²+σ₃²)/2)。用 D-Wave 數字驗證：算術平均 (7.8+3.9)/2=5.85 ≠ Table 1 的 6.1；均方根平均 ≈6.17 ≈ Table 1 的 6.1。**結論：目前公開的 repo 工具函式 `sigma_heuristic` 並不是論文用來產生 Table 1 具體數字的確切方法**（可能是後來簡化過、或 Table 1 的具體 σ 值是手動算好寫死在論文的實驗腳本裡，而非用這個 repo 裡的通用 heuristic）。**實作復刻時，應直接用論文明講的公式（σ₁: 目標平均 Pauli weight 用 p_σ=weight/n 反解 eq.17；σ₃=sqrt(median heuristic)；σ₂=sqrt((σ₁²+σ₃²)/2)），不要照搬 repo 的 `sigma_heuristic` 函式（它算出來的 σ₂ 會偏低）。**

---

## 7. Data-dependent 初始化 —— 核對結果：與 <user> 印象一致，且 repo 有精確實作細節（paper 文字沒寫的常數）

出處：`paper.txt:430`（Sec. 8.1.2）＋ repo `iqpopt/utils.py::initialize_from_data`

### 7.1 論文文字（單體/一體 one-body、二體 two-body）

- **單體（one-body，單 qubit 閘，生成子為 X_j）**：初始化參數為 **arcsin(sqrt(⟨xⱼ⟩))**，⟨xⱼ⟩ 是訓練資料第 j 維的平均值（0/1 二元資料的均值即 marginal probability）。這保證若其他所有參數設為 0，模型會等於一個和訓練資料有相同 single-qubit marginal 的乘積分布（product distribution）。
- **二體（two-body，兩 qubit 閘，作用在 j,k 上）**：初始化參數**正比於**訓練資料第 j、k 維的**共變異數（covariance）**，其中資料要先轉成 **±1 編碼**（pm1 encoding）而非 0/1。比例常數是留給使用者調的 hyperparameter（論文原文：「the specific constant of proportionality is left as a free hyperparameter」）。
- 若還有更高階（>2-body）的閘，參數用零均值常態分布初始化，標準差是另一個 hyperparameter。

**核對：與 <user> 印象一致**（「單體用 arcsin(sqrt(mean))、二體用 ±1 編碼共變異數」皆確認屬實，用詞正是 one-body/two-body 對應的「single qubit gates」/「two-qubit gates」）。

### 7.2 Repo 精確實作（paper 文字沒寫清楚的比例常數細節）

`iqpopt/utils.py::initialize_from_data`：
```python
cov_mat = np.cov((2*data-1).T)   # pm1 編碼的共變異數矩陣
np.fill_diagonal(cov_mat, 0.)
max_cov = np.max(np.abs(cov_mat))
means = np.mean(data, axis=0)

if scale == -1:   # 預設 heuristic（論文文字中"free hyperparameter"的預設值）
    scale = np.sqrt(data.shape[-1]/np.sum(cov_mat**2)*2) * max_cov / np.pi

# 單體閘: params.append(np.arcsin(np.sqrt(means[gen[0]])))
# 二體閘: params.append(cov_mat[gen[0], gen[1]] * scale * np.pi / max_cov)
# 其他: np.random.normal(0, param_noise)
```
這給出論文文字中未明講的**預設比例常數 heuristic**：`scale = sqrt(d/Σcov²·2) · max_cov/π`，其中 d 是資料維度。這是實作復刻時可以直接借用的預設值（若不想自己重新調參）。

---

## 8. Bitflip baseline 定義

出處：`paper.txt:255-278`（Sec. 5, Def. 5.1）

**機制**：對每個 IQP 的參數化閘 exp(iθⱼ X_{gⱼ})，其量子行為是把 |x⟩ 相干地變成 cos(θⱼ)|x⟩ + i sin(θⱼ)|x⊕gⱼ⟩（eq. 28）——即以疊加態同時保留翻轉前後兩種結果。**Stochastic bitflip circuit** 則是把這個量子閘換成古典隨機版本（eq. 29）：
|x⟩⟨x| → cos²(θⱼ) |x⟩⟨x| + sin²(θⱼ) |x⊕gⱼ⟩⟨x⊕gⱼ|
也就是「以機率 sin²(θⱼ) 翻轉 gⱼ 指定的那些 bit，否則以機率 cos²(θⱼ) 保持不變」的**逐 gate 獨立古典隨機翻轉**過程（Def 5.1：初始化全 0 bitstring → 一連串這種隨機翻轉閘 → 讀出最終 bitstring）。

**與 IQP 模型的關係**：作者證明可以把一個 stochastic bitflip circuit 構造成一個特殊的（帶 ancilla qubit 的）parameterised IQP circuit，再對 ancilla 取 partial trace，因此可以推導出其 ⟨Z_a⟩ 的封閉形式（**eq. 32**）：
⟨Z_a⟩ = ∏_{j | {X_{gⱼ}, Z_a}=0} cos(2θⱼ)
（只是 eq. 27 完整 IQP expectation value 公式中的「第一項／古典項」，量子版本 eq.27 還多了 Ω 集合帶來的額外 coherence 項）。

**用途（下界對照 baseline）**：這讓 bitflip 模型可以用**跟 IQP 模型完全相同**的訓練流程（同一套 gate set、同一個 IQPopt 套件、同樣的 MMD loss 與資料依賴初始化）訓練，唯一差別只在 expectation value 公式（eq. 32 vs eq. 27），因此是一個「乾淨對照組」：**若 bitflip 表現明顯比 IQP 差，代表 coherence（量子疊加）本身對模型表現有實質貢獻**；反之若兩者相近，代表 coherence 在該任務上作用不大。論文結論（`paper.txt:653`）：bitflip 模型在除了低維資料集外普遍表現差很多，被視為「coherence 對表現至關重要」的證據。

---

## 9. 三個實驗設定與數字（BinaryBlobs / D-Wave / Binarized MNIST），及與 2603.06440 Table 1 的比對

原文六個實驗中，2603.06440 的 Table 1 引用了其中三個。逐項核對如下（原文出處 `paper.txt`，2603.06440 出處 `<WORK>/data/papers/2603.06440/paper.txt`）。

### 9.1 BinaryBlobs（原文稱 "Binary blobs dataset"，Experiment 2）

出處：`paper.txt:364,398-406,607-610`

- 16-bit 資料集（16 qubits）
- Gate set：Table 2 的 "Largest Pauli weight of gate generators" 對 exp.1,2 = [2,4,6]（rounded default 2）→ 即 **local IQP gates，最高到 6-body**（`paper.txt:610`：「the model using all gates acting on six qubits or less」）
- **14892 個參數**（`paper.txt:610`）
- Baseline：**EBM（1700 個參數）**、RBM、Bitflip
- **Log-likelihood**：EBM **−5.34** vs IQP **−6.35**（`paper.txt:610`），EBM 表現較好
- Bitflip baseline 表現差（"performed badly, often producing bitstrings far from the ground truth"）
- MMD 對 mode imbalance 敏感（RBM 樣本視覺上更討喜但 MMD² 反而較差，因為 MMD² 也反映 mode imbalance 而非只是視覺品質）

**與 2603.06440 Table 1 比對**（2603.06440 `paper.txt:1006-1020`）：**完全一致，無數字落差**。2603.06440 轉述「Local IQP gates (up to 6-body), 14892 params」「EBM (1700 params), RBM, Bitflip」「EBM achieves better likelihood than IQP (log-likelihood: EBM −5.34 vs. IQP −6.35)」— 逐字比對，數字、baseline 組成、結論方向皆與 Ref[38] 原文吻合。

### 9.2 D-Wave（Experiment 3）

出處：`paper.txt:368,398-428,612-615`

- **484-qubit** 資料集，來自 D-Wave Advantage processor（Pegasus 拓撲），**100 μs** 量子退火（quantum annealing）取樣，資料取自 Scriva et al. (2023) 的 repo
- 訓練/測試資料：10000 / 60000 點
- Gate set：**all two-qubit and single-qubit gates**（全部一體與二體閘）on 484 qubits
- 參數量：**"over 100000 parameters"**（>10⁵，原文沒有給更精確的位數）
- Baseline：**RBM、EBM、Bitflip**（三者皆嘗試訓練）
- 結果（`paper.txt:614`）：沒有模型完美擬合，但 IQP 相對表現最好；RBM 疑似 mode collapse（許多對角元素的變異數接近 0）；EBM 表現差；**「Both the RBM and EBM models scored worse than a random distribution for the test MMD²」**；bitflip 模型訓練失敗，covariance matrix 與隨機分布無法區分
- **原文沒有給出精確的 MMD² 數值**（只在 Fig. 7/8 以圖呈現，正文只有質性描述）

**與 2603.06440 Table 1 比對**（2603.06440 `paper.txt:1025-1040`）：**一致，無數字落差**。2603.06440 轉述「All 1- and 2-qubit gates on 484 qubits, >10⁵ params」「RBM, EBM, Bitflip」「No model fits perfectly ... IQP performs best ... RBM/EBM can score worse than random on test MMD; EBM fails」— 皆與原文吻合。注意兩篇都沒有給出具體 MMD² 數字（只在圖裡），這不是轉述缺失，是原文本身就沒有把精確數字寫進正文。

### 9.3 Binarized MNIST（Experiment 4）

出處：`paper.txt:370,398-428,616-620`

- **784-bit**（784 qubits）二值化 MNIST（閾值化後攤平）
- 訓練/測試：50000 / 10000 點
- Gate set：**all-to-all two-qubit gates**
- 參數量：**307720**（`paper.txt:619`：精確整數，非約略值）
- Baseline：**只有 Bitflip**（「we trained only the IQP and bitflip models on this dataset」— 因為 MNIST 在 ML 文獻已被過度研究，作者選擇不再訓練 RBM/EBM）
- 結果：IQP 學到了大部分結構（covariance matrix 呼應真實分布但相關性較弱）；用「noisy reference」實驗做量化比較：對 ground truth 加不同程度的 per-pixel 雜訊（雜訊參數 p，機制見 `paper.txt:618`），**IQP 的 test MMD² 表現大約等同於 p=0.3 的雜訊資料**；bitflip 模型完全學不到結構，MMD² 很差

**與 2603.06440 Table 1 比對**（2603.06440 `paper.txt:1042-1055`）：**一致，無數字落差**。2603.06440 轉述「All-to-all 2-qubit gates, 307,720 params」「Bitflip only」「Test MMD is comparable to a noisy-data reference (reported around noise level p≈0.3)」— 皆準確對應原文。

### 9.4 小結

**核對後結論：2603.06440 的 Table 1 轉述在數字、baseline 組成、質性結論方向上，對這三個實驗（BinaryBlobs / D-Wave / MNIST）都與 Ref[38] 原文完全一致，沒有發現失真或誤植**。唯一值得注意的是 D-Wave 和 MNIST 兩個實驗，Ref[38] 原文本身就只在圖表（Fig 7/8）裡呈現精確 MMD² 數值，正文沒有寫出具體數字，因此兩篇論文的正文轉述都停留在「質性描述＋MNIST 的 p≈0.3 對照」層級，不是轉述遺漏，是原始資料本身的呈現方式。

---

## 10. 抓不到 / 原文沒有的項目

1. **「Walsh coefficient」術語**：全文 grep 無此詞，P_σ(a) 只被稱為 product-Bernoulli 分布，見 §4.2。
2. **D-Wave、MNIST 實驗的精確 MMD² 數值**：原文正文沒有寫出精確數字，只在 Fig. 7 / Fig. 8 圖表中呈現（本次抓取的是 HTML/純文字版，圖片內容無法用 grep 抽取，若需要精確數字要另外看圖或原始 repo 的實驗紀錄）。
3. **GitHub repo 是否包含產生 Table 1 具體 σ 數字、初始化 scale 常數的原始實驗腳本**：只抓到套件核心程式碼（`iqp_optimizer.py`, `training.py`, `utils.py`, `gen_qml/*.py`），沒有找到論文六個實驗（2D Ising/Binary Blobs/D-Wave/MNIST/Scale free/Genomic）各自的訓練腳本或 config（可能在論文提到的「the accompanying repository」——這可能是另一個獨立 repo 而非 `iqpopt` 本體，本次未進一步搜尋，超出 5 分鐘 best-effort 時間預算）。
4. **`sigma_heuristic()` 這個 repo 函式與論文 Table 1 實際使用方法之間的落差原因**：只能確認兩者數字對不上（見 §6.3），無法進一步確認是 repo 之後被簡化過、還是論文原始腳本另有寫法——這點原文和 repo 都沒有明講。
5. **RBM/EBM 對 D-Wave 資料集的訓練細節（例如 batch size、learning rate 的實際選定值，而非 Table 2 的搜尋範圍）**：Table 2 只給出**搜尋範圍**，各實驗實際選中的超參數值原文說要看「the accompanying repository」，未進一步抓取。
6. **`iqp_optimizer.py` 內部 `op_expval` 等核心方法的逐行對照**：抓到了檔案（20033 bytes）但基於時間預算，本次沒有逐行核對 eq. 21/22 在該檔案裡的具體實作位置（只核對了 `gen_qml/iqp_methods.py` 的 `loss_estimate_iqp`/`mmd_loss_iqp`，這兩個檔案已足以覆蓋 MMD loss 與多 bandwidth 平均的核心邏輯）。
