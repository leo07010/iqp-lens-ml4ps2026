# arXiv 2603.06440 復刻結果 — IQP 生成模型於 turbulence D2

Job **342220**（main，55 arms，0 failures，31m55s）與 job **342454**（post-hoc 重建，30 arms，4m46s）；
探索性分析另有 **342514**（§3(c) last-write 重評）、**342542**（§3(d) A kernel 掃描 + §3(e) QCLI/CCI）、
**342685**（§3(d) B 協定歸因）——完整 job 表見附錄。
協定與判定規則寫在跑之前：`docs/IQP_TURBULENCE_REPLICATION_PREREGISTRATION.md`。
Root-cause 分析（全部數字的出處）：`docs/IQP_TURBULENCE_REPLICATION_DIAGNOSIS.md`。
程式：`experiments/13_iqp_turbulence_replication/`。原文無公開程式碼，整條 pipeline 由論文文字重建。

## TL;DR

1. **未重現**（pre-registered 慣例下）。11-snapshot regime 下 RBM 與 DCGAN-1.2M 在**兩個指標上都**贏過我們的 IQP；pre-reg 的 "not reproduced" 條款觸發。
2. 但論文的 IQP PDF–JS **在 last-write-wins 聚合慣例下是對得上的**（完整 200-τ、3 seeds，job 342514：
   0.0322 ± 0.0006 / 0.0476 ± 0.0004 vs 論文 0.0238 / 0.0221），而論文從未定義聚合規則。但這**不改變
   not-reproduced 的判定**：同一慣例下 RBM-12k 兩個指標仍全面贏過 IQP（0.0225 / 0.6384），排序與論文相反，
   詳見 §3(c)。真正的問題在另一層：**我們的 IQP 連自己的 training anchor 都擬合得不好**
   （corr **0.69（抽樣聚合場）／0.77（exact E[z|x,y] 場）**——後者才是與 §3(d) 可比的量）。
   §3(d) 已把成因釘死：**不是「論文未指定的 σ」**，而是論文協定本身（stage 1 凍住 θ_lat 約 −0.08、
   5×10⁴ 抽樣 pmf 約 −0.04）加上這個模型族／MMD 目標約 **0.88–0.89** 的天花板。
3. **Conv-MMD² 欄不可比**：把 ground truth 的像素空間打散就得到 0.704，而論文 Table 2 整欄（0.896–1.472）都在這條線以上，我們的 IQP/RBM/CTRL-marginal（0.73–0.83）也是。飽和區內的排序沒有資訊。
4. **論文的評估設計無法支持它的主張**：held-out 任務被零參數插值解到 corr 0.9998；兩個指標對 τ 排列都不變（打亂 GT 得 0/0），latent adaptation 的核心宣稱從頭到尾沒被量到。
5. **主跑 IQP 是論文 Algorithm 1 的忠實產物，anchor 擬合差主要來自協定、不是我們的實作或 kernel 選擇。**
   單 anchor 的乾淨歸因（§3(d)，job 342542/342685）：exact 場 corr 0.77 之中，約 **0.08** 來自 stage 1 凍住
   θ_lat（含全部 18 個單體項）、約 **0.04** 來自 5×10⁴ 抽樣 pmf，而 Algorithm 1 的 stage 2（對 anchor 再調
   latent 3000 步）**補不回來**（+0.0001）。把這兩項都放開（訓全部參數、用全 pool pmf）到 corr 0.886；
   再換 kernel／K 最多 **0.892**——那是模型族＋closed-form MMD 目標的上限，不到 0.95。
   我們的 IQP 仍通過 8 項獨立數學驗證且訓練已收斂（15k 步即到頂），行為與「論文目標函數的最佳解」一致：
   那個 kernel 把 99.3% 權重放在 |s|≤3，而 D2 的 Walsh 能量 93% 在更高階——這解釋的是**高階誤差的型態**，
   不是 corr 只有 0.77 的主因。
6. **選題前提（P5）測了，但這個指標支持不了它。** D2 的 pooled I_QCLI **0.167** / I_CCI **0.882**，數值上確實落在
   論文說的「真實資料 ≈ 0.2」附近；但**同樣本數的 i.i.d. uniform 18-bit 亂數在 M ≥ 10⁴ 時 I_QCLI 反而更高**
   （0.471 vs 0.162 @ M = 5×10⁴；全樣本 0.621 vs 0.167）。在 n = 18、M ≳ 10⁴ 這個 regime，指標的 null 遠離 0，
   沒有配對 null 就不能宣稱「turbulence 是 IQP-compatible」。詳見 §3(e)。
7. **Baseline 對齊到論文字面設定（單 seed，job 345214）**：論文的 DCGAN@11 PDF–JS 數字對應我們
   pipeline 裡訓練 300–1000 步（未訓練完成）的狀態，但 RBM 掃過 latent 讀法與訓練配方兩個維度，
   沒有任何一組設定能同時對上論文的兩個數字；Conv2D-MMD² 的絕對值依賴隨機初始化的 encoder，
   跨實作不可比，只有同一 encoder 下的相對排序有意義。詳見 §3(f)。

---

## 1. 設定摘要

**資料**：D2 = `2D_velocity_snapshot_sub_domain_2`（Khojasteh et al. 2022，DOI 10.15454/GLNRHK），1000 個 snapshot，
取中間 z-plane（z = 3.0）的 308×328 網格，z-channel = |v| = sqrt(u²+v²+w²)。每個 snapshot 的 code pool = 其 101,024
個原生格點量化成 (6,6,6) bits；N_t = 5×10⁴ tuples。Anchors = t ∈ {1, 100, …, 1000}（11 個）；評估 = 另外 200 個
unseen τ（numpy seed 0 抽一次，所有 arm 共用），200 生成場 vs 200 ground-truth 場，重建網格 64×64。

**Arms**：IQP-31k（|s|≤6）、IQP-63k（|s|≤7）、RBM-12k、DCGAN-51k/0.2M/1.2M（@11 與 @100）各 3 seeds；
DCGAN data-scaling（N ∈ {10,30,50,70,90,100}，單 seed）；6 個 IQP-31k 敏感度 arm（單 seed）；
兩個論文沒有的 control：CTRL-interp（零參數，鄰近 anchor 線性插值）與 CTRL-marginal（18-bit product-Bernoulli，198 params，3 seeds）。

**指標**：PDF–JS（B=50，joint min/max，log₂）與 Conv2D-feature MMD²（3×Conv-ReLU stride 2 + GAP，固定 seed 0 隨機初始化，
median-heuristic σ，biased estimator，N=M=200）——皆照 Appendix I 實作。

| 類別 | 內容 |
|---|---|
| 與論文一致 | 18 qubits、(6,6,6)-bit 編碼、d_lat = 50、3×10⁴ core steps + 3000 latent steps/anchor、11 anchors、200 unseen τ、10⁵ synthesis samples、PDF–JS B=50、Conv2D-MMD² N=M=200 median heuristic、RBM/DCGAN 參數量 |
| 自行決定（原文未給，14 項，見 `IQP_REPLICATION_SPEC_2603.06440.md` §11） | optimizer/lr（Adam 1e-4）、training kernel 的 σ（Ref [38] 三帶寬規則）、core init、PDF–JS log base、Conv2D encoder 通道數、**聚合規則（mean）**、RBM/DCGAN 完整架構與超參、哪 100 個 snapshot、held-out τ 清單、seeds |
| 刻意偏離（pre-reg 已揭露） | (i) 用 Walsh–Hadamard **精確模擬**取代論文每步 500 樣本的 MMD estimator（去掉估計雜訊，不改變模型族的表達力）；(ii) 加兩個 control（CTRL-interp / CTRL-marginal），因為沒有它們「IQP 贏 RBM/DCGAN」無法解讀 |

IQP 每個 arm 約 10.4–11.5 分鐘（單 GPU，n=18 精確模擬，11 ms/step）；stage-1 MMD² 從 0.153 收斂到
6.07e-05（31k）／7.68e-05（63k），三個 seed 一致。RBM 55 s、DCGAN 98 s。

**程式正確性**：提交前以 PennyLane statevector oracle、explicit-kernel MMD、finite-difference gradients、
brute-force QCLI/CCI、手算指標等 8 項獨立驗證全部通過（機器精度）。並在提交前抓到並修掉兩個 bug：
(i) d2 模式下數個 arm 從 mean-binned 的 64×64 場重新推導 z 量化範圍，而非用編碼 code pool 的原生 |v| 範圍
（18.8% decode-scale 錯位；現在全部統一取 (a_z, b_z) = (0, 2.034849)）；(ii) kernel 權重在 β → 0 的 NaN guard。

---

## 2. 確認性結果（pre-registered）

Mean ± std over 3 seeds（std 為 population std，與 `evaluate.py` 一致）。論文欄位取自 Table 2。

| Model (Params)(# TD) | 我們 PDF–JS ↓ | 論文 PDF–JS ↓ | 我們 MMD² (Conv2D) ↓ | 論文 MMD² ↓ |
|---|---|---|---|---|
| RBM-12k (11988)(11) | 0.2139 ± 0.0058 | 0.168770 | 0.7282 ± 0.0157 | 1.472481 |
| IQP-31k (31180)(11) | 0.2700 ± 0.0030 | 0.023784 | 0.8237 ± 0.0046 | 0.952420 |
| IQP-63k (63004)(11) | 0.2366 ± 0.0010 | 0.022084† | 0.8161 ± 0.0129 | 0.942267† |
| DCGAN-51k (55339)(11) | **0.0550 ± 0.0149** | 0.366758 | 0.8799 ± 0.0347 | 1.397969 |
| DCGAN-0.2M (202303)(11) | **0.0809 ± 0.0316** | 0.328273 | 0.9246 ± 0.0763 | 1.407441 |
| DCGAN-1.2M (1213889)(11) | **0.0547 ± 0.0692** | 0.244852 | **0.3819 ± 0.3652** | 1.446292 |
| DCGAN-51k (55339)(100) | 0.0716 ± 0.0437 | 0.135584 | 0.9686 ± 0.0546 | 1.327760 |
| DCGAN-0.2M (202303)(100) | 0.0424 ± 0.0353 | 0.027695 | 0.7861 ± 0.1695 | 0.895912 |
| DCGAN-1.2M (1213889)(100) | 0.0479 ± 0.0095 | 0.012204 | 0.3090 ± 0.0454 | 0.964874 |
| **CTRL-interp (0)(11)** | **0.0012** | — | **0.0022** | — |
| CTRL-marginal (198)(11) | 0.4205 ± 0.0003 | — | 0.8226 ± 0.0017 | — |

粗體 = 在該 regime 內贏過 IQP-63k 的 baseline。†為論文原有標記（原文未給圖例）。

**DCGAN data-scaling study**（單 seed，論文 Fig. data-scaling 的對應）：

| N_train | 51k JS / MMD² | 0.2M JS / MMD² | 1.2M JS / MMD² |
|---|---|---|---|
| 10 | 0.0175 / 0.913 | 0.0110 / 0.691 | 0.0080 / 0.338 |
| 30 | 0.0375 / 0.861 | 0.0125 / 0.408 | 0.0188 / 0.109 |
| 50 | 0.0258 / 1.104 | 0.0170 / 0.394 | 0.0449 / 0.418 |
| 70 | 0.0113 / 0.958 | 0.0383 / 0.844 | 0.0360 / 0.659 |
| 90 | 0.0288 / 0.929 | 0.0398 / 0.928 | 0.0444 / 0.286 |
| 100 | 0.0315 / 0.863 | 0.0276 / 0.973 | 0.0378 / 0.465 |

摘要：**沒有任何 scaling 趨勢**。三個容量在 N=10 就已經達到 PDF–JS 0.008–0.018，之後隨 N 增加是**微幅變差**的噪音。
論文的「DCGAN 要 ~100 snapshots 才追上」在我們的重建裡不成立，因為 DCGAN 在 N=10 就已經贏 IQP。

### 逐條套用 pre-registered decision rule

| 條款 | 判定 | 依據 |
|---|---|---|
| **Reproduced**（兩個指標上每個 IQP arm 都贏過所有 11-snapshot baseline，超過 max(s) 且至少一個 100-snapshot DCGAN 追平） | ✗ 不觸發 | IQP 在 PDF–JS 上輸給**每一個** 11-snapshot baseline（0.2366 vs 最差的 baseline 0.0809） |
| **Partially reproduced**（只在一個指標上成立，或成立但落在 seed noise 內） | ✗ 不觸發 | 兩個指標都沒有「IQP 贏過全部 11-snapshot baseline」 |
| **Not reproduced**（有一個 11-snapshot baseline 在兩個指標上都追平或贏過 IQP） | ✓ **觸發，且有兩個 baseline** | RBM-12k：0.2139 < 0.2366 且 0.7282 < 0.8161。DCGAN-1.2M@11：0.0547 < 0.2366 且 0.3819 < 0.8161 |
| **Uninterpretable**（override；CTRL-interp 在兩個指標上贏過所有 trained model，或 CTRL-marginal 在兩個指標上都與 IQP 落在 seed noise 內） | ✓ **第一個子句觸發** | CTRL-interp 0.0012 / 0.0022，優於任何 trained arm 的最佳值（PDF–JS 0.0080、MMD² 0.109，兩者都出自 DCGAN-1.2M 的 scaling 單 seed run）。第二個子句只在 MMD² 成立（CTRL-marginal 0.8226 vs IQP 0.8237 / 0.8161，遠在 seed noise 內），PDF–JS 不成立，故該子句不觸發 |

**判定：not reproduced，且 uninterpretable override 觸發。** 依 pre-reg，override 蓋過其他判定：
在這個評估設計下，指標量到的東西不是論文說它量的東西，因此「IQP vs RBM vs DCGAN」的排序本身不具科學意義。

### 預測評分（P1–P5，寫在跑之前）

| # | 預測 | 事前機率 | 結果 |
|---|---|---|---|
| P1 | IQP 在 PDF–JS 上贏過所有 11-snapshot baseline | ~0.7 | ✗ **錯**。IQP 是所有 trained arm 裡 PDF–JS 最差的（只贏 CTRL-marginal） |
| P2 | IQP 在 Conv2D-MMD² 上贏過所有 11-snapshot baseline | ~0.5 | ✗ **錯**。贏 DCGAN-51k/0.2M，輸 RBM 與 DCGAN-1.2M |
| P3 | 100-snapshot DCGAN-0.2M/1.2M 至少在一個指標上追平或贏 IQP | ~0.7 | ✓ **對**（但意義被削弱：它們在 11 snapshots 就已經贏了，不需要 100） |
| P4 | CTRL-interp 在 PDF–JS 上有競爭力、甚至贏過所有模型 | — | ✓ **對，而且更強**：兩個指標上都贏過所有模型（0.0012 / 0.0022） |
| P5 | D2 pooled QCLI 落在 ~0.2 附近；CCI 未知 | — | ◐ **數值對，但指標無解釋力**（已補測，job 342542）。pooled I_QCLI **0.167** / I_CCI **0.882**，確實近 0.2；但同樣本數的 i.i.d. uniform null 在 M ≥ 10⁴ 時 QCLI 更高（0.471 @ M = 5×10⁴），指標在 n = 18 的有限樣本 regime 分不出結構與亂數。詳見 §3(e) |

---

## 3. 探索性分析（post-hoc，明確標示）

以下五組都是**看過主結果之後**才決定要做的，不進入 confirmatory 判定。(a) 在做之前已寫進 pre-reg addendum
（含事前預測），(b) 的 arm 本身是 pre-registered 但只有單 seed，(c) 與 (d) 是主結果之後才設計、判讀規則
寫在跑之前，(e) 是 pre-registered 的 P5 補測（測量本身是 post-hoc 執行）。

### (a) 重建規則：exact 條件期望 vs 10⁶ 樣本（job 342454）

Pre-reg 用的是 10⁵ 樣本 + cell mean（每格約 24 個樣本）。論文說「aggregate ... 10⁵–10⁶ samples」而
"probability-aggregation" 更自然的讀法是機率加權條件期望 E[z | x, y]，可從 q_θ 精確算出、零採樣雜訊。
**不重新訓練**，只用已存的 θ 重建 200 個 hold-out 場。

| Metric | Arm (3 seeds) | pre-reg（1e5 + mean） | **+exact E[z\|x,y]** | **+1e6 samples** |
|---|---|---|---|---|
| PDF–JS | IQP-31k @11 | 0.2700 ± 0.0030 | 0.3240 ± 0.0043 | 0.3180 ± 0.0041 |
| PDF–JS | IQP-63k @11 | 0.2366 ± 0.0010 | 0.2921 ± 0.0008 | 0.2862 ± 0.0008 |
| PDF–JS | CTRL-marginal @11 | 0.4205 ± 0.0003 | 0.8682 ± 0.0012 | 0.7081 ± 0.0002 |
| MMD² | IQP-31k @11 | 0.8237 ± 0.0046 | 0.8278 ± 0.0056 | 0.8307 ± 0.0055 |
| MMD² | IQP-63k @11 | 0.8161 ± 0.0129 | 0.8321 ± 0.0190 | 0.8322 ± 0.0155 |
| MMD² | CTRL-marginal @11 | 0.8226 ± 0.0017 | 0.8164 ± 0.0011 | 0.8203 ± 0.0014 |

**Addendum 裡寫的預測是錯的**：我們預測 exact 會讓 PDF–JS 大幅下降（以為 0.24 有很大一塊是聚合雜訊）。
實際上 exact 讓 PDF–JS **變差** 23%（0.2366 → 0.2921），MMD² 幾乎不動。原因與 Q1 的天花板分析一致：
10⁵ 樣本的聚合雜訊本來就只有 4e-5 量級，去掉它不會有幫助；而那點雜訊原本**偶然地把過窄的 model 直方圖
撐寬了一點**，反而拉近了與 GT 的 contrast。CTRL-marginal 的 0.42 → 0.87 是同一效應的極端版。
結論：pre-registered 的判定不因重建規則改變，方向反而更不利於 IQP。

### (b) 敏感度 arm（IQP-31k，單 seed；baseline = iqp_K6_seed0 0.2660 / 0.8188）

| Run | 改了什麼 | PDF–JS | MMD² | stage-1 final loss |
|---|---|---|---|---|
| `iqp_sens_lat_init_data` | θ_lat⁽¹⁾ 用 data init 取代 N(0,0.01) | **0.1880** (−29%) | 0.8105 | **2.37e-05**（最低） |
| `iqp_sens_sigma_ref38_16q` | σ = 0.6, 0.3（Ref [38] 16-qubit 表值） | **0.2107** | 0.8849（變差） | 1.19e-04 |
| `iqp_sens_full_grid` | 用完整原生 pool 取代 5×10⁴ 抽樣 | 0.2441 | 0.8260 | 4.54e-05 |
| `iqp_sens_sigma_meanweight2_6` | σ 對應 mean Pauli weight 2.0 與 6.0 | 0.2481 | 0.8161 | 1.64e-04 |
| `iqp_sens_random_core_init` | random core init | 0.2498 | 0.7955 | 4.45e-05 |
| `iqp_sens_lr1e-3` | lr 1e-3 | 0.2872（最差） | 0.8191 | 2.48e-04（最差） |

排序與訓練損失一致（最低 loss 的 arm 也有最好的 PDF–JS），但**最好的敏感度 arm 0.188 仍比 pipeline 天花板高 4700 倍**，
也仍輸給每一個 DCGAN。改用 exact 重建後這個排序還會重排（`sigma_ref38_16q` 掉到 0.3357，變成第二差），
再次說明 PDF–JS 在這個尺度上量的是 contrast、不是 fidelity。

### (c) 「取最後一個樣本」（last-write-wins）聚合下的全 arm 重評（job 342514，完整 200 τ、3 seeds）

Job **342514**（8gpus，單 GPU，3 min 19 s，COMPLETED）把 last-write-wins 聚合套到全部有存 tuple/θ 的 arm，
用完整 200 個 hold-out τ（取代先前只有 60 τ 的先行版本），IQP 與 CTRL-marginal 都是 3 seeds。

**方法不對稱，先說清楚**：IQP-31k/63k 與 CTRL-marginal 是對*已存好的* θ／replay 重新取樣、重新聚合、
重新評分（不重訓；seed 與 342454 不同）。**RBM 沒有存 tuple，所以是重新訓練後取樣**——同一架構重跑三個
seed，取樣過程本身非確定性；RBM 的 last-write 數字因此跟 IQP/CTRL-marginal 不是同一種操作下的產物，只能
當粗略對照，不是嚴格的 apples-to-apples。DCGAN 與 CTRL-interp 不是 tuple-based，聚合慣例對它們無意義，
數字沿用 §2、不受影響。

以下 mean ± std 全部由 `results_d2_last.jsonl` 依 `run_dir` 前綴（去掉 `_seed{n}`）分組重算（population std，
與全文一致）。**⚠ 不要用 `logs_d2/evaluate_last_merged.log` 的摘要表**：它按 `(arm, train_snapshots)` 聚合，
把 6 個 `iqp_sens_*` 單 seed arm 混進了 `IQP-31k+last` 的平均（該表顯示 0.0466 ± 0.0315，是被稀釋後的數字，
不是真正的 3-seed 值 0.0322 ± 0.0006）。

| Arm (Params) | PDF–JS mean-agg | PDF–JS last-write | MMD² mean-agg | MMD² last-write | 論文 PDF–JS | 論文 MMD² |
|---|---|---|---|---|---|---|
| IQP-31k (31180) | 0.2700 ± 0.0030 | **0.0322 ± 0.0006** | 0.8237 ± 0.0046 | 0.7096 ± 0.0103 | 0.023784 | 0.952420 |
| IQP-63k (63004) | 0.2366 ± 0.0010 | **0.0476 ± 0.0004** | 0.8161 ± 0.0129 | 0.7390 ± 0.0074 | 0.022084† | 0.942267† |
| RBM-12k (11988) | 0.2139 ± 0.0058 | 0.0225 ± 0.0022 | 0.7282 ± 0.0157 | 0.6384 ± 0.0188 | 0.168770 | 1.472481 |
| CTRL-marginal (198) | 0.4205 ± 0.0003 | 0.1486 ± 0.0002 | 0.8226 ± 0.0017 | 0.7993 ± 0.0182 | — | — |
| DCGAN-1.2M@11 (1213889) | 0.0547 ± 0.0692 | 不適用（非 tuple-based，同 mean-agg） | 0.3819 ± 0.3652 | 同左 | 0.244852 | 1.446292 |
| CTRL-interp (0) | 0.0012 | 不適用（零參數，非 tuple） | 0.0022 | 同左 | — | — |

†為論文原有標記（見 §2）。

**敏感度 arm（單 seed，last-write 下；mean-agg 對照取自 §3(b)）：**

| Sensitivity arm | 改了什麼 | PDF–JS last | MMD² last | PDF–JS mean-agg | MMD² mean-agg |
|---|---|---|---|---|---|
| `iqp_sens_lat_init_data` | θ_lat 用 data init | **0.0258**（最好） | 0.6665 | 0.1880 | 0.8105 |
| `iqp_sens_full_grid` | 完整原生 pool 取代抽樣 | 0.0286 | 0.6916 | 0.2441 | 0.8260 |
| `iqp_sens_random_core_init` | random core init | 0.0283 | 0.7025 | 0.2498 | 0.7955 |
| `iqp_sens_sigma_meanweight2_6` | σ 對應 mean weight 2.0/6.0 | 0.0337 | 0.7168 | 0.2481 | 0.8161 |
| `iqp_sens_lr1e-3` | lr 1e-3 | 0.0850 | 0.7793 | 0.2872 | 0.8191 |
| `iqp_sens_sigma_ref38_16q` | σ = Ref[38] 16-qubit 表值 | 0.1212（最差） | 0.8928（最差） | 0.2107 | 0.8849 |

排序與 §3(b) 的 mean-agg 排序大致一致（`lat_init_data` 最好、`sigma_ref38_16q` 最差），聚合慣例沒有改變
敏感度實驗本身的結論。

**解讀**：在 last-write 聚合慣例下，我們的 IQP PDF–JS 確實落到論文數字附近——0.0322 / 0.0476 vs 論文
0.023784 / 0.022084，跟 mean-agg 與論文之間原本約 10 倍的差距相比，已經收斂到同一個量級。但這**不改變
§2 的 not-reproduced 判定**：RBM-12k 在兩個指標上仍全面贏過兩個 IQP arm（0.0225 / 0.6384 vs
0.0322–0.0476 / 0.7096–0.7390），排序與論文相反。DCGAN-1.2M@11 的均值 0.0547 ± 0.0692 因為單一崩潰
seed（見 §4 第 6 點）而顯得跟 IQP 接近，但拿掉那個 seed，另外兩個 seed 的 PDF–JS（0.0037、0.0080）與
MMD²（0.127、0.121）仍遠優於兩個 IQP arm 的 last-write 值——DCGAN 在典型情況下依然贏 IQP，只是聚合後的
均值被 outlier 拉高、看起來接近。MMD² 則沒有同一個「對上論文」的故事：last-write 把我們的 IQP MMD² 從
0.82 壓到 0.71/0.74，方向上更靠近 §5 第 4 點建立的飽和下限（GT 自己空間打散後的 0.704），而不是更靠近
論文的 0.94/0.95；換句話說 last-write 讓 MMD² 更貼近「結構被摧毀」那條線，TL;DR/§5 已經建立的「MMD² 欄
不可比、飽和區內沒排序資訊」結論不變。所以「聚合慣例」解釋了 IQP **絕對數字**為何跟論文對得上，但不改變
「現象未重現」的判定；論文的 IQP 優勢，對應的比較組看起來是一個比 RBM、也比一個未特別調參的 DCGAN 都弱的
baseline。另有一個值得記錄、但**還不下結論**的現象：我們原本以為 last-write 會把 PDF–JS 退化成單純的
逐格 z-marginal 比較，理論上邊際對齊的模型應該天生佔便宜；但 CTRL-marginal 正是刻意對 18-bit 邊際做 exact
fit 的 arm，last-write 下卻是所有 trained arm 裡最差的（0.1486，比兩個 IQP arm 的 0.03–0.05、RBM 的
0.0225 都高）。這代表「退化成純 z-marginal 比較」這個假說站不住——last-write 底下顯然還有 tuple 落點的
(x, y) 空間結構在起作用——但目前證據不足以講清楚機制是什麼，留給未來的診斷，這裡不強行下結論。

### (d) Anchor-fit kernel 掃描與協定歸因（job 342542 / 342685）

**目的**：分辨 §4.5 的「trained IQP 落在資料的 order-≤3 截斷上」是**論文未指定 σ 造成的可修復現象**，
還是**這個受限 IQP 模型族在 D2 上的表達力上限**；並進一步把「模型族上限」與「論文協定本身的限制」分開。
這是整份分析裡唯一還沒被實驗釘死的因果分岔，也決定了 §7 第 3 條（kernel 階數）值不值得做。

**設計**：把問題縮到最乾淨的單元——**只擬合單一 anchor t = 1，不做插值**
（§4.2 已證明 held-out 與 anchor 分數一致到 1%，所以 anchor fit 就是瓶頸本身）。分兩批跑：

- **A（kernel × K × lr 掃描，job 342542，8 GPU，19m18s）**：訓練**全部**參數（core + latent）、用**全** pool pmf、
  Adam、1e5 步。這批問的是「拿掉協定限制之後，這個模型族最好能到哪」。
- **B（協定歸因，job 342685，4 GPU，6m04s）**：固定 `auto3` / K = 6 / lr 1e-4 / 3×10⁴ 步，
  只切換論文協定的兩個開關（stage 1 是否凍住 θ_lat、pmf 用全 pool 還是 5×10⁴ 抽樣），
  外加 Algorithm 1 的 stage 2（對 anchor 再調 latent 3000 步）。

**兩批的 corr 一律用 exact E[z | x, y] 場對 GT 計算**（原因見下方「一項修正」）。

#### A. kernel × K × lr 掃描（`logs_d2/anchor_fit*.log`）

| config | final MMD² ↓ | corr ↑ | pdf_js (mean-agg) ↓ | ⟨Z_s⟩ rel-err @ \|s\| = 1 / 2 / 4 / 6 / 8 |
|---|---|---|---|---|
| `auto3`（K = 6, lr 1e-4；**現行設定**） | **1.81e-05** | 0.876 | **0.184** | 0.008 / 0.063 / 0.480 / 0.821 / 0.970 |
| `flat_K7` | 3.19e-05 | **0.892** | 0.187 | 0.608 / 0.537 / 0.449 / 0.474 / 0.499 |
| `sigma0.3_K7_lr1e-3` | 3.19e-05 | **0.892** | 0.189 | 0.604 / 0.530 / 0.445 / 0.469 / 0.494 |
| `flat_lr1e-3` | 4.73e-05 | 0.846 | 0.212 | 0.622 / 0.587 / 0.499 / 0.547 / 0.609 |
| `sigma0.3` | 4.85e-05 | 0.841 | 0.210 | 0.631 / 0.591 / 0.507 / 0.552 / 0.614 |
| `sigma0.15` | 4.90e-05 | 0.835 | 0.206 | 0.646 / 0.608 / 0.514 / 0.559 / 0.620 |
| `flat` | 4.90e-05 | 0.835 | 0.206 | 同 `sigma0.15` |
| `auto3_K7` | 6.09e-05 | 0.802 | 0.229 | 0.023 / 0.187 / 0.607 / 0.849 / 0.993 |

參考點：GT 對自己 corr = 1.0；把**資料自己的精確 pmf** 截斷到 Walsh order ≤ 3 再走同一條 pipeline 是
corr **0.692** / pdf_js 0.193（§4.5）。**每個 config 都在 15k 步就到頂**，剩下的 85k 步沒有再改善——
「收斂不足／lr 造成早停」這個解釋被排除，步數不是因素。

**A 的解讀**：

1. **事前寫下的判讀規則給出的答案是「模型族上限」，不是「σ 未指定」。** 規則說「若把 kernel 推向高階
   （σ = 0.15 或 flat）能把 corr 拉過 0.85，結論就是 σ 的問題」。實測 `flat` / `sigma0.15` 只到 **0.835**，
   沒有越線；越線的反而是現行的 `auto3`（0.876）與兩個 K = 7 組（0.892）。**推高階不但沒救，還略差。**
2. **兩種 kernel 對應兩種誤差型態，但天花板一樣。** `auto3` 把低階（|s| ≤ 2）擬到 < 7% 誤差、高階整個放掉
   （|s| = 8 誤差 97%）；`flat` / 小 σ 則把誤差**平均攤到各階**（各階 0.45–0.65）。前者 MMD² 好 2.7×、
   pdf_js 也較好，後者 corr 高一點點——但兩條路線的 corr 全落在 **0.84–0.89**，沒有任何一種逼近 0.95。
3. **K = 7（63k params）只在 `flat`/σ0.3 下有幫助**（0.835 → 0.892），在 `auto3` 下反而更差（0.876 → 0.802）。
   多出來的 32k 參數在低階偏差的目標下被浪費掉；**加參數不是出路**。
4. 結論：**這個模型族 + closed-form Gaussian-kernel MMD 目標，在 D2 anchor 上的天花板約 0.88–0.89。**

#### B. 協定歸因（`logs_d2/anchor_fit_protocol*.log`；全部 `auto3` / K = 6 / lr 1e-4 / 3×10⁴ 步）

| config | corr ↑ | pdf_js ↓ | MMD² ↓ |
|---|---|---|---|
| `paper_stage1`（θ_lat 凍在 N(0,0.01)、只訓 core；全 pool pmf） | 0.807 | 0.243 | 4.54e-05 |
| `paper_stage1_plus_stage2`（＋Algorithm 1 stage 2：對 anchor 再調 latent 3000 步） | 0.808 | 0.252 | 4.54e-05 |
| `paper_stage1_subsample5e4`（= **主跑 stage-1 的精確設定**） | 0.769 | 0.267 | 6.21e-05 |
| `allparams_subsample5e4`（放開全部參數；pmf 仍為 5×10⁴ 抽樣） | **0.886** | **0.178** | **1.70e-05** |

**歸因鏈是接得起來的**：主跑 `iqp_K6_seed0` 用 exact 場重算 anchor t = 1 得 corr **0.774** / pdf_js 0.268，
與 `paper_stage1_subsample5e4` 的 0.769 / 0.267 對得上（差異在 seed 與 latent-stage 細節）。

| 因子 | 對 anchor corr 的效果 | 依據 |
|---|---|---|
| stage 1 **凍住 θ_lat**（含全部 18 個單體項） | 約 **−0.08** | 0.886（allparams）→ 0.807（stage1） |
| pmf 用 **5×10⁴ 抽樣**而非全 pool | 約 **−0.04** | 0.807 → 0.769 |
| Algorithm 1 的 **stage 2**（latent adaptation） | **+0.0001**（MMD² 完全不動、pdf_js 反而變差） | 0.807 → 0.808 |
| 模型族／目標函數上限 | 0.886 → 至多 **0.892** | A 表 |

（誠實註記：這兩個因子**沒有完全交叉**——B 缺「全參數 + 全 pool pmf、3×10⁴ 步」那一格。
A 表裡最接近的對應 config（`auto3`、全參數、全 pool、1e5 步）是 0.876，與 0.886 同一量級，
所以上表的加總只當**近似歸因**，不是精確的方差分解。）

**B 的解讀**：主跑 IQP 的 anchor corr 0.77 **不是我們實作錯了，而是論文 Algorithm 1 的忠實產物**——
「stage 1 凍住 θ_lat、只訓 core」是 Algorithm 1 明寫的設計，N_t = 5×10⁴ 也是論文明寫的數字。
把這兩項都放開能到 0.886，再換 kernel/K 最多 0.892。
**擬合不好的主因是協定，其次是模型族／目標函數的上限；training kernel 的 σ 不是主因。**
另外，Algorithm 1 賣點所在的 stage 2 在**它自己的 training anchor 上**都量不到作用（見 §5 第 8 點）。

#### 一項修正（重要，影響先前的敘述）

診斷文件與本文件先前寫的「anchor corr 0.69」是**抽樣聚合場**的 corr（11 個 anchor 平均 0.691）；
同一份診斷若改用 **exact 條件期望場**算則是 **0.776**。§3(d) 兩批實驗全程用 exact 場，
所以正確的對照鏈是 **0.77（主跑）→ 0.886（放開協定）→ 0.892（再換 kernel/K）**，不是從 0.69 起跳。

連帶要修的是 §4.5：「trained IQP ≡ 資料的 order-≤3 截斷（0.688 vs 0.693）」這個等號**只在抽樣場上成立**。
在 exact 場上，主跑 IQP（0.774）**明顯優於** order-≤3 截斷（0.692）。正確的敘述是
「主跑 IQP 的高階 ⟨Z_s⟩ 誤差很大（|s| = 6 時 86%），**但並不等價於** order-≤3 截斷」。

### (e) P5 補測：D2 的 QCLI / CCI（job 342542 內的 CPU 階段，`outputs_anchor_fit/qcli_cci_d2.json`）

**目的**：補上 pre-reg 的 P5。論文選 turbulence 的理由，是它落在 QCLI/CCI 平面的「IQP-compatible」右上象限
（Fig. 5，真實資料 QCLI ≈ 0.2）。這是整篇論文的**選題前提**，值得單獨檢驗。

**pooled**（11 個 anchor 各抽 M 個 code 合併）：

| M | D2 I_QCLI | D2 I_CCI | 同樣本數 i.i.d. uniform 18-bit 的 I_QCLI（null） |
|---|---|---|---|
| 10³ | 0.058 | 0.914 | 0.020 |
| 10⁴ | 0.146 | 0.889 | 0.167 |
| 5×10⁴ | 0.162 | 0.884 | **0.471** |
| 10⁵ | 0.166 | 0.882 | **0.620** |
| 全部（1,111,264 codes） | 0.167 | 0.882 | **0.621** |

**per-snapshot**（M = 5×10⁴）：t = 1 → 0.090 / 0.908；t = 500 → 0.094 / 0.908；t = 1000 → 0.090 / 0.909。

**判定**：P5 的**數值**部分成立——D2 的 pooled I_QCLI 0.167 確實落在論文說的「真實資料約 0.2」附近，
I_CCI 也高（0.88–0.91），表面上就是 Fig. 5 的右上象限。**但這個指標在此 regime 不具解釋力**：
同樣本數的**均勻亂數**（結構為零的 null）在 M ≥ 10⁴ 時 I_QCLI 就**高過** D2（0.471 vs 0.162 @ M = 5×10⁴；
全樣本 0.621 vs 0.167），而 D2 的值反而穩定在 0.16–0.17。在 n = 18 qubits、M ≳ 10⁴ 的有限樣本 regime，
指標的 null 遠離 0（pre-reg 已預告這個 finite-sample 效應），所以
「turbulence 落在 IQP-compatible 象限」這個選題前提**無法用這個指標、在沒有配對 null 的情況下被支持**。
P5 記為 **數值近 0.2 ✓，但指標無解釋力**；要真的檢驗它，需要同 (n, M) 的 null 分布或有限樣本修正（§7 第 4 條）。

### (f) Baseline 對齊到論文字面設定（job 345214，單 seed）

**目的**：§2 的 RBM/DCGAN baseline 全面贏過論文同名 baseline（§6 已指出兩者差距很大）。但論文只給了
架構層級的描述——RBM：11 個 anchor、18-bit tuple、latent = 「固定攤平參數順序（weights and biases）
的前 50 個元素」、其餘凍住；DCGAN：Radford et al. 架構、51k/0.2M/1.2M 參數、latent 128、quantized 2D
fields、11 vs 100 snapshots——完全沒給訓練配方（steps、lr、RBM hidden size 以外的細節、DCGAN input
size）。這些沒寫的部分是我們自行選的標準預設值。這裡問的是：**在論文有寫的架構限制內，換一套訓練
配方，能不能對上論文的 baseline 數字？** 設計與事前宣告見
`docs/IQP_TURBULENCE_REPLICATION_PREREGISTRATION.md`（2026-09-06 addendum，<user> 要求）。

**設計**（單 seed = 0）：
- **DCGAN@11 訓練步數掃描**：3 個容量（51k/0.2M/1.2M）× 4 個 G-steps（300/1000/3000/20000，20000 =
  主跑設定）× seed 0，其餘不變（`--n-train 11`）。
- **RBM latent 讀法 × 訓練配方**：latent ∈ {`biases`（主跑設定：18 個 visible bias + 前 32 個 hidden
  bias）、`literal`（論文句子最字面的讀法：攤平 [W, b_v, b_h] 的前 50 個元素 = visible unit 0 的全部
  50 個權重）} × recipe ∈ {`pcd_adam`（主跑設定：PCD-1 + Adam）、`cd1_sgd`（CD-1 + SGD lr 0.05,
  batch 64, 30 epochs-equivalent）}，4 組合、seed 0。

Job **345214**（8gpus，8 GPU，3 min 23 s，COMPLETED，16 jobs，0 failures）。

**DCGAN@11 訓練步數掃描**（PDF–JS / MMD²，mean-agg，同 §2 慣例）：

| G-steps | 51k | 0.2M | 1.2M |
|---|---|---|---|
| 300 | 0.4329 / 1.0884 | 0.5238 / 0.8307 | 0.3630 / 0.8843 |
| 1000 | 0.2592 / 1.1200 | 0.3472 / 0.8412 | 0.0389 / 0.5921 |
| 3000 | 0.0280 / 0.8457 | 0.0862 / 0.9196 | 0.1691 / 0.9458 |
| 20000（主跑設定） | 0.0147 / 1.1322 | 0.0175 / 0.8921 | 0.0096 / 0.3828 |
| **論文 Table 2** | **0.366758 / 1.397969** | **0.328273 / 1.407441** | **0.244852 / 1.446292** |

**RBM latent 讀法 × 訓練配方**（PDF–JS / MMD²）：

| Latent × recipe | PDF–JS | MMD² |
|---|---|---|
| biases + pcd_adam（主跑設定） | 0.2062 | 0.7060 |
| biases + cd1_sgd | 0.3119 | 0.8154 |
| literal + pcd_adam | 0.2665 | 0.8295 |
| literal + cd1_sgd | 0.3245 | 0.8842 |
| **論文 Table 2** | **0.168770** | **1.472481** |

**解讀**：

1. **DCGAN@11：論文數字落在我們的欠訓練區間，不是某個容量特別對。** 論文的 PDF–JS（0.245–0.367）
   在三個容量的 300–1000 步左右都能對上（例如 1.2M @300 步 = 0.3630、51k @1000 步 = 0.2592、
   0.2M @300 步 = 0.5238 附近都有落點）——也就是說**論文的 GAN@11 對應我們 pipeline 裡「還沒訓練
   完成」的狀態**，不是某個特定容量的固有性質。一旦訓到 3000 步以上，PDF–JS 全部掉到 0.01–0.17，
   20000 步（主跑設定）掉到 0.01–0.02。MMD² 是另一個故事：我們掃過的 300–20000 步、三個容量，
   MMD² 全部落在 0.38–1.13，**沒有任何一組**碰到論文的 1.40–1.45；趨勢也不是單調的（例如 1.2M
   從 300 步的 0.8843 先掉到 1000 步的 0.5921、又回升到 3000 步的 0.9458，20000 步再掉到 0.3828），
   與 §5 第 6 點一致：小樣本 DCGAN 的訓練軌跡本身不穩定。**Conv2D-MMD² 的絕對值取決於隨機初始化的
   encoder 與 median-heuristic σ，這是我們自己的隨機種子決定的常數，跨實作不可比**（§5 第 5 點、
   §6 已提過）：只有同一個 encoder 下的相對排序才有意義，不能拿我們的絕對值去對論文的絕對值。
2. **RBM：四種設定都比論文的 PDF–JS 差，沒有一種能同時對上兩個指標。** mean-agg 下四組 PDF–JS
   落在 0.2062–0.3245，全部比論文的 0.168770 差；MMD² 落在 0.7060–0.8842，全部遠低於論文的
   1.472481——而論文的 1.472481 其實比我們在 §5 第 5 點建立的 iid uniform 噪音場參考點（0.927）
   還高，等於「比雜訊還差」。唯一另一個對照點是 last-write 聚合慣例（§3(c)）：同一個
   biases+pcd_adam 設定換算成 last-write 後 PDF–JS 掉到 0.0225、比論文好一個數量級，但 MMD² 是
   0.6384，一樣搆不到論文的 1.47。latent 讀法（biases vs literal）、訓練配方（pcd_adam vs
   cd1_sgd）、聚合慣例（mean vs last-write）三個維度都掃過了，**沒有一個組合能同時重現論文 RBM
   的兩個數字**。字面讀法（literal）確實讓 RBM 變差（biases+pcd_adam 0.2062 → literal+pcd_adam
   0.2665，+29%），方向對，但幅度遠不足以解釋論文的 0.168770；論文 MMD² 1.472481 本身就已經超出
   我們任何設定能到達的範圍。最可能的解釋是 Conv2D-MMD² 的絕對尺度在論文未公開的 encoder 初始化下
   本來就不一樣，或者論文自己的 RBM baseline 訓練沒調好——作者在 baseline 部分沒給任何調參細節。
3. **對「baseline 是否跟論文一樣」這個問題的誠實回答**：論文文字寫明的架構限制（tuple 表示、
   anchor 數、latent 描述、參數量）全部一致；論文沒寫的訓練配方，DCGAN 這邊能靠**少訓練 10–60 倍
   步數**把 PDF–JS 拉到論文數字，但 MMD² 始終對不上；RBM 這邊掃過 latent 讀法與訓練配方兩個維度，
   沒有任何設定能同時對上兩個指標。這印證並強化 TL;DR 第 2 點與 §5 的判斷：**論文報的 baseline
   數字，最可能對應弱化或未訓練完成的版本**，不是我們的重建方式有系統性錯誤。
4. **限制**：本節全部是單 seed（seed 0）。DCGAN 短步數（300–1000 步）在小樣本下變異可能很大——
   §2 主跑的 DCGAN-1.2M@11 三個 seed 裡就有一個訓練崩潰（PDF–JS 0.153、MMD² 0.898，§5 第 6 點），
   這裡單 seed 掃描抓到的任何一個「剛好對上論文」的點都可能是同一種變異的產物，不應解讀成「論文
   用了 XX 步」這麼精確的結論。RBM 四組設定同理只有單 seed，但 biases+pcd_adam（0.2062）與
   literal+pcd_adam（0.2665）差 6 個百分點，超過 <3% 的噪音門檻，方向性的結論（字面讀法變差）
   可以解讀；更精細的比較仍需要多 seed。

---

## 4. Root cause（數字全部出自 `IQP_TURBULENCE_REPLICATION_DIAGNOSIS.md`）

1. **不是採樣/聚合的問題。** 用 held-out snapshot 自己的經驗 pmf 當「完美模型」走完全相同的
   採樣 → 聚合 → 評分路徑，在 10⁵ 樣本下得到 PDF–JS **4e-5**、MMD² **2e-4**（GT vs GT = 0/0）。
   IQP 的 0.24 / 0.83 高出天花板**四個數量級**。原生網格 101,024 點落進 4096 格，每格 16–30 點，
   格內 z 散布 0.0303 vs 格間 0.4487，24 樣本的格均值標準誤只有訊號的 1.4%。
2. **模型連自己的 training anchor 都沒學會**，所以不是 latent interpolation 的問題：
   anchor-set 分數（0.2338 / 0.2668）與 held-out 分數（0.2354 / 0.2660）**一致到 1%**。
   每個 anchor 上 corr(生成場, GT) 平均 **0.691（抽樣聚合場）**；改用**精確條件期望場**是 **0.776**。
   （兩個數字量的是同一件事的兩種重建方式。§3(d) 一律用 exact 場，因此與 §3(d) 可比的主跑 anchor corr
   是 **0.77**，不是 0.69；本文件其餘處提到主跑 anchor corr 時同此註記。）
3. **codec 是對的，joint 是錯的。** 六個 z-qubit 的 ⟨Z_j⟩ 與資料吻合（t=1：model [0.128, 0.125, 0.154,
   −0.023, 0.294, 0.262] vs data [0.120, 0.115, 0.162, −0.027, 0.305, 0.267]），k_z 直方圖 JS 0.014–0.047。
   但 ⟨Z_s⟩ 的相對誤差隨 Pauli weight 從 **|s|=1 的 5% 升到 |s|=6 的 86%**（此為主跑、凍 latent 的模型；§3(d) 全參數版在 |s|=1 只有 0.8%，見 A 表）。
4. **原因是目標函數的低階偏差。** auto3 三帶寬 Gaussian kernel 的 Walsh 權重 c_s 隨 |s| 指數衰減
   （3.93e-01 → 1.91e-08）；**kernel 加權後 99.3% 的權重落在 |s| ≤ 3**，而資料未加權的 ⟨Z_s⟩² 能量
   有 **93% 在 |s| ≥ 3 之上（46.8% 甚至在 |s| > 7）**。模型在做 objective 要它做的事：訓練後的 MMD²
   比 exact product-Bernoulli 好 10×，只比 estimator 自身的雜訊地板高 7×。
   §3(d) A 表把這個機制直接量了出來：`auto3` 的 ⟨Z_s⟩ 相對誤差是 0.008（|s|=1）→ 0.970（|s|=8），
   而 `flat`/小 σ 把誤差平均攤成各階 0.45–0.65。**這解釋的是誤差型態，不是 corr 的高低**——兩種型態的
   corr 都在 0.84–0.89。
5. **【已由 §3(d) 定案，並修正先前的敘述】主因是協定，不是 σ，也不等於 order-≤3 截斷。**
   先前寫的是「把**精確資料 pmf** 投影到 Walsh order ≤ L 走同一條 pipeline，order-3 截斷得 corr 0.688 /
   std 0.240 / PDF–JS 0.182，而訓練好的 IQP-63k 是 0.693 / 0.219 / 0.235，兩者**在數值上重合**」。
   這個等號**只在抽樣聚合場上成立**：改用 exact 條件期望場，order-≤3 截斷是 **0.692**、主跑 IQP 是 **0.774**，
   模型**明顯優於**截斷。正確的敘述是「**主跑 IQP 的高階 ⟨Z_s⟩ 誤差很大（|s| = 6 時 86%），但並不等價於
   order-≤3 截斷**」。
   至於 corr 只有 0.77 的成因，§3(d) 已經拆開（兩批實驗、單一 anchor、exact 場）：
   - **σ 被排除**：四種 training kernel × 兩個 K 全部落在 corr 0.80–0.89；把 kernel 推向高階
     （`flat`/σ0.15）不但沒有跨過 0.85，還比現行 `auto3`（0.876）差。每個 config 15k 步就到頂，也不是收斂問題。
   - **主因是論文協定**：stage 1 凍住 θ_lat 約 **−0.08**、5×10⁴ 抽樣 pmf 約 **−0.04**，
     Algorithm 1 的 stage 2 補回 **+0.0001**。放開這兩項到 corr 0.886。
   - **其次是模型族／目標函數上限**：放開協定後再怎麼換 kernel/K，最多 **0.892**。
   第 4 點的 kernel 低階偏差仍然成立，但它解釋的是**高階誤差的型態**，不是 corr 0.77 的主因。
6. **論文唯一沒指定、且真的有影響的旋鈕（training kernel 的 σ）只把 PDF–JS 從 0.266 推到 0.211**，
   而且代價是 MMD² 變差。沒有任何一組超參數能跨過那四個數量級。
   §3(d) 在單 anchor 的乾淨設定下得到同樣的量級：σ 在 anchor corr 上的全部影響只有約 ±0.05（0.835–0.892）。
7. **PDF–JS 的量級差距則是純粹的慣例問題。** 同一批 IQP 樣本，只把聚合從 mean 換成 last-write-wins
   （60 τ 診斷）：IQP-31k 0.2646 → **0.0281**，IQP-63k 0.2363 → **0.0446**，正好夾住論文的 0.0238 / 0.0221。
   等價地，模型 raw tuple 的 z-marginal（完全不做場重建）PDF–JS 是 0.0141 / 0.0294，也在同一區間。
   但這**救不了 MMD²**（0.694，仍在「結構被摧毀」的水準）。

---

## 5. 對論文的實質發現

這一節是這次復刻真正的產出：問題不在誰贏，而在這個 benchmark 量不到它宣稱量的東西。

1. **任務本身是平凡的。** D2 在整個 1000-snapshot 視窗幾乎凍結：相鄰 anchor corr 0.993，
   最遠的兩張 snapshot 也有 0.929。因此 held-out τ 的「合成」可以用零參數解掉——
   線性插值 corr **0.99978**（RMS 只有場標準差的 1.9%），連「複製最近的 anchor」都有 0.99917。
   CTRL-interp 的 0.0012 / 0.0022 就是這個事實的量化。
2. **兩個指標都對時間排列不變。** 把 200 個 GT 場的 τ 標籤完全打亂後評分：**0.00000 / 0.00000**。
   一個把時間標籤全弄錯、但輸出正確場**集合**的模型可以拿滿分。Algorithm 1 的整個賣點
   （latent adaptation 讓模型能合成 unseen time）**從頭到尾沒有被測量**。
3. **PDF–JS 對空間結構完全不敏感。** 把每個 GT 場的像素在空間上打散 → PDF–JS **0.00000**。
   它是純粹的 contrast/直方圖檢定。各 arm 的場標準差（GT 0.446；DCGAN-1.2M 0.445、CTRL-interp 0.446、
   RBM 0.270、IQP 0.219、CTRL-marginal 0.118）幾乎完全解釋了 PDF–JS 的排序。
4. **Conv-MMD² 在 >0.70 飽和，且在該區間與 fidelity 反相關。** 空間打散的 GT 得 0.704；
   IQP 0.83、CTRL-marginal 0.82、RBM 0.71 全都在這條線上或之外。Q2c 的 order-truncation 掃描更直接：
   order-1 模型（corr 0.001）得 0.828，order-12 模型（corr 0.981）得 0.972——**越像反而分數越差**。
5. **論文 Table 2 的 MMD² 整欄都在飽和區。** 0.895–1.472，最好的 IQP 是 0.942；
   我們的「空間打散」參考點是 0.704、iid uniform noise 是 0.927、而 CTRL-interp 能到 0.002。
   *在我們對其 spec 的實作裡*，這個指標明明可以逼近 0，但論文所有模型都沒做到。
   （警告：MMD² 絕對值依賴 encoder 初始化與正規化，跨實作只能當指示，不是證明。）
6. **DCGAN@11 的「崩潰」不是 regime 效應，是未調的不穩定 baseline。** 11 張訓練圖彼此 99.3% 相關，
   且與每個 held-out 目標 99.9% 相關，所以**記住這 11 張就是最佳解**——而小 DCGAN 的近乎完全 mode collapse
   （0.2M@11 的 self-corr 0.9989）對只比較兩組特徵分布的指標**完全隱形**。
   我們三個 seed 裡確實有一個（1.2M@11 seed 2）不穩定崩掉（PDF–JS 0.153、MMD² 0.898、對 anchor 的 max-corr 只有 0.33），
   這也是 DCGAN-1.2M@11 那個 ±0.365 巨大 std 的來源。論文看起來是單次 run，很可能就是抽到這種 seed。
7. **作者自己在 Appendix I 承認 PDF–JS 可被騙**：「Without using feature space settings, it is possible that
   the score is very low while the generated image looks very different but with similar distribution of
   occurred values/colors.」我們的量化結果（像素打散 → 0.00000）正是這句話的極端證實；
   問題在於論文仍把 PDF–JS 當作 Table 2 的主指標之一，而它指定的另一個指標又處在飽和區。
8. **Algorithm 1 的 stage 2（latent adaptation）在它自己的 training anchor 上都量不到作用。**
   §3(d) B 在最乾淨的設定下（單 anchor、exact 場、其餘全部固定）：對 anchor 再跑 3000 步 latent adaptation，
   corr 從 0.807 動到 **0.808**、MMD² 完全不動（4.54e-05 → 4.54e-05）、pdf_js 反而變差（0.243 → 0.252）。
   這與第 2 點互相印證：論文的核心機制不只在**評估**上沒被測到（兩個指標對 τ 排列不變），
   在**訓練端**也沒有可量測的效果。真正吃掉分數的是 stage 1 把 θ_lat（含全部 18 個單體項）凍住這件事本身
   （約 −0.08 corr），而 stage 2 補不回來。
9. **QCLI 這個「資料集適不適合 IQP」的選題指標，在 n = 18 沒有配對 null 就不可解釋。**
   §3(e)：D2 的 pooled I_QCLI 0.167 看似落在論文的 IQP-compatible 象限，但同樣本數的均勻亂數
   在 M ≥ 10⁴ 時 QCLI 更高（全樣本 0.621 vs 0.167）。指標的有限樣本 null 遠離 0，因此
   「某資料集 QCLI ≈ 0.2，所以適合 IQP」這類論證，在沒有報告同 (n, M) 的 null 分布之前不能成立。
10. **論文的 baseline 數字，最可能對應弱化或未訓練完成的版本，不是我們重建有誤。** §3(f)
    （job 345214，單 seed）把論文沒寫的訓練配方掃過一輪：DCGAN@11 的 PDF–JS 數字能用**少訓練
    10–60 倍步數**（300–1000 步 vs 我們主跑的 20000 步）對上，但 MMD² 在任何步數都搆不到論文的
    1.40–1.45；RBM 掃過 latent 讀法（biases/literal）× 訓練配方（pcd_adam/cd1_sgd）四種組合，
    沒有一種能同時對上論文的兩個數字，論文 RBM 的 MMD² 1.472481 甚至高於我們的 iid uniform
    噪音場參考點（0.927，第 5 點）。這強化了第 6 點與 TL;DR 第 2 點的判斷：**論文的 IQP
    「優勢」，比較組看起來就是一組偏弱的 baseline**。

---

## 6. 誠實的限制

- **14 項自行決定的設計選擇，任何一項都可能是差距來源。** 影響最大的三個：
  (i) **tuple 來源**——我們從 Eulerian 網格點抽 tuple；Khojasteh et al. 的資料集同時含 Lagrangian 粒子軌跡，
  若論文其實是從粒子抽 tuple，資料 pmf 會完全不同（那會是稀疏、非均勻覆蓋的 (x,y)，其 Walsh 譜也不同）；
  (ii) **training kernel 的 σ**——原文完全沒給，我們用 Ref [38] 的規則，敏感度掃描顯示它能動 0.266 → 0.211；
  (iii) **聚合規則**——已證明它單獨就能造成 9× 的 PDF–JS 差異。
- **exact 模擬 vs 論文的 500-sample estimator**。我們去掉了估計雜訊。這只會讓我們的 IQP 更容易、不會更難，
  但也意味著我們沒有重現論文可能存在的「estimator 雜訊當正則化」效應。
- **跨實作的絕對數字不可直接比。** PDF–JS 的 log base、Conv2D encoder 的通道/初始化、量化區間、
  哪 100 個 snapshot、held-out τ 清單全部是我們自己選的。§2 的表格並排只是「為了記錄」，
  pre-reg 也明說判定看的是**排序**不是絕對值。
- **RBM 與 CTRL-marginal 在主跑沒有存 tuple**，所以 §3(c)（job 342514）的 last-write 重評對這兩個 arm 是
  **重新訓練後取樣**（RBM 的 PCD 在 GPU 上非確定性），不是同一組參數的重評；IQP 則是從存好的 θ 重取樣。
  結論不受影響：同一慣例下 RBM（0.0225 / 0.6384）仍全面贏 IQP。
- **§3(b) 的敏感度 arm 是單 seed**，其中 <3% 的差異不應解讀。
- **P5（D2 的 QCLI/CCI）已補跑（§3(e)），但結論是「這個指標測不出來」，不是「前提成立」。**
  數值上 D2 的 I_QCLI 0.167 / I_CCI 0.882 確實近論文說的 0.2，可是同樣本數的均勻亂數 QCLI 更高
  （0.621 vs 0.167），指標在 n = 18、M ≳ 10⁴ 的 null 遠離 0。要真的檢驗「turbulence 屬於
  IQP-compatible 象限」，需要配對 null 或解析的有限樣本修正——本次沒有做。
  所以論文選 turbulence 的理由**仍未被驗證，也未被推翻**。
- **§3(d) 的兩個協定因子沒有完全交叉**（缺「全參數 + 全 pool pmf、3×10⁴ 步」那一格），
  所以 −0.08 / −0.04 的拆分是近似歸因、不是精確的方差分解；且兩批都只跑**單一 anchor（t = 1）、單 seed**，
  §3(d) 內 <0.02 的 corr 差異不應解讀。
- **DCGAN 只跑 20k G-steps、last checkpoint、沒有調參**。它們贏 IQP 這件事因此更強（未調的 baseline 就贏了），
  但「DCGAN-1.2M 是最好的模型」這種話我們不能講。

---

## 7. 下一步

1. ~~把聚合規則做成全 arm 的開關並重評~~ **已完成**（§3(c)，job 342514，完整 200 τ、3 seeds）：
   聚合慣例確實能移動 PDF–JS 一個數量級（IQP 0.2700 → 0.0322），但它**同樣改善 RBM**（0.2139 → 0.0225），
   排序不變、not-reproduced 判定不變。往後任何跨模型的 PDF–JS 比較都必須先聲明聚合規則，
   而論文從未定義它——這一點應該回饋給原作者。
2. **換一個時間感知的 paired 指標**（例如 per-τ 的 corr / RMSE，或 energy spectrum / structure function 這種
   物理層級診斷）。現行兩個指標對 τ 排列不變，因此 Algorithm 1 的核心宣稱在現行設計下**原則上無法被檢驗**。
   注意論文自己在 Limitations 也承認 PDF–JS 與 Conv-MMD² 是 distribution-level proxy、不能取代物理診斷。
3. ~~先跑 anchor-fit kernel 實驗把「σ 未指定」與「模型族上限」分開~~ **已完成（§3(d)，job 342542/342685），
   而且結論是「不要再調 kernel」。** 四種 training kernel × 兩個 K 全部落在 corr 0.80–0.89，
   推高階（`flat`/σ0.15）比現行 `auto3` 還差，K = 7 只在 `flat` 下有用；每個 config 15k 步就到頂。
   **若目標是「讓 IQP 真的擬合 anchor」，該改的是協定而不是 kernel**：
   (i) stage 1 不要凍住 θ_lat（含 18 個單體項）——這是最大的單一因子（約 +0.08 corr）；
   (ii) 用全 pool pmf 取代 5×10⁴ 抽樣（約 +0.04）。兩者都放開之後仍只有 0.886，
   所以再往上走要換**模型族或目標函數**（closed-form Gaussian-kernel MMD 的低階偏差是結構性的），
   不是加參數（K = 6 → 7 沒有系統性幫助）。
   注意這一切都建立在「指標本身可信」之上——現行指標下「更像資料」不見得分數更好
   （order-12 的 MMD² 比 order-1 還差），所以**順序仍是先修指標（第 2 條）、再修 objective**。
4. **QCLI/CCI 要配 null 才能用**（§3(e)）。若要沿用論文「用 QCLI/CCI 判斷資料集適不適合 IQP」的論證，
   必須在**同一個 (n, M)** 下報告 i.i.d. null 的 QCLI 分布（或做有限樣本修正），比較的是
   「資料 vs 自己的 null」而不是「絕對值 vs 0.2」。在此之前這個指標不該用來選資料集——
   本次 D2（0.167）就低於同樣本數的均勻亂數（0.621）。

**對 DeepLense 影像用 IQP 的含意**：這次最可移轉的教訓不是「IQP 不行」，而是
**closed-form Gaussian-kernel MMD 這個 training objective 有結構性的低階偏差**——它把權重集中在 |s|≤3，
而自然影像（lensing 的 arc、substructure）的資訊同樣大量存在於高階 Walsh 係數。要把 IQP 用在 DeepLense 影像上，
先做一件便宜的事：對目標資料算 kernel-weighted 與 unweighted 的 Walsh 能量分布（就是 §4.4 那兩行數字）。
如果又是「99% 權重對上 7% 能量」，那不論電路多大都會停在低階截斷上，換 ansatz 或加參數沒用
（§3(d)：K = 6 → 7 沒有系統性幫助）。但要注意 §3(d) 也顯示**單純換 kernel 救不回來**——
四種 kernel 的 anchor corr 全在 0.80–0.89——所以真正要換的是 objective（或模型族），不是 kernel 的 σ。
另外，本次的 CTRL-interp 教訓直接適用：**任何 quantum generative 的宣稱都要先擺一個零參數的 trivial baseline**，
否則不知道自己贏的是模型還是任務的平凡性。

---

## 附錄：runs、路徑、指令

| Job | 內容 | Partition | 時間 | Log |
|---|---|---|---|---|
| **342220** | main：55 arms（IQP×6、RBM×3、DCGAN×18、scaling×18、sens×6、ctrl×4），0 failures | 8gpus (8 GPU) | 2026-09-05 17:10:40 → 17:42:35（31m55s） | `logs_d2/sbatch-342220.out` |
| **342454** | post-hoc 重建：15 arms × 2 modes = 30 rows，不重訓 | 8gpus (1 GPU) | 2026-09-05 18:09:22 → 18:14:08（4m46s） | `logs_d2/resynth-342454.out` |
| **342514** | §3(c) last-write 全 arm 重評：IQP/CTRL-marginal 由存好的 θ／replay 重取樣，RBM 為重訓後取樣（非確定性），200 τ | 8gpus (1 GPU) | 2026-09-05 18:22:03 → 18:25:22（3m19s） | `logs_d2/resynth_last-342514.out` |
| **342542** | §3(d) A：anchor t=1 的 kernel × K × lr 掃描（8 configs，全參數、全 pool pmf、1e5 步）＋ §3(e) 的 QCLI/CCI 診斷（同 job 的 CPU 階段） | 8gpus (8 GPU) | 2026-09-05 18:33:28 → 18:52:46（19m18s） | `logs_d2/anchor_fit*.log` |
| **342685** | §3(d) B：協定歸因（4 configs＝凍/不凍 θ_lat × 全 pmf/5×10⁴ 抽樣，含 Algorithm 1 stage 2） | 8gpus (4 GPU) | 2026-09-05 18:58:49 → 19:04:53（6m04s） | `logs_d2/anchor_fit_protocol*.log` |
| **345214** | §3(f) baseline 對齊：RBM latent×recipe 4 configs ＋ DCGAN@11 訓練步數掃描 3 tiers×4 steps，16 jobs，單 seed | 8gpus (8 GPU) | 2026-09-06 04:22:01 → 04:25:24（3m23s） | `logs_d2/sbatch-align-345214.out` |

（342454 的 sbatch 檔頭寫 `-p dev`，實際由 sacct 與 job log 確認跑在 `8gpus`；提交時被覆寫。帳號 <YOUR_ACCOUNT>。）

```
cd <HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication
sbatch submit_d2_main.sbatch          # -> 342220
sbatch submit_resynth.sbatch          # -> 342454
sbatch submit_resynth_last.sbatch     # -> 342514
sbatch submit_anchor_fit.sbatch       # -> 342542  (§3(d) A + §3(e))
sbatch submit_anchor_fit_protocol.sbatch  # -> 342685  (§3(d) B)
sbatch submit_baseline_align.sbatch   # -> 345214  (§3(f))
```

| 產物 | 路徑 |
|---|---|
| 主結果（55 rows JSONL） | `experiments/13_iqp_turbulence_replication/results_d2_main.jsonl` |
| 重建結果（30 rows JSONL） | `experiments/13_iqp_turbulence_replication/results_d2_resynth.jsonl` |
| 每個 arm 的 200 個生成場 + θ | `experiments/13_iqp_turbulence_replication/outputs_d2_main/<arm>/`、`outputs_d2_resynth/<arm>__<mode>/` |
| 訓練/評估 log | `experiments/13_iqp_turbulence_replication/logs_d2/` |
| §3(d) A 程式／輸出／log | `experiments/13_iqp_turbulence_replication/anchor_fit_experiment.py`；`outputs_anchor_fit/`；`logs_d2/anchor_fit.log`（含摘要表）、`logs_d2/anchor_fit_{auto3,auto3_K7,flat,flat_K7,flat_lr1e-3,sigma0.15,sigma0.3,sigma0.3_K7_lr1e-3}.log` |
| §3(d) B 程式／輸出／log | `experiments/13_iqp_turbulence_replication/anchor_fit_protocol.py`；`outputs_anchor_fit_protocol/`；`logs_d2/anchor_fit_protocol.log`（含摘要表）、`logs_d2/anchor_fit_protocol_{paper_stage1,paper_stage1_plus_stage2,paper_stage1_subsample5e4,allparams_subsample5e4}.log` |
| §3(e) QCLI/CCI 程式與結果 | `experiments/13_iqp_turbulence_replication/qcli_cci_d2.py`；`outputs_anchor_fit/qcli_cci_d2.json`（含 pooled、per-snapshot 與 i.i.d. uniform null） |
| §3(d)/(e) 的 sbatch | `submit_anchor_fit.sbatch`、`submit_anchor_fit_protocol.sbatch` |
| §3(f) 程式／輸出／log | `experiments/13_iqp_turbulence_replication/baseline_rbm_variants.py`、`test_baseline_variants.py`；`outputs_d2_baseline_align/`；`results_d2_baseline_align.jsonl`；`logs_d2/baseline_align_summary.log`、`logs_d2/sbatch-align-345214.out` |
| §3(f) 的 sbatch | `submit_baseline_align.sbatch` |
| D2 原始資料與 cache | `<WORK>/data/turbulence_khojasteh2022/`（`2D_velocity_snapshot_sub_domain_2.zip`, 4,116,034,340 B；`d2_cache/`） |
| 圖 | `docs/figures/iqp_turbulence_replication.png` |
| 原文 | `<WORK>/data/papers/2603.06440/paper.txt`、`paper.html` |

**圖說**（`docs/figures/iqp_turbulence_replication.png`）：左側 6 列 × 3 欄的場對照（GT / IQP-63k / RBM-12k /
DCGAN-1.2M@11 / CTRL-interp / CTRL-marginal，τ = 101, 502, 947，共用色階 0–1.70）。GT、DCGAN-1.2M 與 CTRL-interp
清楚呈現圓柱尾流結構；IQP 與 RBM 只剩一塊低值方形加上高頻雜訊；CTRL-marginal 是純雜訊。
右上是 200 個 hold-out τ 的像素強度直方圖（PDF–JS 實際在比的東西），顯示 tuple-sampling 系列
（IQP/RBM/CTRL-marginal，std 0.22/0.27/0.12）比 GT（0.446）少掉 50–75% 的 contrast，而在影像空間工作的
DCGAN 保住了（0.445）。右下是 metric-scale calibration bar chart，紅色虛線 0.704 = GT 自己被空間打散後的
Conv-MMD²，IQP/CTRL-marginal/RBM 全部在線上或線外。

**⚠ 讀 `logs_d2/evaluate.log` 表格時的注意事項**：那張表按 `(arm, train_snapshots)` 聚合，
所以 (i) `IQP-31k (11)` 那列把 3 個 seed **和 6 個敏感度 arm** 一起平均（0.2486 ± 0.0299），
真正的 3-seed 值是 **0.2700 ± 0.0030**；(ii) `DCGAN-* (100)` 那三列把 3 個 seed **和 scaling 的單 seed n=100 run**
一起平均（4 個 run）。本文件 §2 的所有數字都是直接從 JSONL 依 `run_dir` 重算的 3-seed 值，不是那張表的值。
