# JanusMesh：合併 `trellis` 與 `synctweedies` 成單一環境

目標：clone 後只建立 **一個** conda 環境（例如 `janusmesh`），同時能跑 `example_text.py` 與 `SyncTweedies/main.py`。

## 為什麼不能「全自動貼上就好」

- 兩邊若 **Python 主版本不同**（例如 3.8 vs 3.10），幾乎一定要選一邊再重裝另一邊的套件。
- **PyTorch / CUDA** 與 `spconv`、`cupy-cuda11x` 等必須同一條 CUDA 線，否則 runtime 會炸。
- `conda env export` 會帶 **`prefix:`** 與大量 **build string**，不適合直接 commit；合併前應刪除 `prefix`，並盡量用 **`--from-history`** 或手動精簡。

建議流程：**先匯出快照 → 再合併成一份「你機器上驗證過」的 `environment.yml` +（可選）`requirements-janusmesh.txt`**。

---

## 步驟 1：在你本機匯出兩個現有環境

在 repo 根目錄執行（會寫入 `environment/exports/`，已列在 `.gitignore` 可選；若要 commit 快照可改掉 gitignore）：

```bash
cd /project2/siangling/JanusMesh
bash scripts/export_env_snapshots.sh trellis synctweedies
```

預設環境名稱就是上面兩個；若你的 env 名字不同：

```bash
bash scripts/export_env_snapshots.sh 你的TrellisEnv 你的SyncEnv
```

產物說明：

| 檔案 | 用途 |
|------|------|
| `*_from_history.yml` | **較適合當合併基底**（只記你曾 `conda install` 的規格，較短） |
| `*_full.yml` | 完整鎖版本，給自己備份／對 diff |
| `*_pip_freeze.txt` | 該環境 `pip list --format=freeze`，合併 pip 依賴時用 |

---

## 步驟 2：合併成 `janusmesh`

### 作法 A：手動（最穩，推薦第一次）

1. 新建測試環境：  
   `conda create -n janusmesh python=3.xx`（**建議選 Trellis 實際在用的 Python 版本**，再拉 SyncTweedies 的 pip 套件上來測）。
2. 依 **Trellis** 需求裝 PyTorch / spconv / flash-attn / xformers 等（以你現在能跑 Trellis 的指令為準）。
3. 依 **SyncTweedies** 的 `SyncTweedies/environment.yml` 裡 **`pip:`** 區塊，分批 `pip install`；遇到衝突就 **只升級/降級衝突包**，直到 `python example_text.py ...` 與 `conda run -n janusmesh python SyncTweedies/main.py ...` 都能跑。
4. 驗證通過後匯出：  
   ```bash
   conda activate janusmesh
   conda env export --from-history > environment.yml
   pip freeze > requirements-janusmesh.txt
   ```  
   編輯 `environment.yml`：**刪除最後的 `prefix:`**，再 commit。

### 作法 B：`conda-merge`（自動合併兩份 yml）

```bash
conda activate base
pip install conda-merge
conda-merge environment/exports/trellis_from_history.yml environment/exports/synctweedies_from_history.yml > environment-merged-raw.yml
```

接著打開 `environment-merged-raw.yml`：**刪 `prefix:`、解決重複的 python/torch 行**，存成根目錄的 `environment.yml`（`name: janusmesh`），再：

```bash
conda env create -f environment.yml
```

若合併檔仍過長，可只保留 `channels` + `dependencies` 裡的 **python / 少數 conda 包**，其餘改放到 `pip:` 或 `requirements-janusmesh.txt`。

---

## 步驟 3：程式裡改用單一環境名稱

合併成功後，把 `example_text.py` 裡 `run_synctweedies_mesh(..., conda_env_name="synctweedies")` 改成 **`"janusmesh"`**（或你實際取的環境名）。

更進階：若 Trellis 與 SyncTweedies **永遠同一個 interpreter**，可改成用 **`sys.executable`** 直接 `subprocess` 呼叫 `SyncTweedies/main.py`，就不必 `conda run`（可再優化）。

---

## 給 repo 使用者的最終指令

在 README 寫：

```bash
git clone <你的 repo>
cd JanusMesh
conda env create -f environment.yml
conda activate janusmesh
python example_text.py --help
```

若你選擇 **pip 補充檔**：

```bash
conda activate janusmesh
pip install -r requirements-janusmesh.txt
```

---

## CUDA 版本注意

- SyncTweedies 範例 yml 使用 **CUDA 11** 相關 pip（如 `nvidia-cublas-cu11`、`cupy-cuda11x`）。若 Trellis 使用 **CUDA 12**，合併時需改成同一條線的套件，**不要混 11 與 12**。

---

## 合併 pip 時常見錯誤（你遇到的兩個）

### 1) `diff_gaussian_rasterization==0.0.0`：PyPI 沒有

`0.0.0` 幾乎一定是 **在本機從原始碼編譯安裝** 的（3D Gaussian Splatting 的 rasterizer），**不會出現在 PyPI**。

**作法：**從 `trellis` 的 freeze 清單裡 **刪掉這一行**，在已裝好 **torch + CUDA** 的 `janusmesh` 裡改裝官方來源：

```bash
conda activate janusmesh
# 必須先已安裝 torch；--no-build-isolation 讓編譯階段能 import 到環境裡的 torch
pip install "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git" --no-build-isolation
```

若編譯失敗，確認機器有 **nvcc**（CUDA toolkit），必要時設好 `CUDA_HOME`。若仍報錯，可再試加上與你 GPU 相符的架構，例如：`export TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6"`（依機器調整）。

#### `The detected CUDA version (13.x) mismatches ... PyTorch (11.8)`

代表 **PATH 裡的 `nvcc` 版本** 和 **PyTorch 編譯時用的 CUDA（例如 11.8）** 不一致。請二選一（優先 A）：

**A. 在 conda 環境內裝與 PyTorch 一致的 nvcc（11.8），並強制用這套編譯**

```bash
conda activate janusmesh
conda install -n janusmesh cuda-nvcc=11.8 -c nvidia -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
nvcc --version   # 應顯示 release 11.8
pip install "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git" --no-build-isolation
```

若 `cuda-nvcc=11.8` 解析失敗，可改試：`conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y`（仍以官方文件為準）。

**B. 若機器已有 `/usr/local/cuda-11.8`（或 module 提供 cuda/11.8）**

```bash
export CUDA_HOME=/usr/local/cuda-11.8   # 改成你機器上實際路徑
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
pip install "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git" --no-build-isolation
```

不要讓 **CUDA 13 的 nvcc** 留在 PATH 最前面又去編 **cu118** 的 PyTorch extension。

### 2) `igl==2.2.1`：多半是 **conda-forge** 的套件，不是 pip

舊的 `synctweedies` 環境裡 `igl` 常來自 **`conda install -c conda-forge igl`**，對 pip 來說會變成「找不到 wheel」。

**作法：**從 `synctweedies` 的 freeze 清單裡 **刪掉 `igl==...` 那一行**，改由 conda 裝：

```bash
conda install -n janusmesh -c conda-forge igl -y
```

### 3) 重新產生過濾後的 `pip` 清單（範例）

在 repo 根目錄、`EXPORTS` 指到 `environment/exports` 的前提下：

```bash
grep -Ev '^(torch|torchvision|torchaudio|nvidia-|triton)(==|@)' \
  "$EXPORTS/trellis__pip_freeze.txt" | grep -v '^diff_gaussian_rasterization==' \
  > /tmp/janusmesh_trellis_pip.txt

grep -Ev '^(torch|torchvision|torchaudio|nvidia-|triton)(==|@)' \
  "$EXPORTS/synctweedies__pip_freeze.txt" | grep -v '^igl==' \
  > /tmp/janusmesh_st_pip.txt
```

**建議安裝順序：**先 **`conda install igl`**（若尚未裝）→ **`pip install` diff-gaussian-rasterization（git）**（若尚未裝）→ 再 **`pip install -r /tmp/janusmesh_trellis_pip.txt`** → 最後 **`pip install -r /tmp/janusmesh_st_pip.txt`**。

其他在 freeze 裡出現 **`==0.0.0`** 的套件，多半也是本地編譯，同樣要 **從清單移除後改用手動 / git 安裝**。

若你需要，我可以依你匯出的兩份 `*_from_history.yml` + `*_pip_freeze.txt` 幫你做一次「合併草稿」（你貼檔案或放在 `environment/exports/` 後叫我讀）。
