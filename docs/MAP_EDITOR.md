# 自訂地圖：編輯、分享與訓練

**適用前提：所有使用者都已依 [GDG_FlyMaze 原專案](https://github.com/changshencoding/GDG_FlyMaze) 完成 Ubuntu/WSL2、Python 3.12、uv 與訓練環境安裝，既有專案位於 `~/GDG_FlyMaze`。** 這份說明是在該環境加入自訂地圖；不需要重新 clone，也不需要重裝 FlyGym／PyTorch。

原專案環境安裝與自訂地圖擴充安裝是兩件事。首次使用 JSON 地圖，先完成下方第 1 步；已裝過擴充或已同步本次程式碼的人從第 2 步開始。

沒有指定自訂地圖時，原四張地圖、七維 observation、動作、reward 與成功白名單維持原本設定。`test_straight_unknown` 仍保留作 held-out test。

## 下載內容

| 檔案 | 用途 |
|---|---|
| [custom_race_a.json](../maps/custom_race_a.json) | 地圖 A：8×6 格、12 格通道、最短路線 110 mm |
| [custom_raceb.json](../maps/custom_raceb.json) | 地圖 B：10×8 格、45 格通道、最短路線 220 mm，含支路 |
| [FlyMaze_Map_Editor.zip](../downloads/FlyMaze_Map_Editor.zip) | 安裝到既有原專案的一次性擴充；包含兩張地圖、編輯器及範例 |

JSON 檔可在 GitHub 按 **Raw** 另存。ZIP 請按 **Download raw file**。

兩張使用者提供的地圖均保留原始 `map_id`、通道與 `timeout_seconds=14.0`。路線較長不代表時限會自動加長；如需調整時限，請儲存副本，並記錄新的實驗設定。

## 1. 首次使用：在既有原專案加裝擴充

將上面的 ZIP 下載到 Ubuntu 的 `~/Downloads`，然後執行：

```bash
cd ~/Downloads
unzip FlyMaze_Map_Editor.zip
python3 FlyMaze_Map_Editor/install_map_editor.py ~/GDG_FlyMaze
```

安裝程式會核對相容的原版 `maps.py`，備份後加入地圖支援、編輯器、範例與測試。同名檔案若內容不同會停止，避免覆蓋後續修改。ZIP 已包含 A、B 地圖，安裝成功後兩張 JSON 會在原專案的 `maps/`，不用另行複製。

原始程式不會讀取自訂地圖的兩個環境變數；尚未安裝擴充就直接訓練，仍會使用原本地圖。這個擴充只安裝一次，以後分享地圖只需 JSON。

## 2. 檢查已加入的兩張地圖

完成擴充安裝後，在原本的專案根目錄執行：

```bash
cd ~/GDG_FlyMaze
uv run python -m flymaze.map_editor --check maps/custom_race_a.json
uv run python -m flymaze.map_editor --check maps/custom_raceb.json
```

若擴充已安裝或已同步本次程式碼，可直接從本節開始。格式與連通檢查只確認地圖定義有效，不等於目前模型能通關。

## 3. 使用自訂地圖從頭訓練

在 A 地圖從頭訓練：

```bash
cd ~/GDG_FlyMaze
FLYMAZE_MAP_FILE="$PWD/maps/custom_race_a.json" \
FLYMAZE_TRAIN_MAP=custom_race_a \
uv run flymaze train-ppo \
  --config configs/ppo.toml \
  --output-directory artifacts/ppo-custom-race-a
```

在 B 地圖從頭訓練：

```bash
cd ~/GDG_FlyMaze
FLYMAZE_MAP_FILE="$PWD/maps/custom_raceb.json" \
FLYMAZE_TRAIN_MAP=custom_raceb \
uv run flymaze train-ppo \
  --config configs/ppo.toml \
  --output-directory artifacts/ppo-custom-race-b
```

原本 scratch PPO 的入口使用 `train_l_left`。為維持訓練程式，本擴充只在明確設定兩個變數時，把這個 registry 槽指向自訂地圖；地圖本身的 ID 保持 `custom_*`。訓練評估也使用同一張自訂地圖，而 held-out 測試仍使用原本的 `test_straight_unknown`。

| 參數 | 要填的內容 |
|---|---|
| `FLYMAZE_MAP_FILE` | 實際 JSON 的絕對路徑 |
| `FLYMAZE_TRAIN_MAP` | JSON 裡的 `map_id`，必須一致 |
| `--output-directory` | 本次結果目錄，建議使用新資料夾 |

環境變數只作用於該次命令與它產生的子程序。載入時會印出 JSON 絕對路徑與 SHA-256。以上是從頭訓練，不是接續既有 checkpoint 的指令。

如將上述變數用於 `train-multitask-ppo`，只會替換其中的左彎訓練槽，另兩張地圖仍沿用原設定；建議先驗證上面的 scratch PPO 流程。

## 4. 之後分享新地圖：只需傳 JSON

大家完成第 1 步的首次擴充安裝後，將新地圖 JSON 傳給對方即可，不必再傳編輯器，也不必改 Python。假設對方下載到 `~/Downloads`：

```bash
cd ~/GDG_FlyMaze
mkdir -p maps
cp ~/Downloads/custom_friend.json maps/
uv run python -m flymaze.map_editor --check maps/custom_friend.json

FLYMAZE_MAP_FILE="$PWD/maps/custom_friend.json" \
FLYMAZE_TRAIN_MAP=custom_friend \
uv run flymaze train-ppo \
  --config configs/ppo.toml \
  --output-directory artifacts/ppo-custom-friend
```

`custom_friend` 是範例名稱，請換成實際 JSON 裡的 `map_id`。`configs/ppo.toml` 可以沿用，只有要調整訓練參數時才需要改。

## 還原首次安裝前的載入方式

要還原安裝前的載入方式，在解壓縮的套件旁執行：

```bash
python3 FlyMaze_Map_Editor/install_map_editor.py ~/GDG_FlyMaze --restore
```

還原只恢復 maps.py，保留自訂 JSON 與編輯器檔案。若安裝後又手動修改 maps.py，還原會停止，避免覆蓋後續工作。

## 用現有模型查看新地圖

此時只需要 `FLYMAZE_MAP_FILE`，不需要替換訓練槽：

```bash
FLYMAZE_MAP_FILE="$PWD/maps/custom_race_a.json" \
uv run flymaze visualize-ppo \
  --model-path models/ppo_flymaze_mvp.zip \
  --map-id custom_race_a --persist --show-odor
```

命令列評估：

```bash
FLYMAZE_MAP_FILE="$PWD/maps/custom_race_a.json" \
uv run flymaze evaluate-ppo \
  --model-path models/ppo_flymaze_mvp.zip \
  --map-id custom_race_a --seeds 0 1 2 \
  --output artifacts/custom_race_a_evaluation.json
```

新訓練的模型改用對應的 `artifacts/ppo-custom-race-a/ppo_flymaze.zip`。模型不會包含地圖 JSON；請保留同一份 JSON，必要時複製到訓練結果旁備份。

## 開啟地圖編輯器

```bash
uv run python -m flymaze.map_editor --no-browser
```

開啟終端機顯示的完整網址（包含 token），按 Ctrl+C 結束服務。

1. 從「已儲存的地圖」載入 A／B，或使用範本重新設計。
2. 拖曳繪製通道，用右鍵或擦除工具刪除。放置起點與食物。
3. 2D 顯示相對起點的 mm 座標與 BFS 最短路線；3D 只是外觀預覽，沒有執行果蠅物理。
4. 按「儲存至專案」寫入 maps/，同名檔案會先確認覆寫。
5. 也可匯出 JSON 分享，或匯入收到的 JSON 繼續編輯。

快捷鍵：1 通道、2 擦除、3 起點、4 食物、Ctrl+Z 復原、Ctrl+Shift+Z 重做。

## 尺度與地圖生成

- 單層水平格狀通道，寬高各 3–12 格。
- 每格固定 10×10 mm，牆厚 1 mm、高 3 mm；不支援 2×2 全開房間。
- 所有通道須透過共用邊相連；對角相鄰不算連通。
- 起點換算成世界座標 `(0, 0, 0.8)`，果蠅朝 +X，起點右方必須有通道。
- 食物位於另一個通道格中心；每回合時限為 1–300 秒。
- ID 以 `custom_` 開頭，不接受四張原地圖的名稱。
- 有多條路時取確定性的 BFS 最短路線，沿用原 polyline reward shaping；其他路線可能獲得不同塑形獎勵。

| 內容 | 來源 |
|---|---|
| 牆壁 | 通道格未相連的外邊，合併共線邊後建立 MuJoCo box |
| 物理接觸 | 原 `add_explicit_wall_pairs` 與原碰撞身體範圍 |
| 拓撲 | 每格一節點，共用邊產生水平／垂直連線 |
| 路線 | BFS 最短路線轉為原本 PolylineRoute |
| 氣味 | 相同格子產生自由空間遮罩；保留 0.25 mm 網格與原擴散參數 |
| 食物成功 | 原 food_world_configurator 與 head／左右 forefeet 白名單 |

## 驗證狀態

製作端以 Python 3.12 執行標準函式庫測試、前端狀態測試、語法檢查與安裝／還原檢查。前端狀態測試使用 DOM/canvas test double，並非真實瀏覽器視覺驗收。

氣味求解與 MuJoCo 牆壁編譯採用實際數值函式庫；匯入用 FlyGym 型別替身不等同完整果蠅模擬。因此尚未宣稱兩張地圖能通關，也尚未完成 Ubuntu 端的實體果蠅 reset/step、完整 PPO、release checkpoint regression 與建置驗收。正式訓練前請在 Ubuntu 執行專案要求的完整驗證。

可先執行相關測試：

```bash
uv run pytest tests/unit/test_maps.py tests/unit/test_custom_maps.py \
  tests/unit/test_map_editor.py tests/unit/test_shared_maps.py \
  tests/integration/test_custom_map_env.py
```
