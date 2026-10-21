# Concept XAI: Concept-Embedded Explainable AI Model
<img width="890" height="475" alt="grid_batch_010" src="https://github.com/user-attachments/assets/69d6aac7-9401-4f3c-8689-777c3f5d646e" />

This is a prototype implementation of a Concept-Embedded Explainable AI (XAI) model designed to solve the ultimate dilemma of conventional deep learning: "We know *Where* it looked, but we don't know *What* it saw."

While achieving a 91% test accuracy on the MNIST dataset, this model combines backward propagation (BP) during inference with a pre-trained "Concept Extraction Layer" to directly and visually output which geometric parts (meanings) the AI relied on to reach its conclusion.

## 💡 Background & Motivation
In mission-critical fields such as medical diagnosis and infrastructure/machinery failure detection, applying black-box deep learning models directly is unacceptable due to the high risk of misclassification.
- Limitations of Conventional DL: While techniques like Grad-CAM can show where the image was observed, there is no direct mechanism to decode whether the extracted features correspond to meaningful shapes (What). Even with standard DL where BP identifies important weight positions, the exact semantic meaning of those weights remains unknown.
- Black-box Internal Reasoning: Traditionally, the internal logical process of deep learning—such as determining that an image represents the digit "1" because it contains two consecutive vertical bars—has been completely opaque to human observers.

## Core Approach & Key Findings
By placing a "Geometric Feature Extraction Layer (Concept Embedding Layer)" as the first layer, this architecture avoids learning arbitrary features via standard BP at runtime. Instead, it utilizes pre-trained, fixed models designed to extract general geometric concepts universally.

### 1. Visualization of Human-like Thought Processes
By embedding pre-trained concepts and linking them with inference-time gradients, we successfully decoded the previously invisible internal reasoning process of the model.
Our evaluation revealed that the AI does not simply look at the image holistically; rather, it identifies digits based on combinations of a few local features (geometric primitives). For instance, when classifying the digit "1," the model internally captures elements like "multiple vertical bars forming a stroke," visually proving that the AI performs structural cognition and reasoning much closer to humans.

### 2. New Possibilities for Explainable AI (XAI)
This approach completely translates extracted features into human-interpretable geometric concepts (lines, crosses, X-shapes, L-shapes, etc.), demonstrating the powerful potential that embedding pre-trained concepts directly into models can successfully bridge high performance and true explainability.

## 🏗️ Architecture Overview
<code>
[Input Image: 28x28]
↓ (Dynamic Patch Extraction: unfold)
[Layer 1: Geometric Feature Extraction Layer (Pre-trained & Fixed)]
↓ (16-32 dim Spatial Concept Map)
[Layer 2: Spatial Concept Processing Block (Spatial Concept Network)]
↓
[Output: Classification & XAI Automated Audit (Why / Because)]
</code>
       
### 1. Layer 1: Geometric Filters & Concept Dictionary (Pre-trained & Fixed Base)
Instead of letting the model learn black-box features freely, it uses filter groups that universally extract predefined, refined geometric and topological concepts.
#### Input Size: $7 \times 7 \times 1$ patch
#### Covered Concepts (10 Classes):
- Line Types (4): Vertical line, Horizontal line, Diagonal up, Diagonal down
- Intersection & Special Types (2): Cross (+), X-shape (×)
- L-Shape Types (4): L-shape (Top-Left), L-shape (Top-Right), L-shape (Bottom-Right), L-shape (Bottom-Left)

### 2. Layer 2 & Beyond: Spatial Concept Processing Block
#### Block Characteristics
Accepts a spatial concept map $(B, C_{\text{concept}}, H_{\text{grid}}, W_{\text{grid}})$ that preserves the 2D spatial extension and stroke topology of characters, processing them via fully connected layers and pooling mechanisms.

#### Advantages
Breaks through the limitations of "Bag-of-Concepts" (frequency-only approaches). By performing layout learning that preserves spatial relationships, it achieves high accuracy (MNIST test accuracy 91%) with a lightweight structure.

## 🔍 XAI & Automated Audit
By analyzing spatial gradients obtained from backward propagation (BP) during inference on trained models, the reasoning behind the model's decisions is extracted visually and quantitatively.

### Automated Output Management (space/output/)
- Grid Images (grid_batch_XXX.png): Pairs original images with mapped visualizations showing which geometric concepts triggered the model's decisions.
- Detailed Metadata (JSON): Fully records the coordinates of each detected bounding box, maximum gradient contribution, Top-3 concept labels, and their confidence scores (Softmax probabilities).

## 📁 Directory Structure
<code>
CXAI-Model
│   app.toml
│   pyproject.toml
│   README.md
│   run_cxai.bat
│
├── dataset         # MNIST binary data
├── log             # Execution logs
├── space           # Checkpoints and output directory
│   ├─ label_image      # Base concept reference images (5x scaled)
│   └─ output           # Visualization grid images (.png) & inference logs (.json)
└─ src
    └─ cxai
        │   main.py         # CLI & interactive menu controller
        ├── com             # Configuration & common modules
        └─ core             # Model & pipeline implementation
            ├── gnn_main_model.py
            ├── gnn_main_model_helper.py
            ├── gnn_preset_model.py
            └── mnist_dataloader.py
</code>

## 🚀 Usage (CLI Menu Execution)
This system can be easily run from the repository root using an interactive CLI menu or command-line arguments.

### 1. Startup Method
python -m src.cxai.main

Or use the provided batch file (Windows):
run_cxai.bat

### 2. Interactive Menu
Once launched, choose an option to execute step-by-step:

<code>
=================================================================
    Concept XAI - MNIST Step-by-Step Verification System
=================================================================
    [1] Pre-train geometric feature filters and concept dictionaries
    [2] Train the MNIST integrated model
    [3] Test and evaluate the MNIST classification model (achieves 91% accuracy)
    [4] Evaluate training data, map gradients, and output grid images and JSON files
    ---------------------------------------------------------
    [9] Initialize (reset) all checkpoints and caches
    [q] Quit
=================================================================
Enter choice:
</code>
       
- [1]: Pre-train geometric feature filters and concept dictionaries
- [2]: Train the MNIST integrated model
- [3]: Test and evaluate the MNIST classification model (achieves 91% accuracy)
- [4]: Evaluate training data, map gradients, and output grid images and JSON files
- [9]: Initialize (reset) all checkpoints and caches

## 🗺️ Future Prospects
- Medical Image Diagnosis: Detecting lesions based on strict morphological evidence required by physicians (such as specific contours or closed-loop structures) in cellular images and tissue scans.
- Industrial & Medical Equipment Fault Diagnosis: Defining micro-fracture or anomaly patterns in mechanical parts and biological tissues as "constructive geometric concepts" to predict failures with evidence.
- Dynamic Module Extension: Developing an adaptive AI foundation where upper-layer concept filters can be dynamically switched and extended depending on the diagnostic target.

## Execution result
### Geometric Concept Model Training
<code>
🚀 [Geometric Train] Starting training with 10 Clean Geometric Concepts (50 epochs)...
📁 Saved 10 centralized base concept reference images (5x enlarged) to: D:\workspace\CXAI-Model\space\label_image
  - Total samples (augmented): 624
Epoch [1/50] Loss: 2.2000 | Train Acc: 34.91% | Val Acc: 22.34%
Epoch [5/50] Loss: 0.8222 | Train Acc: 75.85% | Val Acc: 88.30%
Epoch [10/50] Loss: 0.4202 | Train Acc: 86.98% | Val Acc: 96.81%
Epoch [15/50] Loss: 0.2415 | Train Acc: 92.08% | Val Acc: 95.74%
Epoch [20/50] Loss: 0.2074 | Train Acc: 93.77% | Val Acc: 95.74%
Epoch [25/50] Loss: 0.1807 | Train Acc: 93.96% | Val Acc: 95.74%
Epoch [30/50] Loss: 0.1431 | Train Acc: 95.85% | Val Acc: 95.74%
Epoch [35/50] Loss: 0.1507 | Train Acc: 94.53% | Val Acc: 95.74%
Epoch [40/50] Loss: 0.1374 | Train Acc: 95.85% | Val Acc: 95.74%
Epoch [45/50] Loss: 0.1407 | Train Acc: 94.91% | Val Acc: 95.74%
Epoch [50/50] Loss: 0.1369 | Train Acc: 95.28% | Val Acc: 95.74%

=== Geometric Concept Dictionary Test ===

--- Per-Class Accuracy (acc[label]) ---
  cross                :  66.67% 🔴 WEAK
  l_shape_dr           :  85.71% 🟡 FAIR
  horizontal_line      :  90.91% 🟢 GOOD
  vertical_line        : 100.00% 🟢 GOOD
  diag_up              : 100.00% 🟢 GOOD
  diag_down            : 100.00% 🟢 GOOD
  x_shape              : 100.00% 🟢 GOOD
  l_shape_ul           : 100.00% 🟢 GOOD
  l_shape_ur           : 100.00% 🟢 GOOD
  l_shape_dl           : 100.00% 🟢 GOOD

--- Confusion Matrix & Misclassification Analysis ---
  horizontal_line  ==>  misclassified as  l_shape_ul  (1 times)
  cross  ==>  misclassified as  l_shape_dl  (1 times)
  l_shape_dr  ==>  misclassified as  vertical_line  (1 times)
  l_shape_dr  ==>  misclassified as  horizontal_line  (1 times)
✅ Geometric filter saved to: D:\workspace\CXAI-Model\space\checkpoint_geometric_filter.pth
✅ Concept dictionary model saved to: D:\workspace\CXAI-Model\space\checkpoint_concept_dictionary.pth
</code>

## MNIST Integrated Model Training
<code>
Loading Training dataset...
Data loaded successfully (Total samples: 60000, Shape: 28x28)
Loading Test dataset...
Data loaded successfully (Total samples: 10000, Shape: 28x28)
💡 Training with 60000 samples (Configured max: -1).
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000100.json (Detected: 6 objects)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000200.json (Detected: 10 objects)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000300.json (Detected: 6 objects)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000400.json (Detected: 5 objects)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000500.json (Detected: 5 objects)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000600.json (Detected: 8 objects)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000700.json (Detected: 9 objects)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000800.json (Detected: 3 objects)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000900.json (Detected: 8 objects)
Epoch [1/10]: 100%|██████████████████████| 938/938 [00:25<00:00, 36.83it/s, acc=49.22%, loss=1.4230]
👉 Epoch 1 Completed | Loss: 1.4230 | Accuracy: 49.22%
Epoch [2/10]: 100%|██████████████████████| 938/938 [00:25<00:00, 36.96it/s, acc=69.06%, loss=0.9160]
👉 Epoch 2 Completed | Loss: 0.9160 | Accuracy: 69.06%
 * * * *
Epoch [10/10]: 100%|█████████████████████| 938/938 [00:25<00:00, 37.46it/s, acc=83.86%, loss=0.5097]
👉 Epoch 10 Completed | Loss: 0.5097 | Accuracy: 83.86%
✅ Integrated classification model saved to: D:\workspace\CXAI-Model\space\checkpoint_mnist_spatial_nn.pth
[2026/10/21 21:55:00] [INFO] === [MNIST Classification Model Training] Completed ===
</code>
       
## MNIST Model Testing & Evaluation
<code>
Loading Training dataset...
Data loaded successfully (Total samples: 60000, Shape: 28x28)
Loading Test dataset...
Data loaded successfully (Total samples: 10000, Shape: 28x28)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000100.json (Detected: 6 objects)
Evaluating: 100%|█████████████████████████████████████████████████| 157/157 [00:03<00:00, 47.33it/s]

📊 [Test Results] Overall Accuracy: 92.35% (Evaluated samples: 10000)

--- Per-Class Test Accuracy (MNIST Digits 0-9) ---
  Digit 0 :  96.73%  (980 samples)
  Digit 1 :  97.97%  (1135 samples)
  Digit 2 :  91.96%  (1032 samples)
  Digit 3 :  94.85%  (1010 samples)
  Digit 4 :  95.82%  (982 samples)
  Digit 5 :  89.24%  (892 samples)
  Digit 6 :  93.95%  (958 samples)
  Digit 7 :  91.05%  (1028 samples)
  Digit 8 :  81.21%  (974 samples)
  Digit 9 :  89.59%  (1009 samples)
[2026/10/21 22:03:07] [INFO] === [MNIST Classification Model Testing] Completed ===
</code>

## CXAI Model Thought Process Visualization
<code>
🧹 Cleared Output directory: D:\workspace\CXAI-Model\space\output
Loading Training dataset...
Data loaded successfully (Total samples: 60000, Shape: 28x28)
Loading Test dataset...
Data loaded successfully (Total samples: 10000, Shape: 28x28)
🚀 Processing Linear-input gradient mapping across 1000 samples...
Mapping & Exporting:   1%|▍                                                           | 7/1000 [00:00<00:32, 30.46it/s]  🖼️ Flushed 10-item grid image: D:\workspace\CXAI-Model\space\output\grid_batch_001.png
Mapping & Exporting:   2%|▉                                                          | 15/1000 [00:00<00:29, 33.92it/s]  🖼️ Flushed 10-item grid image: D:\workspace\CXAI-Model\space\output\grid_batch_002.png
Mapping & Exporting:   3%|█▋                                                         | 28/1000 [00:00<00:27, 35.07it/s]  🖼️ Flushed 10-item grid image: D:\workspace\CXAI-Model\space\output\grid_batch_003.png
 * * * *
Mapping & Exporting: 100%|█████████████████████████████████████████████████████████| 1000/1000 [00:26<00:00, 38.39it/s]
</code>

[日本語版Readme]
# Concept XAI: 概念埋め込み型・解釈可能AIモデル
<img width="890" height="475" alt="grid_batch_010" src="https://github.com/user-attachments/assets/dc0fbc90-a398-426e-a339-7d4b6f3e8b51" />

従来のディープラーニングが抱える最大のジレンマである「Where?（どこを見たか）は分かっても、What?（何を見たのか）が分からない」というブラックボックス問題を解決するための、幾何学的概念埋め込み型・解釈可能AI（XAI）のプロトタイプ実装です。

本モデルは、MNISTデータセットを用いた検証においてテスト精度91%を達成しつつ、推論時の逆伝播（BP）と事前学習済みの「概念抽出レイヤー」を組み合わせることで、AIが「どの幾何学パーツ（意味）を見てその結論に至ったか」を視覚的・定量的に直接出力します。

## 💡 背景と開発動機
医療診断やインフラの故障検知といったミッションクリティカルな分野では、中身がブラックボックスなディープラーニングモデルをそのまま適用することは、誤判定のリスクが高く許容されません。
- 従来のDLの課題: Grad-CAM等で「画像のどこを見たか（Where）」は可視化できても、抽出された特徴が「どのような意味を持つ形状（What）」なのかを重みベクトルから直接デコードする手段（ベクトル $\iff$ ラベル変換）がありませんでした。通常のDLでもBPによって重要な重みの位置は分かりますが、その重みが「何を意味するのか」の対応関係が分からないという課題があります。
- モデル内推論のブラックボックス性: これまで、ディープラーニングが画像から何を読み取ってその数字だと判断しているのか（例えば「画像の中に縦棒が2本続いているから『1』である」といった内部の論理プロセス）は、人間の目には見えませんでした。

## 本アーキテクチャのアプローチと新たな発見
第1層に「幾何学特徴抽出レイヤー（概念埋め込み層）」を配置。この第1層はBPによってその都度適当な特徴を学習するのではなく、あらかじめ汎用的な幾何学概念を学習・固定されたモデルとして構築しています。

### 1. 人間に近い思考プロセスの可視化
モデル内にあらかじめ学習済みの概念を埋め込み、推論時の勾配と結びつけることで、これまで不可視だったモデル内の推論プロセスを完全に解読することに成功しました。
検証の結果、本AIは画像全体をぼんやりと見るのではなく、「画像の中にある数個の局所的な特徴（幾何学パーツ）の組み合わせ」から数字を言い当てていることが明らかになりました。例えば、数字の「1」を判定する際、「縦棒が複数並んでいる（またはストロークが構成されている）」という要素をモデル自身が内部で捉えて推論していることが視覚的に証明され、AIが人間により近い思考や構造的認知を行っていることが分かります。

### 2. 説明可能なAI（XAI）の新たな可能性
このアプローチにより、抽出された特徴量を人の目で見える「幾何学的概念（直線、十字、バツ、L字等）」に完全に翻訳することが可能となり、「モデル内に学習済みの概念を埋め込むことで、高精度と説明可能なAI（XAI）が十分に構築可能である」という強力な可能性を示しています。

## 🏗️ アーキテクチャ概要
<code>
[入力画像: 28x28]
       │
       ▼ (動的パッチ切り出し: unfold)
[第1層: 幾何学特徴抽出レイヤー (事前学習済み・重み固定)]
       │ (16〜32次元の空間概念マップ)
       ▼
[第2層: 空間概念処理ブロック (Spatial Concept Network)]
       │
       ▼
[出力: クラス分類 ＆ XAI自動監査 (Why / Because)]
</code>
       
### 1. 第1層：幾何学フィルターと概念辞書（事前学習・固定ベース）
任意の自由な特徴をブラックボックスに学習させるのではなく、あらかじめ定義された専門的な幾何学的・トポロジカル概念を汎用的に抽出するフィルター群を使用します。
#### 入力サイズ: $7 \times 7 \times 1$ パッチ
#### 網羅する概念（10クラス）:
- 直線系（4個）: 縦線、横線、斜め右上がり、斜め右下がり
- 交差・特殊系（2個）: 十字（＋）、バツ印（×）
- L字系（4個）: L字（左上開き）、L字（右上開き）、L字（右下開き）、L字（左下開き）

### 2. 第2層以降：空間概念処理ブロック
#### ブロック特徴
文字の2次元的な広がりやストロークのトポロジー構造を保持した空間概念マップ $(B, C_{\text{concept}}, H_{\text{grid}}, W_{\text{grid}})$ を受け取り、全結合層やプーリング機構によって統合処理を行います。

#### 優位性
単なる「バッグ・オブ・コンセプト（頻度情報のみ）」の限界を突破し、位置関係を保持したレイアウト学習により、軽量ながら高精度（MNISTテスト正解率 91%）を実現します。

## 🔍 XAI（説明可能なAI）と自動監査
訓練済みモデルに対する推論時の逆伝播（BP）から得られる空間勾配を解析し、モデルが下した判断の根拠を視覚的かつ定量的に抽出します。

### 自動出力管理（space/output/）
- グリッド画像（grid_batch_XXX.png）: 元画像と、AIがどの幾何学的概念に反応して判断したかをマッピングした画像をペアで一覧化。
- 詳細メタデータ（JSON）: 各検出ボックスの座標、最大勾配寄与度、Top-3概念ラベルとその信頼度（Softmax確率）を完全記録。

## 📁 ディレクトリ構成
<code>
CXAI-Model
│   app.toml
│   pyproject.toml
│   README.md
│   run_cxai.bat
│
├── dataset          # MNISTバイナリデータ
├── log              # 実行ログ
├── space            # チェックポイントおよび出力ディレクトリ
│   ├─ label_image      # 概念ごとのベース参照画像（5倍拡大）
│   └─ output           # 視覚化グリッド画像 (.png) および推論ログ (.json)
└─ src
    └─ cxai
        │   main.py         # CLI・インタラクティブメニュー制御
        ├── com             # 設定・共通モジュール
        └─ core             # モデル・パイプライン実装
            ├── gnn_main_model.py
            ├── gnn_main_model_helper.py
            ├── gnn_preset_model.py
            └── mnist_dataloader.py
</code>

## 🚀 使い方（CLIメニューによる簡単実行）
本システムは、リポジトリルートから対話型CLIメニューまたはコマンド引数で手軽に実行できます。

### 1. 起動方法
python -m src.cxai.main

または、付属のバッチファイル（Windows）を使用：
run_cxai.bat

### 2. インターラクティブメニュー
メニューが起動したら、番号を選んでステップ・バイ・ステップで実行可能です。

<code>
=================================================================
    Concept XAI - MNIST Step-by-Step Verification System
=================================================================
    [1] Train Geometric Feature & Dictionary Models
    [2] Train MNIST Spatial Concept Classifier
    [3] Test MNIST Classification Model
    [4] Evaluate Training Data & Extract Label/Concept Anomalies
    ---------------------------------------------------------
    [9] Init - Reset All Checkpoints & Caches
    [q] Quit - Exit Application
=================================================================
Enter choice:
</code>
- [1]: 幾何学特徴フィルターと概念辞書の事前学習
- [2]: MNIST統合モデルのトレーニング
- [3]: MNIST分類モデルのテスト評価（精度91%達成）
- [4]: トレーニングデータの評価・勾配マッピング・グリッド画像（grid_batch_*.png）およびJSONの出力
- [9]: チェックポイント・キャッシュの初期化（リセット）

## 🗺️ 今後の展望
- 医療画像診断への応用: 細胞画像や組織スキャンにおいて、医師が求める厳密な形態学的根拠（特定の輪郭やループ構造の有無など）に基づいた病変検知。
- 産業用・医療機器の故障診断: 機械部品や生体組織の微細な破断・異常パターンを「構成的な幾何学的概念」として定義し、根拠付きで故障予兆を検知。
- 動的モジュール拡張: 診断対象の目的に応じて、上位に重ねる概念フィルターを動的に切り替え・拡張できるアダプティブなAI基盤への発展。


## 実行結果
### 概念モデル構築
<code>
🚀 [Geometric Train] Starting training with 10 Clean Geometric Concepts (50 epochs)...
📁 Saved 10 centralized base concept reference images (5x enlarged) to: D:\workspace\CXAI-Model\space\label_image
  - Total samples (augmented): 624
Epoch [1/50] Loss: 2.2000 | Train Acc: 34.91% | Val Acc: 22.34%
Epoch [5/50] Loss: 0.8222 | Train Acc: 75.85% | Val Acc: 88.30%
Epoch [10/50] Loss: 0.4202 | Train Acc: 86.98% | Val Acc: 96.81%
Epoch [15/50] Loss: 0.2415 | Train Acc: 92.08% | Val Acc: 95.74%
Epoch [20/50] Loss: 0.2074 | Train Acc: 93.77% | Val Acc: 95.74%
Epoch [25/50] Loss: 0.1807 | Train Acc: 93.96% | Val Acc: 95.74%
Epoch [30/50] Loss: 0.1431 | Train Acc: 95.85% | Val Acc: 95.74%
Epoch [35/50] Loss: 0.1507 | Train Acc: 94.53% | Val Acc: 95.74%
Epoch [40/50] Loss: 0.1374 | Train Acc: 95.85% | Val Acc: 95.74%
Epoch [45/50] Loss: 0.1407 | Train Acc: 94.91% | Val Acc: 95.74%
Epoch [50/50] Loss: 0.1369 | Train Acc: 95.28% | Val Acc: 95.74%

=== Geometric Concept Dictionary Test ===

--- Per-Class Accuracy (acc[label]) ---
  cross                :  66.67% 🔴 WEAK
  l_shape_dr           :  85.71% 🟡 FAIR
  horizontal_line      :  90.91% 🟢 GOOD
  vertical_line        : 100.00% 🟢 GOOD
  diag_up              : 100.00% 🟢 GOOD
  diag_down            : 100.00% 🟢 GOOD
  x_shape              : 100.00% 🟢 GOOD
  l_shape_ul           : 100.00% 🟢 GOOD
  l_shape_ur           : 100.00% 🟢 GOOD
  l_shape_dl           : 100.00% 🟢 GOOD

--- Confusion Matrix & Misclassification Analysis ---
  horizontal_line  ==>  misclassified as  l_shape_ul  (1 times)
  cross  ==>  misclassified as  l_shape_dl  (1 times)
  l_shape_dr  ==>  misclassified as  vertical_line  (1 times)
  l_shape_dr  ==>  misclassified as  horizontal_line  (1 times)
✅ Geometric filter saved to: D:\workspace\CXAI-Model\space\checkpoint_geometric_filter.pth
✅ Concept dictionary model saved to: D:\workspace\CXAI-Model\space\checkpoint_concept_dictionary.pth
</code>

## MNISTの学習
<code>
Loading Training dataset...
Data loaded successfully (Total samples: 60000, Shape: 28x28)
Loading Test dataset...
Data loaded successfully (Total samples: 10000, Shape: 28x28)
💡 Training with 60000 samples (Configured max: -1).
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000100.json (Detected: 6 objects)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000200.json (Detected: 10 objects)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000300.json (Detected: 6 objects)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000400.json (Detected: 5 objects)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000500.json (Detected: 5 objects)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000600.json (Detected: 8 objects)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000700.json (Detected: 9 objects)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000800.json (Detected: 3 objects)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000900.json (Detected: 8 objects)
Epoch [1/10]: 100%|██████████████████████| 938/938 [00:25<00:00, 36.83it/s, acc=49.22%, loss=1.4230]
👉 Epoch 1 Completed | Loss: 1.4230 | Accuracy: 49.22%
Epoch [2/10]: 100%|██████████████████████| 938/938 [00:25<00:00, 36.96it/s, acc=69.06%, loss=0.9160]
👉 Epoch 2 Completed | Loss: 0.9160 | Accuracy: 69.06%
・・・
Epoch [10/10]: 100%|█████████████████████| 938/938 [00:25<00:00, 37.46it/s, acc=83.86%, loss=0.5097]
👉 Epoch 10 Completed | Loss: 0.5097 | Accuracy: 83.86%
✅ Integrated classification model saved to: D:\workspace\CXAI-Model\space\checkpoint_mnist_spatial_nn.pth
[2026/10/21 21:55:00] [INFO] === [MNIST Classification Model Training] Completed ===
</code>
       
## MNISTの推論
<code>
Loading Training dataset...
Data loaded successfully (Total samples: 60000, Shape: 28x28)
Loading Test dataset...
Data loaded successfully (Total samples: 10000, Shape: 28x28)
💾 [Epoch 1 JSON Log Saved] D:\workspace\CXAI-Model\space\output\extraction_epoch1_step_000100.json (Detected: 6 objects)
Evaluating: 100%|█████████████████████████████████████████████████| 157/157 [00:03<00:00, 47.33it/s]

📊 [Test Results] Overall Accuracy: 92.35% (Evaluated samples: 10000)

--- Per-Class Test Accuracy (MNIST Digits 0-9) ---
  Digit 0 :  96.73%  (980 samples)
  Digit 1 :  97.97%  (1135 samples)
  Digit 2 :  91.96%  (1032 samples)
  Digit 3 :  94.85%  (1010 samples)
  Digit 4 :  95.82%  (982 samples)
  Digit 5 :  89.24%  (892 samples)
  Digit 6 :  93.95%  (958 samples)
  Digit 7 :  91.05%  (1028 samples)
  Digit 8 :  81.21%  (974 samples)
  Digit 9 :  89.59%  (1009 samples)
[2026/10/21 22:03:07] [INFO] === [MNIST Classification Model Testing] Completed ===
</code>

## CXAIモデルの思考の可視化
<code>
🧹 Cleared Output directory: D:\workspace\CXAI-Model\space\output
Loading Training dataset...
Data loaded successfully (Total samples: 60000, Shape: 28x28)
Loading Test dataset...
Data loaded successfully (Total samples: 10000, Shape: 28x28)
🚀 Processing Linear-input gradient mapping across 1000 samples...
Mapping & Exporting:   1%|▍                                                           | 7/1000 [00:00<00:32, 30.46it/s]  🖼️ Flushed 10-item grid image: D:\workspace\CXAI-Model\space\output\grid_batch_001.png
Mapping & Exporting:   2%|▉                                                          | 15/1000 [00:00<00:29, 33.92it/s]  🖼️ Flushed 10-item grid image: D:\workspace\CXAI-Model\space\output\grid_batch_002.png
Mapping & Exporting:   3%|█▋                                                         | 28/1000 [00:00<00:27, 35.07it/s]  🖼️ Flushed 10-item grid image: D:\workspace\CXAI-Model\space\output\grid_batch_003.png
・・・
Mapping & Exporting: 100%|█████████████████████████████████████████████████████████| 1000/1000 [00:26<00:00, 38.39it/s]
</code>
