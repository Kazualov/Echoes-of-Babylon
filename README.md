**Deep Past Initiative: Machine Translation (Old Assyrian to English)**

This repository contains the solution and research report for the [Deep Past Initiative Machine Translation](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/overview) competition.

**IMPORTANT REQUIREMENT**: All notebooks in this project **MUST** be run exclusively in the **Kaggle Environment**(Kaggle Notebooks) with GPU support.

## **🎯 Task and Data Description**

The task consists of developing a machine translation system to convert transliterated Old Assyrian cuneiform texts into modern English.

Old Assyrian belongs to the morphologically rich Semitic language family and is a low-resource language. The texts primarily consist of administrative and commercial records from Assyrian merchants.

The system must handle noisy transliterations, scribal notations, damaged text fragments, and determinatives.

* **Data Source**: Deep Past Initiative parallel corpus containing over 8,000 texts with metadata.

* **Data Characteristic**: Training data is provided at the document level, whereas quality evaluation is performed at the sentence level.

## **📐 Evaluation Metric**

Translation quality is evaluated using two metrics:

1. **BLEU** — evaluates word-level accuracy.

2. **chrF++** — evaluates character/n-gram level similarity, which is critical for morphologically complex languages.

The final score is calculated as the **Geometric Mean** of micro-averaged BLEU and chrF++ across the entire corpus.

Because the geometric mean is used, the model must achieve high scores on both metrics simultaneously: poor performance on one metric cannot be offset by a high score on the other.

## **💻 Environment and Resource Requirements**

All runs **MUST** be executed in **Kaggle Notebooks**.

* **Accelerator (GPU)**: NVIDIA Tesla T4 x2 or P100 (minimum 16 GB VRAM).

* **Python Runtime**: Python 3.12+ with CUDA-enabled PyTorch.

* **Internet Toggle**:

  * **OFF**: During final evaluation and submission in offline mode (mbart50.ipynb, ensemble-pipeline.ipynb) to simulate Kaggle evaluation conditions.

## **📁 Datasets and Models to Attach in Kaggle**

Before running the notebooks, attach the following resources via the right panel in Kaggle (+ Add Input):

| Resource / Model | Link / Name | Mount Path in Kaggle | Purpose |
| :---- | :---- | :---- | :---- |
| **Competition Data** | [Deep Past Initiative Competition](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation) | /kaggle/input/competitions/deep-past-initiative-machine-translation/ | Competition dataset (train.csv, test.csv).  |
| **3rd-Party mBART-50 Base** | [facebookmbart-large-50-many-to-many-mmt](https://www.kaggle.com/models/nensipansuriya1311/facebookmbart-large-50-many-to-many-mmt) | /kaggle/input/models/nensipansuriya1311/facebookmbart-large-50-many-to-many-mmt/pytorch/default/1 | Pre-downloaded mBART-50 base weights for offline loading.  |
| **Fine-tuned mBART (Ours)** | [akkadian-mbart](https://www.kaggle.com/models/eeee13/akkadian-mbart) | /kaggle/input/models/eeee13/akkadian-mbart/pytorch/subword-seq2seq-v1/1 | Fine-tuned mBART model.  |
| **Qwen2.5-7B Base** | [qwen2.5-7b-instruct](https://www.kaggle.com/models/eeee13/qwen2.5-7b-instruct) | /kaggle/input/models/eeee13/qwen2.5-7b-instruct/pytorch/base-v1/1 | Base Qwen2.5-7B model.  |
| **Qwen Adapter (LoRA)** | [akkadika](https://www.kaggle.com/models/eeee13/akkadika) | /kaggle/input/models/eeee13/akkadika/pytorch/adapter-v1/1 | LoRA adapter for Qwen.  |
| **ByT5 (Fine-tuned)** | akkadian-byt5 | /kaggle/input/models/eeee13/akkadian-byt5/pytorch/byt5-akkadian-v4/1 | Fine-tuned ByT5 model.  |
| **Bitsandbytes Offline** | [bitsandbytes](https://www.kaggle.com/datasets/eeee13/bitsandbytes) | /kaggle/input/datasets/eeee13/bitsandbytes | Offline bitsandbytes package for installation without internet.  |

**Note on ByT5 (Fine-tuned)**: Unfortunately, the fine-tuned ByT5 model is not pre-attached. You will need to train ByT5 using byte-level-seq2seq.ipynb first, and then manually upload and attach the trained model weights to your notebook environment.

## **🚀 Step-by-Step Instructions for Running Notebooks**

All notebooks must be run in the **Kaggle Environment**.

### **1\. pipeline\_QwenQLoRa.ipynb**

*Purpose*: Training and fine-tuning Qwen2.5-7B using QLoRA.

1. Open the notebook in Kaggle, enable **GPU Accelerator**.

2. Attach the competition dataset at:

   /kaggle/input/competitions/deep-past-initiative-machine-translation.

3. Configure the USE\_TRACKING parameter at the beginning of the notebook:

   * **USE\_TRACKING \= False** *(Default recommended option)*:

     * Does NOT read Kaggle Secrets.

     * Does NOT require connection to external tracking platforms.

     * Does NOT upload the adapter to external model hubs.

     * Saves the LoRA adapter locally in the Kaggle directory.

   * **USE\_TRACKING \= True**:

     * Reads Kaggle Secrets, connects to tracking systems, and uploads the adapter to external platforms.

4. Click **Run All** to start training.

### **2\. byte-level-seq2seq.ipynb**

*Purpose*: Training ByT5-small and generating predictions.

1. Launch the Kaggle Notebook with **GPU** enabled (P100 or T4, $\\ge16$ GB VRAM) and **Internet \= ON**.

2. Attach the Deep Past Initiative Machine Translation dataset.

3. Execute the notebook top-to-bottom. Execution time: \~45–60 minutes per ByT5-small run on a GPU P100.

4. *Output files*: preds\_val\_\*.csv, preds\_test\_\*.csv, and ensemble files.

**Important**: Unfortunately, you will need to train ByT5 using this notebook first, and then manually add the trained model weights to the inference notebook.

### **3\. mbart50.ipynb**

*Purpose*: Offline evaluation and inference for mBART-50.

1. Enable **GPU**. Set **Internet \= OFF** (to simulate Kaggle Submission conditions).

2. Attach the competition dataset and the 3rd-Party mBART-50 Base model weights (see the attached resources table).

3. In **Cell 2**, ensure EXECUTION\_MODE is explicitly set to:

   EXECUTION\_MODE \= "offline"

4. Click **Run All**.

### **4\. ensemble-pipeline.ipynb (Final Inference)**

*Purpose*: Final inference pipeline in fully offline mode.

1. Enable **GPU** (GPU T4 x2 or P100). Set **Internet \= OFF**.

2. Attach **all 4 models** (make sure you have manually trained and attached the fine-tuned ByT5 model) and the offline **bitsandbytes** package (see the attached resources table).

3. Verify that all required resources are present in the Input panel on the right.

4. Click **Run All**. Cell \#6 will perform an offline package installation from /kaggle/input/datasets/eeee13/bitsandbytes.

## **🛠️ Data Preparation Pipeline**

The repository includes a separate pipeline for sentence-level dataset preparation:

* Normalizes transliterations and translations.

* Extracts sentence-level pairs from published document transliterations.

* Assigns documents to train/validation splits.

* Generates the resulting dataset and an extraction report.

*Note*: The prepared sentence-level dataset **was not used** for training in the current final submission, as development could not be completed within the current cycle. Training on this dataset is planned as future work. For details, see README\_dataset.md.

## **⚠️ Limitations and Future Work**

### **Limitations**

* **Low-Resource Domain**: Small parallel corpus size. Training was conducted at the document level, while evaluation is performed at the sentence level.

* **Under-trained LLM**: Qwen2.5-7B-Instruct was trained for only 2 epochs due to time constraints.

* **Fixed Hyperparameters**: No systematic hyperparameter search was conducted (learning rate, batch size, warmup schedule).

* **Lack of Cross-Validation**: Evaluation was performed on a single validation split.

* **Automatic Metrics**: BLEU and chrF++ do not always fully reflect translation adequacy for an ancient language.