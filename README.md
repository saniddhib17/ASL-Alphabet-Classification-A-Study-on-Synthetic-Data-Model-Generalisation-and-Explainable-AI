# ASL-Alphabet-Classification-A-Study-on-Synthetic-Data-Model-Generalisation-and-Explainable-AI

## Overview
This project benchmarks and interprets state-of-the-art deep learning models for American Sign Language (ASL) alphabet classification, with a focus on domain generalization from synthetic data to real-world images. Through transfer learning, advanced augmentation, and explainable AI, it addresses key accessibility challenges for the Deaf and Hard-of-Hearing community.

## Tech Stack
| Category               | Tools / Libraries                          |
| ---------------------- | ------------------------------------------ |
| Deep Learning          | PyTorch, Keras, TensorFlow                 |
| Vision & Preprocessing | OpenCV, torchvision                        |
| Data Science           | NumPy, pandas, scikit-learn                |
| Visualization          | Matplotlib, Seaborn, t-SNE, Grad-CAM, SHAP |
| Collaboration & Docs   | Google Colab, Jupyter, Markdown,           |

| Dataset                | Type       | Source (Kaggle link)                                                                           | Size / Classes                        | Characteristics                                               |
| ---------------------- | ---------- | ---------------------------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------- |
| Synthetic ASL Alphabet | Training   | [lexset/synthetic-asl-alphabet](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet) | 27,000 images / 27 classes            | High-res, controlled backgrounds, labeled, diverse conditions |
| Real ASL Alphabet      | Evaluation | [grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)           | 87,000+ images / 29 classes (used 27) | Natural backgrounds, diverse users, uncurated, realistic      |


## Research Questions

- CNN Benchmarking:
How do different pre-trained CNN architectures perform in ASL alphabet classification when trained on synthetic data?

- Domain Generalization:
How well do models trained on synthetic ASL images generalize to real-world ASL images, and what are the per-class challenges?

- Explainability:
Can explainable AI (Grad-CAM, saliency maps) enhance transparency and diagnose model biases in ASL classification?

## Research Gaps and Future Directions 
#### Lack of Comprehensive Benchmarking 
    While individual models have been well-studied, there's a notable absence of comprehensive 
    benchmarking across multiple pre-trained models under identical conditions. Our proposed study 
    aims to fill this gap by systematically comparing ResNet50, MobileNetV2, EfficientNet, and 
    Densenet. 

#### Limited Use of Real-World Datasets 
    Most studies heavily rely on synthetic or controlled datasets. There's a pressing need for more 
    extensive evaluation on diverse, real-world data. Our research will address this by incorporating a 
    substantial real-world test set to validate models trained on synthetic data only. 

#### Lack of focused research on synthetic ASL detection 
    While techniques exist for detecting synthetic images in general, there is a notable absence of 
    research specifically addressing the detection of synthetic sign language imagery. The unique 
    characteristics of hand shapes and gestures in ASL may require specialised approaches beyond 
    general synthetic image detection methods.

#### Explainability Gap 
    Existing ASL classifiers often act as “black boxes," limiting trust among deaf users. Current tools 
    struggle to explain continuous sign sentences or biases in underrepresented groups (e.g., 18% 
    higher misclassifications for darker skin tones) implement fairness-aware algorithms in XAI 
    frameworks to address demographic disparities. 

    Explainable AI techniques have been applied separately to sign language classification and 
    synthetic image detection, but their integration remains largely unexplored. Understanding how 
    models distinguish between real and synthetic ASL signs could provide valuable insights for 
    improving both synthetic data generation and authentication systems. 

## Data Augmentation Impact:
What is the effect of various data augmentation strategies on improving model robustness for synthetic-to-real domain transfer?

- Project Aim

Benchmark pre-trained CNNs (MobileNetV2, ResNet50, EfficientNetB0, DenseNet121) for ASL alphabet recognition.

Evaluate zero-shot generalization from synthetic to real data.

Leverage explainable AI to improve transparency, fairness, and trust.

Quantify the impact of augmentation on domain shift.


## Key Features

- Dual-dataset pipeline: synthetic for training, real for strict zero-shot evaluation.
- Two-frameworks: Keras (mild augmentation), PyTorch (EDA-driven strong augmentation).
- Explainable AI: Grad-CAM, saliency maps for transparency and bias diagnosis.
- Modular, reproducible code and checkpoints.

| Architecture   | Synthetic Validation Accuracy | Real Test Accuracy | Macro F1 (Real) | Generalization Gap  | Notes                                      |
| -------------- | ----------------------------- | ------------------ | --------------- | ------------------- | ------------------------------------------ |
| DenseNet121    | ~99%                          | **43.7%**          | **0.44**        | Highest (0.069)     | Best real-world accuracy, some overfitting |
| ResNet50       | ~99%                          | 42.2%              | 0.39            | 0.0589              | Robust on static, struggles with dynamic   |
| EfficientNetB0 | ~99%                          | 41.7%              | 0.41            | 0.0448              | Good trade-off, lower domain gap           |
| MobileNetV2    | 94%                           | 35.9%              | 0.34            | **Lowest (0.0197)** | Best for lightweight deployment            |

## Additional Results:

Strong augmentation improved real-world accuracy by up to 12 points.
Grad-CAM/saliency maps confirmed hand-region focus for correct predictions; confusion common for visually similar/motion signs.
Persistent domain gap underscores challenge of static-image transfer learning.

How to Run

1. Open in Google Colab.
2. Upload Notebooks.
Mount Google Drive:

from google.colab import drive
drive.mount('/content/drive')

3. Prepare Data:
Download Synthetic Dataset
Download Real Dataset
Upload to Google Drive; adjust paths in code.

4. Run Cells in Order:

For EDA: EDA_TRY_Synthetic.ipynb
For full synthetic-to-real pipeline: FINAL_SYN_REAL.ipynb

Checkpoints & Outputs saved to /models/ and your Drive.

