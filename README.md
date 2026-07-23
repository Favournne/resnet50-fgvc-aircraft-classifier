 Aircraft Variant Classifier

Open In Colab: https://colab.research.google.com/drive/1Y_gx1Tx5eMxjcvTqM9n2LkZQlfjnI78f?usp=sharing

Fine-tuned ResNet50 on the FGVC-Aircraft dataset to classify 100 aircraft variants.

Results
| Metric | Score |
| :--- | :--- |
| Test Accuracy| 61.57% |
| Validation Accuracy | 61.54% |
| Best Confidence | 92% on DC-10 |

- No overfitting (validation and test scores match).
- Top misclassifications occur between visually similar variants (e.g., 737-300 vs 737-700, 717 vs 737).

Setup
 Using Notebook

1. Click the "Open in Colab" button above.
2. Go to `Runtime` → `Change runtime type` → Select GPU.
3. Run all cells (`Runtime` → `Run all`).

Important: The training loop takes about 1 hour on a Colab GPU. Once finished, it will save `best_resnet50_finetuned_extra.pth` to the runtime, load it, and launch the Gradio interface for you to test.

> If you want to skip training, you must download the `.pth` model weights from a hosted source (I don't have one right now, but the notebook will generate it for you).
> 
```bash
git clone https://github.com/your-username/aircraft-classifier.git
cd aircraft-classifier
pip install -r requirements.txt
Usage
Gradio Web App
bash
python app.py
Then open the local URL (or share the public link).

Inference on a Single Image
python
from predict import predict_aircraft
pred_class, confidence = predict_aircraft('path/to/image.jpg')
print(f"{pred_class} ({confidence:.2%})")
Model Architecture
Backbone: ResNet50 (pretrained on ImageNet)

Progressive Unfreezing: Layers 2, 3, 4 were fine-tuned.

Classifier Head: Linear(2048 -> 256) -> ReLU -> Dropout(0.5) -> Linear(256 -> 100)

Loss Function: Weighted Cross-Entropy (balanced for rare classes)

Optimizer: Adam with differential learning rates (1e-4 for head, 1e-5 for layer3/4, 1e-6 for layer2)

Files
aircraft_classifier.ipynb – Full Colab notebook (training, evaluation, visualisation).

app.py – Gradio web interface.

requirements.txt – Python dependencies.

idx_to_class.pkl – Class mapping.

best_resnet50_finetuned_extra.pth – Model weights (download separately – see below).
