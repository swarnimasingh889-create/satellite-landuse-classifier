# 🛰️ Satellite Land-Use Classification with PyTorch

Land-use classifier for Sentinel-2 satellite tiles (EuroSAT, 10 classes) built with PyTorch and transfer learning on ResNet18. Fine-tuning the last residual block raised held-out test accuracy from **86.5% to 96.2%** compared with a frozen-backbone baseline. Includes per-class evaluation and a manual review of misclassified tiles.

**Why this project:** 

I graduated in geography and GIS, so land-use mapping is familiar territory for me, which is why I chose this dataset.

## Dataset
[EuroSAT](https://github.com/phelber/EuroSAT): 27,000 RGB tiles (64×64 pixels) from Sentinel-2, 10 classes:
AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.

Split: 70% train / 15% validation / 15% test (seed 42). The test set was used only after model selection.

## Method
1. **Baseline:** pretrained ResNet18, backbone frozen, new 10-class final layer.
2. **Augmentation:** random flips and rotations, since overhead imagery has no fixed orientation.
3. **Fine-tuning:** unfroze `layer4` with a lower learning rate (1e-4) than the new classifier layer (1e-3).
4. **Extra experiment:** unfreezing `layer3` and training 3 more epochs raised validation accuracy to 97.3%. It was not evaluated on the test set, and part of the gain may come from the extra training.

## Results

| Model | Val accuracy | Test accuracy |
|---|---|---|
| Frozen baseline | 86.0% | 86.5% |
| + fine-tune layer4 | 96.7% | 96.2% |
| + unfreeze layer3 (extra) | 97.3% | not evaluated |

**Baseline (frozen backbone)**
<img width="1350" height="1200" alt="confusion_matrix_baseline" src="https://github.com/user-attachments/assets/a2c73640-5b1d-47b6-8f5b-f9dab77fb479" />

**Fine-tuned (layer4)**
<img width="1350" height="1200" alt="confusion_matrix_finetuned" src="https://github.com/user-attachments/assets/1f80c3f7-7e91-487b-9f0e-5dbda8f91718" />
  

Precision, recall and F1 per class are in the notebook.

## Error analysis
The frozen model confused rivers with highways because both are thin, linear features. Fine-tuning cut those errors substantially. River tiles misclassified as Highway dropped from 29 to 8. In a sample of 8 misclassified River→Highway tiles I reviewed manually, 6 were clear model errors and 2 were genuinely ambiguous. 
 

## Limitations
Tiles are only 64×64 pixels, so the model lacks the wider context an analyst would use. Some labels are ambiguous, which puts a ceiling on achievable accuracy, and more data or a bigger model wouldn’t fix those cases.

## How to run
1. 1. Open [EuroSAT_Project.ipynb](EuroSAT_Project.ipynb) in Google Colab with a GPU runtime.
2. Run all cells in order. The dataset downloads automatically via torchvision.

## Future improvements
- Try a larger backbone (ResNet50 or EfficientNet)
- Run a controlled ablation of layer4 vs layer3 with equal epochs
- Test higher-resolution imagery for the river/highway cases

## Tech
Python, PyTorch, torchvision, scikit-learn, NumPy, Matplotlib, Google Colab

## Author
Swarnima Singh · https://www.linkedin.com/in/swarnima-singh-403a39374
