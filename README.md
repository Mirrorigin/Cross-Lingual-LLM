# Cross-Lingual Sentiment Classification
This project explores sentiment classification on English and Chinese movie reviews using both traditional NLP techniques and multilingual Transformer models. It compares classic feature extraction (Word2Vec, TF-IDF, LDA) with fine-tuning strategies for mBERT, including standard fine-tuning, LoRA, and Adapter modules.
## Prepare Dataset
- [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download)
- [Douban Moviedata-10M](https://moviedata.csuldw.com/)
## Project Structure
```bash
├── raw_data/
│   ├── IMDB_Dataset.csv
│   ├── Douban_Dataset.csv
│   ├── splits/                   # Saved train/dev/test CSVs
│   └── labels/                   # Pickled label lists
├── tokens/                       # Tokenized data
│   ├── zipf/                     # Flattened tokens for Zipf analysis
│   └── sentences/                # Tokenized sentences
├── w2v_models/                   # Trained Word2Vec models
├── results/                      # Training logs and plots
│   └── finetune/                 # Saved fine-tuning results
├── cache/                        # Cached BERT embeddings
├── Figures/
├── DataHandler/
├── Model/
└── Visualization/
```
## Code Description
### Preprocessing
- **DataHandler/data_preprocess.py:** Clean, split and balance Dataset
- **DataHandler/dataset_property.py:** Dataset Property

### Classification Models
- **Model/basic_embedding_models.py:** Run baseline classification
- **Model/mBERT_fine_tuning.py:** Run fine-tuning models
  - Set flags (e.g., `WORD2VEC_CLASSIFY = True`) in the script to toggle models

### Results Analysis
- **Visualization/training_metrics.py:** Plot training losses
- **Visualization/error_analysis.py:** Error Samples

## Notice

This is for Spring 2025 Large Language Models course final project.
