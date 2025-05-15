from gensim.models import Word2Vec
from DataHandler.data_preprocess import save_pickle, load_pickle
from training_pipeline import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as SklearnLDA
from Visualization.plotting_funcs import *

# === Parameters ===
# Set this to True if already train the models
WORD2VEC = True
# Determine Classification model
WORD2VEC_CLASSIFY = False
BERT_CLASSIFY = False
TFIDF_CLASSIFY = False
LDA_CLASSIFY = False
# Set this to True to print details
VERBOSE = True
# Batch size
BATCH_SIZE = 32

# === Preprocessing ===
# === Remember to run Preprocess.py first ===
print("Using existing preprocessed split data...")

# Load the stored labels to avoid re-extracting them
imdb_train_labels = load_pickle("../raw_data/labels/imdb_train_labels.pkl")
imdb_dev_labels = load_pickle("../raw_data/labels/imdb_dev_labels.pkl")
imdb_test_labels = load_pickle("../raw_data/labels/imdb_test_labels.pkl")

douban_train_labels = load_pickle("../raw_data/labels/douban_train_labels.pkl")
douban_dev_labels = load_pickle("../raw_data/labels/douban_dev_labels.pkl")
douban_test_labels = load_pickle("../raw_data/labels/douban_test_labels.pkl")

# Load saved tokenized results
# Word level - flattened tokens from training dataset
imdb_tokens_train = load_pickle(f"../tokens/zipf/imdb_tokens_train.pkl")
douban_tokens_train = load_pickle(f"../tokens/zipf/douban_tokens_train.pkl")
# Sentence level
imdb_sentences_train = load_pickle(f"../tokens/sentences/imdb_sentences_train.pkl")
imdb_sentences_dev = load_pickle(f"../tokens/sentences/imdb_sentences_dev.pkl")
imdb_sentences_test = load_pickle(f"../tokens/sentences/imdb_sentences_test.pkl")
douban_sentences_train = load_pickle(f"../tokens/sentences/douban_sentences_train.pkl")
douban_sentences_dev = load_pickle(f"../tokens/sentences/douban_sentences_dev.pkl")
douban_sentences_test = load_pickle(f"../tokens/sentences/douban_sentences_test.pkl")

# Create mixed EN+ZH training/dev/test set
# text
mixed_sentences_train = imdb_sentences_train + douban_sentences_train
mixed_sentences_dev = imdb_sentences_dev + douban_sentences_dev
mixed_sentences_test = imdb_sentences_test + douban_sentences_test
# labels
mixed_train_labels = imdb_train_labels + douban_train_labels
mixed_dev_labels = imdb_dev_labels + douban_dev_labels
mixed_test_labels = imdb_test_labels + douban_test_labels
# sources
imdb_src_train  = ["imdb"]   * len(imdb_sentences_train)
imdb_src_dev  = ["imdb"]   * len(imdb_sentences_dev)
imdb_src_test = ["imdb"]   * len(imdb_sentences_test)
douban_src_train = ["douban"] * len(douban_sentences_train)
douban_src_dev = ["douban"] * len(douban_sentences_dev)
douban_src_test = ["douban"] * len(douban_sentences_test)

mixed_sources_train = imdb_src_train + douban_src_train
mixed_sources_dev   = imdb_src_dev  + douban_src_dev
mixed_sources_test  = imdb_src_test + douban_src_test

# === Debugging ===
# print("Checking individual review strings for tokenizer compatibility...")
# for i, text in enumerate(douban_train["review"]):
#     try:
#         _ = douban_tokenizer.tokenize(text)
#     except Exception as e:
#         print(f"\nError at index {i}")
#         print("Review:", repr(text))
#         print("Type:", type(text))
#         print("Error:", e)
#         break
# else:
#     print("All reviews are tokenizer-compatible.")

# === Print Statistics of Datasets ===
if VERBOSE:
    print(f"Corpus size (tokens): {len(imdb_tokens_train):,}")
    print(f"Vocabulary size (unique tokens): {len(set(imdb_tokens_train)):,}")

    print(f"Corpus size (tokens): {len(douban_tokens_train):,}")
    print(f"Vocabulary size (unique tokens): {len(set(douban_tokens_train)):,}")

    # Plot Zipf (time-consuming)
    plot_zipf(imdb_tokens_train, douban_tokens_train, dataset_name1="IMDB", dataset_name2="Douban")

# For bert, tf-idf and LDA, they Only work on mixed data
(train_raw, train_labels, train_sources), (dev_raw, dev_labels, dev_sources), (test_raw, test_labels, test_sources) = load_split_raw()

all_metrics = {}
#
#  === Wor2Vec Simple Classification ===
if WORD2VEC_CLASSIFY:
    print(f"\n================Word2Vec plus MLP================")
    print("Using features from wor2vec model, and then training a Classifier...")
    if WORD2VEC:
        print("Loading pre-trained w2v model...")
        mixed_w2v_cbow = Word2Vec.load("../w2v_models/mixed_cbow.model")
        mixed_w2v_skip = Word2Vec.load("../w2v_models/mixed_skipgram.model")

    else:
        print("Start Training w2v model...")
        # Word2Vec-CBOW
        mixed_w2v_cbow = Word2Vec(sentences=mixed_sentences_train, vector_size=100, window=5, min_count=2, workers=4)
        # Word2Vec-SkipGram
        mixed_w2v_skip = Word2Vec(sentences=mixed_sentences_train, vector_size=100, window=5, min_count=2, sg=1,
                                  workers=4)

        # Save model
        mixed_w2v_cbow.save("w2v_models/mixed_cbow.model")
        mixed_w2v_skip.save("w2v_models/mixed_skipgram.model")

        print("All trained Models are successfully saved!")

    # === Show Similar Words ===
    # for word in ["movie", "bad", "great", "love"]:
    #     print_similar_words(imdb_w2v_cbow, word, "IMDB CBOW")
    #     print_similar_words(imdb_w2v_skip, word, "IMDB SkipGram")
    #
    # # print(list(douban_w2v_cbow.wv.key_to_index.keys())[:30])
    # for word in ["电影", "糟糕", "很棒", "喜欢"]:
    #     print_similar_words(douban_w2v_cbow, word, "Douban CBOW")
    #     print_similar_words(douban_w2v_skip, word, "Douban SkipGram")

    word2vec_exps = [
        # Mixed
        (mixed_w2v_cbow, mixed_sentences_train, mixed_sentences_dev, mixed_sentences_test, mixed_train_labels, mixed_dev_labels, mixed_test_labels, mixed_sources_train, mixed_sources_dev, mixed_sources_test, "Mixed_CBow"),
        (mixed_w2v_skip, mixed_sentences_train, mixed_sentences_dev, mixed_sentences_test, mixed_train_labels, mixed_dev_labels, mixed_test_labels, mixed_sources_train, mixed_sources_dev, mixed_sources_test, "Mixed_SkipGram"),
    ]

    for w2v_model, tr_sents, dv_sents, ts_sents, tr_lbls, dv_lbls, ts_lbls, tr_src, dv_src, ts_src, name in word2vec_exps:
        w2v_train_data = Word2VecDataset(tr_sents, tr_lbls, tr_src, w2v_model)
        w2v_dev_data = Word2VecDataset(dv_sents, dv_lbls, dv_src, w2v_model)
        w2v_test_data = Word2VecDataset(ts_sents, ts_lbls, ts_src, w2v_model)

        w2v_train_loader = DataLoader(w2v_train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        w2v_dev_loader = DataLoader(w2v_dev_data , batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        w2v_test_loader = DataLoader(w2v_test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        w2v_baseline_model = SimpleClassifier(input_dim=w2v_model.vector_size)
        w2v_baseline_metrics = run_classification(
            model=w2v_baseline_model,
            train_loader=w2v_train_loader,
            dev_loader=w2v_dev_loader,
            test_loader=w2v_test_loader,
            epoch=100,
            name=name
        )
        all_metrics[f"w2v_{name}"] = w2v_baseline_metrics

#  === mBert Simple Classification without fine-tuning (Using cache to speed up) ===
if BERT_CLASSIFY:
    print(f"\n================mBert plus MLP================")
    print("Using features (cached) from pre-trained mbert model, and then training a Classifier...")

    # Use raw data and get embeddings from the pre-trained bert model
    bert_train_data = BertEmbedsDataset(train_raw, train_labels, train_sources, data_name="train")
    bert_dev_data = BertEmbedsDataset(dev_raw, dev_labels, dev_sources, data_name="dev")
    bert_test_data = BertEmbedsDataset(test_raw, test_labels,test_sources, data_name="test")

    bert_train_loader = DataLoader(bert_train_data, batch_size=BATCH_SIZE, shuffle=True)
    bert_dev_loader = DataLoader(bert_dev_data, batch_size=BATCH_SIZE, shuffle=False)
    bert_test_loader = DataLoader(bert_test_data, batch_size=BATCH_SIZE, shuffle=False)

    # BertDataset stores all CLS vectors in bert_train_data.embeddings, its shape=(N, hidden_size)
    bert_hidden_size = bert_train_data.embeddings.shape[1]
    bert_baseline_model = SimpleClassifier(input_dim=bert_hidden_size)

    bert_baseline_metrics = run_classification(
        model=bert_baseline_model,
        train_loader=bert_train_loader,
        dev_loader=bert_dev_loader,
        test_loader=bert_test_loader,
        epoch=100,
        name="Mixed_BERT_Baseline"
    )
    all_metrics["Mixed_BERT_Baseline"] = bert_baseline_metrics

if TFIDF_CLASSIFY:
    print("\n================ TF–IDF + MLP ================")
    tfidf = TfidfVectorizer(ngram_range=(1,1), max_features=5000)  # 1-gram, 5000-d
    X_tr = tfidf.fit_transform(train_raw)
    X_dev = tfidf.transform(dev_raw)
    X_te = tfidf.transform(test_raw)

    ds_tr_tfidf = FeatureDataset(X_tr, train_labels, train_sources)
    ds_dev_tfidf = FeatureDataset(X_dev, dev_labels, dev_sources)
    ds_te_tfidf = FeatureDataset(X_te, test_labels, test_sources)

    loader_tr = DataLoader(ds_tr_tfidf, batch_size=BATCH_SIZE, shuffle=True)
    loader_dev = DataLoader(ds_dev_tfidf, batch_size=BATCH_SIZE, shuffle=False)
    loader_te  = DataLoader(ds_te_tfidf, batch_size=BATCH_SIZE, shuffle=False)

    model_tfidf = SimpleClassifier(input_dim=X_tr.shape[1])
    metrics_tfidf = run_classification(
        model=model_tfidf,
        train_loader=loader_tr,
        dev_loader=loader_dev,
        test_loader=loader_te,
        epoch=100,
        name="Mixed_TFIDF"
    )
    all_metrics["Mixed_TFIDF"] = metrics_tfidf

if LDA_CLASSIFY:
    print("\n================ LDA + MLP ================")
    # Use CountVectorizer to do Bag‑of‑Words
    count_vec = CountVectorizer(ngram_range=(1, 1), max_features=5000)
    C_tr = count_vec.fit_transform(train_raw)
    C_dev = count_vec.transform(dev_raw)
    C_te = count_vec.transform(test_raw)

    # Component is set to 50
    lda = SklearnLDA(n_components=50, random_state=42)
    Z_tr = lda.fit_transform(C_tr)  # (n_train, 50)
    Z_dev = lda.transform(C_dev)
    Z_te = lda.transform(C_te)

    ds_tr_lda = FeatureDataset(Z_tr, train_labels, train_sources)
    ds_dev_lda = FeatureDataset(Z_dev, dev_labels, dev_sources)
    ds_te_lda = FeatureDataset(Z_te, test_labels, test_sources)

    loader_tr2 = DataLoader(ds_tr_lda, batch_size=BATCH_SIZE, shuffle=True)
    loader_dev2 = DataLoader(ds_dev_lda, batch_size=BATCH_SIZE, shuffle=False)
    loader_te2 = DataLoader(ds_te_lda, batch_size=BATCH_SIZE, shuffle=False)

    model_lda = SimpleClassifier(input_dim=Z_tr.shape[1])
    metrics_lda = run_classification(
        model=model_lda,
        train_loader=loader_tr2,
        dev_loader=loader_dev2,
        test_loader=loader_te2,
        epoch=100,
        name="Mixed_LDA"
    )
    all_metrics["Mixed_LDA"] = metrics_lda

# Save for visulization
save_pickle(all_metrics, "../results/baseline_all_metrics.pkl")