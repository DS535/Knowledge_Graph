import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CKPT = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
MAX_LEN = 512
DATASRC = r"D:\MTech\sem3\NLU\assignments\assignment2\knowledge-net\dataset\train.json"
PREPROCESSED_DIR = "../processed_data"
MODEL_DIR = "../saved_models"

PROCESSED_TRAIN_DOC_FILE_NAME = "train_documnets.pkl"
PROCESSED_TRAIN_TAG_FILE_NAME = "train_tags.pkl"

PROCESSED_TEST_DOC_FILE_NAME = "test_documnets.pkl"
PROCESSED_TEST_TAG_FILE_NAME = "test_tags.pkl"

PROCESSED_VAL_DOC_FILE_NAME = "val_documnets.pkl"
PROCESSED_VAL_TAG_FILE_NAME = "val_tags.pkl"

ner_token_to_idx = {
    "o": 0,

    "bsub": 1,
    "isub": 2,

    "bobj": 3,
    "iobj": 4

}

weights = torch.tensor([0.02, 0.30, 0.18, 0.20, 0.30], requires_grad=False).to(DEVICE)
