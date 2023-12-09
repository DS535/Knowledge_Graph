import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import config
import preprocess
import pickle


def relation_to_sub_map(rel:str):
    map_dict = {
    "PLACE_OF_RESIDENCE" :  "person",
    "PLACE_OF_BIRTH" :  "person",
    "DATE_OF_BIRTH" :  "person",
    "CHILD_OF" :  "person",
    "SPOUSE" :  "person",
    "EDUCATED_AT" :  "person",
    "HEADQUARTERS" :  "person",
    "DATE_OF_DEATH" :  "person",
    "EMPLOYEE_OR_MEMBER_OF" :  "person",
    "FOUNDED_BY" :  "org",
    "DATE_FOUNDED" :  "org",
    "CEO" :  "org",
    "NATIONALITY" :  "person",
    "SUBSIDIARY_OF" :  "org",
    "POLITICAL_AFFILIATION" :  "person"
    }
    
    return map_dict[rel]


def relation_to_obj_map(rel):
    map_dict = {
    "PLACE_OF_RESIDENCE" :  "place",
    "PLACE_OF_BIRTH" :  "place",
    "DATE_OF_BIRTH" :  "date",
    "CHILD_OF" :  "person",
    "SPOUSE" :  "person",
    "EDUCATED_AT" :  "org",
    "HEADQUARTERS" :  "place",
    "DATE_OF_DEATH" :  "date",
    "EMPLOYEE_OR_MEMBER_OF" :  "org",
    "FOUNDED_BY" :  "person",
    "DATE_FOUNDED" :  "date",
    "CEO" :  "person",
    "NATIONALITY" :  "counry",
    "SUBSIDIARY_OF" :  "org",
    "POLITICAL_AFFILIATION" :  "org"
    }
    return map_dict[rel]


def tokenize_doc(tokenizer, document, tags, max_len=config.MAX_LEN):
    cut_off_len = max_len - 2
    doc_input_ids = []
    doc_attention_masks = []
    doc_labels = []

    for word, tag in zip(document, tags):
        tokenized = tokenizer(word, add_special_tokens=False)
        token_cnt = len(tokenized["input_ids"])
        cut_off_len -= token_cnt
        if cut_off_len < 0:
            break
        doc_input_ids.extend(tokenized["input_ids"])
        doc_attention_masks.extend(tokenized["attention_mask"])
        doc_labels.extend([config.ner_token_to_idx[tag]] * token_cnt)

    pad_token = tokenizer.pad_token_id
    pad_len = cut_off_len
    pad_tokens = [pad_token] * pad_len
    pad_atten_masks = [0] * pad_len
    pad_labels = [config.ner_token_to_idx["o"]] * pad_len

    doc_input_ids.extend(pad_tokens)
    doc_attention_masks.extend(pad_atten_masks)
    doc_labels.extend(pad_labels)

    doc_input_ids = [tokenizer.cls_token_id] + doc_input_ids + [tokenizer.sep_token_id]
    doc_attention_masks = [1] + doc_attention_masks + [1]
    doc_labels = [config.ner_token_to_idx["o"]] + doc_labels + [config.ner_token_to_idx["o"]]

    # add cls and
    return doc_input_ids, doc_attention_masks, doc_labels


class NER_Dataset(Dataset):
    def __init__(self, d_type: str):
        super(NER_Dataset, self).__init__()
        self.d_type = d_type.upper()
        assert self.d_type in ['TRAIN', "TEST", "VAL"]
        pkl_paths = preprocess.preprocess_datasets()

        idx = {"TRAIN": 0, "VAL": 1, "TEST": 2}[self.d_type] * 2
        doc_pkl = pkl_paths[idx]
        tag_pkl = pkl_paths[idx+1]

        with open(doc_pkl, "rb") as file:
            self.docs = pickle.load(file)

        with open(tag_pkl, "rb") as file:
            self.tags = pickle.load(file)

        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CKPT)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        input_ids, input_masks, doc_labels = tokenize_doc(self.tokenizer, self.docs[idx], self.tags[idx])
        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=config.DEVICE)
        input_masks_t = torch.tensor(input_masks, dtype=torch.long, device=config.DEVICE)
        doc_labels_t = torch.tensor(doc_labels, dtype=torch.long, device=config.DEVICE)

        return {"input_ids": input_ids_t, "attention_mask": input_masks_t}, doc_labels_t


if __name__ == "__main__":
    ds = NER_Dataset("train")
    print(ds[0])

        