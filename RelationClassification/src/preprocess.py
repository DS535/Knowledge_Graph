import json
import re
import os
from sklearn.model_selection import train_test_split
import config
from config import DATASRC
import pickle


def remove_overlap(entities):
    # (17,27,"Fruit")
    visited_idxs = []
    valid_entities = []
    for ents in entities:
        overlap = False
        for v_st, v_ed in visited_idxs:
            if v_st <= ents[0] <= v_ed or v_st <= ents[1] <= v_ed:
                overlap = True
        if not overlap:
            valid_entities.append(ents)
            visited_idxs.append((ents[0], ents[1]))

    return valid_entities


def read_data():
    data = [json.loads(line)
            for line in open(DATASRC, 'r', encoding="cp866")]

    spacy_train_data = []
    all_relations = {}

    for idx, dataj in enumerate(data):
        dtext = dataj['documentText']
        passgs = dataj["passages"]
        entities = []
        for passg in passgs:

            relations = passg["exhaustivelyAnnotatedProperties"]

            for relation in relations:
                p_id = int(relation["propertyId"])
                p_name = relation["propertyName"]
                if p_id not in all_relations.keys():
                    all_relations[p_id] = p_name

            facts = passg["facts"]
            for fact in facts:
                s_strt = fact["subjectStart"]
                s_end = fact["subjectEnd"]
                s_txt = fact["subjectText"]

                o_strt = fact["objectStart"]
                o_end = fact["objectEnd"]
                o_txt = fact["objectText"]
                f_propid = int(fact["propertyId"])

                if s_strt >= s_end or o_strt >= o_end:
                    continue

                sub_txt = dtext[s_strt:s_end]
                obj_txt = dtext[o_strt:o_end]

                if sub_txt != s_txt or obj_txt != o_txt:
                    continue

                # entities.append((s_strt, s_end, relation_to_sub_map(all_relations[f_propid])))
                # entities.append((o_strt, o_end, relation_to_obj_map(all_relations[f_propid])))

                entities.append((s_strt, s_end, "sub"))
                entities.append((o_strt, o_end, "obj"))

        if entities:
            entities = remove_overlap(entities)
            spacy_train_data.append((dtext, {'entities': entities}))

    return spacy_train_data


def spacy_format_to_token_clf_format(spacy_format_data):
    # [(20, 44, 'sub'), (95, 102, 'obj'), (104, 114, 'obj'), (116, 129, 'obj')]
    documents = []
    tags = []
    for sp_data in spacy_format_data:
        sp_text = re.sub(r"\s", " ", sp_data[0])
        entities = sp_data[1]['entities']
        entities = sorted(entities, key=lambda x: x[0])

        seen_idx = 0
        ent_idx = 0
        entity_spans = []
        while seen_idx < len(sp_text):
            # print(f"seen_idx={seen_idx} an ent_idx={ent_idx}")
            # print(f"entities: {entities}")
            if ent_idx == len(entities):
                entity_spans.append((sp_text[seen_idx:], "o"))
                break

            if seen_idx < entities[ent_idx][0]:
                entity_spans.append((sp_text[seen_idx:entities[ent_idx][0]], "o"))
                seen_idx = entities[ent_idx][0]

            elif seen_idx == entities[ent_idx][0]:
                entity_spans.append(
                    (
                        sp_text[entities[ent_idx][0]: entities[ent_idx][1]],
                        entities[ent_idx][2]
                    )
                )
                seen_idx = entities[ent_idx][1]
                ent_idx += 1

        if entity_spans:
            documnt = []
            tag = []
            for span in entity_spans:
                words = span[0].split()
                documnt.extend(words)
                if span[1] == "o":
                    tag.extend(["o"] * len(words))
                elif span[1] == "sub":
                    if len(words) >= 1:
                        tag__ = ["bsub"] + ["isub"] * (len(words) - 1)
                    else:
                        tag__ = ["bsub"]
                    tag.extend(tag__)

                elif span[1] == "obj":
                    if len(words) >= 1:
                        tag__ = ["bobj"] + ["iobj"] * (len(words) - 1)
                    else:
                        tag__ = ["bobj"]
                    tag.extend(tag__)

            documents.append(documnt)
            tags.append(tag)

    return documents, tags


def write_as_pkl(py_obj, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(py_obj, file)
    return file_path


def preprocess_datasets():
    train_doc_path = os.path.join(config.PREPROCESSED_DIR, config.PROCESSED_TRAIN_DOC_FILE_NAME)
    train_tag_path = os.path.join(config.PREPROCESSED_DIR, config.PROCESSED_TRAIN_TAG_FILE_NAME)
    val_doc_path = os.path.join(config.PREPROCESSED_DIR, config.PROCESSED_VAL_DOC_FILE_NAME)
    val_tag_path = os.path.join(config.PREPROCESSED_DIR, config.PROCESSED_VAL_TAG_FILE_NAME)
    test_doc_path = os.path.join(config.PREPROCESSED_DIR, config.PROCESSED_TEST_DOC_FILE_NAME)
    test_tag_path = os.path.join(config.PREPROCESSED_DIR, config.PROCESSED_TEST_TAG_FILE_NAME)

    all_files_present = True
    for f_path in [train_doc_path, train_tag_path, val_doc_path, val_tag_path, test_doc_path, test_tag_path]:
        if not os.path.exists(f_path):
            all_files_present = False
            break

    if not all_files_present:
        spacy_data = read_data()
        train_data, test_val_data = train_test_split(spacy_data, train_size=0.8, random_state=711)
        test_data, val_data = train_test_split(test_val_data, train_size=0.5, random_state=711)

        train_docs, train_tags = spacy_format_to_token_clf_format(train_data)
        test_docs, test_tags = spacy_format_to_token_clf_format(test_data)
        val_docs, val_tags = spacy_format_to_token_clf_format(val_data)

        write_as_pkl(train_docs, train_doc_path)
        write_as_pkl(train_tags, train_tag_path)

        write_as_pkl(val_docs, val_doc_path)
        write_as_pkl(val_tags, val_tag_path)

        write_as_pkl(test_docs, test_doc_path)
        write_as_pkl(test_tags, test_tag_path)

    return train_doc_path, train_tag_path, val_doc_path, val_tag_path, test_doc_path, test_tag_path


if __name__ == "__main__":
    preprocess_datasets()
