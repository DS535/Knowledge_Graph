import torch
import config
from transformers import AutoModelForTokenClassification
from NER_Dataset import NER_Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import adamw, Adam
import numpy as np
import os
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def loss_fn(output, target, atten_mask, num_labels):
    # output shape (16, 512,5)
    # target shape (16, 512)
    # atten_mask (16, 512)
    loss_func = nn.CrossEntropyLoss(weight=config.weights)
    active_mask = atten_mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_mask,
        target.view(-1),
        torch.tensor(loss_func.ignore_index).type_as(target)
    )
    loss = loss_func(active_logits, active_labels)
    return loss


class NER_Classifier(torch.nn.Module):
    def __init__(self, unfreeze_layers=2):
        super(NER_Classifier, self).__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(
            config.MODEL_CKPT,
            num_labels=5,
            ignore_mismatched_sizes=True
        )

        for param in list(self.model.parameters())[:-unfreeze_layers]:
            param.requires_grad = False

        for param in list(self.model.parameters())[-unfreeze_layers:]:
            param.requires_grad = True

        self.model = self.model.to(config.DEVICE)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        return model_path

    def check_test_accuracy(self):
        self.model.eval()
        test_metrics = []
        test_ds = NER_Dataset("test")
        test_dl = DataLoader(test_ds, 32, shuffle=False)
        for batch_data, batch_label in test_dl:
            model_op = self.model(**batch_data)
            batch_f1 = self.score_batch(model_op.logits, batch_label, batch_data["attention_mask"], "accuracy")
            test_metrics.append(batch_f1)

        return np.mean(test_metrics, axis=0)


    def __match_preds(self, model_op, labels, atten_mask):
        logit_idxes = torch.argmax(model_op, axis=2)
        truth_val = logit_idxes == labels
        final_truth_val = torch.where(
            atten_mask,
            truth_val,
            True
        )
        return final_truth_val

    def score_batch(self, model_op, labels, atten_mask, metric="f1"):
        logit_idxes = torch.argmax(model_op, axis=2).view(-1)
        logit_mask = atten_mask.view(-1) == 1
        logit_mask = logit_mask.view(-1)
        labels = labels.view(-1)

        active_logits = torch.where(
            logit_mask,
            logit_idxes,
            -1
        ).tolist()

        active_labels = torch.where(
            logit_mask,
            labels,
            -1
        ).tolist()
        active_logits = [x for x in active_logits if x != -1]
        active_labels = [x for x in active_labels if x != -1]

        if metric == "f1":
            # batch_metric = f1_score(active_labels, active_logits, average="macro")
            batch_metric = f1_score(active_labels, active_logits, average=None)
        elif metric == "accuracy":
            matrix = confusion_matrix(active_labels, active_logits)
            batch_metric = matrix.diagonal() / matrix.sum(axis=1)
        else:
            raise RuntimeError("Unsupported Metric")

        return batch_metric

    def train_model(self, lr, epochs, batch_size):
        min_val_loss = np.inf
        train_ds = NER_Dataset("train")
        val_ds = NER_Dataset("val")
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size, shuffle=True)
        print(f"Train dataloader: {len(train_dl)}")
        print(f"Val dataloader: {len(val_dl)}")

        optimizer = Adam(self.model.parameters(), lr=lr)

        train_ep_losses = []
        val_ep_losses = []
        epoch_train_f1s = []
        epoch_val_f1s = []
        best_model_path = ""
        for epoch in range(1, epochs + 1):
            train_bt_losses = []
            val_bt_losses = []
            train_f1s = []
            val_f1s = []
            self.model.train()
            batch_cnt = 0
            for batch_data, batch_label in train_dl:
                batch_cnt += 1
                # print(f"epoch: {epoch}, batch:{batch_cnt}")
                model_op = self.model(**batch_data)
                batch_loss = loss_fn(
                    model_op.logits,
                    batch_label,
                    batch_data["attention_mask"],
                    len(config.ner_token_to_idx)
                )

                train_bt_losses.append(batch_loss.detach().cpu().item())
                batch_loss.backward()
                optimizer.step()

                batch_f1 = self.score_batch(model_op.logits, batch_label, batch_data["attention_mask"], "accuracy")
                train_f1s.append(batch_f1)

            self.model.eval()
            for batch_data, batch_label in val_dl:
                model_op = self.model(**batch_data)
                batch_loss = loss_fn(
                    model_op.logits,
                    batch_label,
                    batch_data["attention_mask"],
                    len(config.ner_token_to_idx)
                )
                val_bt_losses.append(batch_loss.detach().cpu().item())

                batch_f1 = self.score_batch(model_op.logits, batch_label, batch_data["attention_mask"], "accuracy")
                val_f1s.append(batch_f1)

            epoch_train_loss = np.mean(train_bt_losses)
            epoch_val_loss = np.mean(val_bt_losses)
            epoch_train_f1 = np.mean(train_f1s, axis=0)
            epoch_val_f1 = np.mean(val_f1s, axis=0)

            print(f"------------{epoch}----------------")
            print(epoch_train_loss)
            print(epoch_val_loss)
            print(epoch_train_f1)
            print(epoch_val_f1)

            train_ep_losses.append(epoch_train_loss)
            val_ep_losses.append(epoch_val_loss)
            epoch_train_f1s.append(epoch_train_f1)
            epoch_val_f1s.append(epoch_val_f1)

            if epoch_val_loss < min_val_loss:
                best_model_name = f"NER_clf_{epoch_val_loss}.pth"
                best_model_path = os.path.join(config.MODEL_DIR, best_model_name)
                self.save_model(best_model_path)
                min_val_loss = epoch_val_loss

        return best_model_path


if __name__ == "__main__":
    # ner_model = NER_Classifier(unfreeze_layers=10)
    # ner_model_path = ner_model.train_model(1e-6, 2, 32)
    # del ner_model

    ner_model_path = r"D:\MTech\sem3\NLU\assignments\assignment2\ER_proj\saved_models\NER_clf_0.9314111386026654.pth"
    ner_model2 = NER_Classifier()
    ner_model2.load_model(ner_model_path)
    test_acc = ner_model2.check_test_accuracy()
    print(test_acc)


