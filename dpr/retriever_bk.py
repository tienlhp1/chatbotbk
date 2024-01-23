import os
import time
import json
import random
import torch
import pandas as pd
import faiss
from datasets import load_dataset
from model import BiEncoder
from util import get_tokenizer
from preprocess import tokenise, preprocess_question
from pyvi.ViTokenizer import tokenize


class DPRRetriever:
    def __init__(
        self, args, q_encoder=None, ctx_encoder=None, biencoder=None, save_type=""
    ):
        start = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.save_type = save_type
        self.dpr_tokenizer = get_tokenizer(self.args.BE_checkpoint)
        if biencoder is not None:
            self.biencoder = biencoder
        elif q_encoder is not None and ctx_encoder is not None:
            self.biencoder = BiEncoder(
                model_checkpoint=self.args.BE_checkpoint,
                q_encoder=q_encoder,
                ctx_encoder=ctx_encoder,
                representation=self.args.BE_representation,
                q_fixed=self.args.q_fixed,
                ctx_fixed=self.args.ctx_fixed,
            )
        else:
            self.biencoder = BiEncoder(
                model_checkpoint=self.args.BE_checkpoint,
                representation=self.args.BE_representation,
                q_fixed=self.args.q_fixed,
                ctx_fixed=self.args.ctx_fixed,
            )
            self.biencoder.load_state_dict(torch.load(self.args.biencoder_path))

        self.biencoder.to(self.device)
        self.q_encoder, self.ctx_encoder = self.biencoder.get_models()
        self.corpus = load_dataset(
            "csv", data_files=self.args.corpus_file, split="train"
        )
        if self.args.index_path:
            self.corpus.load_faiss_index("embeddings", self.args.index_path)
        else:
            self.corpus = self.get_index()
        print("self.corpus: ", self.corpus[:10])
        end = time.time()
        print(end - start)

    def get_index(self):
        self.ctx_encoder.to("cuda").eval()
        with torch.no_grad():
            corpus_with_embeddings = self.corpus.map(
                lambda example: {
                    "embeddings": self.ctx_encoder.get_representation(
                        self.dpr_tokenizer.encode_plus(
                            example["answer"],
                            padding="max_length",
                            truncation=True,
                            max_length=self.args.ctx_len,
                            return_tensors="pt",
                        )["input_ids"].to(self.device),
                        self.dpr_tokenizer.encode_plus(
                            example["answer"],
                            padding="max_length",
                            truncation=True,
                            max_length=self.args.ctx_len,
                            return_tensors="pt",
                        )["attention_mask"].to(self.device),
                    )[0]
                    .to("cpu")
                    .numpy()
                }
            )
        corpus_with_embeddings.add_faiss_index(
            column="embeddings", metric_type=faiss.METRIC_INNER_PRODUCT
        )
        index_path = self.args.biencoder_path.split("/")[-1]
        index_path = "outputs/index/index_" + self.save_type + ".faiss"
        corpus_with_embeddings.save_faiss_index("embeddings", index_path)
        return corpus_with_embeddings

    def retrieve(self, question, top_k=100, segmented=False):
        start = time.time()
        self.q_encoder.to(self.device).eval()

        if segmented:
            tokenized_question = question
        else:
            tokenized_question = tokenise(
                preprocess_question(question, remove_end_phrase=False), tokenize
            )

        with torch.no_grad():
            Q = self.dpr_tokenizer.encode_plus(
                tokenized_question,
                padding="max_length",
                truncation=True,
                max_length=self.args.q_len,
                return_tensors="pt",
            )
            question_embedding = (
                self.q_encoder.get_representation(
                    Q["input_ids"].to(self.device), Q["attention_mask"].to(self.device)
                )[0]
                .to("cpu")
                .numpy()
            )
            scores, retrieved_examples = self.corpus.get_nearest_examples(
                "embeddings", question_embedding, k=top_k
            )
            retrieved_ids = retrieved_examples["id"]
        end = time.time()
        # print(end - start)
        return retrieved_ids, scores

    def test_on_data(self, top_k=[100], segmented=True, train=False):
        result = []
        dtest = pd.read_csv(os.path.join(self.args.data_dir, "test.csv"))
        dval = pd.read_csv(os.path.join(self.args.data_dir, "val.csv"))
        if train:
            dtrain = pd.read_csv(os.path.join(self.args.data_dir, "train.csv"))
            train_retrieved = self.retrieve_on_data(
                dtrain, name="train", top_k=max(top_k), segmented=segmented
            )
        test_retrieved = self.retrieve_on_data(
            dtest, name="test", top_k=max(top_k), segmented=segmented
        )
        val_retrieved = self.retrieve_on_data(
            dval, name="val", top_k=max(top_k), segmented=segmented
        )

        for k in top_k:
            rlt = {}
            strk = str(k)
            rlt[strk] = {}
            test_retrieved_k = [x[:k] for x in test_retrieved]
            val_retrieved_k = [x[:k] for x in val_retrieved]

            print("Testing hit scores with top_{}:".format(k))
            val_hit_acc, val_all_acc = self.calculate_score(dval, val_retrieved_k)
            rlt[strk]["val_hit"] = val_hit_acc
            rlt[strk]["val_all"] = val_all_acc
            print("\tVal hit acc: {:.4f}%".format(val_hit_acc * 100))
            print("\tVal all acc: {:.4f}%".format(val_all_acc * 100))
            test_hit_acc, test_all_acc = self.calculate_score(dtest, test_retrieved_k)
            rlt[strk]["test_hit"] = test_hit_acc
            rlt[strk]["test_all"] = test_all_acc
            print("\tTest hit acc: {:.4f}%".format(test_hit_acc * 100))
            print("\tTest all acc: {:.4f}%".format(test_all_acc * 100))
            result.append(rlt)
        # name = self.args.biencoder_path.split("/")
        save_file = "outputs/testdpr_" + self.save_type + ".json"
        with open(save_file, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    def retrieve_on_data(self, df, name, top_k=100, segmented=False):
        count = 0
        acc = 0
        retrieved_list = []
        if not segmented:
            tokenized_questions = []
            for i in range(len(df)):
                tokenized_question = tokenise(
                    preprocess_question(df["question"][i], remove_end_phrase=False),
                    tokenize,
                )
                tokenized_questions.append(tokenized_question)
            df["tokenized_question"] = tokenized_questions

        for i in range(len(df)):
            tokenized_question = df["tokenized_question"][i]
            retrieved_ids, _ = self.retrieve(tokenized_question, top_k, segmented=True)
            retrieved_list.append(retrieved_ids)
        save_file = "outputs/" + self.save_type + "_" + name + "_retrieved.json"
        with open(save_file, "w") as f:
            json.dump(retrieved_list, f, ensure_ascii=False, indent=4)
        return retrieved_list

    def find_neg(self, df, name, no_negs=3, segmented=True):
        retrieved_list = self.retrieve_on_data(df, name, no_negs + 5, segmented)
        tokenized_ques = []
        ans_id = []
        new_neg = []

        ttokenized_ques = df["tokenized_question"].tolist()
        tans_id = df["ans_id"].tolist()
        tnew_neg = []

        for i in range(len(df)):
            retrieved_ids = retrieved_list[i]
            ans_ids = tans_id[i]
            ans_ids = str(ans_ids)
            ans_ids = [int(x) for x in ans_ids.split(", ")]
            # old_neg_ids = df['neg_ids'][i][1:-1]
            # old_neg_ids = [int(x) for x in old_neg_ids.split(", ")]
            # if not shuffle_negs:
            #    kept_neg_ids = old_neg_ids[:3]
            # else:
            #    kept_neg_ids = random.sample(old_neg_ids, 3)
            new_neg_ids = [
                x for x in retrieved_ids if x not in ans_ids
            ]  # and x not in kept_neg_ids]
            new_neg_ids = new_neg_ids[:no_negs]

            tnew_neg.append(new_neg_ids)

            for x in ans_ids:
                tokenized_ques.append(ttokenized_ques[i])
                ans_id.append(x)
                new_neg.append(new_neg_ids)

        dff = pd.DataFrame()
        dff["tokenized_question"] = tokenized_ques
        dff["ans_id"] = ans_id
        dff["neg_ids"] = new_neg

        dt = pd.DataFrame()
        dt["tokenized_question"] = ttokenized_ques
        dt["ans_id"] = tans_id
        dt["neg_ids"] = tnew_neg
        return dff, dt

    def increase_neg(self, no_negs=3, segmented=True):
        dtrain = pd.read_csv(os.path.join(self.args.data_dir, "ttrain.csv"))
        dval = pd.read_csv(os.path.join(self.args.data_dir, "tval.csv"))
        dtest = pd.read_csv(os.path.join(self.args.data_dir, "ttest.csv"))

        dnew_train, dttrain = self.find_neg(dtrain, "train", no_negs, segmented)
        dnew_val, dtval = self.find_neg(dval, "val", no_negs, segmented)
        dnew_test, dttest = self.find_neg(dtest, "test", no_negs, segmented)

        dnew_train.to_csv(
            "outputs/data/{}/train.csv".format(self.save_type), index=False
        )
        dnew_val.to_csv("outputs/data/{}/val.csv".format(self.save_type), index=False)
        dnew_test.to_csv("outputs/data/{}/test.csv".format(self.save_type), index=False)

        dttrain.to_csv("outputs/data/{}/ttrain.csv".format(self.save_type), index=False)
        dtval.to_csv("outputs/data/{}/tval.csv".format(self.save_type), index=False)
        dttest.to_csv("outputs/data/{}/ttest.csv".format(self.save_type), index=False)

    def calculate_score(self, df, retrieved_list):
        top_k = len(retrieved_list[0])
        all_count = 0
        hit_count = 0
        for i in range(len(df)):
            retrieved_ids = retrieved_list[i]
            ans_ids = [int(x) for x in df["id"][i].split(", ")]
            for a_id in ans_ids:
                if a_id in retrieved_ids:
                    retrieved_ids.remove(a_id)
            if len(retrieved_ids) == top_k - len(ans_ids):
                all_count += 1
            if len(retrieved_ids) < top_k:
                hit_count += 1

        all_acc = all_count / len(df)
        hit_acc = hit_count / len(df)
        return hit_acc, all_acc
