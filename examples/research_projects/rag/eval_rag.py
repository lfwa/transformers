""" Evaluation script for RAG models."""

import argparse
import ast
import logging
import os
import sys
import time

import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

from transformers import BartForConditionalGeneration, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration, RagTokenizer
from transformers import logging as transformers_logging

sys.path.append(os.path.join(os.getcwd()))  # noqa: E402 # isort:skip
from utils_rag import exact_match_score, f1_score  # noqa: E402 # isort:skip

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

transformers_logging.set_verbosity_info()


def infer_model_type(model_name_or_path):
    if "token" in model_name_or_path:
        return "rag_token"
    if "sequence" in model_name_or_path:
        return "rag_sequence"
    if "bart" in model_name_or_path:
        return "bart"
    return None


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def get_scores(args, preds_path, gold_data_path):
    hypos = [line.strip() for line in open(preds_path, "r").readlines()]
    answers = []

    if args.gold_data_mode == "qa":
        data = pd.read_csv(gold_data_path, sep="\t", header=None)
        for answer_list in data[1]:
            ground_truths = ast.literal_eval(answer_list)
            answers.append(ground_truths)
    else:
        references = [
            line.strip() for line in open(gold_data_path, "r").readlines()
        ]
        answers = [[reference] for reference in references]

    f1 = em = total = 0
    for prediction, ground_truths in zip(hypos, answers):
        total += 1
        em += metric_max_over_ground_truths(exact_match_score, prediction,
                                            ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction,
                                            ground_truths)

    em = 100.0 * em / total
    f1 = 100.0 * f1 / total

    logger.info(f"F1: {f1:.2f}")
    logger.info(f"EM: {em:.2f}")


def get_precision_at_k(args, preds_path, gold_data_path):
    k = args.k
    hypos = [line.strip() for line in open(preds_path, "r").readlines()]
    references = [
        line.strip() for line in open(gold_data_path, "r").readlines()
    ]

    em = total = 0
    for hypo, reference in zip(hypos, references):
        hypo_provenance = set(hypo.split("\t")[:k])
        ref_provenance = set(reference.split("\t"))
        total += 1
        em += len(hypo_provenance & ref_provenance) / k

    em = 100.0 * em / total
    logger.info(f"Precision@{k}: {em: .2f}")


def evaluate_batch_retrieval(args, rag_model, questions):
    def strip_title(title):
        if title.startswith('"'):
            title = title[1:]
        if title.endswith('"'):
            title = title[:-1]
        return title

    retriever_input_ids = rag_model.retriever.question_encoder_tokenizer.batch_encode_plus(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )["input_ids"].to(args.device)

    question_enc_outputs = rag_model.rag.question_encoder(retriever_input_ids)
    question_enc_pool_output = question_enc_outputs[0]

    result = rag_model.retriever(
        retriever_input_ids,
        question_enc_pool_output.cpu().detach().to(torch.float32).numpy(),
        prefix=rag_model.rag.generator.config.prefix,
        n_docs=rag_model.config.n_docs,
        return_tensors="pt",
    )
    all_docs = rag_model.retriever.index.get_doc_dicts(result.doc_ids)
    provenance_strings = []
    for docs in all_docs:
        provenance = [strip_title(title) for title in docs["title"]]
        provenance_strings.append("\t".join(provenance))
    return provenance_strings


def time_generation(args, rag_model, questions, tokenizer):
    total_time = []
    with torch.no_grad():
        for question in questions:
            inputs = tokenizer(question, return_tensors="pt")
            input_ids = inputs["input_ids"].to(args.device)
            question_hidden_states = rag_model.question_encoder(input_ids)[0]
            docs_dict = rag_model.retriever(
                input_ids.cpu().numpy(),
                question_hidden_states.cpu().numpy(),
                return_tensors="pt").to(args.device)
            doc_scores = torch.bmm(
                question_hidden_states.unsqueeze(1),
                docs_dict["retrieved_doc_embeds"].float().transpose(
                    1, 2)).squeeze(1)

            start_time = time.time()

            generated = rag_model.generate(
                context_input_ids=docs_dict["context_input_ids"],
                context_attention_mask=docs_dict["context_attention_mask"],
                doc_scores=doc_scores,
            )

            elapsed = time.time() - start_time
            total_time.append(elapsed)
            print(f"Time for question: {elapsed}")

    return np.average(total_time)


def iterate_top(args, rag_model, questions, tokenizer):
    total_time = []
    outputs = []
    with torch.no_grad():
        for question in questions:
            inputs = tokenizer(question, return_tensors="pt")
            input_ids = inputs["input_ids"].to(args.device)
            question_hidden_states = rag_model.question_encoder(input_ids)[0]
            docs_dict = rag_model.retriever(
                input_ids.cpu().numpy(),
                question_hidden_states.cpu().numpy(),
                return_tensors="pt",
                n_docs=10).to(args.device)
            doc_scores = torch.bmm(
                question_hidden_states.unsqueeze(1),
                docs_dict["retrieved_doc_embeds"].float().transpose(
                    1, 2)).squeeze(1)

            start_time = time.time()

            context_input_id_reshape = (
                1, docs_dict["context_input_ids"].size()[1])
            context_att_mask_reshape = (
                1, docs_dict["context_attention_mask"].size()[1])
            retrieved_doc_embed_reshape = (
                docs_dict["retrieved_doc_embeds"].size()[0], 1,
                docs_dict["retrieved_doc_embeds"].size()[2])
            doc_id_reshape = (docs_dict["doc_ids"].size()[0], 1
                              )  # TODO: Might need to be swapped around
            doc_score_reshape = (doc_scores.size()[0], 1
                                 )  # TODO: Might need to be swapped around

            # TODO: Might break when changing to a different batch size
            for i in range(doc_scores.size()[1]):
                context_input_id = docs_dict["context_input_ids"][i].reshape(
                    context_input_id_reshape)
                context_att_mask = docs_dict["context_attention_mask"][
                    i].reshape(context_att_mask_reshape)
                retrieved_doc_embed = docs_dict["retrieved_doc_embeds"][
                    0, i].reshape(retrieved_doc_embed_reshape)
                doc_id = docs_dict["doc_ids"][0, i].reshape(doc_id_reshape)
                doc_score = doc_scores[0, i].reshape(doc_score_reshape)

                generated = rag_model.generate(
                    context_input_ids=context_input_id,
                    context_attention_mask=context_att_mask,
                    doc_scores=doc_score,
                )

                generated_string = tokenizer.batch_decode(
                    generated, skip_special_tokens=True)
                outputs.append(generated_string)

            elapsed = time.time() - start_time
            total_time.append(elapsed)
            print(f"question: {question}")
            print(f"answers: {outputs}")
            print(f"Time for question: {elapsed}")

    return np.average(total_time)


def only_retrieve(args, rag_model, questions):
    with torch.no_grad():
        inputs_dict = rag_model.retriever.question_encoder_tokenizer.batch_encode_plus(
            questions, return_tensors="pt", padding=True, truncation=True)

        input_ids = inputs_dict.input_ids.to(args.device)

        if args.print_docs or args.save_docs_as_kb_file is not None:
            retrieved_docs = []
            retrieved_titles = []
            question_enc_outputs = rag_model.rag.question_encoder(input_ids)
            question_enc_pool_output = question_enc_outputs[0]

            result = rag_model.retriever(
                input_ids,
                question_enc_pool_output.cpu().detach().to(
                    torch.float32).numpy(),
                prefix=rag_model.rag.generator.config.prefix,
                n_docs=rag_model.config.n_docs,
                return_tensors="pt",
            )
            all_docs = rag_model.retriever.index.get_doc_dicts(result.doc_ids)
            for docs in all_docs:
                texts = [text for text in docs["text"]]
                titles = [title for title in docs["title"]]
                retrieved_docs.append(texts)
                retrieved_titles.append(titles)

        if args.save_docs_as_kb_file is not None:
            with open(args.save_docs_as_kb_file, "a+") as f:
                for titles, texts in zip(retrieved_titles, retrieved_docs):
                    for title, text in zip(titles, texts):
                        f.write(f"{title}\t{text}\n")

        if args.print_predictions and args.print_docs:
            for q, d in zip(questions, retrieved_docs):
                logger.info(f"Q: {q}\nDocs: {d}")
        elif args.print_predictions:
            for q in questions:
                logger.info("Q: {}".format(q))

        if args.write_docs_to_file:
            for q, d in zip(questions, retrieved_docs):
                with open(q + "_top1000.txt", "w") as f:
                    f.write(str(d))

        return questions


def evaluate_batch_e2e(args, rag_model, questions):
    with torch.no_grad():
        inputs_dict = rag_model.retriever.question_encoder_tokenizer.batch_encode_plus(
            questions, return_tensors="pt", padding=True, truncation=True)

        input_ids = inputs_dict.input_ids.to(args.device)
        attention_mask = inputs_dict.attention_mask.to(args.device)
        outputs = rag_model.generate(  # rag_model overwrites generate
            input_ids,
            attention_mask=attention_mask,
            num_beams=args.num_beams,
            min_length=args.min_length,
            max_length=args.max_length,
            early_stopping=False,
            num_return_sequences=1,
            bad_words_ids=[
                [0, 0]
            ],  # BART likes to repeat BOS tokens, dont allow it to generate more than one
        )
        answers = rag_model.retriever.generator_tokenizer.batch_decode(
            outputs, skip_special_tokens=True)

        if args.print_docs or args.save_docs_as_kb_file is not None:
            retrieved_docs = []
            retrieved_titles = []
            question_enc_outputs = rag_model.rag.question_encoder(input_ids)
            question_enc_pool_output = question_enc_outputs[0]

            result = rag_model.retriever(
                input_ids,
                question_enc_pool_output.cpu().detach().to(
                    torch.float32).numpy(),
                prefix=rag_model.rag.generator.config.prefix,
                n_docs=rag_model.config.n_docs,
                return_tensors="pt",
            )
            all_docs = rag_model.retriever.index.get_doc_dicts(result.doc_ids)
            for docs in all_docs:
                texts = [text for text in docs["text"]]
                titles = [title for title in docs["title"]]
                retrieved_docs.append(texts)
                retrieved_titles.append(titles)

        if args.save_docs_as_kb_file is not None:
            with open(args.save_docs_as_kb_file, "a+") as f:
                for titles, texts in zip(retrieved_titles, retrieved_docs):
                    for title, text in zip(titles, texts):
                        f.write(f"{title}\t{text}\n")

        if args.print_predictions and args.print_docs:
            for q, a, d in zip(questions, answers, retrieved_docs):
                logger.info(f"Q: {q} - A: {a}\nDocs: {d}")
        elif args.print_predictions:
            for q, a in zip(questions, answers):
                logger.info("Q: {} - A: {}".format(q, a))

        return answers


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        choices=["rag_sequence", "rag_token", "bart"],
        type=str,
        help=
        ("RAG model type: rag_sequence, rag_token or bart, if none specified, the type is inferred from the"
         " model_name_or_path"),
    )
    parser.add_argument(
        "--index_name",
        default=None,
        choices=["exact", "compressed", "legacy", "custom"],
        type=str,
        help="RAG model retriever type",
    )
    parser.add_argument(
        "--index_path",
        default=None,
        type=str,
        help="Path to the retrieval index",
    )
    parser.add_argument(
        "--passages_path",
        default=None,
        type=str,
        help="Path to dataset",
    )
    parser.add_argument("--n_docs",
                        default=5,
                        type=int,
                        help="Number of retrieved docs")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help=
        "Path to pretrained checkpoints or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--eval_mode",
        choices=["e2e", "retrieval"],
        default="e2e",
        type=str,
        help=
        ("Evaluation mode, e2e calculates exact match and F1 of the downstream task, retrieval calculates"
         " precision@k."),
    )
    parser.add_argument("--k",
                        default=1,
                        type=int,
                        help="k for the precision@k calculation")
    parser.add_argument(
        "--evaluation_set",
        default=None,
        type=str,
        required=True,
        help="Path to a file containing evaluation samples",
    )
    parser.add_argument(
        "--gold_data_path",
        default=None,
        type=str,
        required=True,
        help="Path to a tab-separated file with gold samples",
    )
    parser.add_argument(
        "--gold_data_mode",
        default="qa",
        type=str,
        choices=["qa", "ans"],
        help=
        ("Format of the gold data file"
         "qa - a single line in the following format: question [tab] answer_list"
         "ans - a single line of the gold file contains the expected answer string"
         ),
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="predictions.txt",
        help=
        "Name of the predictions file, to be stored in the checkpoints directory",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help=
        "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--recalculate",
        help="Recalculate predictions even if the prediction file exists",
        action="store_true",
    )
    parser.add_argument(
        "--num_beams",
        default=4,
        type=int,
        help="Number of beams to be used when generating answers",
    )
    parser.add_argument("--min_length",
                        default=1,
                        type=int,
                        help="Min length of the generated answers")
    parser.add_argument("--max_length",
                        default=50,
                        type=int,
                        help="Max length of the generated answers")

    parser.add_argument(
        "--print_predictions",
        action="store_true",
        help="If True, prints predictions while evaluating.",
    )
    parser.add_argument(
        "--print_docs",
        action="store_true",
        help="If True, prints docs retried while generating.",
    )
    parser.add_argument(
        "--save_docs_as_kb_file",
        type=str,
        default=None,
        help=
        "File to save retrieved documents (text and title). If None then no saving is performed. Note that this operation appends to the file!"
    )
    parser.add_argument("--write_docs_to_file",
                        action="store_true",
                        help="Write docs to file for each answer.")
    parser.add_argument(
        "--timing",
        action="store_true",
        help="If True, runs timing query for each question instead.")
    parser.add_argument(
        "--iterate_top",
        action="store_true",
        help=
        "If True, runs iteration over top 10 documents and see if output changes."
    )
    parser.add_argument(
        "--only_retrieve",
        action="store_true",
        help=
        "If True, only retrieves top documents and does not feed them through the BERT part of RAG."
    )
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def main(args):
    model_kwargs = {}
    if args.model_type is None:
        args.model_type = infer_model_type(args.model_name_or_path)
        assert args.model_type is not None
    if args.model_type.startswith("rag"):
        model_class = RagTokenForGeneration if args.model_type == "rag_token" else RagSequenceForGeneration
        model_kwargs["n_docs"] = args.n_docs
        if args.index_name is not None:
            model_kwargs["index_name"] = args.index_name
        if args.index_path is not None:
            model_kwargs["index_path"] = args.index_path
        if args.passages_path is not None:
            model_kwargs["passages_path"] = args.passages_path
            model_kwargs["use_dummy_dataset"] = False
    else:
        model_class = BartForConditionalGeneration

    checkpoints = ([
        f.path for f in os.scandir(args.model_name_or_path) if f.is_dir()
    ] if args.eval_all_checkpoints else [args.model_name_or_path])

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    score_fn = get_scores if args.eval_mode == "e2e" else get_precision_at_k
    evaluate_batch_fn = only_retrive if args.only_retrieve else (
        time_generation if args.timing else
        (iterate_top if args.iterate_top else
         (evaluate_batch_e2e if args.eval_mode ==
          "e2e" else evaluate_batch_retrieval)))

    for checkpoint in checkpoints:
        if os.path.exists(args.predictions_path) and (not args.recalculate):
            logger.info(
                "Calculating metrics based on an existing predictions file: {}"
                .format(args.predictions_path))
            score_fn(args, args.predictions_path, args.gold_data_path)
            continue

        logger.info("***** Running evaluation for {} *****".format(checkpoint))
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("  Predictions will be stored under {}".format(
            args.predictions_path))

        if args.model_type.startswith("rag"):
            retriever = RagRetriever.from_pretrained(checkpoint,
                                                     **model_kwargs)
            model = model_class.from_pretrained(checkpoint,
                                                retriever=retriever,
                                                **model_kwargs)
            model.retriever.init_retrieval()
        else:
            model = model_class.from_pretrained(checkpoint, **model_kwargs)
        model.to(args.device)

        timings = []
        if args.timing or args.iterate_top:
            tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        with open(args.evaluation_set,
                  "r") as eval_file, open(args.predictions_path,
                                          "w") as preds_file:
            questions = []
            for line in tqdm(eval_file):
                questions.append(line.strip())
                if len(questions) == args.eval_batch_size:
                    if args.timing or args.iterate_top:
                        answers = evaluate_batch_fn(args, model, questions,
                                                    tokenizer)
                        timings.append(answers)
                    else:
                        answers = evaluate_batch_fn(args, model, questions)
                        preds_file.write("\n".join(answers) + "\n")
                        preds_file.flush()
                    questions = []
            if len(questions) > 0:
                if args.timing or args.iterate_top:
                    answers = evaluate_batch_fn(args, model, questions,
                                                tokenizer)
                    timings.append(answers)
                else:
                    answers = evaluate_batch_fn(args, model, questions)
                    preds_file.write("\n".join(answers))
                    preds_file.flush()

            if not (args.timing or args.iterate_top):
                score_fn(args, args.predictions_path, args.gold_data_path)
        if args.timing:
            print(f"Average time: {np.average(overall)}")


if __name__ == "__main__":
    args = get_args()
    main(args)
