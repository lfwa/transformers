"""Query loaded RAG model from command prompt."""

import argparse

import torch

from transformers import BartForConditionalGeneration, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration, RagTokenizer

def infer_model_type(model_name_or_path):
    if "token" in model_name_or_path:
        return "rag_token"
    if "sequence" in model_name_or_path:
        return "rag_sequence"
    if "bart" in model_name_or_path:
        return "bart"
    return None


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
        "--print_docs",
        action="store_true",
        help="If True, prints docs retried while generating.",
    )
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


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

        if args.print_docs:
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

        if args.print_docs:
            for q, a, d in zip(questions, answers, retrieved_docs):
                print(f"Docs: {d}\nQ: {q} - A: {a}")
        else:
            for q, a in zip(questions, answers):
                print("Q: {} - A: {}".format(q, a))
        print("")
        return answers


def interactive_query(args, rag_model):
    while True:
        # Ask for user input question
        question = input("Enter your question: ")
        evaluate_batch_e2e(args, rag_model, [question])


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

    checkpoint = args.model_name_or_path

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
    
    interactive_query(args, model)


if __name__ == "__main__":
    args = get_args()
    main(args)
