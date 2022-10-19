import os
import pickle
from pathlib import Path
import argparse

import pytorch_lightning as pl
from finetune_rag import GenerativeQAModule
from transformers import RagRetriever
import query


def main(args):
    pl.seed_everything(args.seed)

    model = GenerativeQAModule.load_from_checkpoint(
        checkpoint_path=Path(args.model_dir) / "model.ckpt")

    model.eval()

    pre_qads = query.query(model)

    # Swap out retriever.
    hparams = pickle.load(open(Path(model.output_dir) / "hparams.pkl", "rb"))
    index_name = "custom"
    dataset_path = str(
        Path(args.dataset_dir) /
        "knowledge_base/empty-dataset/my_knowledge_dataset")
    index_path = str(
        Path(args.dataset_dir) /
        "knowledge_base/empty-dataset/my_knowledge_dataset_hnsw_index.faiss")
    use_dummy_dataset = False

    retriever = RagRetriever.from_pretrained(
        hparams.model_name_or_path,
        index_name=index_name,
        passages_path=dataset_path,
        index_path=index_path,
        use_dummy_dataset=use_dummy_dataset)

    model.model.set_retriever(retriever)

    post_qads = query.query(model)
    query.compare_and_print(pre_qads, post_qads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
                        type=str,
                        required=True,
                        help="Path to where model is stored.")
    parser.add_argument("--dataset_dir",
                        default="/mnt/ds3lab-scratch/laanthony/datasets/",
                        type=str,
                        help="Path to where model is stored.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    main(args)
