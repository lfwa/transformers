import os
import pickle
from pathlib import Path

from finetune_rag import GenerativeQAModule
from transformers import RagRetriever
import query


def main():
    model = GenerativeQAModule.load_from_checkpoint(
        checkpoint_path="../rag/experiments/1/model.ckpt")

    pre_qads = query.query(model)

    # Swap out retriever.
    hparams = pickle.load(open(Path(model.output_dir) / "hparams.pkl", "rb"))
    index_name = "custom"
    dataset_path = "../datasets/knowledge_base/empty-dataset/my_knowledge_dataset"
    index_path = "../datasets/knowledge_base/empty-dataset/my_knowledge_dataset_hnsw_index.faiss"
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
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    main()