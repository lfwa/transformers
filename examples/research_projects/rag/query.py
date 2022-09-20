import pickle
from pathlib import Path

from finetune_rag import GenerativeQAModule

questions = [
    "The unusual heat was most noteworthy in Europe, which had its warmest April on record, and Australia, which had its second-warmest.",
    "Climate change is not man made.", "Climate change is man made.",
    "Climate change is a hoax.", "Climate change is real.",
    "The oceans are getting warmer due to climate change.",
    "The oceans are not getting warmer.", "Sea levels are rising",
    "Sea levels are not rising.",
    "Humans emissions and activities have significantly contributed to climate change.",
    "Humans emissions and activities have NOT significantly contributed to climate change.",
    "Climate change causes more extreme weather events.",
    "Climate change does not cause more extreme weather events.",
    "Renewable energy is just an excuse for making coorperations more profit.",
    "Animals will adapt to climate change.",
    "Polar bear numbers are increasing.", "What is climate change?",
    "How do you spell licorice"
]


def query(model):
    for question in questions:
        # Tokenize the question.
        input_ids = model.tokenizer.question_encoder(
            question, return_tensors="pt")["input_ids"].detach()

        question_hidden_states = model.model.question_encoder(input_ids)[0]
        docs_dict = model.model.retriever(
            input_ids.numpy(),
            question_hidden_states.detach().numpy(),
            return_tensors="pt")

        # Give the question to RAG and have it generate an answer!
        generated = model.model.generate(input_ids)

        # Convert the answer tokens back into a single string.
        generated_string = model.add_model_specific_argstokenizer.batch_decode(
            generated, skip_special_tokens=True)[0]

        print("\nQ: " + question)
        print("A: " + generated_string)

        print("Retrieved documents:")
        for doc in model.model.retriever.index.get_doc_dicts(
                docs_dict["doc_ids"])[0]["text"]:
            print(f"  {doc}")


def main():
    model = GenerativeQAModule.load_from_checkpoint(
        checkpoint_path="../rag/experiments/1/model.ckpt")

    query(model)

    if False:
        # Swap out retriever
        hparams = pickle.load(
            open(Path(model.output_dir) / "hparams.pkl", "rb"))

        config_class = RagConfig if model.is_rag_model else AutoConfig
        config = config_class.from_pretrained(hparams.model_name_or_path)

        # set retriever parameters
        config.index_name = "custom"
        config.passages_path = "../datasets/knowledge_base/empty-dataset/my_knowledge_dataset"
        config.index_path = "../datasets/knowledge_base/empty-dataset/my_knowledge_dataset_hnsw_index.faiss"
        config.use_dummy_dataset = False

        retriever = None


if __name__ == "__main__":
    main()
