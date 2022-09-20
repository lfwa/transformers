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
    "Polar bear numbers are increasing.",
    "The increase in global freshwater flow, based on data from 1994 to 2006, was about 18%.",
    "What is climate change?", "How do you spell licorice"
]


def query(model):
    qads = []

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
        generated_string = model.tokenizer.batch_decode(
            generated, skip_special_tokens=True)[0]

        qads.append((question, generated_string, []))

        for doc in model.model.retriever.index.get_doc_dicts(
                docs_dict["doc_ids"])[0]["text"]:
            qads[-1][-1].append(doc)

    return qads


def compare_and_print(pre_qads, post_qads):
    diff = 0
    answers = []

    for pre, post in zip(pre_qads, post_qads):
        pre_q, pre_a, pre_docs = pre
        post_q, post_a, post_docs = post

        print(f"\nQ: {pre_q}")
        print(f"Pre A: {pre_a}")
        print(f"  {pre_docs}")
        print(f"Post A: {post_a}")
        print(f"  {post_docs}")

        diff += int(pre_a != post_a)

        if pre_a not in answers:
            answers.append(pre_a)
        if post_a not in answers:
            answers.append(post_a)

    print("--------------------------------")
    print("SUMMARY")
    print("--------------------------------")
    print(f"Answers given: {answers}")
    print(f"Differing answers before and after swapping retriever: {diff}")
