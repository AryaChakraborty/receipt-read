def ask_csv(question, hf_token, embed_model_name='BAAI/bge-small-en-v1.5', csv_path = 'csv_data/receipts_data.csv', llm_model='mistralai/Mistral-7B-Instruct-v0.2'):

    from beyondllm import source,retrieve,embeddings,llms,generator

    data = source.fit(
        path=csv_path,
        dtype="csv",
        chunk_size=1024,
        chunk_overlap=0)

    embed_model = embeddings.HuggingFaceEmbeddings(
        model_name=embed_model_name
    )

    retriever = retrieve.auto_retriever(
        data=data,
        embed_model=embed_model,
        type="cross-rerank",
        mode="OR",
        top_k=2)

    llm = llms.HuggingFaceHubModel(
        model=llm_model,
        token=hf_token
    )


    system_prompt = f"""
    <s>[INST]
    You are an AI Assistant.
    Please provide direct answers to questions.
    [/INST]
    </s>
    """

    pipeline = generator.Generate(
        question=question,
        retriever=retriever,
        system_prompt=system_prompt,
        llm=llm)

    answer = pipeline.call()

    return answer