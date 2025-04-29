def select_component(
    component_type, component_map, chat_args
):
    components = get_conversation_components(
        chat_args.conversation_id
    )
    previous_component = components[component_type]
    
    if previous_component:
        builder = component_map[previous_component]
        return previous_component, builder(chat_args)
    else: 
        random_name = random_component_by_score(component_type, component_map)
        builder = component_map[random_name]
        return random_name, builder(chat_args)

def build_chat(chat_args: ChatArgs):
    retriever_name, retriever = select_component(
        "retriever",
        retriver_map,
        chat_args
    )
    
    llm_name, llm = select_component(
        "llm",
        llm_map,
        chat_args
    )
    
    memory_name, memory = select_component(
        "memory",
        memory_map,
        chat_args
    )
    
    print(f"Running chain with: memory: {memory_name}, llm: {llm_name}, retriever: {retriever_name}")
    
    set_conversation_components(
        chat_args.conversation_id,
        llm=llm_name,
        retriever=retriever_name,
        memory=memory_name
    )
        
    condense_question_llm = ChatOpenAI(streaming=False)
    
    # trace = langfuse.trace(
    #     CreateTrace(
    #         id=chat_args.conversation_id,
    #         metadata=chat_args.metadata
    #     )
    # )
    
    return StreamingConversationalRerievalChain.from_llm(
        llm=llm,
        condense_question_llm=condense_question_llm,
        memory=memory,
        retriever=retriever,
        # callbacks=[trace.getNewHandler()]
    )
    