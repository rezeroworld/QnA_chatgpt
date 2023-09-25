import docx
import numpy as np
import pandas as pd
import unicodedata
import openai
import os
import time
import tiktoken
import toml

'''
with open('config.toml', 'r') as f:
    config = toml.load(f)

openai.api_key = os.getenv(config['openai']['openai_api_key'])
encoding = tiktoken.get_encoding(config['openai']['encoding'])
separator_len = len(encoding.encode(config['openai']['separator']))

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    
}

f"Context separator contains {separator_len} tokens"

###################################################################################################################

df = pd.read_csv('qa_embeddings.csv', index_col=0, header=0)
q2embed = dict(zip(df.index, df.loc[:, df.columns != "answers"].to_numpy()))
q2a = dict(zip(df.index, df.loc[:,df.columns == "answers"].to_numpy()))
'''

def get_embedding(text, config):
    result = openai.Embedding.create(
      model=config['openai']['embedding_model'],
      input=text
    )
    return result["data"][0]["embedding"]

def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query, contexts, config):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query, config)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

def construct_prompt(question: str, embeds, answers, config) -> str:
    """
    Fetch relevant 
    """
    encoding = tiktoken.get_encoding(config['openai']['encoding'])
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, embeds, config)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.   
        document_section = answers[section_index][0]

        chosen_sections_len += len(document_section) + len(encoding.encode(config['openai']['separator']))
        if chosen_sections_len > config['openai']['max_section_len']:
            break
            
        chosen_sections.append(config['openai']['separator'] + document_section.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n\n A:"

def answer_query_with_context(
    query: str,
    embeds,
    answers,
    config,
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        embeds,
        answers,
        config
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                temperature = config['openai']['temperature'],
                max_tokens = config['openai']['max_tokens'],
                model = config['openai']['completion_model'],
            )

    return (prompt + " " + response["choices"][0]["text"]).strip(" \n")