from typing import Dict, List
import re
import os
import json
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import datetime
import traceback
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])



def transform_messages_for_gemini(messages: List[Dict[str, str]]) -> List[Dict[str, List[str]]]:
    """Transforms messages to the format required by Gemini API."""
    transformed_messages = []
    for message in messages:
        role = message['role']
        content = message['content']
        
        if role == "system":
            # Convert 'system' messages to 'user' messages as Gemini does not support system prompts
            transformed_messages.append({"role": "user", "parts": [content]})
        elif role == "user":
            transformed_messages.append({"role": role, "parts": [content]})
        elif role == "assistant":
            transformed_messages.append({"role": "model", "parts": [content]})

    
    return transformed_messages

def get_completion(messages: List[Dict[str, str]], model: str) -> str:
    transformed_messages = transform_messages_for_gemini(messages)
    
    # Set up the model
    generation_config = {
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 4096,
        "response_mime_type": "text/plain",
    }
    gemini_model = genai.GenerativeModel(
        model_name=model,  # e.g., "gemini-1.5-flash"
        generation_config=generation_config,
        safety_settings={
            'HATE': 'BLOCK_NONE',
            'HARASSMENT': 'BLOCK_NONE',
            'SEXUAL' : 'BLOCK_NONE',
            'DANGEROUS' : 'BLOCK_NONE'
        }
    )
    
    # Start a chat session with the transformed messages
    history = transformed_messages[:-1]
    chat_session = gemini_model.start_chat(history=history)
    
    
    # Get the response
    last_message = transformed_messages[-1]
    response = chat_session.send_message(last_message)
    completion_text = response.text
    print("\ncompletion text")
    print(completion_text)
    print("\n")
    return completion_text

def escape_for_json(input_string):
    escaped_string = json.dumps(input_string)
    return escaped_string

# Load environment variables from .env file
load_dotenv()

#Suppress warnings about elasticsearch certificates
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def run_elasticsearch_query(query, index=["pubmed"]):
    # Retrieve Elasticsearch details from environment variables
    es_host = os.getenv('ELASTICSEARCH_HOST')
    es_user = os.getenv('ELASTICSEARCH_USER')
    es_password = os.getenv('ELASTICSEARCH_PASSWORD')

    # Connect to Elasticsearch
    es = Elasticsearch(
        [es_host],
        http_auth=(es_user, es_password),
        verify_certs=False,  # This will ignore SSL certificate validation
        timeout=120  # Set the timeout to 60 seconds (adjust as needed)
    )

    # Convert the query string to a dictionary
    if isinstance(query, str) and not isinstance(query, dict):
        query_dict = json.loads(query)
    else:
        query_dict = query

    print("\n running es query:")
    print(query_dict)
    print("\n")
    # Execute the query
    response = es.search(query_dict, index=index)

    # Process the response to extract the required information
    results = []
    if response['hits']['hits']:
        for hit in response['hits']['hits']:
            result = {
                "id": "http://www.ncbi.nlm.nih.gov/pubmed/"+str(hit['_id']),
                "title": hit['_source'].get('title', 'No title available'),
                "abstract": hit['_source'].get('abstract', 'No abstract available')
            }
            results.append(result)
    print(f"docs found: {len(results)}")
    return results

def createQuery(query_string: str, size=50): 
    query = {
        "query": {
            "query_string": {
                "query": query_string
            }
        },
        "size": size
    }
    return query

def expand_query_few_shot(df_prior, n, question:str, model:str):
    messages = generate_n_shot_examples_expansion(df_prior, n)
    # Add the user message
    user_message = {
        "role": "user",
        "content": f"""
        Given a biomedical question, generate an Elasticsearch query string that incorporates synonyms and related terms to improve the search results 
        while maintaining precision and relevance to the original question.

        The index contains the fields 'title' and 'abstract', which use the English stemmer. The query string syntax supports the following operators:
        - '+' and '-' for requiring or excluding terms (e.g., +fox -news)
        - '""' for phrase search (e.g., "quick brown")
        - ':' for field-specific search (e.g., title:(quick OR brown))
        - '*' or '?' for wildcards (e.g., qu?ck bro*)
        - '//' for regular expressions (e.g., title:/joh?n(ath[oa]n)/)
        - '~' for fuzzy matching (e.g., quikc~ or quikc~2)
        - '"..."~N' for proximity search (e.g., "fox quick"~5)
        - '^' for boosting terms (e.g., quick^2 fox)
        - 'AND', 'OR', 'NOT' for boolean matching (e.g., ((quick AND fox) OR (brown AND fox) OR fox) AND NOT news)

        Example:
        Question: What are the effects of vitamin D deficiency on the human body?
        Query string: ##(("vitamin d" OR "vitamin d3" OR "cholecalciferol") AND (deficiency OR insufficiency OR "low levels")) AND ("effects" OR "impact" OR "consequences") AND ("human body" OR "human health")##

        Tips:
        - Focus on the main concepts and entities in the question.
        - Use synonyms and related terms to capture variations in terminology.
        - Be cautious not to introduce irrelevant terms that may dilute the search results.
        - Strike a balance between precision and recall based on the specificity of the question.

        Please generate a query string for the following biomedical question and wrap the final query in enclosing ## tags. Example: ##query##
        Question: '''{question}'''
        """
    }
    messages.append(user_message)
    
    print("Prompt Messages:")
    print(messages)
    
    answer =  get_completion(messages, model)
    print("\n Completion:")
    print(answer)
    print("\n")
    return answer

def generate_n_shot_examples_expansion(df, n):
    
    # Initialize the system message
    system_message = {"role": "system", "content": "You are BioASQ-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain."}
    
    # Initialize the list of messages with the system message
    messages = [system_message]
    
    
    if n< 1:
        top_entries = pd.DataFrame()
    else:
        top_entries = df.sort_values(by='f1_score', ascending=False).head(n)
    
    # Loop through each of the top n entries and add the user and assistant messages
    for _, row in top_entries.iterrows():
        question = row['question_body']
        completion = row['completion']
        
        # Replace problematic characters in question
        question = question.replace("/", "\\\\/")
        
        # Add the user message
        user_message = {
            "role": "user",
            "content": f"""
            Given a biomedical question, generate an Elasticsearch query string that incorporates synonyms and related terms to improve the search results 
            while maintaining precision and relevance to the original question.

            The index contains the fields 'title' and 'abstract', which use the English stemmer. The query string syntax supports the following operators:
            - '+' and '-' for requiring or excluding terms (e.g., +fox -news)
            - '""' for phrase search (e.g., "quick brown")
            - ':' for field-specific search (e.g., title:(quick OR brown))
            - '*' or '?' for wildcards (e.g., qu?ck bro*)
            - '//' for regular expressions (e.g., title:/joh?n(ath[oa]n)/)
            - '~' for fuzzy matching (e.g., quikc~ or quikc~2)
            - '"..."~N' for proximity search (e.g., "fox quick"~5)
            - '^' for boosting terms (e.g., quick^2 fox)
            - 'AND', 'OR', 'NOT' for boolean matching (e.g., ((quick AND fox) OR (brown AND fox) OR fox) AND NOT news)

            Example:
            Question: What are the effects of vitamin D deficiency on the human body?
            Query string: ##(("vitamin d" OR "vitamin d3" OR "cholecalciferol") AND (deficiency OR insufficiency OR "low levels")) AND ("effects" OR "impact" OR "consequences") AND ("human body" OR "human health")##

            Tips:
            - Focus on the main concepts and entities in the question.
            - Use synonyms and related terms to capture variations in terminology.
            - Be cautious not to introduce irrelevant terms that may dilute the search results.
            - Strike a balance between precision and recall based on the specificity of the question.

            Please generate a query string for the following biomedical question and wrap the final query in enclosing ## tags. Example: ##query##
            Question: '''{question}'''
            """
        }
        
        # Add the assistant message
        assistant_message = {
            "role": "assistant",
            "content": completion  
        }
        
        messages.extend([user_message, assistant_message])

    return messages

def find_extract_json(text):
    pattern = r'\{.*?\}'
    matches = re.findall(pattern, text, re.DOTALL)
    match = matches[0]
    match_clean = match.replace('\\', "\\\\")
    match_clean = match_clean.replace('\t', "\\t")
    return match_clean

from unicodedata import normalize
def normalize_unicode_string(s, form='NFKC'):
    normalized  = normalize('NFKD', s).encode('ascii','ignore').decode()
    normalized = normalized.lower()
    return normalized


def generate_n_shot_examples_extraction(examples, n):
    """Takes the top n examples, flattens their messages into one list, and filters out messages with the role 'system'."""
    n_shot_examples = []
    for example in examples[:n]:
        for message in example['messages']:
            if message['role'] != 'system':  # Only add messages that don't have the 'system' role
                n_shot_examples.append(message)
    return n_shot_examples

def extract_relevant_snippets_few_shot(examples, n, article:str, question:str, model:str) -> str:
    
    system_message = {"role": "system", "content": "You are BioASQ-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain."}
    messages = [system_message]
    few_shot_examples = generate_n_shot_examples_extraction(examples, n)
    messages.extend(few_shot_examples)
    user_message = {"role": "user", "content": f"""
Given this question: '{question}' extract the exact relevant sentences or longer snippets from the following article 
delimited by tripple backticks that help answer the question. These will be string matched against the original text so make sure that they are exact parphrases character by character.
If no relevant information is present, return an empty array. Return the extracted snippets as a json object containing a string array called 'snippets'. Example return value {{"snippets": ["snippet1", "snippet2"]}}
Extract the snippets only from the following pubmed articles and not from the general context.
```{article}```"""}
    messages.append(user_message)
    print("Prompt Messages:")
    print(messages)
    
    completion = get_completion(messages, model)
    
    json_response = find_extract_json(completion)
    try:
        sentences = json.loads(json_response)
    except Exception as e:
        print(f"Error parsing response as json: {json_response}: {e}")
        traceback.print_exc()
        sentences = {"snippets": []}
    
    
    snippets = generate_snippets_from_sentences(article, sentences['snippets'])
    
    return snippets

def find_offset_and_create_snippet(document_id, text, sentence, section):
    text = normalize_unicode_string(text)
    sentence = normalize_unicode_string(sentence)
    offset_begin = text.find(sentence)
    offset_end = offset_begin + len(sentence)
    return {
        "document": document_id,
        "offsetInBeginSection": offset_begin,
        "offsetInEndSection": offset_end,
        "text": sentence,
        "beginSection": section,
        "endSection": section
    }

def generate_snippets_from_sentences(article, sentences):
    snippets = []

    article_abstract = article.get('abstract') or ''  # This will use '' if 'abstract' is None or does not exist
    article_abstract = normalize_unicode_string(article_abstract)
    article_title = normalize_unicode_string(article.get('title'))

    for sentence in sentences:
        sentence = normalize_unicode_string(sentence)
        if sentence in normalize_unicode_string(article_title):
            snippet = find_offset_and_create_snippet(article['id'], article['title'], sentence, "title")
            snippets.append(snippet)
        elif sentence in normalize_unicode_string(article_abstract):
            snippet = find_offset_and_create_snippet(article['id'], article_abstract, sentence, "abstract")
            snippets.append(snippet)
        else:
            print("\nsentences not found in article: "+sentence+"\n")
            print(article)

    return snippets

def generate_n_shot_examples_reranking(examples, n):
    """Takes the top n examples, flattens their messages into one list, and filters out messages with the role 'system'."""
    n_shot_examples = []
    for example in examples[:n]:
        for message in example['messages']:
            if message['role'] != 'system':  # Only add messages that don't have the 'system' role
                n_shot_examples.append(message)
    return n_shot_examples

def rerank_snippets(examples, n, snippets, question:str, model:str) -> str:
    numbered_snippets = [{'id': idx, 'text': snippet['text']} for idx, snippet in enumerate(snippets)]
    system_message = {"role": "system", "content": "You are BioASQ-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain."}
    messages = [system_message]
    few_shot_examples = generate_n_shot_examples_reranking(examples, n)
    messages.extend(few_shot_examples)
    user_message = {"role": "user", "content": f"""
Given this question: '{question}' select the top 20 snippets that are most helpfull for answering this question from
this list of snippets, rerank them by helpfullness: ```{numbered_snippets}``` return a json object containing one array of only their ids called 'snippets'. For example: {{"snippets: ["id4","id1", "id2"]}}"""}
    messages.append(user_message)
    print("Prompt Messages:")
    print(messages)
    
    response = get_completion(messages,model)
    print(response)
    print("\n")

    completion = get_completion(messages, model)
    json_response = find_extract_json(completion)
    
    try:
        snippets_reranked = json.loads(json_response)
        snippets_idx = snippets_reranked['snippets']
        filtered_array = [snippets[i] for i in snippets_idx]
    except Exception as e:
        print(f"Error parsing response as json: {json_response}: {e}")
        traceback.print_exc()
        filtered_array = snippets
        
    return filtered_array

def simplify_snippets(snippets: List[Dict[str, str]]) -> List[Dict[str, str]]:
    simplified_list = []
    for snippet in snippets:
        # Extract the PMID from the document URL using regex
        pmid_match = re.search(r'pubmed/(\d+)', snippet['document'])
        pmid = pmid_match.group(1) if pmid_match else None
        
        # Extract the text for the snippet
        text = snippet['text']
        
        # Append the simplified snippet to the list
        simplified_list.append({'pmid': pmid, 'snippet': text})
    
    return simplified_list

def generate_ideal_answer_with_citations(question: str, snippets: str, n_shots: int=0) -> str:
    system_message = {"role": "system", "content": "You are Biogen-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain."}
    messages = [system_message]
    few_shot_examples = generate_n_shot_examples_extraction(ideal_examples, n_shots)
    messages.extend(few_shot_examples)
    user_message = {"role": "user", "content": f"""
You will be given a question describing an information need.
You will also be given a list of relevant snippets containing information that should be used to answer that question.
Your goal is to provide a comprehensive, accurate answer based on the given information and cite your sources appropriately.

Instructions:
1. Analyze the question and the provided snippets.
2. Write a coherent and concise answer addressing the question and incorporating information from the snippets.
3. For each statement or fact in your answer, include citations using PMIDs in square brackets [PMID1, PMID2] immediately at the end of the relevant sentence.
4. Ensure each citation corresponds to the source of the information in that sentence.
6. Make sure every PMID you cite is from the provided snippets.
7. Make sure that every sentence is supported by atleast one citation and at most 3 citations.
8. Limit your answer to at most 150 words.

Remember to maintain scientific accuracy while making the answer understandable to a general audience, keep it concise. 
Your response should be informative and directly address the question asked.

Here is an example question and answer:

Example Question: 'why is transferrin and iron low in covid patients but ferritin high?'

Example Narrative: 'The patient is interested in the link between iron and infection, the role iron plays in infection and the implications for COVID-19 course.'

Example Answer: 'During infections, a battle for iron takes place between the human body and the invading viruses [34389110]. 
The immune system cells need iron to defend the body against the infection [34389110]. The virus needs iron to reproduce [35240553]. 
If iron balance is disrupted by the infection, ferritin levels are high [34883281], which signals the disease is severe and may have unfavorable outcomes [34048587, 32681497]. 
Ferritin is maintaining the bodys iron level [35008695]. Some researchers believe that high levels of ferritin not only show the body struggles with infection, but that it might add to the severity of disease [34924800]. 
To help covid patients, the doctors may lower the ferritin levels that are too high using drugs that capture iron [32681497].'

Here is the actual question that you should answer:

Question: {question}

Below are relevant snippets from scientific papers. Use this information to construct your answer:

{snippets}

It is important that every sentence of your answer is supported by atleast one and at most 3 relevant citations (pmids in square brakets)!
Your answer has to be concise and is not allowed to be longer than 150 words!

Please provide your concise answer now:     
"""}
    messages.append(user_message)
    print(messages)
    answer = get_completion(messages, model_ideal)
    print("\ngpt response ideal with citations:")
    print(answer)
    return answer   


def generate_ideal_answer(question:str, snippets:str, n_shots: int):
    system_message = {"role": "system", "content": "You are BioASQ-GPT, an AI expert in question answering, research, and information retrieval in the biomedical domain."}
    messages = [system_message]
    few_shot_examples = generate_n_shot_examples_extraction(ideal_examples, n_shots)
    messages.extend(few_shot_examples)
    user_message = {"role": "user", "content": f"""
            {snippets}\n\n
             '{question}'.
             You are a biomedical expert, write a concise and clear answer to the above question.
             It is very important that the answer is correct.
             The maximum allowed length of the answer is 200 words, but try to keep it short and concise."""}
    messages.append(user_message)
    print(messages)
    answer = get_completion(messages, model_ideal)
    print("\ngpt response ideal:")
    print(answer)
    return answer   

import gradio as gr
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


model_name = "gemini-1.5-flash-002"
model_name_extract = "gemini-1.5-flash-002"
model_name_rerank = "gemini-1.5-flash-002"
model_ideal = "gemini-1.5-flash-002"
n_shot = 3
use_wiki = True

def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Reads a JSONL file and returns a list of examples."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            examples.append(json.loads(line))
    return examples

query_examples = pd.read_csv('2024-03-26_19-24-27_claude-3-opus-20240229_11B1-10-Shot_Retrieval.csv')
snip_extract_examples_file = "Snippet_Extraction_Examples.jsonl"     
snip_extract_examples = read_jsonl_file(snip_extract_examples_file)

snip_rerank_examples_file = "Snippet_Reranking_Examples.jsonl"     
snip_rerank_examples = read_jsonl_file(snip_rerank_examples_file)

ideal_examples_file = "05_QA_Ideal_11B1-3-4_255.jsonl"     
ideal_examples = read_jsonl_file(ideal_examples_file)

def reorder_articles_by_snippet_sequence(relevant_article_ids, snippets):
    ordered_article_ids = []
    mentioned_article_ids = set()

    # Add article IDs in the order they appear in the snippets
    for snippet in snippets:
        document_id = snippet['document']
        if document_id in relevant_article_ids and document_id not in mentioned_article_ids:
            ordered_article_ids.append(document_id)
            mentioned_article_ids.add(document_id)

    # Add the remaining article IDs that weren't mentioned in snippets
    for article_id in relevant_article_ids:
        if article_id not in mentioned_article_ids:
            ordered_article_ids.append(article_id)

    return ordered_article_ids


def get_relevant_snippets(examples, n, articles, question: str, model_name: str):
    def process_article(article):
        snippets = extract_relevant_snippets_few_shot(examples, n, article, question, model_name)
        if snippets:
            article['snippets'] = snippets
        return article if snippets else None

    processed_articles = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        # Submit all tasks to the thread pool
        futures = [executor.submit(process_article, article) for article in articles]
        
        # Collect the results as they complete
        for future in as_completed(futures):
            result = future.result()
            if result:
                processed_articles.append(result)
    
    return processed_articles

def extract_text_wrapped_in_tags(input_string):
    pattern = "##(.*?)##"
    match = re.search(pattern, input_string, re.DOTALL)  
    if match:
        # Remove line breaks from the matched string
        extracted_text = match.group(1).replace('\n', '')
        return extracted_text
    else:
        return "ERROR"

def createQuery(query_string: str, size=50): 
    query = {
        "query": {
            "query_string": {
                "query": query_string
            }
        },
        "size": size
    }
    return query

import time

def create_query(question):
    start_time = time.time()
    completion = expand_query_few_shot(query_examples, n_shot, question, model_name)
    query_string = extract_text_wrapped_in_tags(completion)
    duration = time.time() - start_time
    print(f"Function execution time: {duration} seconds")
    return query_string

def search_expanded_query(expanded_query, question):
    elasticsearch_query = createQuery(expanded_query)
    relevant_articles = run_elasticsearch_query(elasticsearch_query)
    filtered_articles = get_relevant_snippets(snip_extract_examples, n_shot, relevant_articles, question, model_name_extract)
    relevant_snippets = [snippet for article in filtered_articles for snippet in article['snippets']]
    print("relevant snippets:")
    print(relevant_snippets)
    reranked_snippets = rerank_snippets(snip_rerank_examples, n_shot, relevant_snippets, question, model_name_rerank)
    print("reranked snippets:")
    print(reranked_snippets)
    simplyfied_snippets = simplify_snippets(reranked_snippets)
    print("simplified snippets:")
    print(simplyfied_snippets)
    snippets_string =  update_snippets(simplyfied_snippets)
    print("snippet string")
    print(snippets_string)
    return snippets_string


def update_snippets(snippets):
    markdown_snippets = [
        f"{item['snippet']} source: [{item['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{item['pmid']})  \n"
        for item in snippets
    ]
    return ''.join(markdown_snippets)



# Create a custom primary hue based on #9C004B
custom_primary_hue = gr.themes.Color(
    c50="#FCE4EC",  # lightest shade
    c100="#F8BBD0",
    c200="#F48FB1",
    c300="#F06292",
    c400="#EC407A",
    c500="#9C004B",  # primary color
    c600="#D81B60",
    c700="#C2185B",
    c800="#AD1457",
    c900="#880E4F",
    c950="#62002F"   # darkest shade
)


with gr.Blocks(gr.themes.Soft(primary_hue=custom_primary_hue)) as demo:
    n_shots = gr.Number(value=n_shot, visible=False)

    with gr.Column():
        with gr.Row():
            with gr.Column(scale=0):
                gr.Markdown("![](https://www.uni-regensburg.de/typo3conf/ext/ur_template/Resources/Public/dist/Images/ur-logo-wort-bildmarke-grau.svg)")
        gr.Markdown(
        """
        # BioRAGent!
        ### A retrieval augmented generation system showcasing generative query expansion and domain-specific RAG for Q&A
        For questions, feedback and bug-reports please contact Samy.Ateia@sprachlit.uni-regensburg.de
        """)
        
        gr.Markdown("## Enter your Biomedical Question!")

        # First input for the query
        question_input = gr.Textbox(show_label=False, placeholder="Enter your Question here!")
        with gr.Row():
            with gr.Column(scale=9):
                None
            with gr.Column(scale=1):
                #n_shots = gr.Number(label="(Optional) Number of Few-Shot Examples", value=0, precision=0, minimum=0, maximum=10)
                question_search_button = gr.Button("Search", variant="primary")
                

        # Loading spinner (initially hidden)
        loading_spinner = gr.Markdown("Loading...", visible=False)
        
        # Display expanded query
        expanded_query_output = gr.Textbox(label="Expanded Query, You can customize it and search again!")
        with gr.Row():
            with gr.Column(scale=9):
                None
            with gr.Column(scale=1):
                expanded_query_search_button = gr.Button("Search Again", variant="primary")

        # Display answer and answer with citations
        answer_output = gr.Textbox(label="Answer", interactive=False)
        answer_with_citations_output = gr.Textbox(label="Answer with Citations", interactive=False)

         # Output elements
 
        snippets_result = gr.Markdown(label="Search Result Snippets", container=True, value=
            """
            Example Snippet from pubmed source: [38168203](https://pubmed.ncbi.nlm.nih.gov/38168203/)
            """, min_height=50)
        
                # Helper function to disable inputs and show spinner
        def disable_inputs_and_show_spinner():
            return (
                gr.update(visible=True),  # Show spinner
                gr.update(interactive=False),  # Disable question input
                gr.update(interactive=False),  # Disable few_shot_examples_number
                gr.update(interactive=False),  # Disable search button
                gr.update(interactive=False)   # Disable search again button
            )

        # Helper function to re-enable inputs and hide spinner
        def enable_inputs_and_hide_spinner():
            return (
                gr.update(visible=False),  # Hide spinner
                gr.update(interactive=True),  # Enable question input
                gr.update(interactive=True),  # Enable few_shot_examples_number
                gr.update(interactive=True),  # Enable search button
                gr.update(interactive=True)   # Enable search again button
            )

        # Set up button actions
        question_search_button.click(
            disable_inputs_and_show_spinner, 
            inputs=[], 
            outputs=[loading_spinner, question_input, question_search_button, expanded_query_search_button]
        ).success(
            create_query, inputs=question_input, outputs=expanded_query_output
        ).success(
            search_expanded_query, inputs=[expanded_query_output, question_input], outputs=snippets_result
        ).success(
            generate_ideal_answer_with_citations, inputs=[question_input, snippets_result, n_shots], outputs=answer_with_citations_output
        ).success(
            generate_ideal_answer, inputs=[question_input, snippets_result, n_shots], outputs=answer_output
        ).success(
            enable_inputs_and_hide_spinner, inputs=[], outputs=[loading_spinner, question_input, question_search_button, expanded_query_search_button]
        )

        expanded_query_search_button.click(
            disable_inputs_and_show_spinner, 
            inputs=[], 
            outputs=[loading_spinner, question_input, question_search_button, expanded_query_search_button]
        ).success(
            search_expanded_query, inputs=[expanded_query_output, question_input], outputs=snippets_result
        ).success(
            generate_ideal_answer_with_citations, inputs=[question_input, snippets_result, n_shots], outputs=answer_with_citations_output
        ).success(
            generate_ideal_answer, inputs=[question_input, snippets_result, n_shots], outputs=answer_output
        ).success(
            enable_inputs_and_hide_spinner, inputs=[], outputs=[loading_spinner, question_input, question_search_button, expanded_query_search_button]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)