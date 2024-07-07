import io
import os
import json
from typing import List
from openai import OpenAI
from utils import parse_config_file, split_text
from retriever import VectorStoreFromEmbeddings


class AnalyzerLLM:
    def __init__(self, api_key: str, llm_url: str):

        """
        LLM that performs different tasks: analyzes contracts, extracts keywords etc.
        The model is Mixtral MoE 7x8B Instruct v1 available through Nvidia API.

        Parameters:
            - api_key (str): Nvidia API key.
            - base_url (str): Nvidia API.
            - retriever_config (str): config file for the retriever that supports contract analysis.
            - retriever (VectorStoreFromEmbeddings): vector store from embeddings built with Nvidia embeddings model and 
                Transformers Datasets library.
            - llm (str): model identifier (name)
            - history (List): conversational history to support follow up clarifications
            - client (OpenAI): OpenAI API for completions

        Return:
            None.
        """

        self.api_key = api_key
        self.base_url = llm_url
        self.retriever_config = parse_config_file('config.yml')
        self.retriever = VectorStoreFromEmbeddings(self.retriever_config)
        self.llm = "mistralai/mixtral-8x7b-instruct-v0.1"
        self.history = []
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url)

    def analyze_topics(self, texts: list) -> dict:
        '''
        extracts key terms from each chunk of splitted text

        Parameter:
        - list of text chunks as Document objects

        Returns:

        json dict of key terms in the text.

        '''
        system = """Analyze the text and define the list of topics discussed in the text. \
                     Answer only with a json object containing a list of the relevant topics, \
                     "in the following format: \
                      {{"paragraph title": "...",
                      "key topics": [ ... ] # list of key words describing paragraph topics
                       }}"""

        topics = []
        for text in texts:
            completion = self.client.chat.completions.create(
                model=self.llm,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": f'text to analyze: {text.page_content}'}],
                temperature=0.5,
                top_p=1,
                max_tokens=1024,
                stream=False
            )
            try:
                topics.append(json.loads(completion.choices[0].message.content))
            except:
                # currently I just print it out to see json errors in the output;
                # however, LLM structured outputs can be improved e.g. by the use of parsers, prompt tuning, GraphRAG etc.
                print(completion.choices[0].message.content)

        return topics

    def analyze_task(self, task) -> str:

        '''
        the llm is asked to analyze the task and respond if its complies with the contract

        Parameter:
        - task can be a tuple of (task, expenses) or just text for follow up cliarifications

        Returns:

        LLM's analysis of contract compliance as text.

        '''
        system = "you will be given a description of a completed task and a budget allowed for the task." \
                 "you will have to analyze the corresponding contract clauses and respond whether " \
                 "the task and the budget conform to the contract conditions. explain your decision. " \
                 "if it is unclear to you from the description of the completed task alone" \
                 "whether it contradicts the contract, ask a clarifyung question back to the user. "
        # retrieve contract clauses
        retrieved = '\n'.join(self.retriever.search_vectorstore(task[0], 3))
        # call mixtral
        if isinstance(task, tuple):
            # if it is a new task, previous conversation is reset as it might be irrelevant
            self.history = []
            user = f"this is the completed task: {task[0]} and this is the budget for the task: {task[1]}. \
               These are the relevant contract clauses: {retrieved}."
        else:
            user = f"This is a follow up clarification: {task}."
        system_message = [{"role": "system", "content": system}]
        session_messages = [{"role": "user", "content": user}]
        completion = self.client.chat.completions.create(
            model=self.llm,
            messages=system_message + self.history + session_messages,
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=False
        )
        agent_response = completion.choices[0].message.content
        self.history = self.history + session_messages + [{"role": "assistant", "content": agent_response}]
        return agent_response


if __name__ == "__main__":
    api_key = os.getenv('NVIDIA_API_KEY')
    llm_url = 'https://integrate.api.nvidia.com/v1'
    analyzer_llm = AnalyzerLLM(api_key=api_key, llm_url=llm_url)
    topics_json = analyzer_llm.analyze_topics(split_text('contract.txt'))
    output_file = io.open("topics.json", 'w', encoding='utf-8')
    output_file.write(json.dumps(topics_json))