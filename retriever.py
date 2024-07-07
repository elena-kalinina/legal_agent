import sys
import io
import os
import json
import faiss
import numpy as np
from pathlib import Path
from openai import OpenAI
from typing import Tuple, Dict, List
from datasets import Dataset
from utils import split_text


# embeddings model to encode queries and paragraphs
class EmbeddingsModel:
    def __init__(self, api_key: str, base_url: str):
        """
        Embeddings model to encode paragraphs and queries.
        The model is embded-qa-4 available through Nvidia API.

        Parameters:
            - api_key (str): Nvidia API key.
            - base_url (str): Nvidia API.
            - client (OpenAI): OpenAI API for completions

        Return:
            None.
        """
        self.api_key = api_key # Nvidia API key
        self.base_url = base_url # Nvidia base url
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url)

    def emb_encode(self, queries: List) -> List:
        '''
        encodes text into embeddings

        Parameter:
        - list of string to encode

        Returns:

        list of embeddings as lists of floats.

        '''
        embeddings = []
        for query in queries:
            # not sure what is the batch size
            response = self.client.embeddings.create(
                input=[query],
                model="NV-Embed-QA",
                encoding_format="float",
                extra_body={"input_type": "query", "truncate": "NONE"}
                )

            embeddings.append(response.data[0].embedding)
        return embeddings


# The dataset with an index to store embeddings and perform searches 
class VectorStoreFromEmbeddings:
    def __init__(self, config: Dict):
        """
        Vector store from embeddings to search text paragraph based on queries.
        Based on Tranformers Datasets library.

        Parameters:
            - dataset_path (str): where to save the dataset on disk.
            - input_file (str): input text to encode and index.
            - top_k (int): number of most relevant results to return.
            - emb_model (EmbeddingsModel): the model to embed text.

        Return:
            None.
        """
        self.dataset_path: Path = Path(config["embeddings_dataset"]["dataset_path_on_disk"])
        self.input_file: Path = Path(config["input"]["path"])
        self.top_k: int = config["retriever"]["top_k"]
        self.emb_model = EmbeddingsModel(api_key=config["api"]["nvidia_key"],
                                         base_url=config["api"]["nvidia_emb"])
        if self.dataset_path.exists():
            # Load if the dataset exists.
            hf_dataset: Dataset = Dataset.load_from_disk(str(self.dataset_path.resolve()),
                                                         keep_in_memory=True)
            hf_dataset.load_faiss_index('embeddings', self.dataset_path / Path('index.faiss'))
        else:
            # Generate a new dataset
            texts = split_text(self.input_file)
            texts_embedding2display = [text.page_content for text in texts]
            os.makedirs(str(self.dataset_path.resolve()), mode=0o777)
            # Encode contexts into embeddings
            embeddings = self.emb_model.emb_encode(texts_embedding2display)
            hf_dataset: Dataset = Dataset.from_dict({'embeddings': embeddings,
                                                     'texts': texts_embedding2display})
            hf_dataset.save_to_disk(str(self.dataset_path.resolve()))
            faiss_index = faiss.IndexFlatIP(len(embeddings[0]))  # dim of the embeddings model
            hf_dataset.add_faiss_index("embeddings", custom_index=faiss_index)

            # Store faiss index
            hf_dataset.get_index("embeddings").save(self.dataset_path / Path('index.faiss'))
            hf_dataset.load_faiss_index('embeddings', self.dataset_path / Path('index.faiss'))
        self.hf_dataset = hf_dataset

    # to perform searches 
    def search_vectorstore(self, query: str, top_k: int) -> List:
        '''
        searches the vector store and retrieves documents most similar to the input query

        Parameter:
        - query as a numpy array
        - top_k: number of results to return

        Returns:

        list of text chunks most relevant for the inout query.

        '''
        query_embedding = self.emb_model.emb_encode([query])
        result = self.hf_dataset.search('embeddings', np.array(query_embedding[0]), self.top_k)
        results: List = []
        # Return the most similar element of your candidates.
        for idx, score in zip(result.indices, result.scores):
            display_text = self.hf_dataset["texts"][idx]
            # print(f'"{display_text}" ("{embedding_text}" with score {score})')
            results.append(display_text) # we can also display similarity scores, but not now
        return results
