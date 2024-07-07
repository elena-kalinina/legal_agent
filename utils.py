import yaml
import io
from typing import Dict, List
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_config_file(config_file_path: Path) -> Dict:
    '''
    parses the config file

    Parameter:
    - file path

    Returns:

    dictionary of config parameters.

    '''
    fp = open(config_file_path, 'r')
    configuration_parameters: Dict = yaml.load(fp, Loader=yaml.FullLoader)
    fp.close()
    return configuration_parameters


def split_text(fname: str) -> List:
    '''
    splits the text into chunks to be embedded for retrieval.
    splits based on paragraph breaks.

    Parameter:
    - filename

    Returns:

    list of text chunks.

    '''

    with io.open(fname, 'r', encoding='utf-8') as f: # io because we can have different languages
        contract = f.read()

    # initialize the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.create_documents([contract])
    return texts
        