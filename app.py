import os
import yaml
import gradio as gr
from contract_agent import AnalyzerLLM
from retriever import VectorStoreFromEmbeddings
from utils import parse_config_file

base_url = "https://integrate.api.nvidia.com/v1"
api_key = os.getenv("NVIDIA_API_KEY")
llm_agent = AnalyzerLLM(api_key=api_key, llm_url=base_url)


def agent_response(query: str, fname: str) -> str:
    # write fname to config file
    fp = open('config.yml', 'r')
    params: Dict = yaml.load(fp, Loader=yaml.FullLoader)
    params["input"]["path"] = fname
    with open('config.yml', 'w') as outfile:
        yaml.dump(params, outfile, default_flow_style=False)
    # initialize vector store from config, update the vector store
    llm_agent.retriever_config = parse_config_file('config.yml')
    llm_agent.retriever = VectorStoreFromEmbeddings(llm_agent.retriever_config)
    # process query
    if '$' in query:
        proc_query = tuple(query.split(';'))
    else:
        proc_query = query
    response = llm_agent.analyze_task(proc_query)
    return response

# Run Gradio App
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Contract Agent: analyzing legal texts")
    contract_fname = gr.Textbox(label="Contract filename")
    with gr.Row():
        with gr.Column():
            query = gr.Textbox(label="What is your question?")
            btn = gr.Button("Submit")
        with gr.Column():
            res = gr.Textbox(label="Agent Response")
    btn.click(fn=agent_response, inputs=[query, contract_fname], outputs=[res])
gr.close_all()
demo.launch(server_name='0.0.0.0', server_port=5450)