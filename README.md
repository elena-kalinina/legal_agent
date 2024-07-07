# legal_agent

## General info
The legal agent is an LLM that can analyze contracts and respond if a certain requests is in compliance with the contract clauses. The LLM is Mixtral MoE 8x7B Instruct v1 available through Nvidia API (https://build.nvidia.com/mistralai/mixtral-8x7b-instruct). 
The system takes the contract text, splits it into paragraph, embeds each paragraph and indexes them with FAISS index. Embeddings model used is embded-qa-4, available through Nvidia API (https://build.nvidia.com/nvidia/embed-qa-4).
You can provide queries to the system as rows taken from the task file, e.g.: 'Team building event in Colorado;$2,200'. The system will analyze its compliance with the retrieved relevant contract clauses and provide the response.
In case of ambiguity, when it is unclear whether the task conforms or contradicts the contract stipulations, "human in the loop" is introduced: you can provide follow up clarifications and explanations to the system to help it come up with the conclusion. 

## Disclaimer
Because of time pressure and constraints concerning the number of free credits for Nvidia API, the system was tested with a small portion of the contract (file ```contract.txt```). Conversation sequence it was tested for:
- Conference in New York with weekend travel; $3,000
- The flight to New York was booked in business class
- Strategy retreat in the Swiss Alps during winter;$2,400

Also, a few shortcurs were used, e.g.: the file is not uploaded in the app, but the filename is given. 

## Installation and requirements
- Install requirements:
```shell
    pip install -r requirements.txt      
```
- Run the app:
 ```shell
    python app.py      
```
The app will run on localhost:5450. To reproduce the results, indicate ```contract.txt``` as file name and introduce queries as shown above, e.g. Task name + price in USD separated by semicolon: 'Team building event in Colorado;$2,200'. Follow-up clarifications can be added as text.  

