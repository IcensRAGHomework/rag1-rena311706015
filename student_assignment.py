import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder

# for hw02
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.tools import tool
import requests

# for hw03
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from pydantic import BaseModel, Field
from langchain_core.runnables.history import RunnableWithMessageHistory

# for hw04
import base64
from openai import AzureOpenAI
from mimetypes import guess_type

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

# the variables shared by hw01 to hw03

examples = [

    {"input":"今年台灣10月紀念日有哪些?", 
     "output":"""{"Result": [{"date": "2024-10-10","name": "國慶日"}]}"""},
    {"input":"""根據先前的節日清單，這個節日{"date": "10-31", "name": "蔣公誕辰紀念日"}是否有在該月份清單？""", 
     "output":"""{"Result":{"add": true, "reason": "蔣中正誕辰紀念日並未包含在十月的節日清單中。目前十月的現有節日包括國慶日、重陽節、華僑節、台灣光復節和萬聖節。因此，如果該日被認定為節日，應該將其新增至清單中。"}"""}
]
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You know all the holidays, and you will only return valid JSON without enclosing it in '''json'''"),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad", optional=True),
    ]
)
llm = AzureChatOpenAI(
    model=gpt_config['model_name'],
    deployment_name=gpt_config['deployment_name'],
    openai_api_key=gpt_config['api_key'],
    openai_api_version=gpt_config['api_version'],
    azure_endpoint=gpt_config['api_base'],
    temperature=gpt_config['temperature']
)

# hw03
memory={}

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []
        

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in memory:
        memory[session_id] = InMemoryHistory()
    return memory[session_id]


history = get_by_session_id("1")
history_examples = []
for example in examples:
    history_examples.append(HumanMessage(example['input']))
    history_examples.append(AIMessage(example['output']))
history.add_messages(history_examples)



def generate_hw01(question):
    chain = final_prompt | llm
    response = chain.invoke({"input": question})
    
    return response.content
   
def generate_hw02(question):
    tools = [get_holidays_by_month]
    agent = create_openai_functions_agent(
    	llm=llm,
    	prompt=final_prompt,
        tools=tools,
    )
    
    # You can add the verbose=True parameter to get more details. If it logs 'Invoking: get_holidays_by_month with ...', it means the tool was successfully used.
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    response = agent_executor.invoke({"input": question})
    
    # for hw03
    history.add_user_message(question)
    history.add_ai_message(response["output"])
    
    return response["output"] 

def generate_hw03(question2, question3):
    generate_hw02(question2)
    
    chat_history = RunnableWithMessageHistory(
        final_prompt | llm,
        get_by_session_id,
        input_messages_key="input",
        history_messages_key="history",
    )

    response = chat_history.invoke(
        {"input": question3}, 
        config={"configurable": {"session_id": "1"}}
    )
    
    return response.content
   
def generate_hw04(question):    
    image_url=local_image_to_data_url("baseball.png")
    client = AzureOpenAI(
        api_key=gpt_config['api_key'],
        api_version=gpt_config['api_version'],
        base_url=f"{gpt_config['api_base']}openai/deployments/{gpt_config['deployment_name']}"
    )
    response = client.chat.completions.create(
        model=gpt_config['deployment_name'],
        messages=[
            { "role": "system", "content": """You can identify information about the game in the image, and your responses should be in this format {"Result":{"score": 5498}""" },
            { "role": "user", "content": [{"type": "text", "text": "請問中華台北的積分是多少?"}]},
            { "role": "assistant", "content": '{"Result":{"score": 5498}' },
            { "role": "user", "content": [  
                { 
                    "type": "text", 
                    "text": question 
                },
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ] } 
            ], max_tokens=2000     
        )
    
    return response.choices[0].message.content  
   

@tool
def get_holidays_by_month(month):
    """
    get holidays through Calendarific API
    you will only return valid JSON without enclosing it in '''json'''
	Args:
		month: used to specify searching for holidays in that month
	"""
    api_url=f"https://calendarific.com/api/v2/holidays?&api_key=le1Pqi3EMNMZ6cdfczMEw2vuHBWze0NN&country=TW&year=2024&month={month}"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        holidays_data = response.json()
        return holidays_data
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    return f"data:{mime_type};base64,{base64_encoded_data}"
   
def format_response_as_json(response):
    formatted_response = json.loads(response)
    pretty_json = json.dumps(formatted_response, indent=4, ensure_ascii=False)
    
    return pretty_json 
    
 

print(json.loads(generate_hw01("2024年台灣10月紀念日有哪些?")))
#generate_hw02("列出台灣4月的紀念日")
#generate_hw03("台灣9月的紀念日有哪些", '根據先前的節日清單，這個節日{"date": "9-1", "name": "XX紀念日"}是否有在該月份清單？')
print(json.loads(generate_hw04("請問中華台北的積分是多少?")))




#%%  
'''def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response'''