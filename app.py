import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
# from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler

#SETUP STREAMLIT APP
st.set_page_config(page_title="Text to Maths Problem Solver")
st.title("Maths Problem Solver GenAI_Google Gemma")

st.sidebar.image("image.png",width=160)
api_key = st.sidebar.text_input("Groq API KEY", type="password")
if not api_key:
    st.info("Please add your Groq API Key")
    st.stop()

llm = ChatGroq( groq_api_key=api_key,model="Gemma2-9b-It")

##Initialize Tool
wiki = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func=wiki.run,
    description= "A tool for search the Internet to find various topic"
)

## Initialize Math tool

math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Tool to solve maths expression"
)

prompt = """
You are an agent to solve to mathematical question. Logically arrive at the solution with explanation
and display it as point wise for question below
Question:{question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

#Combine all the tools into chain
chain = LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning",
    func=chain.run,
    description="Tool for answering"
)

## initialize Agent

assistant_agent = initialize_agent(
    tools=[wiki_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant",
         "content": "Hi!, I'm a Maths Chatbot"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ## Function to generator response
# def generate_response():
#     response = assistant_agent.invoke({"input":question})
#     return response
question = st.text_area("drianna has 10 pieces of gum to share with her friends. There wasn’t enough gum for all her friends, so she went to the store and got 70 pieces of strawberry gum and 10 pieces of bubble gum. How many pieces of gum does Adrianna have now?")

if st.button("find my answer"):
    if question:
        with st.spinner("Generate response..."):
            st.session_state.messages.append({"role":"user",
                                              "content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])

            st.session_state.messages.append({"role":"assistant", "content":response})
            st.write("### response")
            st.success(response)
    else:
        st.warning("Please enter the question")