from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain

# Initialize the template with dynamic input
template = """You are a helpful assistant. Answer the following question:
{question}

Answer: Let me help you with that."""

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

def get_chain():
    # Initialize the LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",
        model_kwargs={"temperature": 0.5, "max_length": 64}
    )
    
    # Create and return the chain
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def main():
    # Get the chain
    chain = get_chain()
    
    # Example question
    question = "What is the capital of France?"
    
    # Run the chain
    response = chain.run(question=question)
    print(f"Question: {question}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()