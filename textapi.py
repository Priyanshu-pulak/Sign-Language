from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def generate_sentence_from_alphabets(input_file = 'detected_alphabets.txt', output_file = 'convert_sentence.txt'):
    with open(input_file, "r") as file:
        detected_alphabets = file.read().strip()
    
    if not detected_alphabets:
        print("No alphabets detected in the input file.")
        return
    
    llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash-lite')
    
    prompt_template = PromptTemplate(
        template = """
        Convert these detected alphabets into a meaningful sentence or name: {detected_alphabets}
        
        Instructions:
        - Form a proper grammatically correct sentence
        - Make it natural and meaningful
        - Return only the sentence or name, no extra text
        """,
        input_variables = ['detected_alphabets']
    )
    
    parser = StrOutputParser()
    
    chain = prompt_template | llm | parser
    
    generated_sentence = chain.invoke({"detected_alphabets": detected_alphabets})
    
    with open(output_file, "w") as f:
        f.write(generated_sentence)

    print(f"Generated sentence : {generated_sentence}")
    print(f"Sentence saved to {output_file}")
    
    return generated_sentence

if __name__ == "__main__":
    generate_sentence_from_alphabets()


