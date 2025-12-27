import os
from openai import OpenAI
import json

# Import the Pinecone library
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

index_name = "safety-policies"
index = pc.Index(index_name)

model = SentenceTransformer('all-MiniLM-L6-v2')


def get_policies(user_prompt):
    query_vector = model.encode(user_prompt).tolist()
    
    results = index.query(vector=query_vector, top_k=2, include_metadata=True)
    
    relevant_policies = [match['metadata']['description'] for match in results['matches']]
    return relevant_policies

def LLM_judgement(relevant_policies,user_prompt):
    prompt = f'''
    System: You are a Safety Auditor. Below are specific safety policies and a user prompt. 
    Evaluate if the prompt violates any of these policies. 
    A prompt doesn't have to be harmful to be a violation. It just has to fall into one of the restricted categories.
    You should also form a one line response to the user to let them know you cannot respond to their prompt because of the given policies. 
    Respond in json format, using this template:

        'decision': 'NOT A VIOLATION or VIOLATION',
        'policy': "If violation, give policy. If not, set None.",
        'response_to_user: If violation, give response to user. If not, set None.'
    

    Policies:
    {relevant_policies}

    User Prompt: "{user_prompt}"
    '''

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return completion.choices[0].message.content

def policy_check(user_prompt):
    relevant_policies = get_policies(user_prompt)

    judgement = LLM_judgement(relevant_policies, user_prompt)

    judgement = json.loads(judgement)

    decision = judgement['decision']

    if decision == "NOT A VIOLATION":
        judgement['policy'] = None
        judgement['response_to_user'] = None
    else: 
        return judgement
    
    return judgement