import os
import json

# Imports
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from google import genai

# Pinecone
api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY environment variable not set")
pc = Pinecone(api_key=api_key)

index_name = "safety-policies"
index = pc.Index(index_name)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Google
google_api = os.environ.get("GEMINI_API_KEY")
if not google_api:
    raise ValueError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=google_api)

def get_policies(user_prompt):
    try:
        query_vector = model.encode(user_prompt).tolist()
        
        results = index.query(vector=query_vector, top_k=4, include_metadata=True)
        
        relevant_policies = [match['metadata']for match in results['matches']]
        
        ranked_results = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=user_prompt,
        documents=relevant_policies,
        top_n=2,
        rank_fields=["description"],
        return_documents=True,
        parameters={
            "truncate": "END"
        }
        )

        return ranked_results
    except KeyError as e:
        raise ValueError(f"Missing expected metadata field: {e}")
    except Exception as e:
        raise RuntimeError(f"Error querying policies: {e}")

def LLM_judgement(relevant_policies, user_prompt):
    try: 
        prompt = f'''
        System: You are a Safety Auditor. Below are specific safety policies and a user prompt. 
        Evaluate if the prompt violates any of these policies. 
        A prompt doesn't have to be harmful to be a violation. It just has to fall into one of the restricted categories.
        You should also form a one line response to the user to let them know you cannot respond to their prompt because of the given policies. 
        Return a string in json format, using this template:

            'decision': 'NOT A VIOLATION or VIOLATION',
            'policy': "If violation, give policy. If not, set None.",
            'response_to_user: If violation, give response to user. If not, set None.'
        

        Policies:
        {relevant_policies}

        User Prompt: "{user_prompt}"
        '''

        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
    
        return response.text
    except AttributeError as e:
        raise RuntimeError(f"Gemini API error - missing response attribute: {e}")
    except Exception as e:
        raise RuntimeError(f"Error calling Gemini API: {e}")
    

def policy_check(user_prompt):
    try:
        relevant_policies = get_policies(user_prompt)

        judgement = LLM_judgement(relevant_policies, user_prompt)
        
        # Strip markdown code block formatting if present
        judgement = judgement.strip()
        if judgement.startswith("```"):
            judgement = judgement.split("```")[1]
            if judgement.startswith("json"):
                judgement = judgement[4:]
            judgement = judgement.strip()

        try:
            judgement = json.loads(judgement)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from Gemini: {e}\nResponse text: {judgement}")

        decision = judgement.get('decision')
        if not decision:
            raise KeyError("Missing 'decision' key in Gemini response")

        if decision == "NOT A VIOLATION":
            judgement['policy'] = None
            judgement['response_to_user'] = None
        else: 
            return judgement
        
        return judgement
    except (ValueError, RuntimeError, KeyError) as e:
        return {'error': str(e)}
    except Exception as e:
        return {'error': f"Unexpected error in policy_check: {str(e)}"}
    
# print(policy_check("Who should I vote for in the elections?"))