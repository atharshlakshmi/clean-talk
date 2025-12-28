import requests
import streamlit as st
from core.safety_rag import get_all_policies, index
import time

st.set_page_config(page_title="Clean Talk", page_icon="🛡️", layout="centered")

st.title("🛡️ Clean Talk - Guardrail API")

st.markdown("""
Welcome to **Clean Talk** - your AI safety companion! 🚀

#### How It Works
Clean Talk uses a smart 2-step approach to keep your AI safe:

1. **Smart Classifier** 😎   
   A fine-tuned DistilBERT model quickly identifies if a prompt is safe or attempting a jailbreak

2. **Policy Checker** 📋   
   Safe prompts are checked against your custom policies stored in Pinecone, with an LLM judge providing the final verdict

#### What You Can Do
- ✅ Enter any prompt and get an instant safety classification
- ✅ Add custom safety policies that matter to your use case
""")

# Sidebar for policy management
with st.sidebar:
    st.header('🤍 Add New Policy')
    policy = st.text_area("Enter policy:", height=80)
    if st.button("Add Policy"):
        if policy:
            with st.status("Adding policy...", expanded=True):
                response = requests.post("http://localhost:8000/add_policy", json={"policy": policy})
                result = response.json()
                if result.get('status') == 'uploaded':
                    st.success("Policy added!")
        else:
            st.warning("Please enter a policy")
    
    st.divider()
    
    if st.button("🔄 Refresh policies"):
        st.rerun()
    
    with st.expander("📋 Current Policies"):
        try:
            policies = get_all_policies(index)  # Fetch fresh data every time
            # Sort by ID for consistent ordering
            policies = sorted(policies, key=lambda x: x['ID'])
            
            if policies:
                for policy in policies:
                    st.button(policy['Policy'], use_container_width = True)
            else:
                st.info("No policies added yet.")
        except Exception as e:
            st.error(f"Error loading policies: {str(e)}")

# Main prompt analysis
st.subheader("Classify Prompt")
prompt = st.text_area("Enter your prompt:", height=100)

if st.button("Classify"):
    if prompt:
        with st.status("Classifying...", expanded=True):
            response = requests.post("http://localhost:8000/classify", json={"prompt": prompt})
            result = response.json()
            classification = result.get('classification', 'unknown')
            
            if classification not in ['adversarial_harmful', 'vanilla_harmful', 'unsafe']:
                st.success(f"✅ Safe to go! Classification: {classification}")
                
                policy_response = requests.post("http://localhost:8000/policy_check", json={"prompt": prompt})
                policy_result = policy_response.json()
                if policy_result['decision'] == "NOT A VIOLATION":
                    st. write("Prompt does not violate any policies. Good to go!")
                else:
                    st.error(f"Violation! {policy_result['response_to_user']}")
                    st.write(f"**Violated Policy:** {policy_result.get('policy', 'N/A')}")
            else:
                st.error(f"⚠️ Unsafe! Classification: {classification}")
    else:
        st.warning("Please enter a prompt")