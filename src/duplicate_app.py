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

# Chat-like prompt analysis
st.subheader("💬 Safety Check Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter your prompt for safety analysis..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get classification with status indicator
    with st.status("Analyzing prompt...", expanded=False):
        response = requests.post("http://localhost:8000/classify", json={"prompt": prompt})
        result = response.json()
        classification = result.get('classification', 'unknown')
        
        if classification not in ['adversarial_harmful', 'vanilla_harmful', 'unsafe']:
            policy_response = requests.post("http://localhost:8000/policy_check", json={"prompt": prompt})
            policy_result = policy_response.json()
            if policy_result['decision'] == "NOT A VIOLATION":
                response_text = "✅ **Safe** - Prompt does not violate any policies!"
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                with st.chat_message("assistant"):
                    st.markdown(response_text)
            else:
                # Display violation in separate box
                st.error(f"⚠️ **Violation Detected**\n\n{policy_result['response_to_user']}\n\n**Violated Policy:** {policy_result.get('policy', 'N/A')}")
                response_text = "⚠️ Violation detected - see details above"
                st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
            response_text = f"❌ **Unsafe** - Classification: {classification}"
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            with st.chat_message("assistant"):
                st.markdown(response_text)