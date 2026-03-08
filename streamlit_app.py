import streamlit as st
import pickle
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .ticket-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .high-priority {
        color: #d32f2f;
        font-weight: bold;
    }
    .medium-priority {
        color: #f57c00;
        font-weight: bold;
    }
    .low-priority {
        color: #388e3c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        model_path = Path(__file__).parent.parent / "model" / "model.pkl"
        vectorizer_path = Path(__file__).parent.parent / "model" / "vectorizer.pkl"
        
        if not model_path.exists() or not vectorizer_path.exists():
            st.error("❌ Model files not found. Please train the model first.")
            st.stop()
        
        model = pickle.load(open(model_path, 'rb'))
        vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

model, vectorizer = load_models()

# Priority assignment logic
def assign_priority(text):
    text = text.lower()
    
    # High priority keywords
    high_keywords = ["not working", "crashing", "critical", "urgent", "error", "broken", "down"]
    if any(keyword in text for keyword in high_keywords):
        return "High"
    
    # Medium priority keywords
    medium_keywords = ["refund", "billing", "payment", "slow", "issue", "problem"]
    if any(keyword in text for keyword in medium_keywords):
        return "Medium"
    
    # Low priority
    return "Low"

# Get priority color
def get_priority_color(priority):
    colors = {
        "High": "#d32f2f",
        "Medium": "#f57c00",
        "Low": "#388e3c"
    }
    return colors.get(priority, "#666")

# Main UI
st.title("🎫 Support Ticket Classifier")
st.markdown("Automatically classify and prioritize customer support tickets using ML")

# Sidebar - Information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool uses machine learning to:
    - **Classify** tickets into categories
    - **Prioritize** based on keywords and urgency
    - **Streamline** support workflows
    
    **Categories:** Technical, Billing, Account, General Query
    """)
    
    st.divider()
    
    st.header("📊 Classification Metrics")
    if st.checkbox("Show model info"):
        st.info(f"Model type: {type(model).__name__}")
        st.info(f"Vectorizer type: {type(vectorizer).__name__}")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Support Ticket")
    ticket = st.text_area(
        "Paste customer ticket text below:",
        height=150,
        placeholder="e.g., 'My app keeps crashing when I try to upload files...'"
    )
    
    # Character count
    char_count = len(ticket)
    st.caption(f"Characters: {char_count}")

# Classification button
if st.button("🔍 Classify Ticket", use_container_width=True, type="primary"):
    if not ticket.strip():
        st.warning("⚠️ Please enter a support ticket to classify.")
    else:
        try:
            with st.spinner("Analyzing ticket..."):
                # Preprocess
                cleaned = ticket.lower().strip()
                
                # Vectorize
                vect = vectorizer.transform([cleaned])
                
                # Predict
                category = model.predict(vect)[0]
                priority = assign_priority(ticket)
                
                # Display results
                st.divider()
                st.subheader("📊 Classification Results")
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.metric(
                        label="Category",
                        value=category,
                        delta=None
                    )
                
                with result_col2:
                    priority_color = get_priority_color(priority)
                    st.markdown(f"<div style='text-align:center'><h3>Priority: <span style='color:{priority_color}'>{priority}</span></h3></div>", unsafe_allow_html=True)
                
                # Confidence and recommendation
                st.divider()
                st.subheader("💡 Recommendations")
                
                recommendations = {
                    "Technical": "Assign to technical support team. Request system logs if necessary.",
                    "Billing": "Route to billing department. Verify account status.",
                    "Account": "Check account security. May require identity verification.",
                    "General Query": "Provide relevant documentation or FAQ links."
                }
                
                st.info(recommendations.get(category, "Route to general support queue."))
                
                # Priority action
                if priority == "High":
                    st.error(f"🚨 **HIGH PRIORITY** - Immediate action required!")
                elif priority == "Medium":
                    st.warning(f"⚠️ **MEDIUM PRIORITY** - Address within 24 hours")
                else:
                    st.success(f"✅ **LOW PRIORITY** - Standard response time")
        
        except Exception as e:
            st.error(f"❌ Classification error: {str(e)}")

# History/Examples
st.divider()
st.subheader("📚 Example Tickets")

examples = {
    "🔴 High Priority - Technical": "My application keeps crashing whenever I try to upload a file larger than 100MB. This is blocking my work completely.",
    "🟠 Medium Priority - Billing": "I was charged twice for my subscription this month. Please refund the duplicate charge.",
    "🟢 Low Priority - General": "What are your business hours? I wanted to know when I can reach customer support."
}

for label, example in examples.items():
    with st.expander(label):
        st.markdown(f"> *{example}*")
        if st.button(f"Try this example", key=label):
            st.session_state.ticket = example
            st.rerun()