import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Medical RAG Evaluation",
    page_icon="🩺",
    layout="wide"
)

# Dashboard title
st.title("🩺 Medical RAG Evaluation Dashboard")

st.write(
    "Evaluate the performance of your Medical RAG pipeline "
    "using automated evaluation metrics."
)

st.divider()

# Metrics section
st.subheader("RAG Evaluation Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Contextual Recall", "—")

with col2:
    st.metric("Contextual Precision", "—")

with col3:
    st.metric("Faithfulness", "—")

with col4:
    st.metric("Answer Relevancy", "—")

st.divider()

# Evaluation section
st.subheader("Run Evaluation")

st.write(
    "Run your evaluation suite to measure "
    "the performance of your RAG system."
)

if st.button("Run Evaluation Suite", type="primary"):
    st.info("Evaluation integration will be added next.")

st.divider()

st.subheader("Evaluation Results")

st.write("Your individual test case results will appear here.")