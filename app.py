import os
import subprocess
import streamlit as st
from RAG import RAG

def main():
    """Main Streamlit app logic."""
    header = st.container()

    def load_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Load the CSS file
    load_css("Styles/style.css")

    header.title("Textbook Chatbot")
    header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

    # Sidebar for chat history and statistics
    st.sidebar.markdown(f'<h1 class="title-stat">Statistics Reports</h1>', unsafe_allow_html=True)

    # List of statistics to display
    statistics = [
        "Number of correct answers",
        "Number of incorrect answers",
        "User engagement metrics",
        "Response time analysis",
        "Accuracy rate",
        "Common topics or keywords",
        "User satisfaction ratings",
        "Improvement over time",
        "Feedback summary",
        "Statistics per day and overall"
    ]

    # Display statistics in the sidebar
    for stat in statistics:
        st.sidebar.markdown(f"""
            <div class='btn-stat-container'>
                <a href="#" class="btn-stat">{stat}</a>
            </div>
        """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize RAG for processing
    if "rag_instance" not in st.session_state:
        st.session_state.rag_instance = RAG("data/Textbook.pdf")

    # Render existing messages
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)

    # Handle user input
    if prompt := st.chat_input("Ask your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Perform a search using RAG
        results = st.session_state.rag_instance.search_index(prompt)
        response = "\n".join(results) if results else "I'm not sure about that. Can you try rephrasing your question?"

        # Add the assistant's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display the user and assistant messages
        st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class='assistant-message'>
                {response}
                <div class='feedback-buttons'>
                    <span class='feedback-icon'>👍</span>
                    <span class='feedback-icon'>👎</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    if os.environ.get("STREAMLIT_RUNNING") == "1":
        main()
    else:
        os.environ["STREAMLIT_RUNNING"] = "1"  # Set the environment variable to indicate Streamlit is running
        subprocess.run(["streamlit", "run", __file__, "--server.port=5003", "--server.address=0.0.0.0", "--server.baseUrlPath=/team3"])
