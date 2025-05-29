import streamlit as st
import requests
import pandas as pd
import json
import os
from tempfile import NamedTemporaryFile

# Set wide mode as default
st.set_page_config(layout="wide")
# Define fixed file path
FILE_PATH = "tarp_model_202410112037.csv"

API_URL = "http://localhost:8000"

# Custom CSS for styling
st.markdown("""
    <style>
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 5px;
        margin: 10px 0;
    }
    .status-container {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        background-color: #f0f2f6;
    }
    .stProgress > div > div > div > div {
        background-color: #00a0a0;
    }
    </style>
""", unsafe_allow_html=True)

class SimpleFile:
        def __init__(self, path):
            self.name = os.path.basename(path)
            self.size = os.path.getsize(path)
            self.path = path
        
        def getvalue(self):
            with open(self.path, 'rb') as f:
                return f.read()

def save_uploaded_file(uploaded_file):
    """Save uploaded file and return the path"""
    try:
        with NamedTemporaryFile(delete=False, suffix='.csv') as f:
            f.write(uploaded_file.getvalue())
            return f.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def analyze_csv(question: str, file_path: str) -> dict:
    """Make API call to analyze CSV"""
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={"question": question, "file_path": file_path},
            timeout=300  # Increased timeout for complex queries
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def display_analysis_results(question, temp_file_path):
    """Display analysis results in collapsible containers"""
    try:
        # Create columns for the progress steps
        col1, col2, col3 = st.columns(3)
        
        # Initialize all steps as incomplete
        with col1:
            step1_container = st.empty()
            step1_container.markdown("⭕ Generating Query")
        with col2:
            step2_container = st.empty()
            step2_container.markdown("⭕ Executing Query")
        with col3:
            step3_container = st.empty()
            step3_container.markdown("⭕ Generating Response")

        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Query Generation
        step1_container.markdown("🔄 Generating Query")
        progress_bar.progress(33)
        
        # Make API call
        result = analyze_csv(question, temp_file_path)
        
        if result:
            # Mark Step 1 as complete
            step1_container.markdown("✅ Query Generated")
            
            query_container = st.expander("Generated Query", expanded=True)
            with query_container:
                st.code(result["code"], language="python")
            
            # Step 2: Query Execution
            step2_container.markdown("🔄 Executing Query")
            progress_bar.progress(66)
            
            execution_container = st.expander("Query Execution Results", expanded=True)
            with execution_container:
                if isinstance(result["result"], dict) and "plot" in result["result"]:
                    try:
                        import base64
                        plot_data = base64.b64decode(result["result"]["plot"])
                        st.image(plot_data, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying plot: {str(e)}")
                else:
                    try:
                        if isinstance(result["result"], dict):
                            df = pd.DataFrame.from_dict(result["result"], orient='index').transpose()
                        elif isinstance(result["result"], list):
                            df = pd.DataFrame(result["result"])
                        else:
                            df = pd.DataFrame([result["result"]])
                        st.dataframe(df)
                    except Exception as e:
                        st.write(result["result"])
            
            # Mark Step 2 as complete
            step2_container.markdown("✅ Query Executed")
            
            # Step 3: Final Answer Generation
            step3_container.markdown("🔄 Generating Response")
            progress_bar.progress(100)
            
            response_container = st.expander("Final Response", expanded=True)
            with response_container:
                st.markdown(result["answer"])
            
            # Mark Step 3 as complete
            step3_container.markdown("✅ Response Generated")
            
            # Show completion status
            status_text.success("Analysis Complete!")
        else:
            # Mark all steps as failed
            step1_container.markdown("❌ Query Generation Failed")
            step2_container.markdown("❌ Query Execution Failed")
            step3_container.markdown("❌ Response Generation Failed")
            progress_bar.empty()
            status_text.error("❌ Analysis Failed!")
            
    except Exception as e:
        # Mark all steps as failed
        step1_container.markdown("❌ Query Generation Failed")
        step2_container.markdown("❌ Query Execution Failed")
        step3_container.markdown("❌ Response Generation Failed")
        st.error(f"Error during analysis: {str(e)}")
        status_text.error("❌ Analysis Failed!")

def main():
    st.title("CSV Data Analyzer")
    st.write("Upload a CSV file and ask questions about your data!")

    # File upload
    #uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    uploaded_file = SimpleFile(FILE_PATH)
    
    if uploaded_file is not None:
        # Show file details
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / (1024*1024):.2f} MB"
        }
        st.write("File Details:", file_details)

        # Preview the data
        try:
            if isinstance(uploaded_file, SimpleFile):
                df = pd.read_csv(uploaded_file.path)
            else:
                # For st.file_uploader case
                temp_file_path = save_uploaded_file(uploaded_file)
                df = pd.read_csv(temp_file_path)
            #df = pd.read_csv(uploaded_file.path)
            preview_container = st.expander("Preview of your data", expanded=False)
            with preview_container:
                st.dataframe(df.head())
            
            # Save the file
            temp_file_path = save_uploaded_file(uploaded_file)
            
            if temp_file_path:
                # Question input
                question = st.text_input(
                    "What would you like to know about your data?",
                    placeholder="e.g., Plot me trend of torque values over each day of updated time? \n What is the minimum and maximum updated time?",
                    key="question_input"
                )
                
                if st.button("Analyze"):
                    if question:
                        display_analysis_results(question, temp_file_path)
                    else:
                        st.warning("Please enter a question before analyzing")
                
                # Cleanup temporary file
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    st.warning(f"Could not delete temporary file: {e}")
            else:
                st.error("Failed to save uploaded file")
                
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty. Please upload a file containing data.")
        except pd.errors.ParserError:
            st.error("Error parsing the CSV file. Please make sure it's properly formatted.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
                    
    else:
        st.info("Please upload a CSV file to begin analysis.")

if __name__ == "__main__":
    st.cache_data.clear()
    main()
