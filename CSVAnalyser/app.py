from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
import re
from dotenv import load_dotenv, find_dotenv
import os
from langgraph.graph import StateGraph, END, START
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import TypedDict, Annotated
import uvicorn
import base64
import io

# Load environment variables
_ = load_dotenv(find_dotenv())

# Initialize FastAPI app
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    file_path: str

class State(TypedDict):
    question: str
    code: str
    result: str
    answer: str

class CSVAnalyzer:
    def __init__(self):
        # Only use Azure OpenAI
        self.llm = AzureChatOpenAI(
            temperature=0.0,
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2023-05-15"
        )
        self.df = None

    def load_csv(self, file_path: str):
        self.df = pd.read_csv(file_path)
        self.df = self.df.dropna()
        self._process_datetime_columns()
        return self.df

    def _is_datetime(self, series):
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False

        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\s+\d{1,2}:\d{1,2}(:\d{1,2})?'
        ]

        for val in sample:
            if any(re.match(pattern, str(val)) for pattern in date_patterns):
                return True
        return False

    def _process_datetime_columns(self):
        for col in self.df.columns:
            if self._is_datetime(self.df[col]):
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    self.df[col] = pd.to_datetime(self.df[col], format='%m/%d/%Y %H:%M')
                except Exception as e:
                    print(f"Could not convert column '{col}': {e}")

    def write_query(self, state: State):
        prompt = PromptTemplate(
            template="""Given the following pandas DataFrame with columns: {columns}

            Question: {query}

            Return ONLY the Python code without any explanations or text descriptions.
            The code should:
            1. Use the DataFrame named 'df'
            2. Store any final data in a variable named 'result'
            3. Include only the necessary import statements and code
            """,
            input_variables=["query", "columns"]
        )

        parser = StrOutputParser()
        chain = prompt | self.llm | parser
        pandas_code = chain.invoke({
            "query": state["question"],
            "columns": ", ".join(self.df.columns)
        })

        code_blocks = pandas_code.split('```')
        if len(code_blocks) > 1:
            cleaned_code = code_blocks[1].replace('python', '').strip()
        else:
            code_lines = pandas_code.split('\n')
            code_start = 0
            for i, line in enumerate(code_lines):
                if line.strip().startswith(('import', 'from', 'df.', 'df[', 'result')):
                    code_start = i
                    break
            cleaned_code = '\n'.join(code_lines[code_start:])

        return {"code": cleaned_code, "question": state["question"]}

    def execute_query(self, state: State):
        namespace = {
            'df': self.df,
            'pd': pd,
            'np': np,
            'plt': plt,
            'groupby': pd.core.groupby.GroupBy,
            'DataFrame': pd.DataFrame,
            'Series': pd.Series
        }

        exec(state["code"], namespace)

        if 'result' in namespace:
            result = namespace['result']
            if isinstance(result, pd.Series):
                result = result.to_frame()
            elif not isinstance(result, pd.DataFrame):
                if isinstance(result, dict):
                    result = pd.DataFrame([result])
                elif isinstance(result, (list, np.ndarray)):
                    result = pd.DataFrame(result)
                else:
                    result = pd.DataFrame({'result': [result]})

        if 'plt' in state["code"]:
            current_fig = plt.gcf()
            current_axes = current_fig.gca()

            plot_metadata = {
                'plot': None,
                'type': 'plot',
                'title': current_axes.get_title(),
                'xlabel': current_axes.get_xlabel(),
                'ylabel': current_axes.get_ylabel(),
                'has_legend': current_axes.get_legend() is not None
            }

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            plot_metadata['plot'] = plot_data
            result = plot_metadata

        return {"code": state["code"], "result": result}

    def generate_answer(self, state: State):
        is_plot = isinstance(state["result"], dict) and "plot" in state["result"]

        if is_plot:
            image_url = f"data:image/png;base64,{state['result']['plot']}"
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze this plot generated in response to the question: '{state['question']}'. "
                                    "Describe what the visualization shows, including any notable trends or patterns."
                                    f"\n\nPlot Details:\n"
                                    f"- Title: {state['result'].get('title', 'Not specified')}\n"
                                    f"- X-axis: {state['result'].get('xlabel', 'Not specified')}\n"
                                    f"- Y-axis: {state['result'].get('ylabel', 'Not specified')}\n"
                                    f"- Legend: {'Present' if state['result'].get('has_legend', False) else 'Not present'}\n"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ]
        else:
            prompt = (
                "Given the following user question, corresponding Python code, "
                "and result, answer the user question.\n\n"
                f'Question: {state["question"]}\n'
                f'Code: {state["code"]}\n'
                f'Result from code: {state["result"]}'
            )

        response = self.llm.invoke(prompt)
        return {"answer": response.content}

    def setup_workflow(self):
        workflow = StateGraph(State)
        workflow.add_node("write_query", self.write_query)
        workflow.add_node("execute_query", self.execute_query)
        workflow.add_node("generate_answer", self.generate_answer)

        workflow.add_edge(START, "write_query")
        workflow.add_edge("write_query", "execute_query")
        workflow.add_edge("execute_query", "generate_answer")

        return workflow.compile()

# Initialize analyzer
analyzer = CSVAnalyzer()

@app.post("/analyze")
async def analyze_csv(request: QuestionRequest):
    try:
        analyzer.load_csv(request.file_path)
        workflow = analyzer.setup_workflow()
        result = workflow.invoke({"question": request.question})

        response_dict = {
            "answer": result["answer"],
            "code": result.get("code"),
        }

        if isinstance(result.get("result"), dict) and "plot" in result.get("result"):
            response_dict["result"] = result["result"]
        else:
            if hasattr(result.get("result"), 'to_dict'):
                response_dict["result"] = result["result"].to_dict()
            else:
                response_dict["result"] = result.get("result")

        return response_dict
    except Exception as e:
        print(f"Error in analyze_csv: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
