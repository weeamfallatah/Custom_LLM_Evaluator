import json
import re
import pandas as pd
from octoai.client import OctoAI
from typing import List, Dict, Any



# Initialize the OctoAI client with an API token
OCTOAPI_TOKEN = "OCTOAPI_TOKEN"
octo_client = OctoAI(api_key=OCTOAPI_TOKEN)

# Function to run OctoAI model
def run_model_octo_ai(messages):
    completion = octo_client.text_gen.create_chat_completion(
        model="meta-llama-3-70b-instruct",
        messages=messages,
        max_tokens=1000,
        temperature=0,
        top_p=1
    )
    return completion.dict()["choices"][0]['message']['content']

# Class to hold parameter names for test cases
class TestCaseParams:
    """
    A collection of constant values representing the keys used in test cases.
    """
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    QUERY = "query"
    CONTEXT = "context"  

# Class representing an evaluation metric and its associated parameters
class Eval_metric:
    def __init__(self, metric_name: str, criteria: str, uses_context: bool, eval_steps: list, eval_steps_correct_example: list, eval_steps_incorrect_example: list, evaluation_params: list):
        """
        Initializes an evaluation metric with the specified parameters.
        
        Args:
        - metric_name: Name of the metric being evaluated (e.g., accuracy, relevance).
        - criteria: The evaluation criteria.
        - eval_steps: A list of steps to evaluate.
        - eval_steps_correct_example: correct example.
        - eval_steps_incorrect_example: incorrect example.
        - uses_context: Boolean indicating whether evaluation uses context.
        - evaluation_params:  parameters for evaluation.
        """
        self.metric_name = metric_name
        self.criteria = criteria
        self.uses_context = uses_context
        self.eval_steps = eval_steps
        self.eval_steps_correct_example = eval_steps_correct_example
        self.eval_steps_incorrect_example = eval_steps_incorrect_example
        self.evaluation_params = evaluation_params

# Main class responsible for evaluating LLM outputs
class CustomLLMEvaluator:
    
    def __init__(self, model_metric_name: str = "meta-llama-3-70b-instruct"):
        """
        Initializes the evaluator with a specific model name for evaluation.
        
        Args:
        - model_metric_name: The name of the model being evaluated.
        - last_metric_name: Last metric used for evaluation (for tracking).
        - improved_steps_df: Cache for improved steps.
        """
        self.model_metric_name = model_metric_name
        self.last_metric_name = None  
        self.improved_steps_df = None  
    
    # Method to run the language model with a given prompt
    def run_model(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        response = run_model_octo_ai(messages)
        return response if response else "Error: Model response not received."

 # --------------------------------------------- Start Generate evaluation steps -----------------------------------------------------------------

    # Generate evaluation steps for a given metric and criteria
    def generate_eval_steps(self, metric_name: str, criteria: str) -> list:
        """
        Generates detailed, step-by-step evaluation instructions based on a metric and criteria.
        
        Args:
        - metric_name: The evaluation metric (e.g., accuracy).
        - criteria: The criteria for evaluating the metric.

        Returns:
        - A list of evaluation steps.
        """

        step_prompt = f'''
        You are tasked with generating detailed, objective, and step-by-step evaluation instructions specifically tailored to the following evaluation metric and criteria:
        
        Metric: {metric_name}\n
        Criteria: {criteria}\n

        Provide a chain-of-thought reasoning approach to break down the evaluation into clear, logical steps.
        '''
        
        # Run the model with the prompt and split the output into individual steps
        eval_steps = self.run_model(step_prompt).splitlines()
        steps = [step.strip() for step in eval_steps if step.strip()]  # Clean up whitespace
        return steps
 # ----------------------------------------------------- End  -----------------------------------------------------------------------------------

 # --------------------------------------------- Start Generate improving steps -----------------------------------------------------------------

    # Function to generate a step-by-step prompt for improving steps.
    def generate_improving_steps  (self, steps, eval_steps_correct_example=None, eval_steps_incorrect_example=None):
        """
        Refines and improves the given evaluation steps by generating a detailed reasoning prompt.
        
        Args:
        steps (list): The initial list of evaluation steps.
        eval_steps_correct_example (list or None): Example steps for a correct evaluation (optional).
        eval_steps_incorrect_example (list or None): Example steps for an incorrect evaluation (optional).

        Returns:
        - The improved steps generated by the model.
        """

        #optional sections for correct and incorrect evaluation examples
        correct_example_section = f"### Correct Evaluation Example:\n{eval_steps_correct_example}\n" if eval_steps_correct_example else ""
        incorrect_example_section = f"### Incorrect Evaluation Example:\n{eval_steps_incorrect_example}\n" if eval_steps_incorrect_example else ""
        
        # Create a detailed prompt to improve the provided evaluation steps
        step_by_step_prompt = f"""
        You are a reasoning assistant tasked with improving the logic and clarity of the steps provided. Your goals are to:
        1. Clarify and improve the structure.
        2. Identify any unclear, conflicting, or missing steps that may lead to confusion or incorrect results.
        3. Ensure the overall logical flow is coherent, proposing adjustments as needed.
        4. Test the steps using the **Evaluation Examples** to ensure they work without causing confusion.

        ## Instructions:
        - Break down the input steps logically.
        - Review and revise each step critically, ensuring clarity, consistency, and logic.
        - By thinking through each one step-by-step rephrase and reorganize steps for better flow, while keeping the original intent intact.
        - Flag any contradictions, ambiguities, or missing details.
        - After revising, evaluate the steps using both the **Correct Evaluation Example** and the **Incorrect Evaluation Example**. Ask: 
            - "Do these steps correctly handle the correct example without confusion?"
            - "Do these steps properly identify or fail with the incorrect example?" 
        Make adjustments if necessary.
        - Your output must be in the form of a **dictionary**.

        ### Input:
        {steps}

        {correct_example_section}
        {incorrect_example_section}
        Please ensure that your answer is unbiased and does not rely on stereotypes.

        ### Dictionary Output Structure:
        "Revised Steps": (List of improved steps.),
        "Revised Steps against the Correct Example": (Assessment of revised steps with the correct example.),
        "Revised Steps against the Incorrect Example": (Assessment of revised steps with the incorrect example.),
        "Final Steps": (If further improvements are necessary, list them here. If no further improvements are needed, use the 'Revised Steps'.),
        "Reasoning and Modifications": (Explanation of changes made in the Revised Steps and Final Steps),
        "Warnings": (List contradictions, ambiguities, or issues in the Input steps.),
        "Notes": (Any additional notes.)

        *NOTE*: Output must be a dictionary only, without pre or post text. if you have any additional text add it to Notes section.
        """
        # Run the model with the prompt and return the improved steps
        response = self.run_model(step_by_step_prompt)
        return response if response else "Error: Model response not received."
   
    # Function to extract a structured dictionary from the response using regex.
    def extract_improving_steps_response(self, response):
        """
        Extracts structured data (like revised steps, examples, reasoning) from the model's response.
        
        Args:
        - response: The raw response from the model (could be JSON or plain text).

        Returns:
        - A pandas DataFrame containing structured data from the response.
        """   
        all_Response = response # Store the full response

        # Attempt to parse `response` as JSON if it is a string
        if isinstance(response, str):
            try:
                response = json.loads(response)
                extracted_evaluation_response = {
                    "Revised Steps": response.get("Revised Steps", []),
                    "Correct Example": response.get("Correct Example", ""),
                    "Incorrect Example": response.get("Incorrect Example", ""),
                    "Reasoning": response.get("Reasoning", ""),
                    "Final Steps": response.get("Final Steps", []),
                    "Warnings": response.get("Warnings", []),
                    "Notes": response.get("Notes", "")
                }
            except json.JSONDecodeError:
                print("Warning: Response is not a valid JSON string, attempting regex extraction.")
                # Initialize dictionary to store regex extracted fields
                extracted_evaluation_response = {}
        else:
            # If `response` is not already a dictionary, initialize as empty dict
            if not isinstance(response, dict):
                extracted_evaluation_response = {}

        # If JSON parsing fails, extract values using regex patterns
        if 'extracted_evaluation_response' not in locals():  # Only if JSON extraction fails
            extracted_evaluation_response["Revised Steps"] = re.findall(r'"Revised Steps":\s*\[(.*?)\]', all_Response)
            extracted_evaluation_response["Correct Example"] = re.search(r'"Correct Example":\s*"(.*?)"', all_Response).group(1) if re.search(r'"Correct Example":\s*"(.*?)"', all_Response) else ""
            extracted_evaluation_response["Incorrect Example"] = re.search(r'"Incorrect Example":\s*"(.*?)"', all_Response).group(1) if re.search(r'"Incorrect Example":\s*"(.*?)"', all_Response) else ""
            extracted_evaluation_response["Reasoning"] = re.search(r'"Reasoning":\s*"(.*?)"', all_Response).group(1) if re.search(r'"Reasoning":\s*"(.*?)"', all_Response) else ""
            extracted_evaluation_response["Final Steps"] = re.findall(r'"Final Steps":\s*\[(.*?)\]', all_Response)
            extracted_evaluation_response["Warnings"] = re.findall(r'"Warnings":\s*\[(.*?)\]', all_Response)
            extracted_evaluation_response["Notes"] = re.search(r'"Notes":\s*"(.*?)"', all_Response).group(1) if re.search(r'"Notes":\s*"(.*?)"', all_Response) else ""

        
        # Convert extracted data to a DataFrame 
        df = pd.DataFrame([extracted_evaluation_response ])
        df["All_Response"] = all_Response

        print("############################### Revised Steps all_Response ##################################")
        print(all_Response)
        return df
    
    # Get or generate improved steps DataFrame
    def improved_steps_check(self, metric_name, eval_steps, eval_steps_correct_example, eval_steps_incorrect_example):
        """
        Checks if improved evaluation steps are already available for the given metric.
        If not, generates them and stores the result in a DataFrame for future use.

        Args:
            metric_name (str): The name of the evaluation metric (e.g., 'accuracy', 'relevance').
            eval_steps (list): A list of steps used for evaluation.
            eval_steps_correct_example (list or None): A list of correct evaluation steps to use as an example.
            eval_steps_incorrect_example (list or None): A list of incorrect evaluation steps to use as an example.

        Returns:
            pandas.DataFrame: A DataFrame containing the improved evaluation steps, along with structured details 
                            such as reasoning, correct/incorrect examples, and final revised steps.
        """

        if self.last_metric_name != metric_name:     # Check if the metric name has changed since the last call
            # Generate improved steps for the evaluation process
            improved_steps_response = self.generate_improving_steps(eval_steps, eval_steps_correct_example, eval_steps_incorrect_example)
            # Extract and structure the improved steps into a DataFrame
            self.improved_steps_df = self.extract_improving_steps_response(improved_steps_response)
            self.last_metric_name = metric_name
        return self.improved_steps_df
 # ----------------------------------------------------- End  -----------------------------------------------------------------------------------

 # ---------------------------------------------    Start Custom LLM Evaluator  -----------------------------------------------------------------

    # Function to evaluate LLM output based on steps and criteria
    def custom_evaluator_prompt(self, metric_name: str, criteria: str, eval_steps: list = None, eval_steps_correct_example: list = None, eval_steps_incorrect_example: list = None, 
                            evaluation_params: list = None, query: str = None, actual_output: str = None, expected_output: str = None, 
                            context: str = None, threshold: float = 0.5) -> dict:
        """
        Creates a custom evaluation prompt for the model, using steps and examples to guide evaluation.
        
        Args:
        - metric_name: The evaluation metric (e.g., accuracy).
        - criteria: The evaluation criteria.
        - eval_steps: Steps for evaluating the output.
        - eval_steps_correct_example: Steps for a correct example (optional).
        - eval_steps_incorrect_example: Steps for an incorrect example (optional).

        Returns:
        - (intended to return an evaluation prompt or result).
       """
        if not eval_steps:
            eval_steps = self.generate_eval_steps(metric_name, criteria)

        improved_steps_df = self.improved_steps_check(metric_name, eval_steps, eval_steps_correct_example, eval_steps_incorrect_example)
        all_improved_steps_response = improved_steps_df["All_Response"][0]
        final_steps = improved_steps_df["Final Steps"][0]
        steps_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(final_steps)])

        Query = f"**Query:** {query}\n" if TestCaseParams.QUERY in evaluation_params and query else ""
        Context = f"""**Context:**\n'{context}'\n*Consider the alignment of the actual output with the context provided, but ensure that this alignment has a moderate impact on the final score.*\n""" if TestCaseParams.CONTEXT in evaluation_params and context else ""
        Expected_output = f"**Expected Output:**\n{expected_output}\n" if TestCaseParams.EXPECTED_OUTPUT in evaluation_params and expected_output else ""

        evaluator_prompt = f"""
        You are a reasoning assistant tasked with evaluating the following LLM output based on the evaluation steps below.

        **Instructions:** 
        - The Evaluation Criteria is provided to outlines the key aspect under evaluation, but it is NOT a standalone judgment factor.
        - Focus on the evaluation steps to guide the evaluation and avoid making any assumptions beyond them.
        - The alignment with the context, if provided, should have a moderate effect rather than heavily impacting the score.
        - Ensure to think step-by-step before arriving at a score. 
        - Your output must be in the form of a **dictionary**.

        {Query}
        {Context}
        {Expected_output}
        **Actual Output:**\n{actual_output}\n\n

        **Evaluation Criteria:** {criteria}\n
        **Evaluation Steps:**\n{steps_text} \n

        **Dictionary Output Structure:**
        "Score": (Score ranging from 0 to 1 based on the evaluation steps) ,
        "Reasoning": (The reasons that led to choosing this score, including but not limited to context alignment if applicable.) ,
        "Notes": (Any additional notes.)

        *NOTE*: Output must be a dictionary only, without pre or post text. if you have any additional text add it to Notes section.
        """

        # Extract values using regex patterns 
        evaluation_result = self.run_model(evaluator_prompt)
        all_evaluation_result = evaluation_result
     
        print("############################### all_evaluation_result ##################################")
        print(evaluation_result)

        # Attempt to parse `response` as JSON if it is a string
        # Try parsing the output to JSON
        try:
            evaluation_result = json.loads(evaluation_result) if isinstance(evaluation_result, str) else evaluation_result
        except json.JSONDecodeError:
            print("JSON parsing error - attempting regex extraction.")
            
            # Fallback to regex extraction
            extracted_evaluation_result = {
                "Score": re.search(r'"Score":\s*(\d*\.?\d+)', evaluation_result).group(1) if re.search(r'"Score":\s*(\d*\.?\d+)', evaluation_result) else None,
                "Reasoning": re.search(r'"Reasoning":\s*"(.*?)"', evaluation_result).group(1) if re.search(r'"Reasoning":\s*"(.*?)"', evaluation_result) else "",
                "Notes": re.search(r'"Notes":\s*"(.*?)"', evaluation_result).group(1) if re.search(r'"Notes":\s*"(.*?)"', evaluation_result) else ""
            }
        else:
            # Ensure dictionary has expected structure if JSON parse is successful
            extracted_evaluation_result = {
                "Score": evaluation_result.get("Score", ""),
                "Reasoning": evaluation_result.get("Reasoning", ""),
                "Notes": evaluation_result.get("Notes", "")
            }
       

        all_evaluation_result_df = pd.DataFrame([extracted_evaluation_result])
        all_evaluation_result_df["evaluation_result"] = all_evaluation_result
        scores = all_evaluation_result_df["Score"][0] 
        Reasoning = all_evaluation_result_df["Reasoning"][0]  
        final_score = 1 if scores and (threshold is None or scores >= threshold) else 0

        return {
            "query": query,
            "context": context,
            "actual_output": actual_output,
            "expected_output": expected_output,
            "metric_name": metric_name,
            "scores": scores,
            "final_score": final_score,
            "reasoning": Reasoning,
            "evaluation_result": all_evaluation_result,
            "improved_steps_response": all_improved_steps_response
        }

    # Evaluate multiple LLM outputs from a DataFrame
    def custom_Eval(self, TestCases : pd.DataFrame, metric: Eval_metric, threshold: float = None) -> pd.DataFrame:
        """
        Evaluates multiple LLM outputs from a DataFrame of test cases using a specified evaluation metric.

        Args:
            TestCases (pd.DataFrame): A DataFrame containing test cases, where each row represents a test case.
                                    Required columns include 'query', 'actual_output', and 'expected_output'. 
                                    Optionally, it can include 'context'.
            metric (Eval_metric): An instance of the Eval_metric class, containing the evaluation metric and related information.
            threshold (float, optional): A threshold value to determine whether the evaluation result meets the required standard.
                                        If not provided, default evaluation logic applies.

        Returns:
            pd.DataFrame: A DataFrame containing the evaluation results for each test case.
        """

        # Generate evaluation steps if they are not provided in the metric
        if not metric.eval_steps:
            metric.eval_steps = self.generate_eval_steps(metric.metric_name, metric.criteria)
        
        # Check if the required columns are present in the TestCases DataFrame
        required_columns = ['expected_output']
        for col in required_columns:
            if col not in TestCases.columns:
                raise ValueError(f"Missing required column: {col} in DataFrame.")

        results = []  # List to hold evaluation results for each test case
        # Iterate over each test case in the DataFrame and Evaluate the test case using the custom evaluator prompt
        for _, row in TestCases.iterrows():
            result = self.custom_evaluator_prompt(
                metric_name=metric.metric_name,
                criteria=metric.criteria,
                eval_steps=metric.eval_steps,
                eval_steps_correct_example=metric.eval_steps_correct_example,
                eval_steps_incorrect_example=metric.eval_steps_incorrect_example,
                evaluation_params=metric.evaluation_params,
                query=row.get('query'),
                actual_output=row.get('actual_output'),
                expected_output=row.get('expected_output'),
                context=row.get('context'),
                threshold=threshold,
            )
            results.append(result)

        return pd.DataFrame(results)

 # ----------------------------------------------------- End  -----------------------------------------------------------------------------------
