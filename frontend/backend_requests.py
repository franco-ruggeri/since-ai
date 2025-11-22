import requests
import json
import pandas as pd


def get_visualization(user_prompt: str, dataframe: pd.DataFrame) -> dict:
    """
    Converts a dataframe to JSON and sends it to the backend API along with user prompt.
    
    Args:
        user_prompt (str): The user's prompt/query
        dataframe (pd.DataFrame): The dataframe to visualize
        
    Returns:
        dict: The API response
        
    Raises:
        ValueError: If dataframe cannot be converted to JSON
        requests.RequestException: If API request fails
    """
    
    # Convert dataframe to JSON
    try:
        df_json = dataframe.to_json(orient='split')
        df_dict = json.loads(df_json)
    except Exception as e:
        raise ValueError(f"Failed to convert dataframe to JSON: {str(e)}")
    
    # Prepare the API request payload
    payload = {
        "user_prompt": user_prompt,
        "data": df_dict
    }
    
    # Make API request to backend
    try:
        # Replace with your actual backend URL
        backend_url = "http://localhost:8000/visualize" # Swap the url for request
        response = requests.post(backend_url, json=payload)
        response.raise_for_status()
        
        return response.json()
    except requests.RequestException as e:
        raise requests.RequestException(f"API request failed: {str(e)}")
