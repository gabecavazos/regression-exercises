import openai
import requests
import ipywidgets as widgets
from IPython.display import display


# Define a function that sends a request to the ChatGPT API and returns the response
def get_chat_response(prompt, api_key):
    openai.api_key = api_key

    response = requests.post(
        "https://api.openai.com/v1/engines/davinci-codex/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "prompt": prompt,
            "max_tokens": 150,
            "temperature": 0.7,
            "n": 1,
            "stop": "\n",
        },
    )

    response_dict = response.json()
    return response_dict["choices"][0]["text"]


# Define a Jupyter Lab widget that allows you to enter text and get a response from ChatGPT
class ChatGPTWidget:
    def __init__(self, api_key):
        self.api_key = api_key

        # Create input and output boxes, and a send button
        self.input_box = widgets.Text(description="Input:")
        self.output_box = widgets.HTML()
        self.button = widgets.Button(description="Send")

        # Register a callback function for the button click event
        self.button.on_click(self.send)

        # Display the widget
        display(self.input_box, self.button, self.output_box)

    # Callback function for the send button click event
    def send(self, button):
        # Get the text from the input box and send it to ChatGPT
        prompt = self.input_box.value
        response = get_chat_response(prompt, self.api_key)

        # Display the response in the output box
        self.output_box.value = response