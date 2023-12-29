# GPT-OCR Prompt Optimize
The scripts in this project can be used for optimizing the prompt used for correcting OCR engine responses. All the necessary data has been provided under the directory `/data/`
## Setup and Running
1. Clone the repository - `git clone https://github.com/your-username/your-repository.git`
2. Install dependencies.
   `pip install -r requirements.txt`
3. Obtain your OpenAPI key and set it to your local environment variable `OPENAI_API_KEY`
4. An example run script:

 `python src/main.py 'cjeu-35-turbo-instruct' 'Please note that the text to be corrected is in French. Fix spelling mistakes, do not add/remove words, make consistent word spacing, add missing spaces, fix font case issues within words, fix numbering issues, make consistent line breaks for the following text:'`

## Results
text here 

## Warning
The use of OpenAI's API is not free and is subject to the pricing policies set by OpenAI. Usage costs are incurred based on the number of tokens processed. Detailed pricing information can be found on the OpenAI pricing page. If you choose to run or integrate this code, any and all charges incurred for API usage will be billed to the client of this specific project. We urge you to be mindful of potential bugs or unnecessary loops in the code that might lead to excessive and unintended API calls. Such issues can result in unexpected charges to the client. It's recommended to thoroughly test the code in a controlled environment before extensive usage.