# GPT-OCR Prompt Optimize
The scripts in this project can be used for finding an optimized prompt for correcting OCR engine responses. All the raw OCR output has been provided under the directory `/data/`. Each txt file in the data folder represent a single page having a multi-column layout in French language. Running the `main.py` will give you a comparison results between normalized levenshtein distance for Google Vision and Adobe+GPT in both textual and graphical forms. Finding the prompt that gives better accuracy is done manually by trial-and-error method.
## Setup and Running
1. Clone the repository - `git clone https://github.com/your-username/your-repository.git`
2. Install dependencies.
   `pip install -r requirements.txt`
3. Obtain your OpenAPI key and set it to your local environment variable `OPENAI_API_KEY`
4. An example run script:

 ```python
 python src/main.py 'cjeu-35-turbo-instruct' 'Please note that the text to be corrected is in French. Fix spelling mistakes, do not add/remove words, make consistent word spacing, add missing spaces, fix font case issues within words, fix numbering issues, make consistent line breaks for the following text:'
 ```

## Results
After running the script on input data, 
- **textual results** can be obtained from `/data/fr/GPT_<timestamp>/result.txt`
- **graphical results** can be obtained from `/data/GPT_<timestamp>.png`

Note that every run will generate separate result files/folders that can be uniquely identified by a `<timestamp>`

## Warning
The use of OpenAI's API is not free and is subject to the pricing policies set by OpenAI. Usage costs are incurred based on the number of tokens processed. Detailed pricing information can be found on the OpenAI pricing page. If you choose to run or integrate this code, any and all charges incurred for API usage will be billed to the client of this specific project. We urge you to be mindful of potential bugs or unnecessary loops in the code that might lead to excessive and unintended API calls. Such issues can result in unexpected charges to the client. It's recommended to thoroughly test the code in a controlled environment before extensive usage.

## Issue Reporting
If you encounter any bugs or issues while using the scripts, please report them in the Issues section. Include a brief description of the problem and steps to reproduce it. 