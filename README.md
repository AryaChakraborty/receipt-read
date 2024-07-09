## Receipt Image to Text Extraction Project

This project extracts text from receipt images and processes the data to generate structured JSON output.

## How to Use This Project

### 1. Create a Virtual Environment

First, create a virtual environment to manage dependencies.

- For Mac users
```bash
python3 -m venv venv
source venv/bin/activate
```

- For Windows users
```bash
python -m venv venv
venv/scripts/activate
```

### 2. Add the HF_TOKEN in .env File

#### Providing Access Token

- To get your personal access token from HuggingFace Hub, vist [here](https://huggingface.co/settings/tokens)
- **Note** - if you do not have an account in Huggingface Hub, create one by signing up.
- Click the "New Token" button at the bottom to create a new token. Copy the token and paste it after running the next code block.

#### Adding the Access Token

Create a `.env` file in the root directory of the project and add your Hugging Face token.

```
HF_TOKEN=your_hugging_face_token_here
```

### 3. Install Requirements

Install the necessary packages using `pip`.

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Execute the application to process the receipt images.

```bash
python src/app.py
```

### 5. Check the Output

After running the application, check the `outputs` folder for the `result.json` file containing the processed data.

## Additional Information

The experiments for this project were run in the `receipt_image_to_text.ipynb` notebook. This notebook contains detailed steps and code used during the development and testing of the text extraction process.

## License

This project is licensed under the MIT License.