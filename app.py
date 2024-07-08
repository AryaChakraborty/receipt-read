from utils import generate_text_from_image, config_model, extract_receipt_info
import json
from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv('HF_TOKEN')

model, processor, device = config_model()

# Example usage
image_path = "image2"
image_path = f"images/{image_path}.jpg"  # Replace with your image path
extracted_text = generate_text_from_image(model, image_path, processor, device)

receipt_data = json.dumps(extracted_text)
receipt_data = json.loads(receipt_data)

final_data = extract_receipt_info(receipt_data)

# Save the final data to a result.json file
with open('result.json', 'w') as json_file:
    json.dump(final_data, json_file, indent=4)