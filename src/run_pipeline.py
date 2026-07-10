from openpyxl import load_workbook
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image
import csv

# Parameters
GT_PATH = "./dataset/dataset.xlsx"
IMAGE_PATH = "./dataset/"
OUTPUT_PATH = "./results.csv"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

# Prompt
PROMPT_TEXT = (
    "You are an expert product analyst who estimates real-world object "
    "properties from images with high accuracy.\n\n"
    "Silently think through these steps before answering (do not show your reasoning):\n"
    "1. Identify the object's category and material.\n"
    "2. Check specifically: is this a case, cover, sleeve, headphone, cable, sticker, card, or other thin/flat item? "
    "If yes, its height/thickness is almost certainly under 3cm, often under 1cm, do not estimate height like a 3D bulky object.\n"
    "3. Estimate length and width by comparing to a known reference: a credit card is 8.5 x 5.4cm, "
    "a smartphone is ~15cm tall, a soda can is ~12cm tall and ~6.6cm wide.\n"
    "4. Estimate weight independently, based on the object's category and typical real-world weight for that "
    "type of product (not solely derived from your size estimate), small accessories are often under 100g, "
    "kitchen/large items can be 500g-3000g+.\n\n"
    "You MUST give a specific numeric estimate. Never refuse or say it cannot be determined. "
    "Do not default to generic 'average' sizes, commit to a specific estimate based on what you actually see.\n\n"
    "Respond with ONLY numbers separated by commas, nothing else, no units, no words, no reasoning shown, "
    "in this exact order: weight_g,length_cm,width_cm,height_cm"
)

class EstimatorPipeline:
    def __init__(self, model_name):

        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            max_model_len=16384,
            gpu_memory_utilization=0.85,
            limit_mm_per_prompt={"image": 1, "video": 0},
        )

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.sampling_params = SamplingParams(temperature=0, max_tokens=200)
    
    def build_prompt(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT_TEXT},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def estimate(self, image_path):
        image = Image.open(image_path).convert("RGB")
        prompt = self.build_prompt()

        outputs = self.llm.generate(
            {"prompt": prompt, "multi_modal_data": {"image": image}},
            self.sampling_params,
        )
        return outputs[0].outputs[0].text.strip()
    
def get_image_name(ws):
    image_names = []
    for row in ws.iter_rows(min_col=6, max_col=6, min_row=4, max_row=100, values_only=True):
        image_names.append(row[0])
    return image_names

 # ---

def main():
    wb = load_workbook(GT_PATH)
    ws = wb["Objects"]

    pipeline = EstimatorPipeline(MODEL)

    # Get the image names to open the image and predict
    image_names = get_image_name(ws)

    # Loop through and get the predictions
    combined_results = []
    for idx, name in enumerate(image_names):
        print(f"[{idx}/{len(image_names)}] Processing {name}...")
        result = pipeline.estimate(IMAGE_PATH + name)
        weight, length, width, height = result.split(",")
        combined_results.append({
            "image_name": name,
            "weight_g": weight.strip(),
            "length_cm": length.strip(),
            "width_cm": width.strip(),
            "height_cm": height.strip(),
        })

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "weight_g", "length_cm", "width_cm", "height_cm"])
        writer.writeheader()
        writer.writerows(combined_results)

    print(f"\nSaved {len(combined_results)} results to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()