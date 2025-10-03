import argparse
import os
import sys
from PIL import Image
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.model.inference import inference_with_vllm

parser = argparse.ArgumentParser(description="Run dots_ocr demo against a local vLLM server")
parser.add_argument("--ip", type=str, default="localhost", help="vLLM server ip")
parser.add_argument("--port", type=str, default="8000", help="vLLM server port")
parser.add_argument("--model_name", type=str, default="model", help="model name on the server")
parser.add_argument("--prompt_mode", type=str, default="prompt_layout_all_en", help="key into dict_promptmode_to_prompt")
parser.add_argument("--image", type=str, default="demo/demo_image1.jpg", help="path to input image")
parser.add_argument("--timeout", type=int, default=60, help="request timeout seconds")

args = parser.parse_args()

def main():
    if not os.path.exists(args.image):
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        sys.exit(2)

    prompt = dict_promptmode_to_prompt.get(args.prompt_mode)
    if prompt is None:
        print(f"ERROR: unknown prompt_mode '{args.prompt_mode}'. Available keys: {list(dict_promptmode_to_prompt.keys())}", file=sys.stderr)
        sys.exit(3)

    try:
        image = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"ERROR: failed to open image {args.image}: {e}", file=sys.stderr)
        sys.exit(4)

    response = inference_with_vllm(
        image,
        prompt,
        ip=args.ip,
        port=args.port,
        temperature=0.1,
        top_p=0.9,
        model_name=args.model_name,
        timeout_seconds=args.timeout,
    )

    if response is None:
        print("No response (request error).", file=sys.stderr)
        sys.exit(5)

    print("response:")
    print(response)

if __name__ == "__main__":
    main()
