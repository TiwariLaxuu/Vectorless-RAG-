from pageindex import PageIndexClient
import pageindex.utils as utils
import openai, fitz, base64, os

# Get your PageIndex API key from https://dash.pageindex.ai/api-keys
PAGEINDEX_API_KEY = "YOUR_PAGEINDEX_API_KEY"
pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

# Setup OpenAI client
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

async def call_vlm(prompt, image_paths=None, model="gpt-4.1"):
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    messages = [{"role": "user", "content": prompt}]
    if image_paths:
        content = [{"type": "text", "text": prompt}]
        for image in image_paths:
            if os.path.exists(image):
                with open(image, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    })
        messages[0]["content"] = content
    response = await client.chat.completions.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message.content.strip()

def extract_pdf_page_images(pdf_path, output_dir="pdf_images"):
    os.makedirs(output_dir, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    page_images = {}
    total_pages = len(pdf_document)
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        # Convert page to image
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("jpeg")
        image_path = os.path.join(output_dir, f"page_{page_number + 1}.jpg")
        with open(image_path, "wb") as image_file:
            image_file.write(img_data)
        page_images[page_number + 1] = image_path
        print(f"Saved page {page_number + 1} image: {image_path}")
    pdf_document.close()
    return page_images, total_pages

def get_page_images_for_nodes(node_list, node_map, page_images):
    # Get PDF page images for retrieved nodes
    image_paths = []
    seen_pages = set()
    for node_id in node_list:
        node_info = node_map[node_id]
        for page_num in range(node_info['start_index'], node_info['end_index'] + 1):
            if page_num not in seen_pages:
                image_paths.append(page_images[page_num])
                seen_pages.add(page_num)
    return image_paths

import os, requests

# You can also use our GitHub repo to generate PageIndex tree
# https://github.com/VectifyAI/PageIndex

pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"  # the "Attention Is All You Need" paper
pdf_path = os.path.join("../data", pdf_url.split('/')[-1])
os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

response = requests.get(pdf_url)
with open(pdf_path, "wb") as f:
    f.write(response.content)
print(f"Downloaded {pdf_url}\n")

# Extract page images from PDF
print("Extracting page images...")
page_images, total_pages = extract_pdf_page_images(pdf_path)
print(f"Extracted {len(page_images)} page images from {total_pages} total pages.\n")

doc_id = pi_client.submit_document(pdf_path)["doc_id"]
print('Document Submitted:', doc_id)

if pi_client.is_retrieval_ready(doc_id):
    tree = pi_client.get_tree(doc_id, node_summary=True)['result']
    print('Simplified Tree Structure of the Document:')
    utils.print_tree(tree, exclude_fields=['text'])
else:
    print("Processing document, please try again later...")