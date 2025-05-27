import os
import cv2
import sys
import numpy as np
from typing import List
from PIL import Image, ImageDraw



import streamlit as st

# sys.path.append('./')
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM

# Import the required modules from the original code
idmv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "dependencies/IDM-VTON"))
sys.path.append(idmv_path)

from dependency_manager import setup_idmvton

# Setup the dependency
idmvton_manager = setup_idmvton()

# Import modules
from idm_vton_imports import (
    TryonPipeline,
    UNet2DConditionModel_ref,
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
    apply_net,
    Parsing,
    OpenPose,
    convert_PIL_to_numpy,
    apply_exif_orientation
)

# from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
# from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
# from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
# import gradio_demo.apply_net as apply_net
# from preprocess.humanparsing.run_parsing import Parsing
# from preprocess.openpose.run_openpose import OpenPose
# from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

# Optional SAM imports (install with: pip install segment-anything)
from segment_anything import sam_model_registry, SamPredictor

# from describer import Describer
# from utils_mask import get_mask_location

# Set page config
st.set_page_config(
    page_title="Virtual Try-On",
    page_icon="👕",
    layout="wide"
)

# Initialize device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def create_simple_mask(image, mask_type="upper_body"):
    """Create a simple mask for upper body area"""
    width, height = image.size
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    if mask_type == "upper_body":
        # Create a rough upper body mask (chest and torso area)
        # This is a simplified approach - you might want to adjust based on your needs
        top_y = int(height * 0.2)  # Start from 20% down
        bottom_y = int(height * 0.7)  # End at 70% down
        left_x = int(width * 0.25)  # Start from 25% from left
        right_x = int(width * 0.75)  # End at 75% from left

        # Draw rectangle for upper body
        draw.rectangle([left_x, top_y, right_x, bottom_y], fill=255)

        # Make it more natural with an ellipse
        draw.ellipse([left_x, top_y, right_x, bottom_y], fill=255)

    return mask


def create_sam_mask(image, sam_predictor, prompt, points=None):
    """Create mask using SAM model"""

    # Convert PIL to numpy
    image_np = np.array(image)
    sam_predictor.set_image(image_np)

    if points is None:
        # Default points for upper body (you can make this interactive)
        height, width = image_np.shape[:2]
        points = np.array([
            [width // 2, height // 3],  # Center chest
            [width // 3, height // 2],  # Left torso
            [2 * width // 3, height // 2]  # Right torso
        ])

    labels = np.array([1, 1, 1])  # All positive points

    masks, scores, logits = sam_predictor.predict(
        text_prompt = '',
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )

    # Use the mask with highest score
    best_mask = masks[np.argmax(scores)]
    mask_image = Image.fromarray((best_mask * 255).astype(np.uint8))

    return mask_image


@st.cache_resource
def load_sam_model():
    """Load SAM model"""

    sam_checkpoint = "./ckpt/sam/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    if not os.path.exists(sam_checkpoint):
        st.error(f"SAM checkpoint not found: {sam_checkpoint}")
        return None

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def pil_to_binary_mask(pil_image, threshold=0):
    """Convert PIL image to binary mask"""
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i, j] == True:
                mask[i, j] = 1
    mask = (mask * 255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


# def classify_garment_type_with_qwen(image):
#     processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL-Chat")
#     model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", torch_dtype=torch.float16)
#
#     prompt = "<|user|>\n<|image_1|>\nWhat is this dress? Is it upper body clothing like a shirt or t-shirt, or lower body clothing like pants?\n<|endoftext|>\n<|assistant|>"
#
#     inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
#     output = model.generate(**inputs, max_new_tokens=30)
#     answer = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
#     return answer.lower()


@st.cache_resource
def load_models():
    """Load all the required models with caching"""
    base_path = 'yisol/IDM-VTON'

    # Load UNet models
    unet = UNet2DConditionModel.from_pretrained(
        base_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    unet.requires_grad_(False)

    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        base_path,
        subfolder="unet_encoder",
        torch_dtype=torch.float16,
    )
    UNet_Encoder.requires_grad_(False)

    # Load tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )

    # Load scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

    # Load text encoders
    text_encoder_one = CLIPTextModel.from_pretrained(
        base_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )

    # Load image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    )

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        base_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    )

    # Set requires_grad to False
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # Create pipeline
    pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    )
    pipe.unet_encoder = UNet_Encoder

    # Load the Qwen Model for image descriptions and SAM for mask generation
    cloth_describer = Describer()
    mask_predictor = load_sam_model()

    # Load parsing and openpose models (for original method)
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)

    return pipe, cloth_describer, mask_predictor, parsing_model, openpose_model


# def get_mask_location_original(model_parse, keypoint, width=384, height=512):
#     """Original mask generation method from the provided code"""
#     # This is a simplified version - you'd need to implement the full function
#     # with all the label mappings and logic from your provided code
#
#     im_parse = model_parse.resize((width, height), Image.NEAREST)
#     parse_array = np.array(im_parse)
#
#     # For upper body (simplified)
#     parse_mask = (parse_array == 4).astype(np.float32) + (parse_array == 7).astype(np.float32)
#
#     # Create binary mask
#     mask = Image.fromarray((parse_mask * 255).astype(np.uint8))
#     mask_gray = Image.fromarray((parse_mask * 127).astype(np.uint8))
#
#     return mask, mask_gray


def perform_tryon(human_img, garment_img, garment_desc, mask_method="simple", use_auto_crop=False, denoise_steps=30, seed=42):
    """Perform the virtual try-on"""

    # Load models
    pipe, cloth_describer, mask_predictor, parsing_model, openpose_model = load_models()

    # Get description of the cloth
    question = "What is this dress? Is it upper body clothing like a shirt or t-shirt, or lower body clothing like pants? Classify between [upper/lower]"
    cloth_position = cloth_describer.describe_image(garment_img, question=question)

    if 'upper' in cloth_position:
        mask_prompt = "Segment the entire upper body garment in the image. Make sure the segmentation includes the full shirt or t-shirt, including sleeves, shoulders, and neckline. It’s okay if the mask includes parts of the mannequin or background, but do not exclude any visible part of the clothing."
    elif 'lower' in cloth_position:
        mask_prompt = "Segment the entire lower body garment in the image. Include the full pants, skirt, or shorts from waist to hem. The mask should cover the complete garment even if it overlaps with background or legs. Avoid excluding any part of the clothing."
    else:
        mask_prompt = 'Segment the main garment in the image. Ensure that the segmentation includes the entire clothing item as shown, even if some parts are partially occluded or overlap with background or body. The mask can be slightly larger than the garment, but must not miss any part of it.'

    # Move models to device
    if mask_method == "original":
        openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    # Prepare images
    garm_img = garment_img.convert("RGB").resize((768, 1024))
    human_img_orig = human_img.convert("RGB")
    human_img = human_img_orig.resize((768, 1024))

    # # Handle cropping
    # if use_auto_crop:
    #     width, height = human_img_orig.size
    #     target_width = int(min(width, height * (3 / 4)))
    #     target_height = int(min(height, width * (4 / 3)))
    #     left = (width - target_width) / 2
    #     top = (height - target_height) / 2
    #     right = (width + target_width) / 2
    #     bottom = (height + target_height) / 2
    #     cropped_img = human_img_orig.crop((left, top, right, bottom))
    #     crop_size = cropped_img.size
    #     human_img = cropped_img.resize((768, 1024))
    # else:
    #     human_img = human_img_orig.resize((768, 1024))

    # Generate mask based on selected method
    if mask_method == "sam":
        mask = create_sam_mask(human_img, mask_predictor, mask_prompt)
        mask = mask.resize((768, 1024))
    elif mask_method == "original":
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768, 1024))
    else:  # simple method
        mask = create_simple_mask(human_img.resize((768, 1024)))

    # Prepare tensor transforms
    tensor_transfrom = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize([0.5], [0.5]),
                                           ])

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    # Prepare pose image (required for the pipeline)
    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args((
        'show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
        './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v',
        '--opts', 'MODEL.DEVICE', 'cuda'
    ))
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    # Generate try-on result
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # Encode prompts for human
                prompt = "model is wearing " + garment_desc
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                if not isinstance(negative_prompt, List):
                    negative_prompt = [negative_prompt] * 1

                with torch.inference_mode():
                    (prompt_embeds,
                     negative_prompt_embeds,
                     pooled_prompt_embeds,
                     negative_pooled_prompt_embeds,) = pipe.encode_prompt(prompt,
                                                                          num_images_per_prompt=1,
                                                                          do_classifier_free_guidance=True,
                                                                          negative_prompt=negative_prompt,
                                                                          )

                    # Encode prompts for garment
                    prompt_c = "a photo of " + garment_desc
                    (prompt_embeds_c, _, _, _,) = pipe.encode_prompt([prompt_c],
                                                                     num_images_per_prompt=1,
                                                                     do_classifier_free_guidance=False,
                                                                     negative_prompt=negative_prompt)

                    # Prepare tensors
                    pose_img_tensor = tensor_transfrom(pose_img).unsqueeze(0).to(device, torch.float16)
                    garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, torch.float16)
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

                    # Generate images
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device, torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_img_tensor.to(device, torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                        cloth=garm_tensor.to(device, torch.float16),
                        mask_image=mask,
                        image=human_img,
                        height=1024,
                        width=768,
                        ip_adapter_image=garm_img.resize((768, 1024)),
                        guidance_scale=2.0,
                    )[0]

    # # Handle cropping output
    # if use_auto_crop:
    #     out_img = images[0].resize(crop_size)
    #     human_img_orig.paste(out_img, (int(left), int(top)))
    #     return human_img_orig, mask_gray, mask
    # else:
    return images[0], mask_gray, mask


def main():
    st.title("👕 Virtual Try-On")
    st.markdown("Upload a person's photo and a garment image to see the virtual try-on result!")

    # Create columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Person Image")

        # Option to use webcam or upload
        input_method = st.radio(
            "Choose input method:",
            ["Upload Image", "Use Webcam (Coming Soon)"],
            key="input_method"
        )

        if input_method == "Upload Image":
            human_img = st.file_uploader(
                "Upload person image",
                type=['png', 'jpg', 'jpeg'],
                key="human"
            )

            if human_img is not None:
                human_image = Image.open(human_img)
                st.image(human_image, caption="Person Image", use_column_width=True)
        else:
            st.info("Webcam functionality will be added in the next update!")
            human_image = None

    with col2:
        st.subheader("👔 Garment Image")
        garment_img = st.file_uploader(
            "Upload garment image",
            type=['png', 'jpg', 'jpeg'],
            key="garment"
        )

        if garment_img is not None:
            garment_image = Image.open(garment_img)
            st.image(garment_image, caption="Garment Image", use_column_width=True)

    # # Garment description
    # st.subheader("📝 Garment Description")
    # col1, col2 = st.columns([3, 1])
    # with col1:
    #     garment_desc = st.text_input(
    #         "Describe the garment",
    #         placeholder="e.g., Short Sleeve Round Neck T-shirt",
    #         help="This will be used to generate the try-on result"
    #     )
    # with col2:
    #     if st.button("🤖 Auto-Generate (Coming Soon)"):
    #         st.info("Qwen integration for auto-description coming soon!")

    # Masking method selection
    st.subheader("🎯 Mask Generation Method")
    mask_method = st.selectbox(
        "Choose masking method:",
        ["simple", "sam", "original"],
        format_func=lambda x: {
            "simple": "Simple Geometric Mask (Fast)",
            "sam": "SAM - Segment Anything Model (Best Quality)",
            "original": "Original Human Parsing (Slow but Accurate)"
        }[x],
        help="SAM provides the best results but requires additional setup"
    )

    # if mask_method == "sam" and not SAM_AVAILABLE:
    #     st.warning("⚠️ SAM is not installed. Install with: `pip install segment-anything`")
    #     st.info("Also download the SAM checkpoint from Meta's repository")

    # Advanced settings
    # with st.expander("⚙️ Advanced Settings"):
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         use_auto_crop = st.checkbox("Use auto-crop & resizing", value=False)
    #     with col2:
    #         denoise_steps = st.slider("Denoising Steps", 20, 40, 30)
    #         seed = st.number_input("Seed", min_value=-1, max_value=2147483647, value=42)

    # Try-on button
    if st.button("🎯 Try On!", type="primary", use_container_width=True):
        if input_method == "Upload Image" and human_img is not None and garment_img is not None and garment_desc:
            with st.spinner("Generating try-on result... This may take a few moments."):
                try:
                    result_img, mask_img, generated_mask = perform_tryon(
                        human_image,
                        garment_image,
                        # garment_desc,
                        mask_method,
                        # use_auto_crop,
                        # denoise_steps,
                        # seed
                    )

                    # Display results
                    st.subheader("✨ Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(result_img, caption="Try-On Result", use_column_width=True)
                    with col2:
                        st.image(generated_mask, caption="Generated Mask", use_column_width=True)
                    with col3:
                        st.image(mask_img, caption="Processed Mask", use_column_width=True)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please make sure all model files are properly installed and accessible.")
        else:
            st.warning("Please upload both images and provide a garment description.")

    # Information boxes
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "💡 **Masking Methods:**\n"
            "- **Simple**: Fast geometric mask, good for testing\n"
            "- **SAM**: Best quality, requires setup\n"
            "- **Original**: Uses human parsing, most accurate for complex poses"
        )

    with col2:
        st.info(
            "🚀 **Coming Soon:**\n"
            "- Webcam integration\n"
            "- Real-time try-on preview\n"
            "- Multiple garment categories"
        )

    # Footer
    st.markdown("---")
    st.markdown(
        "📚 **Setup Requirements:** This demo requires IDM-VTON model files. "
        "Check out the [source code](https://github.com/yisol/IDM-VTON) and "
        "[model](https://huggingface.co/yisol/IDM-VTON) for setup instructions."
    )


if __name__ == "__main__":
    main()
