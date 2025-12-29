import asyncio
import os
import aiohttp
from google import genai
from google.genai.types import GenerateContentConfig
from openai import AsyncOpenAI
from models.image_prompt import ImagePrompt
from models.sql.image_asset import ImageAsset
from utils.download_helpers import download_file
from utils.get_env import get_pexels_api_key_env
from utils.get_env import get_pixabay_api_key_env
from utils.get_env import get_flux_api_key_env
from utils.get_env import get_flux_url_env
from utils.image_provider import (
    is_image_generation_disabled,
    is_pixels_selected,
    is_pixabay_selected,
    is_gemini_flash_selected,
    is_dalle3_selected,
    is_flux_selected
)
import uuid


class ImageGenerationService:
    # Class-level semaphore shared across all instances
    # This ensures only 1 Flux request runs at a time globally
    _flux_semaphore = None
    _semaphore_lock = None

    def __init__(self, output_directory: str):
        self.output_directory = output_directory
        self.is_image_generation_disabled = is_image_generation_disabled()
        self.image_gen_func = self.get_image_gen_func()
        # Initialize class-level semaphore on first instance creation
        if ImageGenerationService._flux_semaphore is None:
            ImageGenerationService._flux_semaphore = asyncio.Semaphore(1)

    @property
    def flux_semaphore(self):
        """Access the shared class-level semaphore"""
        if ImageGenerationService._flux_semaphore is None:
            ImageGenerationService._flux_semaphore = asyncio.Semaphore(1)
        return ImageGenerationService._flux_semaphore

    def get_image_gen_func(self):
        if self.is_image_generation_disabled:
            return None

        if is_pixabay_selected():
            return self.get_image_from_pixabay
        elif is_pixels_selected():
            return self.get_image_from_pexels
        elif is_gemini_flash_selected():
            return self.generate_image_google
        elif is_dalle3_selected():
            return self.generate_image_openai
        elif is_flux_selected():
            return self.generate_image_flux
        return None

    def is_stock_provider_selected(self):
        return is_pixels_selected() or is_pixabay_selected()

    async def generate_image(self, prompt: ImagePrompt) -> str | ImageAsset:
        """
        Generates an image based on the provided prompt.
        - If no image generation function is available, returns a placeholder image.
        - If the stock provider is selected, it uses the prompt directly,
        otherwise it uses the full image prompt with theme.
        - Output Directory is used for saving the generated image not the stock provider.
        """
        if self.is_image_generation_disabled:
            print("Image generation is disabled. Using placeholder image.")
            return "/static/images/placeholder.jpg"

        if not self.image_gen_func:
            print("No image generation function found. Using placeholder image.")
            return "/static/images/placeholder.jpg"

        image_prompt = prompt.get_image_prompt(
            with_theme=not self.is_stock_provider_selected()
        )
        print(f"Request - Generating Image for {image_prompt}")

        try:
            if self.is_stock_provider_selected():
                image_path = await self.image_gen_func(image_prompt)
            else:
                image_path = await self.image_gen_func(
                    image_prompt, self.output_directory
                )
            if image_path:
                if image_path.startswith("http"):
                    return image_path
                elif os.path.exists(image_path):
                    return ImageAsset(
                        path=image_path,
                        is_uploaded=False,
                        extras={
                            "prompt": prompt.prompt,
                            "theme_prompt": prompt.theme_prompt,
                        },
                    )
            raise Exception(f"Image not found at {image_path}")

        except Exception as e:
            print(f"Error generating image: {e}")
            return "/static/images/placeholder.jpg"

    async def generate_image_openai(self, prompt: str, output_directory: str) -> str:
        client = AsyncOpenAI()
        result = await client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            quality="standard",
            size="1024x1024",
        )
        image_url = result.data[0].url
        return await download_file(image_url, output_directory)

    async def generate_image_flux(self, prompt: str, output_directory: str) -> str:
        # Use semaphore to limit concurrent Flux API requests
        print(f"[FLUX DEBUG] Waiting for semaphore... (prompt: {prompt[:50]}...)")
        async with self.flux_semaphore:
            print(f"[FLUX DEBUG] Semaphore acquired, starting Flux request")
            flux_url = get_flux_url_env()
            flux_api_key = get_flux_api_key_env()

            if not flux_url or not flux_api_key:
                raise Exception("FLUX_URL and FLUX_API_KEY must be set in environment or user config")

            payload = {
                "prompt": prompt,
                "width": 512,
                "height": 512,
                "guidance_scale": 3.5,
                "output_type": "pil",
                "num_inference_steps": 5,
                "max_sequence_length": 512
            }

            headers = {
                'Content-Type': 'application/json',
                'Authorization': flux_api_key
            }

            # Set a 120 second timeout for Flux API (increased from 60)
            timeout = aiohttp.ClientTimeout(total=120)

            print(f"[FLUX DEBUG] Making HTTP POST to {flux_url}")
            async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as session:
                async with session.post(
                    flux_url,
                    headers=headers,
                    json=payload
                ) as response:
                    print(f"[FLUX DEBUG] Received response with status: {response.status}")
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Flux API error: {response.status} - {error_text}")

                    # Check content type to determine response format
                    content_type = response.headers.get('Content-Type', '')
                    print(f"[FLUX DEBUG] Response content type: {content_type}")

                    # If response is an image (JPEG, PNG, etc.)
                    if 'image/' in content_type:
                        image_data = await response.read()
                        # Determine file extension from content type
                        ext = 'jpg' if 'jpeg' in content_type else 'png'
                        image_path = os.path.join(output_directory, f"{uuid.uuid4()}.{ext}")
                        with open(image_path, "wb") as f:
                            f.write(image_data)
                        return image_path

                    # If response is JSON
                    elif 'application/json' in content_type:
                        result = await response.json()

                        # If the API returns a URL
                        if "url" in result:
                            image_url = result["url"]
                            return await download_file(image_url, output_directory)

                        # If the API returns base64 image data
                        elif "image" in result:
                            import base64
                            image_data = base64.b64decode(result["image"])
                            image_path = os.path.join(output_directory, f"{uuid.uuid4()}.png")
                            with open(image_path, "wb") as f:
                                f.write(image_data)
                            return image_path
                        else:
                            raise Exception(f"Unexpected Flux API response format: {result}")

                    else:
                        raise Exception(f"Unexpected content type from Flux API: {content_type}") 
    
    async def generate_image_google(self, prompt: str, output_directory: str) -> str:
        client = genai.Client()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash-image-preview",
            contents=[prompt],
            config=GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )

        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                image_path = os.path.join(output_directory, f"{uuid.uuid4()}.jpg")
                with open(image_path, "wb") as f:
                    f.write(part.inline_data.data)

        return image_path

    async def get_image_from_pexels(self, prompt: str) -> str:
        async with aiohttp.ClientSession(trust_env=True) as session:
            response = await session.get(
                f"https://api.pexels.com/v1/search?query={prompt}&per_page=1",
                headers={"Authorization": f"{get_pexels_api_key_env()}"},
            )
            data = await response.json()
            image_url = data["photos"][0]["src"]["large"]
            return image_url

    async def get_image_from_pixabay(self, prompt: str) -> str:
        async with aiohttp.ClientSession(trust_env=True) as session:
            response = await session.get(
                f"https://pixabay.com/api/?key={get_pixabay_api_key_env()}&q={prompt}&image_type=photo&per_page=3"
            )
            data = await response.json()
            image_url = data["hits"][0]["largeImageURL"]
            return image_url
