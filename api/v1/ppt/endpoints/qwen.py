from typing import Annotated, List
from fastapi import APIRouter, Body, HTTPException

from utils.available_models import list_available_qwen_models

QWEN_ROUTER = APIRouter(prefix="/qwen", tags=["qwen"])


@QWEN_ROUTER.post("/models/available", response_model=List[str])
async def get_available_models(
    url: Annotated[str, Body(embed=True)],
    api_key: Annotated[str, Body(embed=True)],
):
    try:
        return await list_available_qwen_models(url, api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
