"""
OCR Service — two-pass GPT-4o vision pipeline.

Pass 1 — Document Detection:
    Send the image to GPT-4o and ask it to identify which type of ID document
    it is. Returns a structured JSON with detected type + confidence.

Pass 2 — Field Extraction (RAG-Augmented):
    Use the RAG service to retrieve the exact schema for the detected document.
    Send a second call to GPT-4o with the schema-specific extraction prompt,
    so it knows precisely which fields to extract and in what format.

This two-pass design keeps each prompt focused and produces clean, structured JSON.
"""

import os
import json
import logging
import re
from openai import OpenAI

from app.services.rag_service import get_rag_service

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = [
    "aadhaar", "pan", "passport", "voter_id",
    "driving_license", "ration_card", "utility_bill", "bank_passbook",
]

DETECTION_SYSTEM_PROMPT = """You are an expert Indian KYC document analyst with deep knowledge of
all Indian government-issued identity documents. Analyse the image and identify the document type.

Respond ONLY with valid JSON in this exact format:
{
  "detected_type": "<one of: aadhaar | pan | passport | voter_id | driving_license | ration_card | utility_bill | bank_passbook | unknown>",
  "display_name": "<human-readable name>",
  "confidence": <0.0 to 1.0>,
  "detection_reason": "<brief explanation of visual cues that led to this identification>"
}"""


def detect_document_type(image_b64_uri: str) -> dict:
    """
    Pass 1: Send image to GPT-4o and detect the document type.
    Returns a dict with detected_type, display_name, confidence, detection_reason.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": DETECTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Please identify this KYC document. Look for logos, text, layout patterns, "
                            "and formatting clues to determine the exact document type."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_b64_uri, "detail": "high"},
                    },
                ],
            },
        ],
        max_tokens=300,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
        return result
    except json.JSONDecodeError:
        logger.warning("Detection response was not valid JSON: %s", raw)
        return {
            "detected_type": "unknown",
            "display_name": "Unknown Document",
            "confidence": 0.0,
            "detection_reason": raw,
        }


def extract_fields(image_b64_uri: str, doc_type: str, schema: dict) -> dict:
    """
    Pass 2: Given the document type and RAG-retrieved schema, extract all fields.
    Returns a dict mapping field_key -> extracted_value.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Build the field list from the schema
    fields_description = "\n".join(
        f"  - {key}: {info.get('label', key)}"
        + (f" (hint: {info['hint']})" if "hint" in info else "")
        for key, info in schema["fields"].items()
    )

    extraction_system_prompt = f"""You are an expert OCR engine specialised in Indian KYC documents.
You are processing a {schema['display_name']}. Extract the following fields accurately.

Fields to extract:
{fields_description}

{schema['extraction_prompt']}

Rules:
- Return ONLY valid JSON with the field keys listed above as keys.
- Use null for any field that is not visible or not present.
- Do NOT invent or guess values — only extract what is clearly readable.
- For dates, preserve the format as printed on the document.
- For the Aadhaar number, include spaces (XXXX XXXX XXXX format).
- Trim whitespace from all values."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": extraction_system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Please extract all the requested fields from this document. "
                            "Read every text element carefully, including the fine print."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_b64_uri, "detail": "high"},
                    },
                ],
            },
        ],
        max_tokens=800,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        extracted = json.loads(raw)
        # Remove null values but keep empty strings for fields that exist
        return {k: v for k, v in extracted.items() if v is not None}
    except json.JSONDecodeError:
        logger.warning("Extraction response was not valid JSON: %s", raw)
        return {"raw_text": raw}


def process_document(image_b64_uri: str) -> dict:
    """
    Full pipeline: detect → RAG schema retrieval → extract → return result.

    Returns:
        {
          "document_type": str,
          "display_name": str,
          "confidence": float,
          "detection_reason": str,
          "schema": { fields: {...}, ... },
          "extracted_fields": { field_key: value, ... },
          "rag_similarity_score": float,
        }
    """
    rag = get_rag_service()

    # ── Pass 1: Detect document type ────────────────────────────────────────
    logger.info("Pass 1: Detecting document type...")
    detection = detect_document_type(image_b64_uri)
    doc_type = detection.get("detected_type", "unknown")
    logger.info("Detected: %s (confidence=%.2f)", doc_type, detection.get("confidence", 0))

    # ── RAG: Retrieve schema ─────────────────────────────────────────────────
    if doc_type != "unknown" and doc_type in SUPPORTED_TYPES:
        schema = rag.get_schema_by_type(doc_type)
    else:
        # Fall back to semantic search if type is unknown or unrecognised
        query = (
            f"{detection.get('display_name', '')} "
            f"{detection.get('detection_reason', '')} "
            f"{doc_type}"
        )
        schema = rag.retrieve_schema(query)
        logger.info("RAG fallback — retrieved schema: %s (score=%.3f)",
                    schema.get("type"), schema.get("similarity_score"))

    if not schema:
        return {
            "document_type": doc_type,
            "display_name": detection.get("display_name", "Unknown"),
            "confidence": detection.get("confidence", 0),
            "detection_reason": detection.get("detection_reason", ""),
            "schema": {},
            "extracted_fields": {},
            "rag_similarity_score": 0.0,
            "error": "Could not find a matching document schema.",
        }

    # ── Pass 2: Extract fields ────────────────────────────────────────────────
    logger.info("Pass 2: Extracting fields for %s...", schema["display_name"])
    extracted = extract_fields(image_b64_uri, schema["type"], schema)

    return {
        "document_type": schema["type"],
        "display_name": schema["display_name"],
        "confidence": detection.get("confidence", 1.0),
        "detection_reason": detection.get("detection_reason", ""),
        "schema": {
            "fields": schema["fields"],
            "category": schema.get("category", ""),
        },
        "extracted_fields": extracted,
        "rag_similarity_score": schema.get("similarity_score", 1.0),
    }
