"""
PII Redaction using Microsoft Presidio
Mask sensitive data in insurance LLM responses
"""
from dotenv import load_dotenv; load_dotenv()

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    print("Install: pip install presidio-analyzer presidio-anonymizer")
    print("Also run: python -m spacy download en_core_web_lg")

def redact_pii(text: str, entities: list = None) -> dict:
    """Redact PII from text using Presidio."""
    if not PRESIDIO_AVAILABLE:
        return {"redacted_text": text, "entities_found": [], "error": "Presidio not available"}
    
    if entities is None:
        entities = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD",
                    "IBAN_CODE", "MEDICAL_LICENSE", "US_SSN", "DATE_TIME",
                    "IP_ADDRESS", "LOCATION", "NRP"]
    
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    
    results = analyzer.analyze(text=text, entities=entities, language="en")
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={"PERSON": OperatorConfig("replace", {"new_value": "<CUSTOMER_NAME>"}),
                   "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
                   "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
                   "CREDIT_CARD": OperatorConfig("mask", {"chars_to_mask": 12, "masking_char": "*", "from_end": False}),
                   "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"})}
    )
    return {"redacted_text": anonymized.text,
            "entities_found": [{"type": r.entity_type, "score": r.score} for r in results]}

def redact_llm_response(llm_response: str) -> str:
    """Post-process LLM response to remove any PII that slipped through."""
    result = redact_pii(llm_response)
    return result["redacted_text"]

if __name__ == "__main__":
    sample_texts = [
        "Customer Rajesh Kumar (9876543210) filed claim. Policy holder email: rajesh@gmail.com",
        "Card ending 4111111111111111 used for premium payment by Priya Singh at Mumbai.",
    ]
    for text in sample_texts:
        result = redact_pii(text)
        print(f"Original: {text}")
        print(f"Redacted: {result['redacted_text']}")
        print(f"Entities: {result['entities_found']}\n")
