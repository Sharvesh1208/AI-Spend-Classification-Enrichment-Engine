import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import spacy
import numpy as np
from collections import defaultdict
import logging
from transformers import pipeline as hf_pipeline
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")


class EnrichedDescriptionGenerator:
    """
    Model-based enriched description generator using LLM
    """

    def __init__(self, model_name="google/flan-t5-base"):
        """
        Initialize with language model
        Options: google/flan-t5-base (fast), google/flan-t5-large (better), microsoft/phi-2 (best)
        """
        logger.info(f"Loading enriched description model: {model_name}")

        try:
            if "t5" in model_name.lower():
                self.generator = hf_pipeline(
                    "text2text-generation",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            else:
                self.generator = hf_pipeline(
                    "text-generation",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )

            self.model_name = model_name
            logger.info(f"Description generator model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            logger.info("Falling back to rule-based generation")
            self.generator = None
            self.model_name = "rule-based"

    def generate_enriched_description(
            self,
            text: str,
            product: str = None,
            category: str = None,
            vendor: str = None,
            quantity: int = None,
            amount: int = None
    ) -> str:
        """Generate enriched description using language model"""

        if self.generator is None:
            return self._rule_based_description(text, product, category, vendor, quantity, amount)

        try:
            prompt = self._build_prompt(text, product, category, vendor, quantity, amount)

            if "t5" in self.model_name.lower():
                result = self.generator(prompt, max_length=80, num_return_sequences=1)
                description = result[0]['generated_text'].strip()
            else:
                result = self.generator(prompt, max_new_tokens=50, num_return_sequences=1)
                full_text = result[0]['generated_text']
                description = full_text[len(prompt):].strip()

            description = self._clean_description(description)

            if len(description) < 10 or len(description) > 200:
                return self._rule_based_description(text, product, category, vendor, quantity, amount)

            return description

        except Exception as e:
            logger.error(f"Error generating description: {str(e)}")
            return self._rule_based_description(text, product, category, vendor, quantity, amount)

    def _build_prompt(self, text, product, category, vendor, quantity, amount):
        """Build structured prompt for language model"""

        prompt_parts = []
        prompt_parts.append("Create a brief business procurement description.")
        prompt_parts.append(f"Item: {text}")

        if product:
            prompt_parts.append(f"Product: {product}")
        if vendor:
            prompt_parts.append(f"From: {vendor}")
        if quantity:
            prompt_parts.append(f"Qty: {quantity}")
        if amount:
            prompt_parts.append(f"Value: ${amount:,}")

        prompt_parts.append("Description:")

        return " ".join(prompt_parts)

    def _clean_description(self, description: str) -> str:
        """Clean and format generated description"""

        description = re.sub(r'^(Description:|Enriched Description:|Summary:)\s*', '', description, flags=re.IGNORECASE)
        description = description.strip().strip('"\'').rstrip('.')

        if description:
            description = description[0].upper() + description[1:]

        return description

    def _rule_based_description(self, text, product, category, vendor, quantity, amount):
        """Fallback rule-based description generator"""

        parts = []

        # Quantity descriptor
        if quantity:
            if quantity == 1:
                parts.append("Single")
            elif quantity <= 5:
                parts.append(f"{quantity}")
            elif quantity <= 20:
                parts.append(f"Bulk order of {quantity}")
            else:
                parts.append(f"Large procurement of {quantity}")

        # Product
        if product and product != "Unknown Product":
            parts.append(product.lower())
        else:
            parts.append("items")

        # Vendor context
        if vendor and vendor != "N/A":
            vendor_short = vendor.split()[0]
            parts.append(f"from {vendor_short}")

        # Infer purpose
        text_lower = text.lower()
        purpose_keywords = {
            'development': 'for development team',
            'developer': 'for development team',
            'office': 'for office operations',
            'training': 'for training purposes',
            'sales': 'for sales team',
            'design': 'for design department',
            'field': 'for field operations',
            'remote': 'for remote workforce',
            'employee': 'for employee use',
            'staff': 'for staff use'
        }

        purpose_found = False
        for keyword, purpose in purpose_keywords.items():
            if keyword in text_lower:
                parts.append(purpose)
                purpose_found = True
                break

        if not purpose_found and category:
            if 'IT Equipment' in category:
                parts.append("for IT infrastructure")
            elif 'Software' in category or 'Cloud' in category:
                parts.append("for business operations")
            elif 'Mobile' in category:
                parts.append("for mobile workforce")
            elif 'Office' in category or 'Furniture' in category:
                parts.append("for workspace enhancement")
            else:
                parts.append("for business use")

        # Value indicator
        if amount:
            if amount >= 50000:
                parts.append("- major investment")
            elif amount >= 10000:
                parts.append("- significant procurement")

        return " ".join(parts)


class ModelBasedVendorNormalizer:
    """Model-based vendor normalization using embeddings"""

    def __init__(self, known_vendors, similarity_threshold=0.65):
        self.known_vendors = list(known_vendors)
        self.similarity_threshold = similarity_threshold

        self.vendor_embeddings = {}
        for vendor in self.known_vendors:
            variations = self._create_vendor_variations(vendor)
            embeddings = model.encode(variations, convert_to_tensor=True)
            self.vendor_embeddings[vendor] = embeddings

    def _create_vendor_variations(self, vendor_name):
        """Create different variations of vendor name for better matching"""
        variations = [vendor_name]

        if " " in vendor_name:
            first_word = vendor_name.split()[0]
            if len(first_word) > 2:
                variations.append(first_word)

        suffixes = ['Inc.', 'Corp.', 'Corporation', 'Ltd.', 'LLC', 'Co.', 'Group', 'Technologies', 'Systems']
        clean_name = vendor_name
        for suffix in suffixes:
            if suffix in clean_name:
                clean_name = clean_name.replace(suffix, '').strip()
                variations.append(clean_name)

        variations.extend([v.lower() for v in variations])

        return list(set(variations))

    def normalize_vendor_name(self, text, detected_vendor_fragment=None):
        """Normalize vendor name using model-based similarity matching"""
        potential_vendors = self._extract_vendor_candidates(text)

        if detected_vendor_fragment:
            potential_vendors.append(detected_vendor_fragment)

        best_match = None
        best_score = 0.0

        for candidate in potential_vendors:
            if len(candidate) < 2:
                continue

            candidate_embedding = model.encode(candidate, convert_to_tensor=True)

            for vendor, vendor_embeddings in self.vendor_embeddings.items():
                similarities = util.cos_sim(candidate_embedding, vendor_embeddings)
                max_similarity = similarities.max().item()

                if max_similarity > best_score and max_similarity >= self.similarity_threshold:
                    best_match = vendor
                    best_score = max_similarity

        return best_match, best_score

    def _extract_vendor_candidates(self, text):
        """Extract potential vendor names from text"""
        candidates = []

        from_patterns = [
            r'\bfrom\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
            r'\bby\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
            r'\bsupplied\s+by\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
        ]

        for pattern in from_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            candidates.extend(matches)

        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON']:
                candidates.append(ent.text)

        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        candidates.extend(capitalized_words)

        cleaned = []
        stop_words = {'from', 'by', 'the', 'and', 'or', 'purchase', 'order'}

        for candidate in candidates:
            candidate = candidate.strip()
            if (len(candidate) > 2 and
                    candidate.lower() not in stop_words and
                    not candidate.isdigit() and
                    not re.match(r'^PO-\d+$', candidate, re.IGNORECASE)):
                cleaned.append(candidate)

        return cleaned


def extract_amount_and_quantity(text: str):
    """Enhanced extraction that distinguishes between amounts and quantities"""
    text_clean = text.strip()
    po_pattern = r'\bPO-\d+\b'
    text_without_po = re.sub(po_pattern, '', text_clean, flags=re.IGNORECASE)

    amount = None
    quantity = None

    # Look for monetary amounts
    currency_patterns = [
        r'[\$£€¥]\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:USD|EUR|GBP|dollars?|euros?|pounds?)\b',
        r'(\d+(?:\.\d+)?)\s*([KkMm])\b',
    ]

    for pattern in currency_patterns:
        matches = re.findall(pattern, text_without_po, re.IGNORECASE)
        if matches:
            if isinstance(matches[0], tuple) and len(matches[0]) == 2:
                num, unit = matches[0]
                val = float(num)
                if unit.lower() == 'm':
                    val *= 1_000_000
                elif unit.lower() == 'k':
                    val *= 1_000
                amount = int(val)
            else:
                amount_str = matches[-1].replace(',', '') if isinstance(matches[0], str) else matches[-1][0].replace(
                    ',', '')
                amount = int(float(amount_str))
            break

    # Look for quantities
    quantity_patterns = [
        r'\b(?:purchase|order|buy|acquire)\s+(?:of\s+)?(\d+)\s+(?:laptops?|computers?|monitors?|desks?|chairs?)',
        r'\b(\d+)\s+(?:laptops?|computers?|monitors?|desks?|chairs?|units?|pieces?|items?)',
        r'\bquantity[:\s]+(\d+)',
        r'\bqty[:\s]+(\d+)',
    ]

    for pattern in quantity_patterns:
        matches = re.findall(pattern, text_clean, re.IGNORECASE)
        if matches:
            quantity = int(matches[0])
            break

    return amount, quantity


def extract_amount(text: str):
    """Wrapper for backward compatibility"""
    amount, _ = extract_amount_and_quantity(text)
    return amount


def extract_quantity(text: str):
    """Extract quantity"""
    _, quantity = extract_amount_and_quantity(text)
    return quantity


def extract_product(text: str):
    """Extract product information using NLP"""
    text_clean = text.strip()

    noise_patterns = [
        r'PO-\d+', r'\d+[KkMm]\b', r'[\$£€¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?',
        r'invoice\s+\d+', r'\bfrom\s+\w+', r'\bby\s+\w+',
    ]

    for pattern in noise_patterns:
        text_clean = re.sub(pattern, ' ', text_clean, flags=re.IGNORECASE)

    text_clean = ' '.join(text_clean.split())

    product_indicators = {
        'laptop', 'laptops', 'computer', 'computers', 'desktop', 'desktops',
        'monitor', 'monitors', 'screen', 'screens', 'display', 'displays',
        'phone', 'phones', 'smartphone', 'smartphones', 'mobile', 'tablet', 'tablets',
        'server', 'servers', 'printer', 'printers', 'chair', 'chairs', 'desk', 'desks',
        'software', 'license', 'licenses', 'subscription', 'hosting', 'cloud',
    }

    doc = nlp(text_clean)

    for token in doc:
        if token.text.lower() in product_indicators:
            return token.text

    for chunk in doc.noun_chunks:
        chunk_words = set(chunk.text.lower().split())
        if chunk_words.intersection(product_indicators):
            return chunk.text

    for token in doc:
        if token.pos_ == 'NOUN' and len(token.text) > 2:
            return token.text

    return "Unknown Product"


def detect_mode(text: str) -> str:
    """Mode detection function"""
    return model_based_detect_mode(text)


def model_based_detect_mode(text: str, vendor_normalizer=None) -> str:
    """Model-based mode detection"""
    text_lower = text.lower().strip()

    has_po_number = bool(re.search(r'\bpo-\d+\b', text_lower))
    has_quantity_product = bool(re.search(r'\b\d+\s+(?:laptops?|computers?|monitors?|desks?|chairs?)\b', text_lower))
    has_purchase_language = bool(re.search(r'\b(?:purchase|order|buy|acquired?|procured?)\b', text_lower))
    has_vendor_pattern = bool(re.search(r'\b(?:from|by|supplied by)\s+\w+', text_lower))

    has_generic_service = bool(
        re.search(r'^\s*(?:consulting|audit|training|hosting|cloud|software)\s+(?:services?|costs?|fees?)\b',
                  text_lower))
    has_invoice_pattern = bool(re.search(r'invoice\s+(?:from|by)|\s+-\s+invoice\s+\d+', text_lower))

    normalization_score = 0
    missing_data_score = 0

    if has_po_number:
        normalization_score += 1
    if has_quantity_product:
        normalization_score += 1
    if has_purchase_language:
        normalization_score += 1
    if has_vendor_pattern:
        normalization_score += 1

    if has_generic_service:
        missing_data_score += 2
    if has_invoice_pattern:
        missing_data_score += 1

    words = text_lower.split()
    if len(words) <= 4:
        missing_data_score += 0.5
    elif len(words) >= 7:
        normalization_score += 0.5

    return "normalization" if normalization_score > missing_data_score else "missing-data"


def improved_vendor_prediction(text, vendor_embeds, fallback_vendors=None):
    """Enhanced vendor prediction"""
    if not vendor_embeds:
        return None, 0.0

    q_vec = model.encode(text, convert_to_tensor=True)
    best_vendor, vendor_score = None, -1

    for vendor, emb in vendor_embeds.items():
        if emb.dim() > 1:
            scores = util.cos_sim(q_vec, emb)
            score = scores.max().item()
        else:
            score = util.cos_sim(q_vec, emb).item()

        if score > vendor_score:
            best_vendor, vendor_score = vendor, score

    return best_vendor, vendor_score


def normalize_pipeline(text, vendor_embeds, category_embeds):
    """Enhanced normalization pipeline"""
    best_vendor, vendor_score = improved_vendor_prediction(text, vendor_embeds)

    q_vec = model.encode(text, convert_to_tensor=True)
    best_cat, cat_score = None, -1
    for c, emb in category_embeds.items():
        if emb.dim() > 1:
            scores = util.cos_sim(q_vec, emb)
            score = scores.max().item()
        else:
            score = util.cos_sim(q_vec, emb).item()

        if score > cat_score:
            best_cat, cat_score = c, score

    amount, quantity = extract_amount_and_quantity(text)

    return {
        "mode": "normalization",
        "normalized_vendor": best_vendor,
        "vendor_confidence": round(float(vendor_score), 3),
        "predicted_category": best_cat,
        "category_confidence": round(float(cat_score), 3),
        "product": extract_product(text),
        "amount": amount,
        "quantity": quantity
    }


def missing_pipeline(text, category_embeds, vendor_embeds):
    """Enhanced missing-data pipeline"""
    best_vendor, vendor_score = improved_vendor_prediction(text, vendor_embeds)

    q_vec = model.encode(text, convert_to_tensor=True)
    best_cat, cat_score = None, -1
    for c, emb in category_embeds.items():
        if emb.dim() > 1:
            scores = util.cos_sim(q_vec, emb)
            score = scores.max().item()
        else:
            score = util.cos_sim(q_vec, emb).item()

        if score > cat_score:
            best_cat, cat_score = c, score

    amount, quantity = extract_amount_and_quantity(text)

    return {
        "mode": "missing-data",
        "predicted_vendor": best_vendor,
        "vendor_confidence": round(float(vendor_score), 3),
        "predicted_category": best_cat,
        "category_confidence": round(float(cat_score), 3),
        "product": extract_product(text),
        "amount": amount,
        "quantity": quantity
    }


class EnhancedUnifiedPipeline:
    """Enhanced Pipeline with model-based vendor normalization and enriched descriptions"""

    def __init__(self, support_df, vendor_col="gold_vendor_normalized",
                 category_col="gold_category", raw_col="RawInputStyle"):

        logger.info("Initializing Enhanced Unified Pipeline with Enriched Descriptions...")

        self.vendors = support_df.groupby(vendor_col)[raw_col].apply(list).to_dict()
        self.categories = support_df.groupby(category_col)[raw_col].apply(list).to_dict()

        # Initialize model-based vendor normalizer
        self.vendor_normalizer = ModelBasedVendorNormalizer(
            known_vendors=list(self.vendors.keys()),
            similarity_threshold=0.6
        )

        logger.info("Creating vendor embeddings...")
        self.vendor_embeds = {}
        for vendor, examples in self.vendors.items():
            sample_size = min(10, len(examples))
            sample_examples = examples[:sample_size]
            if sample_examples:
                embeddings = model.encode(sample_examples, convert_to_tensor=True)
                self.vendor_embeds[vendor] = embeddings

        logger.info("Creating category embeddings...")
        self.category_embeds = {}
        for category, examples in self.categories.items():
            sample_size = min(10, len(examples))
            sample_examples = examples[:sample_size]
            if sample_examples:
                embeddings = model.encode(sample_examples, convert_to_tensor=True)
                self.category_embeds[category] = embeddings

        # Initialize enriched description generator
        logger.info("Initializing enriched description generator...")
        try:
            self.description_generator = EnrichedDescriptionGenerator(
                model_name="google/flan-t5-base"  # Change to "google/flan-t5-large" for better quality
            )
        except Exception as e:
            logger.error(f"Failed to load description generator: {e}")
            self.description_generator = None

        logger.info(
            f"Enhanced Pipeline loaded: {len(self.vendor_embeds)} vendors, {len(self.category_embeds)} categories")

    def predict_single(self, text: str):
        if not text or pd.isna(text):
            return {
                "mode": "error",
                "error": "Empty or null input",
                "normalized_vendor": None,
                "predicted_category": None,
                "product": None,
                "amount": None,
                "quantity": None,
                "enriched_description": "No data available"
            }

        try:
            mode = detect_mode(text)

            # Enhanced vendor prediction
            normalized_vendor, norm_score = self.vendor_normalizer.normalize_vendor_name(text)
            emb_vendor, emb_score = improved_vendor_prediction(text, self.vendor_embeds)

            # Choose best vendor result
            if norm_score > 0.7:
                best_vendor, vendor_score = normalized_vendor, norm_score
            elif emb_score > 0.4:
                best_vendor, vendor_score = emb_vendor, emb_score
            elif normalized_vendor:
                best_vendor, vendor_score = normalized_vendor, norm_score
            else:
                best_vendor, vendor_score = emb_vendor, emb_score

            # Category prediction
            q_vec = model.encode(text, convert_to_tensor=True)
            best_cat, cat_score = None, -1
            for c, emb in self.category_embeds.items():
                if emb.dim() > 1:
                    scores = util.cos_sim(q_vec, emb)
                    score = scores.max().item()
                else:
                    score = util.cos_sim(q_vec, emb).item()

                if score > cat_score:
                    best_cat, cat_score = c, score

            amount, quantity = extract_amount_and_quantity(text)
            product = extract_product(text)

            # Generate enriched description
            enriched_desc = "Standard procurement item"
            if self.description_generator:
                try:
                    enriched_desc = self.description_generator.generate_enriched_description(
                        text=text,
                        product=product,
                        category=best_cat,
                        vendor=best_vendor,
                        quantity=quantity,
                        amount=amount
                    )
                except Exception as e:
                    logger.error(f"Error generating enriched description: {e}")

            return {
                "mode": mode,
                "normalized_vendor" if mode == "normalization" else "predicted_vendor": best_vendor,
                "vendor_confidence": round(float(vendor_score), 3),
                "predicted_category": best_cat,
                "category_confidence": round(float(cat_score), 3),
                "product": product,
                "amount": amount,
                "quantity": quantity,
                "enriched_description": enriched_desc
            }

        except Exception as e:
            logger.error(f"Error processing text '{text[:50]}...': {str(e)}")
            return {
                "mode": "error",
                "error": str(e),
                "normalized_vendor": None,
                "predicted_category": None,
                "product": None,
                "amount": None,
                "quantity": None,
                "enriched_description": "Processing error"
            }

    def predict_batch(self, data):
        results = []
        total = len(data)

        for i, text in enumerate(data):
            if i % 10 == 0:
                logger.info(f"Processing {i}/{total} records...")

            result = self.predict_single(text)
            results.append(result)

        logger.info(f"Completed processing {total} records")
        return pd.DataFrame(results)


class UnifiedPipeline:
    """Standard Pipeline for backward compatibility"""

    def __init__(self, support_df, vendor_col="gold_vendor_normalized",
                 category_col="gold_category", raw_col="RawInputStyle"):

        logger.info("Initializing Standard Unified Pipeline...")

        self.vendors = support_df.groupby(vendor_col)[raw_col].apply(list).to_dict()
        self.categories = support_df.groupby(category_col)[raw_col].apply(list).to_dict()

        logger.info("Creating vendor embeddings...")
        self.vendor_embeds = {}
        for vendor, examples in self.vendors.items():
            sample_size = min(5, len(examples))
            sample_examples = examples[:sample_size]
            if sample_examples:
                embeddings = model.encode(sample_examples, convert_to_tensor=True)
                self.vendor_embeds[vendor] = embeddings

        logger.info("Creating category embeddings...")
        self.category_embeds = {}
        for category, examples in self.categories.items():
            sample_size = min(5, len(examples))
            sample_examples = examples[:sample_size]
            if sample_examples:
                embeddings = model.encode(sample_examples, convert_to_tensor=True)
                self.category_embeds[category] = embeddings

        logger.info(
            f"Standard Pipeline loaded: {len(self.vendor_embeds)} vendors, {len(self.category_embeds)} categories")

    def predict_single(self, text: str):
        if not text or pd.isna(text):
            return {
                "mode": "error",
                "error": "Empty or null input",
                "normalized_vendor": None,
                "predicted_category": None,
                "product": None,
                "amount": None
            }

        try:
            mode = detect_mode(text)

            if mode == "normalization":
                result = normalize_pipeline(text, self.vendor_embeds, self.category_embeds)
            else:
                result = missing_pipeline(text, self.category_embeds, self.vendor_embeds)

            if 'quantity' in result:
                del result['quantity']

            return result

        except Exception as e:
            logger.error(f"Error processing text '{text[:50]}...': {str(e)}")
            return {
                "mode": "error",
                "error": str(e),
                "normalized_vendor": None,
                "predicted_category": None,
                "product": None,
                "amount": None
            }

    def predict_batch(self, data):
        results = []
        total = len(data)

        for i, text in enumerate(data):
            if i % 10 == 0:
                logger.info(f"Processing {i}/{total} records...")

            result = self.predict_single(text)
            results.append(result)

        logger.info(f"Completed processing {total} records")
        return pd.DataFrame(results)