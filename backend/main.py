from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io
import traceback
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pipeline components
try:
    from procurement_pipeline import (
        EnhancedUnifiedPipeline,
        UnifiedPipeline,
        extract_amount_and_quantity,
        extract_product,
        detect_mode
    )

    PIPELINE_AVAILABLE = True
    logger.info("✅ Pipeline modules imported successfully")
except ImportError as e:
    PIPELINE_AVAILABLE = False
    logger.error(f"❌ Failed to import pipeline: {e}")

app = FastAPI(
    title="Enhanced Procurement Spend Classifier API",
    version="2.0.0",
    description="AI-powered procurement classification with enriched descriptions"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipe = None
pipeline_type = "None"


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipe, pipeline_type

    if not PIPELINE_AVAILABLE:
        logger.error("Pipeline not available. Check procurement_pipeline.py")
        return

    try:
        logger.info("Loading support dataset...")

        # Try to load dataset
        try:
            support_df = pd.read_csv("structured_spend_mixed.csv")
            logger.info(f"Loaded dataset: {len(support_df)} records")
        except FileNotFoundError:
            logger.warning("Dataset not found, creating sample data")
            support_df = create_sample_dataset()

        # Initialize Enhanced Pipeline
        logger.info("Initializing Enhanced Pipeline...")
        pipe = EnhancedUnifiedPipeline(
            support_df,
            vendor_col="gold_vendor_normalized",
            category_col="gold_category",
            raw_col="RawInputStyle"
        )
        pipeline_type = "Enhanced"

        vendor_count = len(pipe.vendor_embeds) if hasattr(pipe, 'vendor_embeds') else 0
        category_count = len(pipe.category_embeds) if hasattr(pipe, 'category_embeds') else 0

        logger.info(f"✅ Pipeline ready: {vendor_count} vendors, {category_count} categories")

    except Exception as e:
        logger.error(f"Startup error: {e}")
        traceback.print_exc()


def create_sample_dataset():
    """Create sample training data"""
    return pd.DataFrame({
        'RawInputStyle': [
            # IT Equipment - with vendors
            'Dell Latitude Laptops PO-1234 15K',
            'HP ProBook Computers PO-5678 12K',
            'Apple MacBook Pro Purchase 25K',
            'Lenovo ThinkPad Order PO-9012 18K',
            'Microsoft Surface Tablets 8K',

            # Software & Cloud
            'Microsoft Office 365 Licenses 5K',
            'Adobe Creative Cloud Subscription 3K',
            'Salesforce CRM License PO-3456 10K',
            'AWS Cloud Hosting Services 15K',
            'Oracle Database License 20K',

            # Mobile Devices
            'iPhone 14 Pro Purchase 30K',
            'Samsung Galaxy Smartphones 12K',
            'iPad tablets for training 8K',

            # Office Equipment
            'Herman Miller Office Chairs 6K',
            'Standing Desks Purchase 4K',
            'Conference Room Equipment 10K',

            # Missing-data mode examples
            'Consulting services - invoice 4581',
            'Professional services - invoice 9999',
            'Equipment maintenance - invoice 1234',
            'Cloud services - invoice 5678',
            'Software licensing - invoice 7890',
        ],
        'gold_vendor_normalized': [
            # Vendors
            'Dell Technologies Inc.',
            'HP Inc.',
            'Apple Inc.',
            'Lenovo Group Limited',
            'Microsoft Corporation',

            'Microsoft Corporation',
            'Adobe Inc.',
            'Salesforce Inc.',
            'Amazon Web Services',
            'Oracle Corporation',

            'Apple Inc.',
            'Samsung Electronics',
            'Apple Inc.',

            'Herman Miller Inc.',
            'Steelcase Inc.',
            'Cisco Systems Inc.',

            'Accenture',
            'Deloitte',
            'IBM',
            'Amazon Web Services',
            'Microsoft Corporation',
        ],
        'gold_category': [
            # Categories
            'IT Equipment > Laptops (UNSPSC: 43211503)',
            'IT Equipment > Laptops (UNSPSC: 43211503)',
            'IT Equipment > Laptops (UNSPSC: 43211503)',
            'IT Equipment > Laptops (UNSPSC: 43211503)',
            'Mobile Devices > Tablets (UNSPSC: 43191502)',

            'IT Services > Software',
            'IT Services > Software',
            'IT Services > Software',
            'IT Services > Cloud Services',
            'IT Services > Software',

            'Mobile Devices > Smartphones (UNSPSC: 43191501)',
            'Mobile Devices > Smartphones (UNSPSC: 43191501)',
            'Mobile Devices > Tablets (UNSPSC: 43191502)',

            'Office Equipment > Furniture',
            'Office Equipment > Furniture',
            'IT Equipment > Network Equipment',

            'Professional Services > Consulting',
            'Professional Services > Consulting',
            'IT Services > Maintenance',
            'IT Services > Cloud Services',
            'IT Services > Software',
        ]
    })


@app.get("/")
async def root():
    """API information"""
    features = []
    if pipeline_type == "Enhanced":
        features = [
            "Enriched description generation using LLM",
            "Model-based vendor normalization",
            "Advanced amount and quantity extraction",
            "Product identification with NLP",
            "Confidence scoring for predictions",
            "Support for both CSV and text input"
        ]

    return {
        "service": "Procurement Spend Classifier API",
        "version": "2.0.0",
        "status": "ready" if pipe else "initializing",
        "pipeline_type": pipeline_type,
        "features": features,
        "endpoints": {
            "health": "/health",
            "predict_batch": "/predict (POST)",
            "predict_text": "/predict-text (POST)",
            "predict_single": "/predict_single (POST)",
            "test": "/test_extraction",
            "stats": "/stats",
            "sample": "/sample_data"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    if pipe is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "message": "Pipeline not initialized"}
        )

    vendor_count = len(pipe.vendor_embeds) if hasattr(pipe, 'vendor_embeds') else 0
    category_count = len(pipe.category_embeds) if hasattr(pipe, 'category_embeds') else 0
    has_description_gen = hasattr(pipe, 'description_generator') and pipe.description_generator is not None

    return {
        "status": "healthy",
        "pipeline_type": pipeline_type,
        "vendors_loaded": vendor_count,
        "categories_loaded": category_count,
        "enriched_descriptions": has_description_gen,
        "model_name": pipe.description_generator.model_name if has_description_gen else "N/A"
    }


@app.post("/predict")
async def predict_batch(file: UploadFile):
    """Batch prediction from CSV file"""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Find text column
        text_col = find_text_column(df.columns)
        if not text_col:
            raise HTTPException(
                status_code=400,
                detail=f"No valid text column found. Expected: text, RawInputStyle, description. Found: {list(df.columns)}"
            )

        logger.info(f"Processing {len(df)} records from column '{text_col}'")

        # Process data
        results = pipe.predict_batch(df[text_col].tolist())

        # Clean results
        results = clean_dataframe(results)
        results['original_text'] = df[text_col].tolist()

        # Reorder columns for better readability
        column_order = [
            'original_text', 'mode', 'enriched_description',
            'normalized_vendor', 'predicted_vendor', 'vendor_confidence',
            'predicted_category', 'category_confidence',
            'product', 'amount', 'quantity'
        ]
        results = results[[col for col in column_order if col in results.columns]]

        logger.info(f"✅ Processed {len(results)} records")

        return {
            "status": "success",
            "pipeline_type": pipeline_type,
            "processed_count": len(results),
            "source_column": text_col,
            "results": results.to_dict(orient="records")
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-text")
async def predict_text(request: dict):
    """Text input prediction endpoint - supports single or multi-line text"""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    if "text" not in request:
        raise HTTPException(status_code=400, detail="Request must contain 'text' field")

    text = request["text"]

    try:
        logger.info(f"Processing text input: {text[:100]}...")

        # Split text into lines if multiple entries
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        if not lines:
            raise HTTPException(status_code=400, detail="No valid text provided")

        # Process each line
        results = []
        for line in lines:
            result = pipe.predict_single(line)
            result['original_text'] = line
            results.append(clean_dict(result))

        logger.info(f"✅ Processed {len(results)} text entries")

        return {
            "status": "success",
            "pipeline_type": pipeline_type,
            "processed_count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Text prediction error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_single")
async def predict_single(request: dict):
    """Single text prediction"""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    if "text" not in request:
        raise HTTPException(status_code=400, detail="Request must contain 'text' field")

    text = request["text"]

    try:
        logger.info(f"Processing: {text[:100]}...")
        result = pipe.predict_single(text)

        return {
            "status": "success",
            "pipeline_type": pipeline_type,
            "original_text": text,
            "result": clean_dict(result)
        }
    except Exception as e:
        logger.error(f"Single prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test_extraction")
async def test_extraction():
    """Test extraction functions with examples"""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    test_cases = [
        {
            "text": "Purchase 10 laptops from Dell for $15000",
            "description": "Should extract quantity, amount, and vendor"
        },
        {
            "text": "Microsoft Office 365 subscription PO-1234 5K",
            "description": "Should detect normalization mode with amount"
        },
        {
            "text": "Consulting services - invoice 9876",
            "description": "Should detect missing-data mode"
        },
        {
            "text": "Order 20 Samsung monitors at $300 each",
            "description": "Should extract both quantity and unit price"
        },
        {
            "text": "Apple MacBook Pro for development team",
            "description": "Should identify product and purpose"
        }
    ]

    results = []
    for case in test_cases:
        text = case["text"]
        try:
            # Test extraction functions
            amount, quantity = extract_amount_and_quantity(text)
            product = extract_product(text)
            mode = detect_mode(text)

            # Full pipeline result
            full_result = pipe.predict_single(text)

            results.append({
                "input": text,
                "description": case["description"],
                "extraction": {
                    "amount": amount,
                    "quantity": quantity,
                    "product": product,
                    "mode": mode
                },
                "pipeline_result": clean_dict(full_result)
            })
        except Exception as e:
            results.append({
                "input": text,
                "description": case["description"],
                "error": str(e)
            })

    return {
        "pipeline_type": pipeline_type,
        "test_cases": results,
        "summary": {
            "total": len(test_cases),
            "successful": len([r for r in results if "error" not in r])
        }
    }


@app.get("/stats")
async def get_statistics():
    """Pipeline statistics"""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    vendors = list(pipe.vendors.keys())[:10] if hasattr(pipe, 'vendors') else []
    categories = list(pipe.categories.keys())[:10] if hasattr(pipe, 'categories') else []

    return {
        "pipeline_type": pipeline_type,
        "total_vendors": len(pipe.vendors) if hasattr(pipe, 'vendors') else 0,
        "total_categories": len(pipe.categories) if hasattr(pipe, 'categories') else 0,
        "sample_vendors": vendors,
        "sample_categories": categories,
        "has_enriched_descriptions": hasattr(pipe, 'description_generator')
    }


@app.get("/sample_data")
async def get_sample_data():
    """Sample data format"""
    return {
        "csv_format": {
            "required_columns": ["text or RawInputStyle or description"],
            "example": "text\n\"Purchase 10 laptops from Dell\"\n\"Microsoft Office license 5K\""
        },
        "json_text_format": {
            "endpoint": "/predict-text",
            "example": {
                "text": "Purchase 10 laptops from Dell for development team\nMicrosoft Office 365 subscription 5K"}
        },
        "json_single_format": {
            "endpoint": "/predict_single",
            "example": {"text": "Purchase 10 laptops from Dell for development team"}
        },
        "test_examples": [
            "Dell Latitude Laptops PO-1234 15K",
            "Purchase 10 monitors from Samsung",
            "Consulting services - invoice 4581",
            "Microsoft Office 365 subscription $5000"
        ]
    }


# Helper functions
def find_text_column(columns):
    """Find the text column in dataframe"""
    possible_cols = ['text', 'RawInputStyle', 'description', 'raw_text', 'input_text']
    for col in possible_cols:
        if col in columns:
            return col
    return None


def clean_dataframe(df):
    """Clean dataframe for JSON serialization"""
    df = df.replace({pd.NA: None, float('nan'): None, float('inf'): None, -float('inf'): None})
    df = df.where(pd.notnull(df), None)
    return df


def clean_dict(d):
    """Clean dictionary for JSON serialization"""
    cleaned = {}
    for k, v in d.items():
        if pd.isna(v) or v in [float('inf'), -float('inf')]:
            cleaned[k] = None
        else:
            cleaned[k] = v
    return cleaned


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Enhanced Procurement Classifier API")
    print("📍 Server: http://127.0.0.1:8000")
    print("📖 Docs: http://127.0.0.1:8000/docs")
    print("📝 Supports: CSV upload & direct text input")

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")