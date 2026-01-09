"""
Gradio app for phishing detection using URLScan.io and Hopsworks.

This app:
1. Loads a trained model from Hopsworks Model Registry
2. Takes a URL as input
3. Scans it using URLScan.io API
4. Extracts features from the scan results
5. Runs inference using the loaded model
6. Returns whether the URL is likely phishing or legitimate
"""

import os
import logging
import gradio as gr
from typing import Tuple

from phising_detection.inference import PhishingDetectionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global inference pipeline
pipeline = None


def initialize_app():
    """Initialize the app by loading the inference pipeline."""
    global pipeline

    try:
        # Get URLScan API key
        urlscan_api_key = os.getenv("URLSCAN_API_KEY")

        # Initialize pipeline
        pipeline = PhishingDetectionPipeline(
            model_name="phishing_detector",
            model_version=None,  # Use latest version
            urlscan_api_key=urlscan_api_key
        )

        # Load model from Hopsworks
        pipeline.load_model_from_hopsworks()
        logger.info("Inference pipeline initialized successfully!")

        return True
    except Exception as e:
        logger.error(f"Failed to initialize app: {e}")
        return False


def gradio_interface(url: str) -> Tuple[str, str, str]:
    """
    Gradio interface function.

    Args:
        url: URL to analyze

    Returns:
        Tuple of (result_html, confidence_html, details_html)
    """
    if not url or not url.strip():
        return "Please enter a URL", "", ""

    # Clean URL
    url = url.strip()

    # Add http:// if no protocol specified
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    # Run prediction using pipeline
    result = pipeline.predict_url(url)

    # Check for errors
    if "error" in result:
        return (
            f'<h2 style="color: orange;">ERROR</h2>',
            "",
            f"<p><strong>Error:</strong> {result['error']}</p>"
        )

    # Format result with color
    prediction = result["prediction"]
    confidence = result["confidence"]

    if prediction == "PHISHING":
        result_html = f'<h2 style="color: red;">PHISHING</h2>'
        color = "red"
    elif prediction == "LEGITIMATE":
        result_html = f'<h2 style="color: green;">LEGITIMATE</h2>'
        color = "green"
    else:
        result_html = f'<h2 style="color: orange;">UNKNOWN</h2>'
        color = "orange"

    confidence_html = f'<h3 style="color: {color};">Confidence: {confidence * 100:.2f}%</h3>'

    # Format details
    details_html = f"""
    <h4>Prediction Details:</h4>
    <ul>
        <li><strong>Phishing Probability:</strong> {result['phishing_probability'] * 100:.2f}%</li>
        <li><strong>Legitimate Probability:</strong> {result['legitimate_probability'] * 100:.2f}%</li>
        <li><strong>URLScan UUID:</strong> {result.get('scan_uuid', 'N/A')}</li>
    </ul>

    <h4>Extracted Features:</h4>
    <ul>
        <li><strong>Domain Age (days):</strong> {result['features'].get('domain_age_days', 'N/A')}</li>
        <li><strong>Secure Percentage:</strong> {result['features'].get('secure_percentage', 'N/A')}%</li>
        <li><strong>Has Umbrella Rank:</strong> {'Yes' if result['features'].get('has_umbrella_rank') else 'No'}</li>
        <li><strong>Umbrella Rank:</strong> {result['features'].get('umbrella_rank', 'N/A')}</li>
        <li><strong>Has TLS:</strong> {'Yes' if result['features'].get('has_tls') else 'No'}</li>
        <li><strong>TLS Valid Days:</strong> {result['features'].get('tls_valid_days', 'N/A')}</li>
        <li><strong>URL Length:</strong> {result['features'].get('url_length', 'N/A')}</li>
        <li><strong>Subdomain Count:</strong> {result['features'].get('subdomain_count', 'N/A')}</li>
    </ul>
    """

    return result_html, confidence_html, details_html


def create_gradio_app():
    """Create and configure the Gradio interface."""

    # Custom CSS for better styling
    css = """
    .output-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    """

    with gr.Blocks(css=css, title="Phishing URL Detection") as demo:
        gr.Markdown(
            """
            # Phishing URL Detection

            Enter a URL to check if it's a phishing website or legitimate.
            This app uses URLScan.io to analyze the website and a machine learning model
            trained on URLScan features to predict if it's phishing.

            **Note:** Scanning a URL can take up to 90 seconds as we wait for URLScan.io to complete the analysis.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                url_input = gr.Textbox(
                    label="URL to Check",
                    placeholder="Enter URL (e.g., example.com or https://example.com)",
                    lines=1
                )
            with gr.Column(scale=1):
                submit_btn = gr.Button("Check URL", variant="primary", size="lg")

        with gr.Row():
            result_output = gr.HTML(label="Prediction")

        with gr.Row():
            confidence_output = gr.HTML(label="Confidence")

        with gr.Row():
            details_output = gr.HTML(label="Details")

        # Example URLs
        gr.Markdown("### Example URLs to Try:")
        gr.Examples(
            examples=[
                ["https://google.com"],
                ["https://github.com"],
                ["https://facebook.com"],
            ],
            inputs=url_input,
        )

        # Connect button to function
        submit_btn.click(
            fn=gradio_interface,
            inputs=url_input,
            outputs=[result_output, confidence_output, details_output]
        )

        gr.Markdown(
            """
            ---
            **Disclaimer:** This tool is for educational and research purposes only.
            The predictions are not 100% accurate and should not be the sole basis for security decisions.
            """
        )

    return demo


def main():
    """Main function to run the Gradio app."""
    logger.info("Starting Phishing Detection App...")

    # Initialize app (load model and URLScan client)
    logger.info("Initializing app...")
    if not initialize_app():
        logger.error("Failed to initialize app. Exiting.")
        return

    # Create and launch Gradio app
    logger.info("Creating Gradio interface...")
    demo = create_gradio_app()

    logger.info("Launching Gradio app...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
