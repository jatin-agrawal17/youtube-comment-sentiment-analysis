# app.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
from dotenv import load_dotenv
import os
from pathlib import Path
import google.generativeai as genai
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile
from datetime import datetime
import uuid
import textwrap
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors



env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
print("DEBUG → YOUTUBE_API_KEY loaded:", bool(YOUTUBE_API_KEY))

if not YOUTUBE_API_KEY:
    raise RuntimeError("YOUTUBE_API_KEY not found in environment")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)

# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def clean_text(text: str) -> str:
    """Remove unicode characters that ReportLab cannot render"""
    return text.encode("ascii", "ignore").decode()


def draw_heading(c, text, y):
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, text)
    return y - 25


def draw_paragraph(c, text, y):
    c.setFont("Helvetica", 11)
    for line in textwrap.wrap(text, 90):
        if y < 60:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = 800
        c.drawString(50, y, line)
        y -= 14
    return y - 10


def add_page_number(c):
    c.setFont("Helvetica", 9)
    c.drawRightString(570, 20, f"Page {c.getPageNumber()}")

def render_structured_text(c, text, y, page_height):
    sections = ["Executive Summary", "Key Insights", "Recommendations", "Conclusion"]

    for i, section in enumerate(sections):
        if section not in text:
            continue

        # Split at current section
        after = text.split(section, 1)[1].strip()

        # Find where the next section starts
        next_index = len(after)
        for next_section in sections[i + 1:]:
            pos = after.find(next_section)
            if pos != -1:
                next_index = pos
                break

        paragraph = after[:next_index].strip()

        # Draw section heading
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, section)
        y -= 22

        # Draw paragraph
        c.setFont("Helvetica", 11)
        for line in textwrap.wrap(paragraph, 95):
            if y < 60:
                add_page_number(c)
                c.showPage()
                c.setFont("Helvetica", 11)
                y = page_height - 60
            c.drawString(50, y, line)
            y -= 14

        y -= 18

    return y



# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    # Set MLflow tracking URI to your server
    mlflow.set_tracking_uri("http://ec2-3-93-194-48.compute-1.amazonaws.com:5000/")  # Replace with your MLflow tracking URI
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)  # Load the vectorizer
    return model, vectorizer

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "1", "./tfidf_vectorizer.pkl")  # Update paths and versions as needed


import requests

@app.route('/fetch-comments', methods=['POST'])
def fetch_comments():
    data = request.get_json()
    video_id = data.get("video_id")

    if not video_id:
        return jsonify({"error": "video_id required"}), 400

    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,
        "textFormat": "plainText",
        "key": YOUTUBE_API_KEY
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        yt_data = r.json()

        comments = []
        for item in yt_data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "text": snippet["textDisplay"],
                "timestamp": snippet["publishedAt"]
            })

        return jsonify({"comments": comments})

    except Exception as e:
        app.logger.error(f"YouTube API error: {e}")
        return jsonify({"error": "Failed to fetch comments"}), 500


@app.route('/')
def home():
    return "Welcome to the flask api"

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Make predictions
        predictions = model.predict(transformed_comments).tolist()  # Convert to list
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Make predictions
        predictions = model.predict(transformed_comments).tolist()  # Convert to list
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500
    
@app.route("/generate_pdf_report", methods=["POST"])
def generate_pdf_report():
    data = request.json

    # ---------- GEMINI PROMPT ----------
    prompt = f"""
Write a professional YouTube Comment Sentiment Analysis Report in plain text English.

IMPORTANT INSTRUCTIONS:
- Do NOT use Markdown
- Do NOT use bullet points
- Do NOT use symbols like -, *, ##, ###, ---
- Write only normal sentences and paragraphs
- Separate sections using a blank line
- Use simple section titles followed by paragraphs

Use the following structure EXACTLY:

Executive Summary
Write one clear paragraph summarizing the overall sentiment and engagement.

Key Insights
Write one or two paragraphs explaining major sentiment trends, audience reactions, and engagement patterns.

Recommendations
Write a paragraph suggesting improvements or next steps based on the analysis.

Conclusion
Write a short concluding paragraph summarizing the findings.

Here is the analysis data you must use:

Total Comments: {data['summary']['totalComments']}
Unique Commenters: {data['summary']['uniqueCommenters']}
Average Comment Length: {data['summary']['avgWordLength']} words
Average Sentiment Score: {data['summary']['normalizedSentimentScore']}/10

Sentiment Distribution:
{data['sentimentCounts']}

Sample Positive Comments:
{data['sentimentSamples']['positive']}

Sample Negative Comments:
{data['sentimentSamples']['negative']}
"""

    model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
    response = model.generate_content(prompt)

    if not response or not response.text:
        return jsonify({"error": "Gemini returned empty response"}), 500

    safe_text = clean_text(response.text)

    # ---------- FILE PATH ----------
    filename = f"youtube_report_{uuid.uuid4().hex}.pdf"
    path = os.path.join(tempfile.gettempdir(), filename)

    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4

    # ---------- COVER PAGE ----------
    c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(width / 2, height - 200,
                        "YouTube Comment Sentiment Analysis")

    c.setFont("Helvetica", 14)
    c.drawCentredString(width / 2, height - 240,
                        "AI-Generated Analytical Report")

    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, height - 300,
                        f"Generated on: {datetime.now().strftime('%d %B %Y')}")

    add_page_number(c)
    c.showPage()

    # ---------- METRICS TABLE ----------
    table_data = [
        ["Metric", "Value"],
        ["Total Comments", data["summary"]["totalComments"]],
        ["Unique Commenters", data["summary"]["uniqueCommenters"]],
        ["Avg Comment Length", f'{data["summary"]["avgWordLength"]} words'],
        ["Avg Sentiment Score", f'{data["summary"]["normalizedSentimentScore"]}/10']
    ]

    table = Table(table_data, colWidths=[260, 200])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("FONT", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (1,1), (-1,-1), "CENTER")
    ]))

    table.wrapOn(c, width, height)
    table.drawOn(c, 80, height - 320)

    add_page_number(c)
    c.showPage()

    # ---------- SENTIMENT PIE CHART ----------
    labels = ["Positive", "Neutral", "Negative"]
    values = [
        data["sentimentCounts"]["1"],
        data["sentimentCounts"]["0"],
        data["sentimentCounts"]["-1"]
    ]

    plt.figure()
    plt.pie(values, labels=labels, autopct="%1.1f%%")
    chart_path = os.path.join(tempfile.gettempdir(), "sentiment_chart.png")
    plt.savefig(chart_path)
    plt.close()

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 80, "Sentiment Distribution")
    c.drawImage(chart_path, 100, height - 450, width=350, height=350)

    add_page_number(c)
    c.showPage()

    # ---------- REPORT TEXT ----------
    y = height - 60
    y = render_structured_text(c, safe_text, y, height)

    add_page_number(c)
    c.save()

    return send_file(
        path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="YouTube_Comment_Analysis_Report.pdf"
    )



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)