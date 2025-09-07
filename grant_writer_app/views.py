from django.http import JsonResponse, HttpResponse
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from transformers import pipeline
import torch
import os
import json
import requests
import logging

from django.http import HttpResponse
import sentry_sdk

# Loading of model with the help of transformer
try:
    print("Loading model pipeline...")
    model_pipeline = pipeline("text-generation", model="daryl149/llama-2-7b-chat-hf", device="cpu", trust_remote_code=True)
    print("Model pipeline loaded successfully!")
except Exception as e:
    print(f"Error loading model pipeline: {e}")
    model_pipeline = None

# View to generate text based on the provided prompt
@csrf_exempt
def generate_response(request):
    if request.method == "POST":
        try:
            # Parse input data
            data = json.loads(request.body)
            print("👉 Payload from React:", data)  # Debug log

            # Collect fields from frontend
            grant_title = data.get("grant_title", "")
            objective = data.get("objective", "")
            audience = data.get("audience", "")
            funding = data.get("funding", "")
            details = data.get("details", "")
            transcription = data.get("transcription", "")

            # Build prompt string
            prompt = f"""
            Grant Title: {grant_title}
            Objective: {objective}
            Audience: {audience}
            Funding: {funding}
            Details: {details}
            Extra Notes: {transcription}
            """.strip()

            # Validate
            if not any([grant_title, objective, audience, funding, details, transcription]):
                return JsonResponse({"error": "No input provided"}, status=400)

            # Check if model pipeline is available
            if model_pipeline:
                response = model_pipeline(prompt, max_length=200, do_sample=True)
                generated_text = response[0].get("generated_text", "")
                return JsonResponse({"response": generated_text}, status=200)
            else:
                # For now, just return mock text (since torch not installed)
                return JsonResponse({"response": f"✅ Backend received data and built prompt:\n{prompt}"}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Only POST method is allowed"}, status=405)


# Dummy form view (to handle requests for testing)
def grant_proposal_form(request):
    return HttpResponse("<h1>Grant Proposal Form</h1><p>Send a POST request to /generate/ with a 'prompt'.</p>")

# View for the user interface to interact with the model
def grant_writer_ui(request):
    response_data = None
    if request.method == "POST":
        prompt = request.POST.get("prompt", "").strip()
        if prompt:
            # Interact with the model's API endpoint
            api_url = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000/grant_writer/generate/")
            headers = {"Content-Type": "application/json"}
            payload = {"prompt": prompt}
            try:
                api_response = requests.post(api_url, json=payload, headers=headers)
                if api_response.status_code == 200:
                    response_data = api_response.json().get("response", "No response generated.")
                else:
                    response_data = f"Error: {api_response.status_code} - {api_response.text}"
            except requests.RequestException as e:
                response_data = f"An error occurred: {e}"

    return render(request, "grant_writer_ui.html", {"response": response_data})


