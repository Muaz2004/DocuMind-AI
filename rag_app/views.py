import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .services.rag_engine import query_rag


@csrf_exempt
def query_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"}, status=400)

    try:
        data = json.loads(request.body)
        question = data.get("question")

        if not question:
            return JsonResponse({"error": "Question is required"}, status=400)

        results = query_rag(question)

        return JsonResponse({
            "question": question,
            "results": results
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    


import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage

from .services.rag_engine import index_uploaded_pdf


UPLOAD_DIR = "rag_app/uploads"


@csrf_exempt
def upload_pdf(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    if "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    file = request.FILES["file"]

    fs = FileSystemStorage(location=UPLOAD_DIR)
    filename = fs.save(file.name, file)
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Index the uploaded PDF
    index_uploaded_pdf(file_path)

    return JsonResponse({
        "message": "File uploaded and indexed successfully",
        "file": filename
    })

