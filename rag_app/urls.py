from django.urls import path
from .views import query_view, upload_pdf

urlpatterns = [
    path("upload/", upload_pdf),
    path("query/", query_view),
]
