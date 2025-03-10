from django.urls import path
from .views import predict_fake_reviews_api, health_check  # Updated function name

urlpatterns = [
    path('predict/', predict_fake_reviews_api, name='predict_fake_reviews_api'),
    path('health/', health_check),
]
