from django.urls import path
from .views import predict_fake_reviews_api, health_check, home  # Updated function name

urlpatterns = [
    path('', home, name='home'),
    path('predict/', predict_fake_reviews_api, name='predict_fake_reviews_api'),
    path('health/', health_check),
]
