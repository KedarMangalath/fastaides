# chatbot/urls.py
from django.contrib import admin
from django.urls import path
from chat import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.chat_view, name='chat'),
    path('process_message/', views.process_message, name='process_message'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)