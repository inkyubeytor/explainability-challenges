from django.urls import path

from . import views

urlpatterns = [
    path("", views.start, name="base"),
    path("graphic", views.graphic, name="graphic"),
    path("start", views.start, name="start"),
    path("submit", views.submit, name="submit"),
    path("result", views.result, name="result"),
]
