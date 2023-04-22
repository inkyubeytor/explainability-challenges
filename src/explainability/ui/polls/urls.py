from django.urls import path

from . import views

urlpatterns = [
    # ex: /polls/
    path("", views.index, name="index"),
    # ex: /polls/5/
    path("graphic", views.graphic, name="graphic"),
    path("start", views.start, name="start"),
    path("submit", views.submit, name="submit"),
    path("result", views.result, name="result"),
]
