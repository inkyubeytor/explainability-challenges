from django.urls import path

from . import views

urlpatterns = [
    path("", views.start, name="base"),
    path("graphic", views.graphic, name="graphic"),
    path("demo", views.demo, name="demo"),
    path("demo_graphic", views.demo_graphic, name="demo_graphic"),
    path("start", views.start, name="start"),
    path("submit", views.submit, name="submit"),
    path("submit_demo", views.submit_demo, name="submit_demo"),
    path("result", views.result, name="result"),
]
