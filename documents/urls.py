from django.urls import path
from . import views

urlpatterns = [
    #path('', views.home, name='home'),
    path('upload/', views.upload_document, name='upload_document'),
    path('list/', views.document_list, name='document_list'),
    path('signup/', views.signup, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('index/', views.home, name='index'),
]