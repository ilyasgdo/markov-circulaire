from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('api/resumes_circulaires/<str:id_variable>/<str:unite>', views.get_resumes),
    path('api/moyenne_unite/<str:id_variable>/<str:unite>', views.get_moyennes_unite),
    path('views/recalculer-intervalle/', views.update_interval, name='recalculer_intervalle'),
    path('views/afficherGraphe/', views.update_interval, name='afficherGraphe'),
]
