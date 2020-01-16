from django.shortcuts import render
from django.http import HttpResponse
from .models import Posts

# Create your views here.
def index(request):
    posts = Posts.objects.all()[:10]

    context = {
        'title':'Latest posts',
        'posts':posts
    }
    return render(request,'posts/index.html',context)

def details(request, id):
    post = Posts.objects.get(id=id)

    contex = {
        'post':post
    }

    return render(request, 'posts/details.html',contex)