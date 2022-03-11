from os import getenv

# server
HOST = getenv('HOST', 'localhost')
PORT = int(getenv('PORT', 8000))