from os import getenv

# server
HOST = getenv('HOST', "0.0.0.0")
PORT = int(getenv('PORT', 8000))