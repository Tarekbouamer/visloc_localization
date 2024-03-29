

from flask import Flask, jsonify, request
import json

from PIL import Image

#declared an empty variable for reassignment
response = ''

#creating the instance of our flask application
app = Flask(__name__)

#route to entertain our post and get request from flutter app
@app.route('/', methods = ['GET', 'POST'])
def nameRoute():

    #fetching the global response variable to manipulate inside the function
    global response

    #checking the request type we get from the app
    if(request.method == 'POST'):
        # file = request.files #getting the response data
        file = request.files
        print(file)
        
        image = Image.open(file["image"]).resize((320, 320))
        print(image.size)
        image.show()

        # image = Image.open(file).resize((32, 32))
        # print(request_data['name'])
        # request_data = json.loads(request_data.decode('utf-8')) #converting it from json to key value pair
        # print(request_data)
        # name = request_data['name'] #assigning it to name
        # print(name)
        # response = f'Hi {name}! this is Python' #re-assigning response with the name we got from the user
        return " " #to avoid a type error 
    else:
        print(response)
        return jsonify({'name' : response}) #sending data back to your frontend app


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
