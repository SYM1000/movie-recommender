from flask.scaffold import F
from movie_recommenderv3 import get_recommendation
from flask import Flask
from flask_restful import Resource, Api, reqparse
import os
from movie_recommenderv3 import get_recommendation

app = Flask(__name__)
api = Api(app)

class Recommendation(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('movie', required=True)
        args = parser.parse_args()

        recommendations = get_recommendation(args['movie'])

        if recommendations == False: # If movie was not found
            return{
                'message': f"'{args['movie']}' was not found"
            }, 401

        return recommendations, 200



api.add_resource(Recommendation, '/recommendation')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))