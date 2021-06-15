from movie_recommenderv4 import get_recommendation_server
from flask import Flask
from flask_restful import Resource, Api, reqparse
import os

app = Flask(__name__)
api = Api(app)

class Recommendation(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('movie', required=True)
        parser.add_argument('sorted', required=False) # need fix
        args = parser.parse_args()

        if args['sorted'] == "true": # need fix
            recommendations = get_recommendation_server(args['movie'], True)
        else:
            recommendations = get_recommendation_server(args['movie'], False)

        if recommendations == False: # If movie was not found
            return{
                'message': f"'{args['movie']}' was not found"
            }, 401

        return recommendations, 200



api.add_resource(Recommendation, '/recommendation')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))