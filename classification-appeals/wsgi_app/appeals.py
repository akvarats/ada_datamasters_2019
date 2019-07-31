import json
import falcon
from wsgi_app.classifier import classify


class IndexResource:

    def on_get(self, req, resp):
        resp.status = falcon.HTTP_301
        resp.set_header('Location', '/static/index.html')


class ClassifierResource:

    def on_post(self, req, resp):
        predicted = classify(req.media['appeal_text'])
        resp.body = json.dumps(predicted, ensure_ascii=False)


def add_routes(app):
    app.add_route('/', IndexResource())
    app.add_route('/classify', ClassifierResource())
    return app


app = falcon.API()
app = add_routes(app)

# from wsgiref.simple_server import make_server
# with make_server('', 8081, app) as httpd:
#     print('port 8081')
#     httpd.serve_forever()
