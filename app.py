from flask import Flask, make_response
from model import Model

app = Flask(__name__)


model = Model()

@app.route('/')
def homepage():
    global model

    return """
    <h1>Sentiment Decider</h1>
    <p>Current Model: {trained}</p>
    <form method="get" action="/trainw2v">
        <button type="submit">Train Word 2 Vec</button>
    </form>
    <form method="get" action="/trainonehot">
        <button type="submit">Train One-Hot</button>
    </form>
    <form method="get" action="/test">
        <button type="submit">Test Current Model</button>
    </form>
    """.format(trained=model.trained)

@app.route('/trainw2v')
def train_w2v():
    global model
    model.train_w2v()
    return """
        <h1>Trained Word 2 Vec</h1>
        <form method="get" action="/test">
            <button type="submit">Test Model</button>
        </form>
        <form method="get" action="/">
            <button type="submit">Go Home</button>
        </form>
        """ 

@app.route('/trainonehot')
def train_onehot():
    global model
    model.train_onehot()
    return """
        <h1>Trained One-Hot</h1>
        <form method="get" action="/test">
            <button type="submit">Test Model</button>
        </form>
        <form method="get" action="/">
            <button type="submit">Go Home</button>
        </form>
        """ 

@app.route('/test')
def test():
    global model
    if model.model is None:
        return make_response("""
            <h1>Model Not Trained</h1>
            <p>Please Train a Model</p>
            <form method="get" action="/">
                <button type="submit">Go Home</button>
            </form>
            """, 400)
    score = model.test_model()
    return """
    <h1>Test Results</h1>
    <p>Model: {trained}</p>
    <p>Score: {score}</p>
    <form method="get" action="/">
        <button type="submit">Go Home</button>
    </form>
    """.format(trained=model.trained, score=score)

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
