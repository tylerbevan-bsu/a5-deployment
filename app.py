from flask import Flask, make_response
from datetime import datetime
app = Flask(__name__)


model = None
trained = 'None'

@app.route('/')
def homepage():
    the_time = datetime.now().strftime("%A, %d %b %Y %l:%M %p")

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
    """.format(trained=trained)

@app.route('/trainw2v')
def train_w2v():
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
    if model is None:
        return make_response("""
            <h1>Model Not Trained</h1>
            <p>Please Train a Model</p>
            <form method="get" action="/">
                <button type="submit">Go Home</button>
            </form>
            """, 400)        
    pass

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
