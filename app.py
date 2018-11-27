from flask import Flask, make_response, Response, session
from flask_session import Session
from model import Model

app = Flask(__name__)

SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

app.secret_key = 'Some super secret string that nobody should ever know'

@app.route('/')
def homepage():
    session['model'] = Model()

    return """
    <h1>Sentiment Decider</h1>
    <p>Current Model: {trained}</p>
    <form method="get" action="/trainw2v">
        <button type="submit">Train Word 2 Vec</button>
    </form>
    <form method="get" action="/trainonehot">
        <button type="submit">Train One-Hot</button>
    </form>
    """.format(trained=session['model'].trained)

@app.route('/trainw2v')
def train_w2v():
    session['model'].train_w2v()
    return """
        <h1>Trained Word 2 Vec</h1>
        <form method="get" action="/test">
            <button type="submit">Test Model</button>
        </form>
        <form method="get" action="/">
            <button type="submit">Restart</button>
        </form>
        """ 

@app.route('/trainonehot')
def train_onehot():
    session['model'].train_onehot()
    return """
        <h1>Trained One-Hot</h1>
        <form method="get" action="/test">
            <button type="submit">Test Model</button>
        </form>
        <form method="get" action="/">
            <button type="submit">Restart</button>
        </form>
        """ 

@app.route('/test')
def test():
    if not 'model' in session:
        return make_response("""
            <h1>Model Not Trained</h1>
            <p>Please Train a Model</p>
            <form method="get" action="/">
                <button type="submit">Restart</button>
            </form>
            """, 400)
    if session['model'] is None:
        return make_response("""
            <h1>Model Not Trained</h1>
            <p>Please Train a Model</p>
            <form method="get" action="/">
                <button type="submit">Restart</button>
            </form>
            """, 400)
    score = session['model'].test_model()
    return """
    <h1>Test Results</h1>
    <p>Model: {trained}</p>
    <p>Score: {score}</p>
    <img src='/plot.png' alt='Plot'>
    <form method="get" action="/">
        <button type="submit">Restart</button>
    </form>
    """.format(trained=session['model'].trained, score=score)

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

@app.route('/plot.png')
def make_plot():
    return Response(session['model'].plot_roc(), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
